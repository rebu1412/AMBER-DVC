"""
Base Encoder to create multi-level conv features and positional embedding.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from misc.detr_utils.misc import NestedTensor


class BaseEncoder(nn.Module):
    def __init__(self, num_feature_levels, vf_dim, hidden_dim):
        super(BaseEncoder, self).__init__()
        self.pos_embed = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        if num_feature_levels > 1:
            # Preserves Temporal Resolution, Dimension Projection (aligning)
            # Provides a high-resolution feature map
            input_proj_list = []
            in_channels = vf_dim
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
            ))
            # Temporal Downsampling, Larger Receptive Field
            # frame/2: Captures medium-term events (e.g., a person walking for a few seconds).
            # frame/4: Captures longer-term events (e.g., an entire scene or activity).
            for _ in range(num_feature_levels - 1):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(vf_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, vf, mask, duration):
        # vf: (Batch, Frame, dim), 
        # mask: (Batch, Frame),  
        # duration: (Batch)
        srcs = []
        masks = []
        poses = []

        vf = vf.transpose(1, 2)  # (b, f, d) --> (b, d, f)
        vf_nt = NestedTensor(vf, mask, duration)
        pos0 = self.pos_embed(vf_nt) # designed to receive a NestedTensor
        src0, mask0 = vf_nt.decompose() # (b, d, f)

        # Level 1
        srcs.append(self.input_proj[0](src0)) # (b, hidden_dim, f)
        masks.append(mask0) # (b, f)
        poses.append(pos0) # (b, hidden_dim, f)
        assert mask is not None

        # Level 2, 3,...
        for l in range(1, self.num_feature_levels):
            if l == 1:
                src = self.input_proj[l](vf_nt.tensors) # [b, hidden_dim, f/2]
            else:
                src = self.input_proj[l](srcs[-1]) # [b, hidden_dim, f/4,8,16]
            m = vf_nt.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-1:]).to(torch.bool)[0]
            pos_l = self.pos_embed(NestedTensor(src, mask, duration)).to(src.dtype)
            srcs.append(src) # (B, hidden dim, F), (B, hidden dim, F/2), (B, hidden dim, F/4), (B, hidden dim, F/8)
            masks.append(mask) # (B, F), (B, F/2), (B, F/4), (B, F/8)
            poses.append(pos_l)

        return srcs, masks, poses

def build_base_encoder(args):
    base_encoder = BaseEncoder(args.num_feature_levels, args.feature_dim, args.hidden_dim)
    return base_encoder

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.max_duration = 256
        self.duration_embed_layer = nn.Linear(self.max_duration, self.max_duration)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors # [b, d, f]
        mask = tensor_list.mask # [b, d]
        duration = tensor_list.duration # [b,]
        assert mask is not None

        not_mask = ~mask # [b, d]
        x_embed = not_mask.cumsum(1, dtype=torch.float32) # [b, d]
        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:] + eps) * self.scale

        # pos for dim
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # (npf, )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # (npf, )
        pos_x = x_embed[:, :, None] / dim_t # [b, d, npf]
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        # [b, d, npf]

        # pos for frame
        dur_embed = self.duration_embedding(duration).reshape(-1,1,self.max_duration).expand_as(pos_x)
        # [b, d, npf]
        
        pos = torch.cat((pos_x, dur_embed), dim=2).permute(0, 2, 1)
        # [b, npf*2, d]
        return pos

    def duration_embedding(self, durations):
        out = torch.zeros(len(durations), self.max_duration, device=durations.device)
        durations = durations.int()
        for ii in range(len(durations)):
            out[ii, :durations[ii]] = 1
        out = self.duration_embed_layer(out)
        return out

def build_position_encoding(position_embedding, N_steps):
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding

if __name__ == '__main__':
    batch_size, seq_len, d_model = 2, 200, 3072
    vf = torch.rand(batch_size, seq_len, d_model).to('cuda')
    mask = torch.rand(batch_size, seq_len).to('cuda') > 0.5
    duration = torch.rand(batch_size).to('cuda')

    num_feature_levels, feature_dim, hidden_dim = 4, 3072, 512
    base_encoder = BaseEncoder(num_feature_levels, feature_dim, hidden_dim).to('cuda')

    srcs, masks, poss = base_encoder(vf, mask, duration)
    for src, mask, pos in zip(srcs, masks, poss):
        print(src.shape) #  [2, 512, 200]  [2, 512, 100]  [2, 512, 50]  [2, 512, 25]
        print(mask.shape) # [2, 200]       [2, 100]       [2, 50]       [2, 25]
        print(pos.shape) #  [2, 512, 200]  [2, 512, 100]  [2, 512, 50]  [2, 512, 25]
    # python -m pdvc.base_encoder
