import torch
import torch.nn as nn
from functools import partial
from typing import List
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba as ViM
    from mamba_ssm.modules.mamba_new import Mamba as DBM
except:
    print('Please install mamba_ssm before using Mamba encoder')

class Decoder(nn.Module):
    def __init__(self,
                 num_layers=3,
                 n_res_gr=2,
                 FSM_block = 1,
                 d_model=512, 
                 d_ffn=1024,
                 d_state: int = 16,
                 d_conv: int = 2,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 activation="silu",
                 norm_type="layernorm",
                 dropout_rate=0.1,
                 num_feature_levels=4
                 ):

        super().__init__()
        self.num_layers = num_layers
        self.n_res_gr = n_res_gr
        self.FSM_block = FSM_block
        self.d_model = d_model
        self.hidden_dim = d_ffn
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.layers = nn.ModuleList([
            DecoderLayer(
                n_res_gr=n_res_gr,
                d_model=d_model, 
                d_ffn=d_ffn,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                activation=activation,
                norm_type=norm_type,
                dropout_rate=dropout_rate,
            ) for _ in range(num_layers)
        ])

        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize level_embed
        nn.init.normal_(self.level_embed) # Positional embeddings often initialized from normal distribution

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Parameter) and m is self.res_scale:
                nn.init.constant_(m, 0.1)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings if provided."""
        return tensor if pos is None else tensor + pos
    
    def prepare_decoder_inputs(self, srcs, masks, pos_embeds, query_embed):
        '''
        Prepares input tensors for an encoder by processing multi-level feature inputs.
        Flattens source features, masks, and positional embeddings across levels, adds
        level-specific positional embeddings, and computes valid ratios and level indices.

        Args:
            srcs (list[torch.Tensor]): List of source feature tensors, each of shape (bs, d, f).
            masks (list[torch.Tensor]): List of binary mask tensors, each of shape (bs, f).
            pos_embeds (list[torch.Tensor]): List of positional embedding tensors, each of shape (bs, d_m, f).
            query_embed [torch.Tensor]: [num_queries, d]

        Returns:
            tuple: Contains the following:
                - src_flatten (torch.Tensor): (bs, L1+L2+..., c).
                - temporal_shapes (torch.Tensor): (num_level,).
                - level_start_index (torch.Tensor): Start indices for each level, shape (num_level,).
                - valid_ratios (torch.Tensor): Valid ratios for each level, shape (bs, num_level).
                - lvl_pos_embed_flatten (torch.Tensor): Flattened positional embeddings, shape (bs, L1+L2+..., d_m).
                - mask_flatten (torch.Tensor): Flattened masks, shape (bs, L1+L2+...).
        '''
        srcs_list = []
        query_idx_list = []
        mask_flatten = []
        lvl_pos_embed_flatten = []

        bs, _, _ = srcs[0].shape
        num_queries = query_embed.shape[0] 
        queries_tokens = query_embed.unsqueeze(0).repeat(bs, 1, 1).unsqueeze(2) # [bs, num_queries, 1, d]

        # Iterate Over feature Levels
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, dim, f = src.shape
            group = f//num_queries
            src = src.transpose(1, 2).view(bs, num_queries, group, dim) # [bs, num_queries, (f//num_queries), d]
            src_query = torch.cat([src, queries_tokens], dim=2) # [bs, num_queries, (f//num_queries)+1, d]
            src_query = src_query.view(bs, -1, dim)[:, :-1, :] # [bs, f+num_queries-1, dim]
            # queries index
            indices = torch.arange(group, src_query.shape[1], group+1)
            
            # Add Level-Specific Positional Embedding
            pos_embed = pos_embed.transpose(1, 2).view(bs, num_queries, group, dim) #（bs, Fx, d）
            query_pos_embed = torch.zeros(bs, num_queries, 1, dim, dtype=pos_embed.dtype, device=pos_embed.device)
            pos_embed = torch.cat([pos_embed, query_pos_embed], dim=2).view(bs, -1, dim)[:, :-1, :]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) #（bs, L, d_m）+ (1, 1, d_m)

            srcs_list.append(src_query)
            query_idx_list.append(indices)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(srcs_list, 1)  # (bs, L1+L2+L3+L4, c)
        mask_flatten = torch.cat(mask_flatten, 1)  # (bs, L1+L2+L3+L4)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (bs, L1+L2+L3+L4, c)

        return srcs_list, query_idx_list, src_flatten, mask_flatten, lvl_pos_embed_flatten

    def forward(self, raw_res, full_res, pos=None):
        '''
        Args:
            raw_res (List[torch.Tensor]): List of resolution features, each of shape (B, F_i, d),
                                        where F_i in (F, F/2, F/4, F/8) represents temporal resolutions.
            full_res (torch.Tensor): Concatenated features of shape (B, F+F/2+F/4+F/8, d).
            pos (torch.Tensor, optional): Positional encodings of shape (B, F+F/2+F/4+F/8, d).
    
        Returns:
            res_list (List[List[torch.Tensor]]): List of resolution features for each layer.
            res_full_list (List[torch.Tensor]): List of full-resolution features for each layer.
        '''
        res_list = []
        res_full_list = []
        # layer-to-layer
        resolution = raw_res
        full_resolution = self.with_pos_embed(full_res, pos)
        # print(full_res)
        # print(full_resolution)
        # hehe
        for layer in self.layers:
            resolution, full_resolution = layer(resolution, full_resolution)
            res_list.append(resolution)
            res_full_list.append(full_resolution)

        return res_list, res_full_list

class DecoderLayer(nn.Module): 
    def __init__(self,
                 n_res_gr=2,
                 FSM_block = 2,
                 d_model=512, 
                 d_ffn=1024,
                 d_state: int = 16,
                 d_conv: int = 2,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 activation="silu",
                 norm_type="layernorm",
                 dropout_rate=0.1,
                 ):
        super().__init__()

        self.n_res_gr = n_res_gr
        self.FSM_block = FSM_block

        self.linear = nn.ModuleList([
            ffn(in_features=d_model, 
                hidden_features=d_ffn, 
                activation=activation,
                norm_type="layernorm", 
                dropout_rate=dropout_rate)
            for _ in range(n_res_gr)
        ])

        # FSMBlocks for each resolution group: n_res_gr groups, each with FSM_block blocks
        self.FSM_blocks = nn.ModuleList([
            nn.ModuleList([
                FSMBlock(d_model=d_model, 
                         d_ffn=d_ffn, 
                         d_state=d_state, 
                         d_conv=d_conv,
                         expand=expand, 
                         dt_rank=dt_rank, 
                         activation=activation, 
                         norm_type=norm_type,
                         dropout_rate=dropout_rate)
                for _ in range(FSM_block)
            ]) for _ in range(n_res_gr)
        ])

        # Cross-res communication block
        self.CTGA = CTGABlock(
            n_res_gr=n_res_gr,
            d_model=d_model,
            d_ffn=d_ffn,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
        )

    def forward(self, res, full_res):
        """
        Args:
            res: [(bs, F, d), (bs, F/2, d), (bs, F/4, d), (bs, F/8, d)]
            full_res: (bs, F+F/2+F/4+F/8, d), concatenated features
        """
        out_FSM = []

        # Each res GROUP
        for gr_id, indices in enumerate(range(0, len(res), 2)):
            res1, res2 = res[indices], res[indices+1]
            res1, res2 = self.linear[gr_id](res1), self.linear[gr_id](res2) # shared FFN
            for block_id in range(self.FSM_block):
                res1, res2 = self.FSM_blocks[gr_id][block_id](res1, res2)
            out_FSM.extend([res1, res2])
        
        # Cross-res communication
        out_res, out_full = self.CTGA(out_FSM, full_res)
        return out_res, out_full

class CTGABlock(nn.Module):        
    def __init__(self,
                 n_res_gr=2,
                 d_model=512, 
                 d_ffn=1024,
                 d_state: int = 16,
                 d_conv: int = 2,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 activation="silu",
                 norm_type="layernorm",
                 dropout_rate=0.1,
                 ):
        super().__init__()
        self.hidden_dim = d_model
        self.n_res_gr = n_res_gr
        self.scale = self.hidden_dim ** 0.5

        # Each Local res
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) if norm_type == "layernorm" else nn.RMSNorm(d_model)
            for _ in range(n_res_gr)
            ])
        self.ssm = nn.ModuleList([
            DBM(d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dt_rank=dt_rank, use_fast_path=True)
            for _ in range(n_res_gr)
        ])

        # Communication
        self.query = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_res_gr*2)
        ])

        self.ffn = nn.ModuleList([
            ffn(in_features=d_model, hidden_features=d_ffn, activation=activation,
            norm_type="layernorm", dropout_rate=dropout_rate)
            for _ in range(n_res_gr)
            ])
        self.softmax = nn.Softmax(dim=-1)

        # GLOBAL block
        self.norm_global = nn.LayerNorm(d_model) if norm_type == "layernorm" else nn.RMSNorm(d_model)
        self.ssm_global = DBM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            use_fast_path=True
            )
        self.ffn_global = ffn(
            in_features=d_model, 
            hidden_features=d_ffn, 
            activation=activation,
            norm_type="layernorm", 
            dropout_rate=dropout_rate
            )
        self.dropout = nn.Dropout(dropout_rate)

        # Resolution-to-global fusion
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)  # Learnable scaling
        self.norm_gate = nn.LayerNorm(d_model * 2) if norm_type == "layernorm" else nn.RMSNorm(d_model * 2)
        self.global_res_gate = nn.Linear(d_model * 2, 2)

    def forward(self, x_list: List[torch.Tensor], x_flatten: torch.Tensor):
        """
        Args:
            x_list: List of tensors, each shape (B, F, d)
                    where F in (f, f/2, f/4, f/8)
            x_flatten: (B, F_all, d)
            pos: positional encoding tensor or None
        Returns:
            out_list: List of tensors (B, L, d), fused with cross-task attention
        """
        split_indices = [(x_list[i].size(1), x_list[i+2].size(1)) for i in range(0, len(x_list)//2)]

        # Each resolution group PATH (0-2 | 1-3)
        resolution_list = []
        for i in range(self.n_res_gr):
            fused = torch.cat([x_list[i], x_list[i+2]], dim=1)
            fused_ = self.norms[i](fused)
            fused_ = self.ssm[i](fused_)
            fused_ = self.dropout(fused_)
            res1 =  fused_[:, :split_indices[i][0], :] + x_list[i]
            res2 =  fused_[:, split_indices[i][0]:, :] + x_list[i+2]
            resolution_list.extend([res1, res2])
        # Global PATH
        fused_global_ = self.norm_global(x_flatten)
        fused_global_ = self.ssm_global(fused_global_)
        fused_global_ = x_flatten + self.dropout(fused_global_) # (B, F_all, d)

        # Cross Attention
        queries = [self.query[i](res) for i, res in enumerate(resolution_list)]
        scores = [torch.matmul(q, fused_global_.transpose(-1, -2)) / self.scale for q in queries] # (B, F_i, F_all)
        attention = [self.softmax(s) for s in scores] # (B, F_i, F_all)
        out_list = [res + self.dropout(torch.matmul(attn, fused_global_)) for res, attn in zip(resolution_list, attention)]

        # FFN
        out = []
        for i in range(0, len(out_list), 2):
            res1 = self.ffn[i//2](out_list[i])
            res2 = self.ffn[i//2](out_list[i+1])
            out.extend([res1, res2])
        out = sorted(out, key=lambda x: x.shape[1], reverse=True)

        # GATED - Fused between Global and local resolution
        out_concat = torch.cat(out, dim=1)
        gate_input = self.norm_gate(torch.cat([fused_global_, out_concat], dim=-1)) # (B, F_all, d_model * 2)
        gate_scores = self.global_res_gate(gate_input)  # (B, F_all, 2)
        gate = torch.softmax(gate_scores, dim=-1)  # (B, F_all, 2)
        fused_global_ = gate[..., 0:1] * fused_global_ + self.res_scale * gate[..., 1:2] * self.dropout(out_concat)
        fused_global_ = self.dropout(self.ffn_global(fused_global_))

        return out, fused_global_

class FSMBlock(nn.Module):
    def __init__(self,
                 d_model=512, 
                 d_ffn=1024,
                 d_state: int = 16,
                 d_conv: int = 2,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 activation="silu",
                 norm_type="layernorm",
                 dropout_rate=0.1,
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model) if norm_type == "layernorm" else nn.RMSNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) if norm_type == "layernorm" else nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.ssm = DBM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            use_fast_path=True
            )
        
        self.ffn = ffn(
            in_features=d_model, 
            hidden_features=d_ffn, 
            activation=activation,
            norm_type="layernorm",
            dropout_rate=dropout_rate
            )

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        '''
        Args:
            - x: [B, f_resolution, d_model]
            - z: [B, f_resolution/2, d_model]
        '''
        fused = torch.cat([x, z], dim=1) # (Bs, 3/2*f_resolution, d_model)

        # shared ssm block
        fused_z = self.norm1(fused)
        fused_ = self.ssm(fused_z) # (Bs, 3/2*f_resolution, d_model)
        fused_ = self.dropout(fused_)
        fused = fused_z + fused_ # (Bs, 3/2*f_resolution, d_model)
        res1, res2 = torch.split(fused, [x.size(1), z.size(1)], dim=1)

        # FFN
        res1 = x + self.dropout(self.ffn(self.norm2(res1)))
        res2 = z + self.dropout(self.ffn(self.norm2(res2)))

        return res1, res2

class ffn(nn.Module):
    def __init__(self, 
                 in_features=512, 
                 hidden_features=1024, 
                 activation="silu",
                 norm_type="layernorm",
                 dropout_rate=0.):
        super().__init__()

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_features, in_features)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(in_features) if norm_type == "layernorm" else nn.RMSNorm(in_features)

    def forward(self, x):
        x_ = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = x + self.dropout2(x_)
        x = self.norm(x)
        return x
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    elif activation == "silu":
        return F.silu
    else:
        raise NotImplementedError(f"Activation {activation} not supported")
    
def build_decoder(args):
    return Decoder(
        num_layers=args.dec_layers,
        n_res_gr=2, 
        FSM_block=args.FSM_block, 
        d_model=args.hidden_dim,
        d_ffn=args.ff_dim, 
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_rank=args.dt_rank,
        activation=args.act,
        norm_type=args.norm,
        dropout_rate=args.dropout_rate 
    )