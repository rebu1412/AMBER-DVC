import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
from dataclasses import dataclass, field
import copy

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.ops.triton.layernorm import RMSNorm
from transformers import AutoTokenizer
print(Mamba)

# print(RMSNorm)
# https://github.com/state-spaces/mamba/issues/229

# d_model: 768
# n_layer: 24
# vocab_size: 50277
# ssm_cfg: {}
# rms_norm: True
# residual_in_fp32: True
# fused_add_norm: True
# pad_vocab_size_multiple: int = 8

@dataclass
class MambaConfig:

    d_model: int = 768  #############
    d_intermediate: int = 0
    n_layer: int = 24  #############
    vocab_size: int = 50277  #############
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True   #############
    residual_in_fp32: bool = True   #############
    fused_add_norm: bool = True   #############
    pad_vocab_size_multiple: int = 8   #############
    tie_embeddings: bool = True

def create_block(
    d_model,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(
        Mamba, 
        layer_idx=layer_idx,
        **factory_kwargs
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(module, n_layer, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

class MambaCaptioner(nn.Module):
    def __init__(
            self, 
            opt: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ) -> None:
        super(MambaCaptioner, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.opt = opt

        self.d_model = opt.d_model # Add to cfg
        self.n_layer = opt.n_layer
        self.vocab_size = opt.vocab_size + 1  # +1 for <EOS> token
        self.rms_norm = opt.rms_norm
        self.residual_in_fp32 = opt.residual_in_fp32
        self.fused_add_norm = opt.fused_add_norm
        self.pad_vocab_size_multiple = opt.pad_vocab_size_multiple

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size = self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)

        self.input_encoding_size = opt.input_encoding_size
        self.drop_prob_lm = opt.drop_prob
        self.max_caption_len = opt.max_caption_len
        self.ss_prob = 0.0  # Schedule sampling probability

        # Embedding layer for tokens
        self.embedding = nn.Embedding(self.vocab_size, self.input_encoding_size)
        
        # Input projection: combine token embedding and video features (hs)
        self.input_projection = nn.Linear(self.input_encoding_size + opt.hidden_dim, self.d_model)
        
        # Mamba backbone
        self.layers = nn.ModuleList([
            create_block(
                d_model=self.d_model,
                norm_epsilon=1e-5,
                rms_norm=self.rms_norm,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
                layer_idx=i,
                **factory_kwargs
            ) for i in range(self.n_layer)
        ])
        self.norm_f = (nn.LayerNorm if not self.rms_norm else RMSNorm)(
            self.d_model, eps=1e-5, **factory_kwargs
        )

        # Output head
        self.logit = nn.Linear(self.d_model, self.vocab_size, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        
        self.apply(partial(_init_weights, n_layer=self.n_layer))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

    def build_loss(self, input, target, mask):
        '''
        input: [match_pairs, max_caption_length, vocab_size]
        target: [match_pairs, max_caption_length]
        mask: [match_pairs]
        '''
        one_hot = F.one_hot(target, self.vocab_size).float()
        max_len = input.shape[1]
        output = -(one_hot[:, :max_len] * input * mask[:, :max_len, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output

    def forward(self, hs, cap_tensor):
        '''
        hs: [bs, match_pairs, hidden_dim]
        cap_tensor: [match_pairs, max_caption_length], tokenized
        '''
        seq = cap_tensor.long()
        vid_num, query_num, _ = hs.shape
        assert vid_num == 1
        
        # Flatten query for batch processing
        query = hs.view(-1, hs.shape[-1])  # [match_pairs, hidden_dim]
        outputs = []
        
        for i in range(seq.size(1) - 1): # max_caption_length
            if self.training and i >= 1 and self.ss_prob > 0.0:
                # sample from the prior output probability distribution instead of using the target token
                sample_prob = torch.zeros(query_num, device=hs.device).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    # Using gt, not using Schedule sampling, 
                    it = seq[:, i].clone()
                else:
                    # Using Schedule sampling
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].clone()
                    prob_prev = torch.exp(outputs[-1])
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                # SOS token
                it = seq[:, i].clone() # [num_gt_cap]
            
            if i >= 1 and seq[:, i].sum() == 0:  # break if all the sequences end
                break
            
            # [match_pairs, vocab_size]
            output, state = self.get_logprobs_state(it, query)
            outputs.append(output)
        
        # [match_pairs, max_seq_length, vocab_size]
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, query):
        '''
        it: [match_pairs], token IDs
        query: [match_pairs, hidden_dim], video query features
        '''
        xt = self.embedding(it)  # [match_pairs, input_encoding_size]
        input_feats = torch.cat([xt, query], dim=-1)  # [match_pairs, input_encoding_size + hidden_dim]
        hidden_states = self.input_projection(input_feats)  # [match_pairs, d_model]
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual)
        output = self.dropout(hidden_states) # [match_pairs, d_model]
        logprobs = F.log_softmax(self.logit(output), dim=-1)
        return logprobs, hidden_states

    def sample(self, hs, opt={}):
        '''
        hs: [bs, match_pairs, hidden_dim]
        opt: sampling options (sample_max, temperature)
        '''
        vid_num, query_num, _ = hs.shape
        assert vid_num == 1
        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)
        
        query = hs.view(-1, hs.shape[-1])  # [match_pairs, hidden_dim]
        seq = []
        seqLogprobs = []
        
        it = torch.zeros(query_num, dtype=torch.long, device=hs.device)  # <BOS>
        
        for t in range(self.max_caption_len + 1):
            logprobs, state = self.get_logprobs_state(it, query)
            
            if t >= 1:
                if sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                else:
                    prob_prev = torch.exp(logprobs / temperature)
                    it = torch.multinomial(prob_prev, 1).view(-1)
                    sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)).view(-1)
                
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs)
        
        if not seq:
            return [], []
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class Opt:
    d_model: int = 768  # Mamba d_model
    n_layer: int = 24  # Number of Mamba layers
    vocab_size: int = 50277  # Matches EleutherAI/gpt-neox-20b or GPT-2 tokenizer
    input_encoding_size = 512  # Token embedding size
    hidden_dim = 512  # Dimension of hs (video query features)
    drop_prob = 0.5  # Dropout probability
    max_caption_len = 20  # Maximum caption length
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8

if __name__ == '__main__': # python -m pdvc.CaptioningHead.mamba
    opt = Opt()

    # Load tokenizer (mamba-130m typically uses a GPT-2-like tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or "<|endoftext|>"

    # Create dummy inputs
    batch_size = 1  # Must be 1 as per MambaCaptioner assertion
    match_pairs = 3  # Number of events/regions to caption
    hidden_dim = opt.hidden_dim
    max_caption_len = opt.max_caption_len
    hs = torch.rand(batch_size, match_pairs, hidden_dim).to('cuda')

    # Dummy captions (YouCook2-like)
    captions = [
        "Sauté the onions in a pan",
        "Bake the cake at 350 degrees",
        "Chop the vegetables finely"
    ]
    cap_tensor = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_caption_len,
        return_tensors="pt"
    )["input_ids"].to('cuda')

    print(cap_tensor.shape)
    print(hs.shape)

    # # Initialize MambaCaptioner
    captioner = MambaCaptioner(opt).to('cuda' if torch.cuda.is_available() else 'cpu')
    print(captioner)

    # Test forward pass (training mode)
    # captioner.train()
    # outputs = captioner(hs, cap_tensor)
    # print(f"Forward output shape: {outputs.shape}")  # Expected: [match_pairs, max_caption_len-1, vocab_size]

    # # Compute loss
    # loss = captioner.build_loss(outputs, cap_tensor[:, 1:], mask[:, 1:])
    # print(f"Training loss: {loss.item()}")

    # # Test sample method (inference mode)
    # captioner.eval()
    # seq, seq_logprobs = captioner.sample(hs, opt={'sample_max': 1, 'temperature': 1.0})
    # print(f"Sampled sequence shape: {seq.shape}")  # Expected: [match_pairs, generated_length]
    # print(f"Sampled logprobs shape: {seq_logprobs.shape}")  # Expected: [match_pairs, generated_length]

    # # Decode generated captions
    # generated_captions = [tokenizer.decode(s, skip_special_tokens=True) for s in seq]
    # print(f"Generated captions: {generated_captions}")
