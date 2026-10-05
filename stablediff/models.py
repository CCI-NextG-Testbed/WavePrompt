import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import complex.complex_module as cm


def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x):
        x = self.projection(x)
        embedding = torch.view_as_complex(
            self.embedding.to(x.device).contiguous()
        )
        return x * embedding

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table

class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, d_context=768, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.CosineComplexMultiHeadAttention(
            hidden_dim, num_heads, bias=True, eps=float(block_kwargs.get("eps", 1e-8)))

        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn_2 = cm.CosineComplexCrossAttention(
            n_heads=num_heads,
            d_embed=hidden_dim,
            d_cross=d_context,
            in_proj_bias=True,
            eps=float(block_kwargs.get("eps", 1e-8))
        )
        self.norm3 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 9*hidden_dim, bias=True)
        )

        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, context, t):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t).chunk(9, dim=1)

        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(mod_x)

        mod_x = modulate(self.norm2(x), shift_mca, scale_mca)
        x = x + gate_mca.unsqueeze(1) * self.attn_2(mod_x, context)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp))

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_zero)

    def forward(self, x, context, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
    
class tfdiff_Simple(nn.Module):

    def __init__(self, params, device):
        super().__init__()
        self.params = params
        self.device = device

        self.input_dim = params.input_dim
        self.output_dim = getattr(params, "output_dim", params.input_dim)
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.dropout = params.dropout
        self.mlp_ratio = params.mlp_ratio

        # Embeddings
        self.p_embed = PositionEmbedding(params.sample_rate, self.input_dim, self.hidden_dim) # Position Embedding for complex space preservation
        self.t_embed = DiffusionEmbedding(params.max_step, params.embed_dim, self.hidden_dim) # Time Embedding


        self.bits_token = nn.Linear(1, self.hidden_dim * 2)
        init_weight_xavier(self.bits_token)

        # Blocks + head
        self.blocks = nn.ModuleList(
            [DiA(self.hidden_dim, self.num_heads, self.dropout) for _ in range(params.num_block)]
        )
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)
    
    def _encode_bits_seq(self, bits, device, N):
        """
        bits: [B,N] 0/1 (or [B,N,1])
        return: [B,N,H,2]
        """
        if bits is None:
            return None
        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=torch.float32, device=device)
        else:
            bits = bits.to(device).float()

        if bits.ndim == 1:
            bits = bits.unsqueeze(0)  # [1,N]
        if bits.shape[1] != N:
            # If mismatch, you need a mapping from samples->symbols (oversampling etc.)
            # For now, truncate/pad as a safe fallback:
            if bits.shape[1] > N:
                bits = bits[:, :N]
            else:
                pad = torch.zeros(bits.shape[0], N - bits.shape[1], device=device)
                bits = torch.cat([bits, pad], dim=1)

        bits = bits.unsqueeze(-1)  # [B,N,1]
        B = bits.shape[0]
        b = self.bits_token(bits)              # [B,N,2H]
        b = b.view(B, N, self.hidden_dim, 2)   # [B,N,H,2]
        return b

    def forward(self, x, context, t, bits_input=None):
        device = x.device

        B, N = x.shape[0], x.shape[1]

        x = self.p_embed(x)

        t = self.t_embed(t)

        b_seq = self._encode_bits_seq(bits_input, device, N)
        if b_seq is not None:
            if not torch.is_complex(b_seq):
                b_seq = torch.view_as_complex(b_seq.contiguous())
            x = x + b_seq

        for block in self.blocks:
            x = block(x, context, t)

        x = self.final_layer(x, context, t)

        return x
