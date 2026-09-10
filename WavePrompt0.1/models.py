import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

import complex.complex_module as cm
import complex.complex_layers as cm_l


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
        return cm.complex_mul(x, self.embedding.to(x.device))

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table

class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        attn_eps = float(block_kwargs.get("eps", 1e-8))
        self.attn = cm.CosineComplexMultiHeadAttention(
            hidden_dim, num_heads, bias=True, eps=attn_eps)
        #self.attn = cm.ComplexMultiHeadAttention(
        #    hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(mod_x)
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
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

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
    
class tfdiff_Simple(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = torch.device("cpu")

        self.input_dim = params.input_dim
        self.output_dim = getattr(params, "output_dim", params.input_dim)
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.dropout = params.dropout
        self.mlp_ratio = params.mlp_ratio

        # Embeddings
        self.p_embed = PositionEmbedding(params.sample_rate, self.input_dim, self.hidden_dim) # Position Embedding for complex space preservation
        self.t_embed = DiffusionEmbedding(params.max_step, params.embed_dim, self.hidden_dim) # Time Embedding

        # Optional conditioning projection (real -> complex hidden)
        self.text_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        text_dim = self.text_encoder.get_sentence_embedding_dimension()
        # project real text embedding to complex [B, H, 2]
        self.text_proj = nn.Linear(text_dim, self.hidden_dim * 2)
        init_weight_xavier(self.text_proj)

        self.bits_token = nn.Linear(1, self.hidden_dim * 2)
        init_weight_xavier(self.bits_token)

        # Blocks + head
        self.blocks = nn.ModuleList(
            [DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(params.num_block)]
        )
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def _encode_text(self, prompts, device):
        """
        prompts: list[str] or already a tensor
        Returns: complex conditioning vector [B, H, 2]
        """
        if isinstance(prompts, (list, tuple)):
            # SentenceTransformer handles batching internally
            text_emb = self.text_encoder.encode(
                prompts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )   # [B, D_text], real
        elif isinstance(prompts, torch.Tensor):
            # assume already [B, D_text] real embeddings
            text_emb = prompts.to(device)
        else:
            # single string
            text_emb = self.text_encoder.encode(
                [prompts],
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )   # [1, D_text]

        B = text_emb.shape[0]
        # project to 2*hidden_dim and reshape to complex [B, H, 2]
        text_proj = self.text_proj(text_emb)              # [B, 2H]
        text_proj = text_proj.view(B, self.hidden_dim, 2) # [B, H, 2]
        return text_proj
    
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

    def forward(self, x, t, cond):
        device = x.device

        # cond is dict: {'prompt': label/list[str], 'bits_cond' or 'bits': [B,N]}
        prompt_input = cond.get("prompt") if isinstance(cond, dict) else cond
        bits_input   = None
        if isinstance(cond, dict):
            bits_input = cond.get("bits_cond", cond.get("bits"))

        # x expected [B,N,1,2]
        B, N = x.shape[0], x.shape[1]

        # tokenize signal
        x = self.p_embed(x)        # [B,N,H,2]

        # timestep embedding
        t = self.t_embed(t)        # [B,H,2]

        # prompt is global "c" like your reference
        c_prompt = self._encode_text(prompt_input, device)  # [B,H,2]
        if c_prompt is None:
            c = t
        else:
            if c_prompt.shape[0] == 1 and B > 1:
                c_prompt = c_prompt.expand(B, -1, -1)
            c = t + c_prompt       # [B,H,2]

        # bits are per-token injection (unambiguous)
        b_seq = self._encode_bits_seq(bits_input, device, N)  # [B,N,H,2] or None
        if b_seq is not None:
            x = x + b_seq

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x
