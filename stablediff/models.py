import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

import complex.complex_module as cm
import complex.complex_layers as cm_l
import complex.complex_functions as cm_f


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

class ComplexUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            cm.ComplexSwitchSequential(cm_l.ComplexConv1d(4, 320, kernel_size=3, padding=1)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(320, 320), cm.ComplexUNet_AttentionBlock(8, 40)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(320, 320), cm.ComplexUNet_AttentionBlock(8, 40)),
            cm.ComplexSwitchSequential(cm_l.ComplexConv1d(320, 320, kernel_size=3, stride=2, padding=1)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(320, 640), cm.ComplexUNet_AttentionBlock(8, 80)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(640, 640), cm.ComplexUNet_AttentionBlock(8, 80)),
            cm.ComplexSwitchSequential(cm_l.ComplexConv1d(640, 640, kernel_size=3, stride=2, padding=1)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(640, 1280), cm.ComplexUNet_AttentionBlock(8, 160)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(1280, 1280), cm.ComplexUNet_AttentionBlock(8, 160)),
            cm.ComplexSwitchSequential(cm_l.ComplexConv1d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(1280, 1280)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = cm.ComplexSwitchSequential(
            cm.ComplexUNet_ResidualBlock(1280, 1280),
            cm.ComplexUNet_AttentionBlock(8, 160),
            cm.ComplexUNet_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(2560, 1280)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(2560, 1280)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(2560, 1280), cm.ComplexUpSample()),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(2560, 1280), cm.ComplexUNet_AttentionBlock(8, 160)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(2560, 1280), cm.ComplexUNet_AttentionBlock(8, 160)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(1920, 1280), cm.ComplexUNet_AttentionBlock(8, 160), cm.ComplexUpSample()),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(1920, 640), cm.ComplexUNet_AttentionBlock(8, 80)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(1280, 640), cm.ComplexUNet_AttentionBlock(8, 80)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(960, 640), cm.ComplexUNet_AttentionBlock(8, 80), cm.ComplexUpSample()),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(960, 320), cm.ComplexUNet_AttentionBlock(8, 40)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(640, 320), cm.ComplexUNet_AttentionBlock(8, 40)),
            cm.ComplexSwitchSequential(cm.ComplexUNet_ResidualBlock(640, 320), cm.ComplexUNet_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):

        skip_connections = []

        for layers in self.encoder:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoder:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = layers(x, context, time)

        return x

class ComplexUNet_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = cm_l.ComplexGroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = cm_l.ComplexConv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = cm_f.complex_silu(x)
        x = self.conv(x)
        return x

class stablediff_Simple(nn.Module):

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

        self.time_dim = 1280

        # Embeddings
        self.p_embed = PositionEmbedding(params.sample_rate, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(params.max_step, params.embed_dim, self.time_dim)

        self.bits_token = nn.Linear(1, 4)
        init_weight_xavier(self.bits_token)

        self.unet = ComplexUNet()
        self.final = ComplexUNet_OutputLayer(320, 4)

    def _encode_bits_seq(self, bits, device, N):
        """
        bits: [B, original_length]
        return: [B,4,N]
        """

        if bits is None:
            return None

        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=torch.float32, device=device)
        else:
            bits = bits.to(device).float()

        if bits.ndim == 1:
            bits = bits.unsqueeze(0)

        # [B,L] -> [B,1,L]
        bits = bits.unsqueeze(1)

        # Match latent sequence length
        bits = F.interpolate(
            bits,
            size=N,
            mode="nearest"
        )

        # [B,1,N] -> [B,N,1]
        bits = bits.transpose(1, 2)

        # Learn 1 bit feature -> 4 latent channels
        bits = self.bits_token(bits)

        # [B,N,4] -> [B,4,N]
        bits = bits.transpose(1, 2)

        return bits

    def forward(self, latent, time, cond):
        device = latent.device

        B, C, N = latent.shape

        prompt_input = cond.get("prompt") if isinstance(cond, dict) else cond

        bits_input = None
        if isinstance(cond, dict):
            bits_input = cond.get("bits_cond", cond.get("bits"))

        time = self.t_embed(time)

        b_seq = self._encode_bits_seq(bits_input, device, N)

        if b_seq is not None:
            b_seq = b_seq.to(latent.dtype)
            latent = latent + b_seq

        output = self.unet(latent, prompt_input, time)
        output = self.final(output)

        return output