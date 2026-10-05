import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math
import complex.complex_layers as cm_l
import complex.complex_functions as cm_f


def apply_complex(F_r, F_i, X):

    if torch.is_complex(X):
        X_r = X.real
        X_i = X.imag

    elif X.shape[-1] == 2:
        X_r = X[..., 0]
        X_i = X[..., 1]

    else:
        X_r = X
        X_i = torch.zeros_like(X)

    real = F_r(X_r) - F_i(X_i)
    imag = F_r(X_i) + F_i(X_r)

    return torch.complex(real, imag)

def apply_complex_sep(F_r, F_i, X):

    if torch.is_complex(X):
        X_r = X.real
        X_i = X.imag

    elif X.shape[-1] == 2:
        X_r = X[..., 0]
        X_i = X[..., 1]

    else:
        X_r = X
        X_i = torch.zeros_like(X)

    real = F_r(X_r)
    imag = F_i(X_i)

    return torch.complex(real, imag)

def _split_heads_complex(x, num_heads):

    """
    x: [B, S, hidden_dim] complex

    returns: [B, H, S, head_dim] complex
    """

    B, S, D = x.shape

    assert D % num_heads == 0

    hd = D // num_heads

    return x.view(B, S, num_heads, hd).permute(0, 2, 1, 3).contiguous()


def _merge_heads_complex(x):

    """
    x: [B, H, S, head_dim] complex

    returns: [B, S, hidden_dim] complex
    """

    B, H, S, hd = x.shape

    return x.permute(0, 2, 1, 3).contiguous().view(B, S, H * hd)

@torch.jit.script
def complex_mul(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.mul(X_r, Y_r) - torch.mul(X_i, Y_i)
    Z_i = torch.mul(X_r, Y_i) + torch.mul(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_bmm(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.bmm(X_r, Y_r) - torch.bmm(X_i, Y_i)
    Z_i = torch.bmm(X_r, Y_i) + torch.bmm(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_softmax(X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F.softmax(X_r, dim=-1), F.softmax(X_i, dim=-1)), dim=-1)

@torch.jit.script
def transpose_qkv(x, num_heads: int):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1, 2)
    x = x.transpose(1, 2)
    return x.reshape(-1, x.shape[2], x.shape[3], 2)

@torch.jit.script
def transpose_output(x, num_heads: int):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2], 2)
    x = x.transpose(1, 2)
    return x.reshape(x.shape[0], x.shape[1], -1, 2)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, X):
        device = X.device
        dtype = X.dtype
        mask = torch.ones(*X.shape[-3:], device=device, dtype=dtype)
        mask = F.dropout1d(mask, p=0.5, training=self.training)
        return torch.mul(X, mask)


class ComplexGELU(nn.Module):
    def __init__(self, approximate='none'):
        super().__init__()
        self.gelu_r = nn.GELU(approximate)
        self.gelu_i = nn.GELU(approximate)
    
    def forward(self, X):
        return apply_complex_sep(self.gelu_r, self.gelu_i, X)


class ComplexSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu_r = nn.SiLU()
        self.silu_i = nn.SiLU()

    def forward(self, X):
        return apply_complex_sep(self.silu_r, self.silu_i, X)


class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_r = nn.ReLU()
        self.relu_i = nn.ReLU()

    def forward(self, X):
        return apply_complex_sep(self.relu_r, self.relu_i, X)

class NativeComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.complex(
            self.relu(x.real),
            self.relu(x.imag)
        )

class ComplexAvgPool3d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool_r = nn.AvgPool3d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.avg_pool_i = nn.AvgPool3d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, X):
        return apply_complex_sep(self.avg_pool_r, self.avg_pool_i, X)


class ComplexFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.flt_r = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        self.flt_i = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, X):
        return apply_complex_sep(self.flt_r, self.flt_i, X)


class NaiveComplexBatchNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm3d, self).__init__()
        self.bn_r = nn.BatchNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = nn.BatchNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, X):
        return apply_complex_sep(self.bn_r, self.bn_i, X)


class NaiveComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(NaiveComplexLayerNorm, self).__init__()
        self.ln_r = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.ln_i = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, X):
        return apply_complex_sep(self.ln_r, self.ln_i, X)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.l_r = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)
        self.l_i = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)

    def forward(self, X):
        return apply_complex(self.l_r, self.l_i, X)


class ComplexMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ComplexGELU, bias=True, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ComplexLinear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.drop1 = ComplexDropout(dropout)
        self.fc2 = ComplexLinear(hidden_features, out_features, bias)
        self.drop2 = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ComplexConv3d(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv_r = nn.Conv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dtype=torch.float32,
        )
        self.conv_i = nn.Conv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dtype=torch.float32,
        )

    def forward(self, X):
        return apply_complex(self.conv_r, self.conv_i, X)

class ComplexResidual3d(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv1 = ComplexConv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv2 = ComplexConv3d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv3 = ComplexConv3d(
            input_channels, num_channels, kernel_size=1, padding=0, stride=stride
        )
        self.bn1 = NaiveComplexBatchNorm3d(num_channels)
        self.bn2 = NaiveComplexBatchNorm3d(num_channels)
        self.relu1 = ComplexReLU()
        self.relu2 = ComplexReLU()

    def forward(self, X):
        Y = self.relu1(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y)) + self.conv3(X)
        return self.relu2(Y)


# [32 32 10 10 3] -> [32 10 32*10*3]
class ComplexSegment(nn.Module):
    def __init__(self, input_channels, seg_channels, seg_size):
        super().__init__()
        self.seg_conv = ComplexResidual3d(
            input_channels,
            seg_channels,
            kernel_size=seg_size,
            padding=(0, 0, 0),
            stride=seg_size,
        )
        self.flt = ComplexFlatten(start_dim=2, end_dim=-1)

    def forward(self, X):
        Y = self.seg_conv(X)
        Y = Y.transpose(1, 2)
        Y = self.flt(Y)
        return Y


class Complex2Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, X):
        X = self.linear1(X)
        X = self.linear2(F.relu(X))
        return X.squeeze(dim=-1)


class ComplexDotProductAttention(nn.Module):
    """
    Query shape: [batch_size, query_num, query_key_dim]
    Key shape: [batch_size, key_value_num, query_key_dim]
    Value shape: [batch_size, key_value_num, value_dim]
    """
    def __init__(self, dropout, **kwargs):
        super(ComplexDotProductAttention, self).__init__(**kwargs)
        self.dropout = ComplexDropout(dropout)

    def forward(self, queries, keys, values):
        query_key_dim = queries.shape[-2]
        self.attention_weights = complex_softmax(
            complex_bmm(queries, keys.transpose(1, 2)) / math.sqrt(query_key_dim)
        )
        Y = complex_bmm(self.dropout(self.attention_weights), values)
        return Y

class ComplexMultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size,
        num_hiddens,
        num_heads,
        dropout,
        key_size=None,
        value_size=None,
        bias=False,
        **kwargs
    ):
        super(ComplexMultiHeadAttention, self).__init__(**kwargs)
        key_size = key_size or query_size
        value_size = value_size or query_size
        self.num_heads = num_heads
        self.attention = ComplexDotProductAttention(dropout=dropout)
        self.w_q = ComplexLinear(query_size, num_hiddens, bias=bias)
        self.w_k = ComplexLinear(key_size, num_hiddens, bias=bias)
        self.w_v = ComplexLinear(value_size, num_hiddens, bias=bias)
        self.w_o = ComplexLinear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        Y = self.w_o(output_concat)
        return Y

class AttnMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)
        return ((V.unsqueeze(-1) * K.unsqueeze(-2)).cumsum(-3) * Q.unsqueeze(-2)).sum(-1)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        # As in preprint
        grad_Q = ((V.unsqueeze(-1) * K.unsqueeze(-2)).cumsum(-3) * grad_output.unsqueeze(-1)).sum(-2)
        grad_K = ((grad_output.unsqueeze(-1) * Q.unsqueeze(-2))
                  .flip(-3).cumsum(-3).flip(-3) * V.unsqueeze(-1)).sum(-2)
        grad_V = ((grad_output.unsqueeze(-1) * Q.unsqueeze(-2))
                  .flip(-3).cumsum(-3).flip(-3) * K.unsqueeze(-2)).sum(-1)
        return grad_Q, grad_K, grad_V

class CosineAttentionCausal(nn.Module):
    def __init__(self, num_heads, eps=1e-8):
        super().__init__()
        self.eps = eps
        # norm_const of shape (1, H, 1, 1) like preprint
        self.norm_const = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, Q, K, V, s=None):
        B, H, S, Dk = Q.shape
        if s is None:
            # match "sequence length at current timestep"
            s = torch.tensor(float(S), device=Q.device, dtype=Q.dtype).view(1, 1, 1, 1)

        # cosine normalization
        Qn = F.normalize(Q, dim=-1, p=2, eps=self.eps)
        Kn = F.normalize(K, dim=-1, p=2, eps=self.eps)

        # V scaling per head
        # Preprint line: V = V / s ** norm_const.sigmoid()
        scale = s ** torch.sigmoid(self.norm_const)   # [1,H,1,1] broadcast over B,S,Dv
        Vt = V / scale

        return AttnMul.apply(Qn, Kn, Vt)

class CosineAttentionCross(nn.Module):
    def __init__(self, num_heads, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.norm_const = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, Q, K, V, s=None):

        B, H, Sq, Dk = Q.shape
        Sk = K.shape[2]

        if s is None:
            s = torch.tensor(
                float(Sk),
                device=Q.device,
                dtype=Q.dtype
            ).view(1, 1, 1, 1)

        Qn = F.normalize(Q, dim=-1, p=2, eps=self.eps)
        Kn = F.normalize(K, dim=-1, p=2, eps=self.eps)

        scale = s ** torch.sigmoid(self.norm_const)

        Vt = V / scale

        scores = torch.matmul(
            Qn,
            Kn.transpose(-1, -2)
        )

        weights = F.softmax(scores, dim=-1)

        return torch.matmul(weights, Vt)


class CosineComplexMultiHeadAttention(nn.Module):
    """
    Complex multi-head attention:
      - Uses real and imaginary Q/K components independently
      - Applies cosine attention to V_real and V_imag separately
      - Returns native complex output
    """
    def __init__(self, hidden_dim, num_heads, bias=True, eps=1e-8):
        super().__init__()

        self.num_heads = num_heads
        self.eps = eps

        self.w_q = ComplexLinear(hidden_dim, hidden_dim, bias=bias)
        self.w_k = ComplexLinear(hidden_dim, hidden_dim, bias=bias)
        self.w_v = ComplexLinear(hidden_dim, hidden_dim, bias=bias)
        self.w_o = ComplexLinear(hidden_dim, hidden_dim, bias=bias)

        self.attn = CosineAttentionCausal(num_heads=num_heads, eps=eps)

    def forward(self, x, s=None):
        """
        x:       [B, S, hidden_dim] complex
        returns: [B, S, hidden_dim] complex
        """

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        qh = _split_heads_complex(q, self.num_heads)  # [B,H,S,hd]
        kh = _split_heads_complex(k, self.num_heads)  # [B,H,S,hd]
        vh = _split_heads_complex(v, self.num_heads)  # [B,H,S,hd]

        Qr = qh.real
        Kr = kh.real

        Qi = qh.imag
        Ki = kh.imag

        Vr = vh.real
        Vi = vh.imag

        Or = self.attn(Qr, Kr, Vr, s=s)
        Oi = self.attn(Qi, Ki, Vi, s=s)

        out_h = torch.complex(Or, Oi)  # [B,H,S,hd]

        out = _merge_heads_complex(out_h)  # [B,S,hidden_dim]
        out = self.w_o(out)

        return out


class CosineComplexCrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True, eps=1e-8):
        super().__init__()
        self.q_proj = ComplexLinear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = ComplexLinear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = ComplexLinear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = ComplexLinear(d_embed, d_embed, bias=out_proj_bias)

        self.attn = CosineAttentionCross(num_heads=n_heads, eps=eps)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y, s=None):
        """
        x: [B, S, d_embed] complex
        y: [B, T, d_cross] real or complex

        returns:
           [B, S, d_embed] complex
        """

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        qh = _split_heads_complex(q, self.n_heads)  # [B,H,S,hd]
        kh = _split_heads_complex(k, self.n_heads)  # [B,H,T,hd]
        vh = _split_heads_complex(v, self.n_heads)  # [B,H,T,hd]

        Qr = qh.real
        Kr = kh.real

        Qi = qh.imag
        Ki = kh.imag

        Vr = vh.real
        Vi = vh.imag

        Or = self.attn(Qr, Kr, Vr, s=s)
        Oi = self.attn(Qi, Ki, Vi, s=s)

        out_h = torch.complex(Or, Oi)  # [B,H,S,hd]

        out = _merge_heads_complex(out_h)  # [B,S,d_embed]
        out = self.out_proj(out)

        return out
    
class ComplexPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=10000):
        super(ComplexPositionalEncoding, self).__init__()
        self.dropout = ComplexDropout(dropout)
        pcode = torch.zeros((1, max_len, hidden_dim, 2), dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, hidden_dim, dtype=torch.float32) / hidden_dim
        )
        pcode[:, :, :, 0] = torch.cos(pos)
        pcode[:, :, :, 1] = torch.sin(pos)
        self.register_buffer("pcode", pcode, persistent=False)

    def forward(self, X):
        X = complex_mul(X, self.pcode[:, : X.shape[1], :, :].to(X.device))
        Y = self.dropout(X)
        return Y


class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.linear1 = ComplexLinear(input_dim, hidden_dim)
        self.relu = ComplexReLU()
        self.linear2 = ComplexLinear(hidden_dim, output_dim)

    def forward(self, X):
        Y = self.linear2(self.relu(self.linear1(X)))
        return Y


class ComplexAddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(ComplexAddNorm, self).__init__(**kwargs)
        self.dropout = ComplexDropout(dropout)
        self.ln = NaiveComplexLayerNorm(normalized_shape)

    def forward(self, X, Y):
        Y = self.ln(self.dropout(Y) + X)
        return Y


class ComplexEncoderBlock(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexEncoderBlock, self).__init__(**kwargs)
        self.attention = ComplexMultiHeadAttention(
            key_dim, query_dim, value_dim, hidden_dim, num_heads, dropout, use_bias
        )
        self.addnorm1 = ComplexAddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input_dim, ffn_hidden_dim, ffn_hidden_dim)
        self.addnorm2 = ComplexAddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.attention(X, X, X)
        Z = self.addnorm1(X, Y)
        return self.addnorm2(Z, self.ffn(Y))


class ComplexTransformerEncoder(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexTransformerEncoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.pos_encoding = ComplexPositionalEncoding(hidden_dim, dropout)
        self.blks = nn.Sequential()
        for n in range(num_layers):
            self.blks.add_module(
                "Block" + str(n),
                ComplexEncoderBlock(
                    key_dim,
                    query_dim,
                    value_dim,
                    hidden_dim,
                    norm_shape,
                    ffn_input_dim,
                    ffn_hidden_dim,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    def forward(self, X, *args):
        X = self.pos_encoding(X * math.sqrt(self.hidden_dim))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class ComplexUpSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cm_f.complex_upsample(x, scale_factor=2)

class ComplexUNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_feature = cm_l.ComplexGroupNorm(32, in_channels, eps=1e-6)
        self.conv_feature = cm_l.ComplexConv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time = ComplexLinear(1280, out_channels)

        self.groupnorm_merged = cm_l.ComplexGroupNorm(32, out_channels, eps=1e-6)
        self.conv_merged = cm_l.ComplexConv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = cm_l.ComplexConv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t):

        residue = x

        feature = self.groupnorm_feature(x)
        feature = cm_f.complex_silu(feature)
        feature = self.conv_feature(feature)

        t = cm_f.complex_silu(t)
        t = self.linear_time(t)
        t = t.unsqueeze(-1)

        merged = feature + t

        merged = self.groupnorm_merged(merged)
        merged = cm_f.complex_silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class ComplexUNet_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()

        channels = n_head * n_embd

        self.groupnorm = cm_l.ComplexGroupNorm(32, channels, eps=1e-6)
        self.conv_input = cm_l.ComplexConv1d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = cm_l.NaiveComplexLayerNorm(channels)
        self.attention_1 = CosineComplexMultiHeadAttention(
            channels, n_head, bias=True, eps=1e-8
        )

        self.layernorm_2 = cm_l.NaiveComplexLayerNorm(channels)
        self.attention_2 = CosineComplexCrossAttention(
            n_head, channels, d_context, in_proj_bias=True, eps=1e-8
        )

        self.layernorm_3 = cm_l.NaiveComplexLayerNorm(channels)

        self.linear_geglu_1 = cm_l.ComplexLinear(channels, channels * 8)
        self.linear_geglu_2 = cm_l.ComplexLinear(channels * 4, channels)

        self.conv_output = cm_l.ComplexConv1d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        x = x.transpose(1, 2)  # [B, N, C]

        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)

        x += residue_short

        residue_short = x

        x = self.layernorm_2(x)
        x = self.attention_2(x, context)

        x += residue_short

        residue_short = x

        x = self.layernorm_3(x)

        x = self.linear_geglu_1(x)

        if not torch.is_complex(x) and x.shape[-1] == 2:
            x = torch.view_as_complex(x.contiguous())

        x, gate = x.chunk(2, dim=-1)

        x = x * cm_f.complex_gelu(gate)
        x = self.linear_geglu_2(x)

        if not torch.is_complex(x) and x.shape[-1] == 2:
            x = torch.view_as_complex(x.contiguous())

        x += residue_short

        x = x.transpose(1, 2)  # [B, C, N]

        return self.conv_output(x) + residue_long
    
class ComplexSwitchSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)


    def forward(self, x, context, t):
        for layer in self:
            if isinstance(layer, ComplexUNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ComplexUNet_ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x
