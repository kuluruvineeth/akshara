import torch
import torch.nn.functional as F
from torch import nn

from akshara.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def rope_tables(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), frequencies)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (i, i + head_dim / 2), the "half" layout Llama and Qwen use; x is (batch, heads, seq, head_dim)."""
    first, second = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps) if config.qk_norm else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))
