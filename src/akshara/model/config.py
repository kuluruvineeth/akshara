from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 65_536
    d_model: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    ffn_dim: int = 1536
    max_seq_len: int = 2048
    rope_theta: float = 1e5
    norm_eps: float = 1e-5
    qk_norm: bool = False
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


TIERS = {
    "S": ModelConfig(),
    "M": ModelConfig(d_model=1536, n_layers=28, n_heads=12, n_kv_heads=4, ffn_dim=4096, max_seq_len=4096,
                     rope_theta=5e5, qk_norm=True),
    "L": ModelConfig(d_model=2048, n_layers=36, n_heads=16, n_kv_heads=4, ffn_dim=11008, max_seq_len=4096,
                     rope_theta=5e5, qk_norm=True),
}  # fmt: skip
