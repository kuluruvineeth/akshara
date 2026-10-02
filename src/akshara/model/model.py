import torch
from torch import nn

from akshara.model.config import ModelConfig
from akshara.model.layers import Block, RMSNorm, rope_tables


class Akshara(nn.Module):
    """A Llama/Qwen3-style decoder: pre-RMSNorm, GQA with RoPE, SwiGLU, no biases, optionally tied embeddings.

    Parameter names follow transformers' Llama/Qwen3 layout, so export is a renaming-free copy.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = rope_tables(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[1]
        if seq > self.config.max_seq_len:
            raise ValueError(f"sequence of {seq} tokens is longer than max_seq_len={self.config.max_seq_len}")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos[:seq], self.rope_sin[:seq]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
