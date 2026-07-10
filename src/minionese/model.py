from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    context_length: int = 256
    d_model: int = 160
    n_layers: int = 4
    n_heads: int = 5
    d_ff: int = 640
    dropout: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.triu(mx.full((length, length), -1e9, dtype=scores.dtype), k=1)
        attended = mx.softmax(scores + mask, axis=-1) @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.output(attended)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        return x + self.down(nn.gelu(self.up(self.mlp_norm(x))))


class MinioneseTransformer(nn.Module):
    """A small, original, pre-norm decoder-only Transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = [DecoderBlock(config) for _ in range(config.n_layers)]
        self.final_norm = nn.LayerNorm(config.d_model)

    def __call__(self, tokens: mx.array) -> mx.array:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError(f"sequence length {length} exceeds context {self.config.context_length}")
        positions = mx.arange(length)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Weight tying keeps capacity in the decoder instead of duplicating a large head.
        return x @ self.token_embedding.weight.T


def count_parameters(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(value.size for _, value in tree_flatten(model.parameters()))


def create_model(config: ModelConfig, seed: int = 2026) -> MinioneseTransformer:
    mx.random.seed(seed)
    return MinioneseTransformer(config)
