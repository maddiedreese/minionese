from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .tokenizer import format_conversation, load_tokenizer


torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


@dataclass(frozen=True)
class TorchModelConfig:
    vocab_size: int = 512
    context_length: int = 256
    d_model: int = 160
    n_layers: int = 4
    n_heads: int = 5
    d_ff: int = 640
    dropout: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "TorchModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class TorchCausalSelfAttention(nn.Module):
    def __init__(self, config: TorchModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(
            torch.full((length, length), -1e9, dtype=scores.dtype, device=x.device),
            diagonal=1,
        )
        attended = torch.softmax(scores + mask, dim=-1) @ v
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class TorchDecoderBlock(nn.Module):
    def __init__(self, config: TorchModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = TorchCausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.down(torch.nn.functional.gelu(self.up(self.mlp_norm(x))))


class TorchMinioneseTransformer(nn.Module):
    def __init__(self, config: TorchModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(TorchDecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context {self.config.context_length}"
            )
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.token_embedding.weight.T


class MinioneseRuntime:
    """Portable CPU runtime for the trained safetensors export."""

    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)
        config = TorchModelConfig.load(artifact_dir / "model.json")
        self.model = TorchMinioneseTransformer(config)
        state = load_file(artifact_dir / "best.safetensors", device="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.tokenizer = load_tokenizer(artifact_dir)

    @torch.inference_mode()
    def reply(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        top_k: int = 24,
        seed: int = 2026,
    ) -> str:
        if not 1 <= max_new_tokens < self.model.config.context_length:
            raise ValueError("max_new_tokens must be between 1 and context_length - 1")
        ids = self.tokenizer.encode(
            format_conversation(prompt), add_special_tokens=False
        ).ids
        ids = ids[-(self.model.config.context_length - max_new_tokens) :]
        eos_id = self.tokenizer.token_to_id("<eos>")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            tokens = torch.tensor([ids + generated], dtype=torch.long)
            logits = self.model(tokens)[0, -1].float().clone()
            for token_id in set(generated[-12:]):
                logits[token_id] = (
                    logits[token_id] / 1.12
                    if logits[token_id] > 0
                    else logits[token_id] * 1.12
                )
            if temperature <= 0:
                next_id = int(torch.argmax(logits))
            else:
                scores, candidates = torch.topk(logits, k=min(top_k, logits.numel()))
                probabilities = torch.softmax(scores / temperature, dim=-1)
                selected = torch.multinomial(probabilities, 1, generator=generator)
                next_id = int(candidates[selected])
            if next_id == eos_id:
                break
            generated.append(next_id)

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return re.split(r"<(?:user|assistant|bos|eos|pad)>", text, maxsplit=1)[0].strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Minionese inference on CPU with PyTorch")
    parser.add_argument("message", help="short English or Minionese prompt")
    parser.add_argument("--model", default="artifacts/final", help="model artifact directory")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    runtime = MinioneseRuntime(args.model)
    print(runtime.reply(args.message, temperature=args.temperature, seed=args.seed))


if __name__ == "__main__":
    main()
