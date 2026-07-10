from __future__ import annotations

import re

import mlx.core as mx
import numpy as np

from .tokenizer import format_conversation


def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 48, temperature: float = 0.65, top_k: int = 24, seed: int = 2026) -> str:
    ids = tokenizer.encode(format_conversation(prompt), add_special_tokens=False).ids
    ids = ids[-(model.config.context_length - max_new_tokens) :]
    eos_id = tokenizer.token_to_id("<eos>")
    rng = np.random.default_rng(seed)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        tokens = mx.array(np.asarray([ids + generated], dtype=np.int32))
        logits = np.asarray(model(tokens)[0, -1].astype(mx.float32))
        # Mild repetition penalty prevents tiny models from getting stuck in loops.
        for token_id in set(generated[-12:]):
            logits[token_id] = logits[token_id] / 1.12 if logits[token_id] > 0 else logits[token_id] * 1.12
        if temperature <= 0:
            next_id = int(np.argmax(logits))
        else:
            cutoff = np.argpartition(logits, -top_k)[-top_k:]
            scores = logits[cutoff] / temperature
            scores = scores - scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            next_id = int(rng.choice(cutoff, p=probabilities))
        if next_id == eos_id:
            break
        generated.append(next_id)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = re.split(r"<(?:user|assistant|bos|eos|pad)>", text, maxsplit=1)[0].strip()
    return text

