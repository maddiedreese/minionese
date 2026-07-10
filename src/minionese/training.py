from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import read_jsonl
from .model import MinioneseTransformer, ModelConfig, count_parameters, create_model
from .tokenizer import format_conversation, load_tokenizer


def encode_rows(rows: list[dict[str, str]], tokenizer, context_length: int) -> list[tuple[list[int], list[int]]]:
    encoded: list[tuple[list[int], list[int]]] = []
    assistant_id = tokenizer.token_to_id("<assistant>")
    for row in rows:
        ids = tokenizer.encode(format_conversation(row["prompt"], row["reply"]), add_special_tokens=False).ids[:context_length]
        try:
            reply_start = ids.index(assistant_id) + 1
        except ValueError as exc:
            raise RuntimeError("tokenizer did not preserve <assistant> as a special token") from exc
        mask = [0] * reply_start + [1] * (len(ids) - reply_start)
        if len(ids) >= 3 and sum(mask) >= 2:
            encoded.append((ids, mask))
    return encoded


def make_batch(examples: list[tuple[list[int], list[int]]], indices: np.ndarray, pad_id: int) -> tuple[mx.array, mx.array, int, int]:
    selected = [examples[int(i)] for i in indices]
    width = max(len(ids) for ids, _ in selected)
    tokens = np.full((len(selected), width), pad_id, dtype=np.int32)
    masks = np.zeros((len(selected), width), dtype=np.float32)
    for row, (ids, mask) in enumerate(selected):
        tokens[row, : len(ids)] = ids
        masks[row, : len(mask)] = mask
    return mx.array(tokens), mx.array(masks), int(masks[:, 1:].sum()), int((tokens != pad_id).sum())


def masked_loss(model: MinioneseTransformer, tokens: mx.array, mask: mx.array) -> mx.array:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    target_mask = mask[:, 1:]
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (losses * target_mask).sum() / mx.maximum(target_mask.sum(), mx.array(1.0))


def evaluate_loss(model: MinioneseTransformer, examples, pad_id: int, batch_size: int = 128, max_batches: int | None = None) -> float:
    losses: list[float] = []
    for batch_no, start in enumerate(range(0, len(examples), batch_size)):
        if max_batches is not None and batch_no >= max_batches:
            break
        idx = np.arange(start, min(start + batch_size, len(examples)))
        tokens, mask, _, _ = make_batch(examples, idx, pad_id)
        loss = masked_loss(model, tokens, mask)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _learning_rate(step: int, total_steps: int, peak: float, warmup: int, minimum: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def train(
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    steps: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    warmup_steps: int = 150,
    validate_every: int = 250,
    seed: int = 2026,
) -> dict:
    data_dir, output = Path(data_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = ModelConfig.load(config_path)
    tokenizer = load_tokenizer(tokenizer_dir)
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != config.vocab_size:
        raise ValueError(f"tokenizer has {actual_vocab} tokens but model config requires {config.vocab_size}")
    train_rows = read_jsonl(data_dir / "train.jsonl")
    validation_rows = read_jsonl(data_dir / "validation.jsonl")
    train_examples = encode_rows(train_rows, tokenizer, config.context_length)
    validation_examples = encode_rows(validation_rows, tokenizer, config.context_length)
    pad_id = tokenizer.token_to_id("<pad>")

    model = create_model(config, seed)
    parameter_count = count_parameters(model)
    optimizer = optim.AdamW(learning_rate=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, masked_loss)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_step = 0
    assistant_tokens_seen = 0
    training_tokens_seen = 0
    started = time.time()
    history: list[dict] = []

    for step in range(steps):
        lr = _learning_rate(step, steps, learning_rate, warmup_steps, min_learning_rate)
        optimizer.learning_rate = mx.array(lr)
        indices = rng.integers(0, len(train_examples), size=batch_size)
        tokens, mask, batch_reply_tokens, batch_training_tokens = make_batch(train_examples, indices, pad_id)
        loss, gradients = loss_and_grad(model, tokens, mask)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        assistant_tokens_seen += batch_reply_tokens
        training_tokens_seen += batch_training_tokens

        should_report = step == 0 or (step + 1) % 50 == 0
        should_validate = (step + 1) % validate_every == 0 or step + 1 == steps
        if should_report:
            elapsed = max(time.time() - started, 1e-6)
            print(f"step {step + 1:5d}/{steps} loss {float(loss.item()):.4f} lr {lr:.2e} tok/s {training_tokens_seen / elapsed:,.0f}", flush=True)
        if should_validate:
            val_loss = evaluate_loss(model, validation_examples, pad_id, max_batches=16)
            record = {"step": step + 1, "train_loss": float(loss.item()), "validation_loss": val_loss, "training_tokens_seen": training_tokens_seen, "assistant_tokens_seen": assistant_tokens_seen, "learning_rate": lr}
            history.append(record)
            print(f"validation step {step + 1}: loss {val_loss:.4f} perplexity {math.exp(min(val_loss, 20)):.2f}", flush=True)
            model.save_weights(str(output / "last.safetensors"))
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step + 1
                model.save_weights(str(output / "best.safetensors"))

    config.save(output / "model.json")
    report = {
        "pretrained": False,
        "initialization": "random",
        "seed": seed,
        "parameters": parameter_count,
        "architecture": asdict(config),
        "steps": steps,
        "batch_size": batch_size,
        "training_tokens_seen": training_tokens_seen,
        "assistant_tokens_seen": assistant_tokens_seen,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "best_validation_perplexity": math.exp(min(best_loss, 20)),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_trained_model(model_dir: str | Path) -> MinioneseTransformer:
    model_dir = Path(model_dir)
    config = ModelConfig.load(model_dir / "model.json")
    model = create_model(config)
    weights = model_dir / "best.safetensors"
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model
