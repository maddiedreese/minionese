from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import INTENTS, read_jsonl
from .generation import generate_reply
from .tokenizer import load_tokenizer
from .training import encode_rows, evaluate_loss, load_trained_model


STYLE_MARKERS = {
    "amiko", "banana", "bello", "poopaye", "tank", "yu", "me", "para", "si", "po", "ka",
    "tulaliloo", "gelato", "wahaha", "o", "bon", "bene", "vamos", "dormi", "hana", "dul", "sae",
    "alegria", "molto", "mucho", "beedo", "bee", "do", "pwede", "kanpai", "minionese",
}


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _style_ok(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & STYLE_MARKERS) or bool(re.search(r"\b\w+-(?:o|a)\b", text.lower()))


def _unigram_baseline(train_examples, test_examples, vocab_size: int) -> float:
    counts = Counter()
    total = 0
    for ids, mask in train_examples:
        for token, active in zip(ids[1:], mask[1:]):
            if active:
                counts[token] += 1
                total += 1
    denom = total + vocab_size
    loss, number = 0.0, 0
    for ids, mask in test_examples:
        for token, active in zip(ids[1:], mask[1:]):
            if active:
                loss -= math.log((counts[token] + 1) / denom)
                number += 1
    return loss / max(number, 1)


def evaluate(model_dir: str | Path, tokenizer_dir: str | Path, data_dir: str | Path, output_path: str | Path, samples_per_intent_language: int = 8) -> dict:
    model = load_trained_model(model_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    rows = read_jsonl(Path(data_dir) / "test.jsonl")
    train_rows = read_jsonl(Path(data_dir) / "train.jsonl")
    train_examples = encode_rows(train_rows, tokenizer, model.config.context_length)
    test_examples = encode_rows(rows, tokenizer, model.config.context_length)
    pad_id = tokenizer.token_to_id("<pad>")
    test_loss = evaluate_loss(model, test_examples, pad_id)
    baseline_loss = _unigram_baseline(train_examples, test_examples, model.config.vocab_size)

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["intent"], row["input_language"])].append(row)
    samples, relevant, styled, clean = [], 0, 0, 0
    by_language = {"english": {"passed": 0, "total": 0}, "minionese": {"passed": 0, "total": 0}}
    by_intent = {name: {"passed": 0, "total": 0} for name in INTENTS}
    for (intent, language), candidates in sorted(buckets.items()):
        for index, row in enumerate(candidates[:samples_per_intent_language]):
            reply = generate_reply(model, tokenizer, row["prompt"], temperature=0.0, seed=10_000 + index)
            groups = INTENTS[intent].keywords
            intent_ok = all(any(_contains_phrase(reply, phrase) for phrase in group) for group in groups)
            style_ok = _style_ok(reply)
            clean_ok = bool(reply) and "<" not in reply and ">" not in reply
            passed = intent_ok and style_ok and clean_ok
            relevant += int(intent_ok)
            styled += int(style_ok)
            clean += int(clean_ok)
            by_language[language]["total"] += 1
            by_language[language]["passed"] += int(passed)
            by_intent[intent]["total"] += 1
            by_intent[intent]["passed"] += int(passed)
            samples.append({"intent": intent, "input_language": language, "prompt": row["prompt"], "reply": reply, "intent_ok": intent_ok, "style_ok": style_ok, "clean_ok": clean_ok})
    total = len(samples)
    for values in list(by_language.values()) + list(by_intent.values()):
        values["rate"] = values["passed"] / max(values["total"], 1)
    report = {
        "test_loss": test_loss,
        "test_perplexity": math.exp(min(test_loss, 20)),
        "unigram_baseline_loss": baseline_loss,
        "beats_unigram_baseline": test_loss < baseline_loss,
        "behavioral_samples": total,
        "intent_relevance_rate": relevant / max(total, 1),
        "minionese_style_rate": styled / max(total, 1),
        "clean_completion_rate": clean / max(total, 1),
        "overall_pass_rate": sum(v["passed"] for v in by_language.values()) / max(total, 1),
        "by_input_language": by_language,
        "by_intent": by_intent,
        "samples": samples,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report

