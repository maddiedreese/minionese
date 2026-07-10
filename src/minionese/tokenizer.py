from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .data import read_jsonl


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"]


def train_tokenizer(train_jsonl: str | Path, output_dir: str | Path, vocab_size: int = 512) -> dict:
    """Train byte-level BPE from the training split only."""
    rows = read_jsonl(train_jsonl)
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS, initial_alphabet=ByteLevel.alphabet(), show_progress=True)

    def corpus():
        for row in rows:
            yield f"<bos><user>{row['prompt']}<assistant>{row['reply']}<eos>"

    tokenizer.train_from_iterator(corpus(), trainer=trainer, length=len(rows))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"))
    metadata = {
        "type": "byte-level-bpe",
        "trained_from": str(Path(train_jsonl)),
        "training_examples": len(rows),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "pretrained": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def format_conversation(prompt: str, reply: str | None = None) -> str:
    text = f"<bos><user>{prompt}<assistant>"
    return text if reply is None else f"{text}{reply}<eos>"

