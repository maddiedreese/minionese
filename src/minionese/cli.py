from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minionese", description="Train and chat with a tiny from-scratch Minionese model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check the local MLX environment")

    prepare = sub.add_parser("prepare", help="generate the original conversational dataset")
    prepare.add_argument("--output", default="artifacts/data")
    prepare.add_argument("--train-count", type=int, default=60_000)
    prepare.add_argument("--validation-count", type=int, default=4_000)
    prepare.add_argument("--test-count", type=int, default=4_000)
    prepare.add_argument("--seed", type=int, default=2026)

    tok = sub.add_parser("train-tokenizer", help="train byte-level BPE from scratch")
    tok.add_argument("--train", default="artifacts/data/train.jsonl")
    tok.add_argument("--output", default="artifacts/tokenizer")
    tok.add_argument("--vocab-size", type=int, default=512)

    train = sub.add_parser("train", help="train the randomly initialized Transformer")
    train.add_argument("--data", default="artifacts/data")
    train.add_argument("--tokenizer", default="artifacts/tokenizer")
    train.add_argument("--config", default="configs/model.json")
    train.add_argument("--output", default="artifacts/checkpoints")
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--validate-every", type=int, default=250)
    train.add_argument("--seed", type=int, default=2026)

    ev = sub.add_parser("evaluate", help="evaluate loss and held-out conversations")
    ev.add_argument("--model", default="artifacts/checkpoints")
    ev.add_argument("--tokenizer", default="artifacts/tokenizer")
    ev.add_argument("--data", default="artifacts/data")
    ev.add_argument("--output", default="artifacts/evaluation.json")
    ev.add_argument("--samples-per-intent-language", type=int, default=8)

    for name in ("reply", "chat"):
        cmd = sub.add_parser(name, help="generate one reply" if name == "reply" else "start an interactive local chat")
        cmd.add_argument("message", nargs="?", help="message for reply mode")
        cmd.add_argument("--model", default="artifacts/checkpoints")
        cmd.add_argument("--tokenizer", default="artifacts/tokenizer")
        cmd.add_argument("--temperature", type=float, default=0.0, help="0 is reliable greedy decoding; use 0.5-0.8 for more variety")
        cmd.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        import mlx.core as mx
        from .model import ModelConfig, count_parameters, create_model

        model = create_model(ModelConfig.load("configs/model.json"))
        report = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(), "mlx_device": str(mx.default_device()), "parameters": count_parameters(model), "ready": platform.machine() == "arm64"}
        print(json.dumps(report, indent=2))
    elif args.command == "prepare":
        from .data import write_dataset
        print(json.dumps(write_dataset(args.output, args.train_count, args.validation_count, args.test_count, args.seed), indent=2))
    elif args.command == "train-tokenizer":
        from .tokenizer import train_tokenizer
        print(json.dumps(train_tokenizer(args.train, args.output, args.vocab_size), indent=2))
    elif args.command == "train":
        from .training import train
        report = train(args.data, args.tokenizer, args.config, args.output, steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate, validate_every=args.validate_every, seed=args.seed)
        print(json.dumps(report, indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        report = evaluate(args.model, args.tokenizer, args.data, args.output, args.samples_per_intent_language)
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    else:
        from .generation import generate_reply
        from .tokenizer import load_tokenizer
        from .training import load_trained_model
        model, tokenizer = load_trained_model(args.model), load_tokenizer(args.tokenizer)
        if args.command == "reply":
            if not args.message:
                raise SystemExit("reply requires a message")
            print(generate_reply(model, tokenizer, args.message, temperature=args.temperature, seed=args.seed))
        else:
            print("Bello! Type /quit to leave.")
            while True:
                try:
                    message = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nPoopaye!")
                    break
                if message.lower() in {"/quit", "/exit", "quit", "exit"}:
                    print("Minion: Poopaye!")
                    break
                if message:
                    print("Minion:", generate_reply(model, tokenizer, message, temperature=args.temperature, seed=args.seed))


if __name__ == "__main__":
    main()
