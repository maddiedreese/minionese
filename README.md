# Minionese LM

A tiny conversational decoder-only Transformer trained entirely from scratch on an original Minionese-inspired dataset. It accepts short English or Minionese messages and replies in a playful, consistent Minionese project dialect.

This project does **not** use pretrained weights, film subtitles, screenplay text, iMessages, or code from the project that inspired the experiment. Minionese is an improvised fictional dialect rather than a language with fixed grammar, so this is explicitly an unofficial fan research project—not an official translator or an Illumination product.

## Architecture

- 4 causal decoder layers
- 160 model width, 5 attention heads, 640 MLP width
- learned positional embeddings and tied token/output embeddings
- 512-token custom byte-level BPE tokenizer
- 256-token context window
- MLX training on Apple Silicon, plus PyTorch CPU inference on Linux, macOS, and Windows

The exact parameter count is printed by `minionese doctor` and saved with every training report.

## Trained result

The included local artifact was trained from random initialization for 5,000 batches and 9,981,697 sampled non-padding tokens. On 400 held-out behavioral prompts it achieved 99.0% intent/style/clean pass rate (99.5% for English inputs and 98.5% for Minionese inputs), with test perplexity 1.116 versus a 4.688 unigram baseline. See [the full results](docs/RESULTS.md) and `artifacts/evaluation.json`.

## Set up and reproduce

Python 3.11+ and Apple Silicon are required to reproduce training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[apple,dev]'
minionese doctor
minionese prepare
minionese train-tokenizer
minionese train
minionese evaluate
```

Training defaults to 5,000 batches, approximately 8–10 million sampled non-padding tokens depending on generated prompt lengths. The tokenizer and every model parameter begin from random/untrained state. Artifacts are written below `artifacts/`; large generated data and weights are ignored by Git.

The ready-to-use export is in `artifacts/final`. To use it explicitly:

```bash
minionese reply "hello" --model artifacts/final --tokenizer artifacts/final
```

## Portable CPU inference

The checked-in safetensors export can also run with PyTorch on a CPU-only Linux,
macOS, or Windows machine; MLX and Apple Silicon are not required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference]'
minionese-cpu "hello" --model artifacts/final
```

Expected greedy-decoding output:

```text
Bello! Tulaliloo, amiko!
```

Try the hosted model at [notaminion.com](https://notaminion.com).


## Chat

```bash
minionese reply "I'm having a wonderful day"
minionese reply "bello, como estas?"
minionese chat
```

Greedy decoding (`--temperature 0`) is the default because it is substantially more reliable at this model size. Set `--temperature 0.6` for varied replies.

The model supports the conversational intents represented in the dataset. It does not understand arbitrary English vocabulary, and compound or out-of-domain prompts can produce a mismatched or malformed response. This is a measured limitation of training a 1.35M-parameter model from zero, not a general assistant.

See [the implementation and evaluation plan](docs/PLAN.md) for the dataset specification, completion gates, and limitations.
