# Minionese LM implementation plan

## Goal and acceptance criteria

Build a local conversational model whose tokenizer and approximately 1–2M model parameters are initialized and trained from scratch. It must accept short English or Minionese messages and produce a relevant, short Minionese reply. It is deliberately not a factual assistant, translator, or general-purpose model.

The work is complete when the repository contains a reproducible data pipeline, tokenizer, decoder-only Transformer, MLX trainer, local chat command, exported weights, and an evaluation report that demonstrates intent relevance and Minionese style on held-out prompts.

## Research-derived language specification

Minionese is not a standardized constructed language. Creator and voice actor Pierre Coffin describes it as improvised speech whose recognizable fragments and melody matter more than formal grammar. Reported source languages include English, Spanish, French, Italian, and Indonesian; commonly documented expressions also draw from Korean, Japanese, and Filipino.

This project therefore defines an original, reproducible project dialect:

- Preserve a small documented core: `bello` (hello), `poopaye` (goodbye), `tank yu` (thank you), `para tu` (for you), `gelato`, `banana`, `hana/dul/sae`, and the onomatopoeic `bee do` alarm.
- Use short subject–verb–object clauses, frequent omitted articles, expressive repetition, exclamations, and playful open syllables.
- Mix a controlled set of transparent words from the source languages, while keeping enough familiar anchors for a reader to infer intent.
- Favor warm, childlike, mischievous, food-oriented responses. Do not imitate a particular film scene or reproduce scripts/subtitles.
- Treat spelling as flexible but internally consistent enough for a 1.3M-parameter model to learn.

The model is a style-and-intent responder. There is no defensible notion of exact translation because canonical Minionese has neither fixed grammar nor a complete lexicon.

## Files and responsibilities

- `src/minionese/data.py`: original intent inventory, English/Minionese prompt variants, response realization, deterministic split generation, and JSONL writing.
- `src/minionese/tokenizer.py`: train/load the byte-level BPE tokenizer using only the training split.
- `src/minionese/model.py`: causal decoder-only Transformer and parameter accounting.
- `src/minionese/training.py`: assistant-masked examples, batching, AdamW training, validation, checkpointing, and export.
- `src/minionese/generation.py`: prompt formatting and autoregressive decoding.
- `src/minionese/evaluation.py`: held-out loss plus behavioral intent/style checks.
- `src/minionese/cli.py`: `doctor`, `prepare`, `train-tokenizer`, `train`, `evaluate`, `reply`, and `chat` commands.
- `tests/`: deterministic data, split, tokenizer, model, and generation smoke tests.

## Data design

Generate original conversational pairs from semantic intents rather than scraping copyrighted subtitle or screenplay text. Each intent has independently authored English prompts, Minionese prompts, semantic response atoms, and surface-style variants. Productive slot substitution (food, activities, moods, times, companions) creates tens of thousands of combinations.

Cover the supported conversational vocabulary in training; a 1.35M-parameter model trained from zero cannot infer the meaning of an English word it has never encountered. Validation and test wrap semantic utterances in unseen full phrasings, measuring surface/compositional robustness without pretending to test zero-shot lexical knowledge. Keep a fixed seed and save dataset statistics and a SHA-256 digest. Train the tokenizer only on `train.jsonl`.

Format each sequence as:

`<bos><user>{message}<assistant>{reply}<eos>`

Compute loss only on assistant reply tokens. This makes the limited parameter budget learn conditional response behavior rather than spend most updates reconstructing user text.

## Model and training

The target architecture is 4 pre-norm decoder blocks, `d_model=160`, 5 heads, `d_ff=640`, learned positional embeddings, GELU MLPs, and tied token input/output weights. Context is 256 tokens. Use next-token cross-entropy, AdamW, gradient clipping, warmup followed by cosine decay, deterministic seeds, validation-based best checkpoint selection, and MLX Metal execution.

Start with a smoke run. Then train for enough batches to expose the model to about 8M non-padding tokens. If held-out loss stalls or outputs collapse, adjust learning rate, intent balance, or decoding before increasing model size.

## Evaluation gates

- Provenance: tokenizer is trained locally; model begins from random parameters; no pretrained weights are loaded.
- Architecture: four decoder layers, 256-token context, total trainable parameters in the 1–2M range.
- Language: at least 95% of sampled replies contain no formatting markers and end normally; at least 80% satisfy Minionese-style lexical/phonetic heuristics.
- Relevance: held-out prompts across every intent are scored against intent-specific keyword groups; target macro intent success is at least 70%.
- Input modes: report English→Minionese and Minionese→Minionese success separately.
- Baseline: trained test loss must beat a unigram/token-frequency baseline.
- Manual transcript: include at least ten unseen prompts covering greeting, emotion, food, plans, help, thanks, and farewell.

## Known limitations

The output is an original Minionese-inspired project dialect, not an official Illumination translator. A model this small has narrow conversational coverage and limited compositional reasoning. It should answer out-of-domain factual questions playfully rather than invent authoritative facts.
