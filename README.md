# akshara

An open English + Telugu language model family: data recipes, tokenizer, model, training and evaluation in one
repository, so a single commit pins how every released model was made.

## Idea

Akshara is built end to end in the open: corpora prepared with [dataflow](https://github.com/kuluruvineeth/dataflow),
our own tokenizer, our own trainer, and every result measured and published.

Telugu is spoken by tens of millions of people but is poorly served by open tokenizers and data. Akshara treats it as
a first-class language, starting from the tokenizer: it keeps Telugu words whole instead of cutting them at every
vowel sign.

## Tokenizer

```bash
uv run akshara-tokenizer train artifacts/tokenizer/tokenizer.json   # byte-level BPE, 65,536 tokens
uv run akshara-tokenizer fertility artifacts/tokenizer/tokenizer.json HuggingFaceTB/SmolLM2-135M
```

`fertility` reports tokens per word and bytes per token on the FLORES-200 devtest sets for English and Telugu.

The first version is published as [`kuluruvineeth/akshara-tokenizer-v0`](https://huggingface.co/kuluruvineeth/akshara-tokenizer-v0):
1.69 tokens per Telugu word and 1.32 per English word. The
[Telugu Tokenizer Leaderboard](https://huggingface.co/spaces/kuluruvineeth/telugu-tokenizer-leaderboard) compares
it with 26 other tokenizers.

## Development

dataflow is used as a local editable dependency, so clone both repositories side by side:

```bash
git clone https://github.com/kuluruvineeth/dataflow
git clone https://github.com/kuluruvineeth/akshara
cd akshara
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
