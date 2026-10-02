import unicodedata
from collections.abc import Iterable, Iterator
from itertools import chain

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 65_536
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|image|>",
    "<|audio|>",
    *(f"<|reserved_{i}|>" for i in range(53)),
]
TELUGU_BLOCK = range(0x0C00, 0x0C80)


def telugu_characters() -> list[str]:
    return [chr(code) for code in TELUGU_BLOCK if unicodedata.category(chr(code)) != "Cn"]


def seed_lines(characters: list[str], repeats: int) -> Iterator[str]:
    for character in characters:
        yield " ".join([character] * repeats)


def empty_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(texts: Iterable[str], vocab_size: int = VOCAB_SIZE, seed_repeats: int = 1000) -> Tokenizer:
    tokenizer = empty_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(chain(texts, seed_lines(telugu_characters(), seed_repeats)), trainer)
    return tokenizer
