from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dataflow.utils.hashing import hash64
from dataflow.utils.text import simplify_text

MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def words_of(text: str) -> list[str]:
    return simplify_text(text).split()


def ngram_hashes(words: list[str], n: int) -> np.ndarray:
    """One 64-bit hash per n-gram, combined from per-word hashes so it costs one hash per word, not per n-gram."""
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    word_hashes = np.fromiter((hash64(word) for word in words), dtype=np.uint64, count=len(words))
    count = len(words) - n + 1
    hashes = word_hashes[:count].copy()
    for offset in range(1, n):
        hashes = hashes * MULTIPLIER + word_hashes[offset : offset + count]
    return hashes


@dataclass
class NgramIndex:
    """Sorted n-gram hashes of every eval item, each paired with the item it came from."""

    n: int
    hashes: np.ndarray
    items: np.ndarray

    @classmethod
    def build(cls, texts: Iterable[str], n: int) -> "NgramIndex":
        hashes, items = [np.empty(0, dtype=np.uint64)], [np.empty(0, dtype=np.uint32)]
        for item, text in enumerate(texts):
            unique = np.unique(ngram_hashes(words_of(text), n))
            hashes.append(unique)
            items.append(np.full(len(unique), item, dtype=np.uint32))
        all_hashes, all_items = np.concatenate(hashes), np.concatenate(items)
        order = np.argsort(all_hashes, kind="stable")
        return cls(n, all_hashes[order], all_items[order])

    def __len__(self) -> int:
        return len(self.hashes)

    def contains(self, hashes: np.ndarray) -> np.ndarray:
        if not len(self.hashes):
            return np.zeros(len(hashes), dtype=bool)
        positions = np.minimum(np.searchsorted(self.hashes, hashes), len(self.hashes) - 1)
        return self.hashes[positions] == hashes

    def items_of(self, ngram_hash: int) -> np.ndarray:
        value = np.uint64(ngram_hash)
        start, end = np.searchsorted(self.hashes, value, "left"), np.searchsorted(self.hashes, value, "right")
        return self.items[start:end]

    def save(self, path: str | Path) -> None:
        np.savez(path, n=self.n, hashes=self.hashes, items=self.items)

    @classmethod
    def load(cls, path: str | Path) -> "NgramIndex":
        with np.load(path) as data:
            return cls(int(data["n"]), data["hashes"], data["items"])
