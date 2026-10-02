import bisect
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

Piece = tuple[int, int, int]


class TokenFiles:
    """Merged token files from dataflow (`.ds` tokens, `.ds.index` document ends, `.ds.meta`), memory-mapped."""

    def __init__(self, folder: str | Path):
        self.paths = sorted(Path(folder).glob("*.ds"))
        if not self.paths:
            raise FileNotFoundError(f"no .ds files in {folder}")
        metas = [json.loads(Path(f"{path}.meta").read_text()) for path in self.paths]
        self.eos_token_id = metas[0]["eos_token_id"]
        self.tokens = [np.memmap(path, dtype=f"<u{meta['token_bytes']}", mode="r") for path, meta in zip(self.paths, metas)]
        self.ends = [np.fromfile(f"{path}.index", dtype="<u8").astype(np.int64) for path in self.paths]
        self.first_document = np.cumsum([0, *(len(ends) for ends in self.ends)])

    def span(self, file: int, document: int) -> tuple[int, int]:
        ends = self.ends[file]
        return (int(ends[document - 1]) if document else 0), int(ends[document])

    def read(self, piece: Piece) -> np.ndarray:
        length, file, start = piece
        return np.asarray(self.tokens[file][start : start + length])


class PackedLoader:
    """Rows of `seq_len + 1` tokens packed best-fit from whole documents, sharded by document across ranks.

    The buffer holds pieces (length, file, start): documents longer than a row are split into row-sized pieces. A row
    is filled with the longest piece that still fits, again and again; when none fits, the longest piece is cut (long
    documents are split anyway) and its rest goes back to the buffer, so no token is dropped. `state_dict()` is the stream position plus the buffer,
    enough to continue with exactly the same batches.
    """

    def __init__(
        self,
        folder: str | Path,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 5000,
        repeat: bool = True,
    ):
        self.files = TokenFiles(folder)
        self.batch_size = batch_size
        self.row = seq_len + 1
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.repeat = repeat
        self.epoch, self.file, self.document = 0, 0, 0
        self.buffer: list[Piece] = []

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "file": self.file, "document": self.document, "buffer": list(self.buffer)}

    def load_state_dict(self, state: dict) -> None:
        self.epoch, self.file, self.document = state["epoch"], state["file"], state["document"]
        self.buffer = [tuple(piece) for piece in state["buffer"]]

    def next_document(self) -> bool:
        """Add this rank's next document to the buffer; False when the stream (and `repeat`) is exhausted."""
        while True:
            if self.file == len(self.files.paths):
                if not self.repeat:
                    return False
                self.epoch, self.file, self.document = self.epoch + 1, 0, 0
            if self.document == len(self.files.ends[self.file]):
                self.file, self.document = self.file + 1, 0
                continue
            file, document = self.file, self.document
            self.document += 1
            if (self.files.first_document[file] + document) % self.world_size == self.rank:
                start, end = self.files.span(file, document)
                for offset in range(start, end, self.row):
                    bisect.insort(self.buffer, (min(self.row, end - offset), file, offset))
                return True

    def buffered(self) -> int:
        return sum(piece[0] for piece in self.buffer)

    def pack_row(self) -> np.ndarray | None:
        while (len(self.buffer) < self.buffer_size or self.buffered() < self.row) and self.next_document():
            pass
        if self.buffered() < self.row:
            return None
        parts, remaining = [], self.row
        while remaining:
            fits = bisect.bisect_right(self.buffer, (remaining, len(self.files.paths), 0)) - 1
            if fits >= 0:
                piece = self.buffer.pop(fits)
            else:
                length, file, start = self.buffer.pop()
                piece = (remaining, file, start)
                bisect.insort(self.buffer, (length - remaining, file, start + remaining))
            parts.append(self.files.read(piece))
            remaining -= piece[0]
        return np.concatenate(parts)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        rows = []
        for _ in range(self.batch_size):
            row = self.pack_row()
            if row is None:
                raise StopIteration
            rows.append(row)
        return torch.from_numpy(np.stack(rows).astype(np.int64))
