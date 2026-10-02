import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dataflow.pipeline.decont import NgramIndex, covered_fraction, text_ngrams

from akshara.evals.freeze import text_of
from akshara.evals.sets import PROMPT_FIELDS

ITEM_THRESHOLD = 0.5
MAX_ITEMS = 10
SOURCE_THRESHOLD = 0.02


def per_benchmark(manifest: dict, items: np.ndarray) -> list[dict]:
    """Contaminated items, counted per eval set, from global item ids."""
    rows = []
    for entry in manifest["sets"]:
        first, count = entry["first_item"], entry["rows"]
        contaminated = int(((items >= first) & (items < first + count)).sum())
        rows.append(
            {
                "name": entry["name"],
                "suite": entry["suite"],
                "rows": count,
                "contaminated": contaminated,
                "share": contaminated / count if count else 0.0,
            }
        )
    return rows


class ItemPrompts:
    """The prompt of any eval item by global id: its `PROMPT_FIELDS` from the frozen row, or the whole item text for
    sets without them. Read from the frozen sets on first use."""

    def __init__(self, manifest: dict, sets_folder: str | Path):
        self.entries = manifest["sets"]
        self.starts = np.array([entry["first_item"] for entry in self.entries])
        self.sets_folder = Path(sets_folder)
        self.loaded: dict[str, list[str]] = {}

    def load(self, name: str) -> list[str]:
        path = self.sets_folder / f"{name}.parquet"
        if name not in PROMPT_FIELDS:
            return pq.read_table(path, columns=["text"]).column("text").to_pylist()
        rows = (json.loads(row) for row in pq.read_table(path, columns=["row"]).column("row").to_pylist())
        return [text_of({field: row.get(field) for field in PROMPT_FIELDS[name]}) for row in rows]

    def __getitem__(self, item: int) -> str:
        entry = self.entries[int(np.searchsorted(self.starts, item, "right")) - 1]
        if entry["name"] not in self.loaded:
            self.loaded[entry["name"]] = self.load(entry["name"])
        return self.loaded[entry["name"]][item - entry["first_item"]]


def instruction_contamination(
    prompts: Iterable[str],
    index: NgramIndex,
    manifest: dict,
    sets_folder: str | Path,
    item_threshold: float = ITEM_THRESHOLD,
    max_items: int = MAX_ITEMS,
) -> list[dict]:
    """Tulu 3's rule for a training source: an eval item is matched when more than half of its prompt's words sit in
    n-grams that also occur in the source's prompts. Prompts on both sides, never responses or answers. N-grams shared
    by more than `max_items` eval items are templates and never count."""
    found = [text_ngrams(prompt, index.n) for prompt in prompts]
    hashes = np.unique(np.concatenate([np.empty(0, dtype=np.uint64), *found]))
    matched = np.setdiff1d(hashes[index.contains(hashes)], index.shared_ngrams(max_items))
    prompts_of = ItemPrompts(manifest, sets_folder)
    contaminated = [
        item
        for item in index.items_with(matched)
        if covered_fraction(prompts_of[item], index.n, matched) > item_threshold
    ]
    return per_benchmark(manifest, np.array(contaminated, dtype=np.int64))


def source_is_contaminated(report: list[dict], source_threshold: float = SOURCE_THRESHOLD) -> bool:
    """Drop the whole source if it contaminates more than 2% of any eval set."""
    return any(row["share"] > source_threshold for row in report)
