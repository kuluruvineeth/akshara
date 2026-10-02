import json
from collections import Counter

import numpy as np
import pytest
import torch
from dataflow.io import get_datafolder
from dataflow.pipeline.tokens.tokenizer import write_index_and_meta

from akshara.data import PackedLoader

EOS = 0


def write_files(folder, documents: list[list[int]], files: int = 2) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    datafolder = get_datafolder(str(folder))
    for part in range(files):
        chunk = documents[part::files]
        tokens = np.concatenate([np.array(document, dtype="<u2") for document in chunk])
        (folder / f"merged_{part:03d}.ds").write_bytes(tokens.tobytes())
        ends = np.cumsum([len(document) for document in chunk]).tolist()
        write_index_and_meta(datafolder, f"merged_{part:03d}.ds", ends, {"token_bytes": 2, "eos_token_id": EOS})


def labelled(lengths: list[int]) -> list[list[int]]:
    """Document k is the token k + 1 repeated, ending in EOS, so every token says which document it came from."""
    return [[k + 1] * (length - 1) + [EOS] for k, length in enumerate(lengths)]


def test_rows_pack_whole_documents_when_they_fit(tmp_path):
    lengths = [6, 4, 7, 3, 5, 5, 2, 8, 9, 1] * 4
    write_files(tmp_path, labelled(lengths))
    loader = PackedLoader(tmp_path, batch_size=2, seq_len=9, repeat=False)
    batches = list(loader)
    assert all(batch.shape == (2, 10) for batch in batches)
    rows = torch.cat(batches).tolist()
    counts = [Counter(row) for row in rows]
    whole = sum(count[label] == lengths[label - 1] - 1 for count in counts for label in set(count) - {EOS})
    pieces = sum(len(set(count) - {EOS}) for count in counts)
    assert whole / pieces > 0.9


def test_no_token_is_lost_and_the_leftover_is_less_than_a_row(tmp_path):
    documents = labelled([13, 2, 30, 7, 7, 1, 25, 4] * 3)
    write_files(tmp_path, documents)
    rows = torch.cat(list(PackedLoader(tmp_path, batch_size=1, seq_len=15, repeat=False))).flatten().tolist()
    everything = Counter(token for document in documents for token in document)
    emitted = Counter(rows)
    assert not emitted - everything
    assert 0 <= sum((everything - emitted).values()) < 16


def test_documents_longer_than_a_row_are_split_into_pieces(tmp_path):
    write_files(tmp_path, labelled([25, 3]), files=1)
    loader = PackedLoader(tmp_path, batch_size=1, seq_len=9, repeat=False)
    loader.next_document()
    assert [length for length, _, _ in loader.buffer] == [5, 10, 10]


def test_ranks_see_disjoint_documents_that_cover_everything(tmp_path):
    lengths = [5] * 40
    write_files(tmp_path, labelled(lengths))
    seen = []
    for rank in range(2):
        rows = torch.cat(list(PackedLoader(tmp_path, batch_size=1, seq_len=9, rank=rank, world_size=2, repeat=False)))
        seen.append(set(rows.flatten().tolist()) - {EOS})
    assert not seen[0] & seen[1]
    assert seen[0] | seen[1] == set(range(1, 41))


def test_resume_continues_with_exactly_the_same_batches(tmp_path):
    rng = np.random.default_rng(0)
    write_files(tmp_path, labelled(rng.integers(2, 40, size=300).tolist()))
    reference = PackedLoader(tmp_path, batch_size=4, seq_len=31, buffer_size=50)
    expected = [next(reference) for _ in range(12)]
    first = PackedLoader(tmp_path, batch_size=4, seq_len=31, buffer_size=50)
    for _ in range(5):
        next(first)
    state = json.loads(json.dumps(first.state_dict()))
    resumed = PackedLoader(tmp_path, batch_size=4, seq_len=31, buffer_size=50)
    resumed.load_state_dict(state)
    for batch in expected[5:]:
        assert torch.equal(next(resumed), batch)


def test_training_data_repeats_across_epochs(tmp_path):
    write_files(tmp_path, labelled([5] * 4), files=1)
    loader = PackedLoader(tmp_path, batch_size=1, seq_len=9, buffer_size=1)
    for _ in range(5):
        next(loader)
    assert loader.epoch >= 1


def test_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        PackedLoader(tmp_path, batch_size=1, seq_len=8)
