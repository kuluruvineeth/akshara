import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dataflow.pipeline.decont import NgramIndex
from dataflow.utils.cache import cached_download
from huggingface_hub import HfApi, get_session, hf_hub_download
from huggingface_hub.utils import build_hf_headers

from akshara.evals.sets import EVAL_SETS, NOT_FROZEN, EvalSet

PARQUET_LISTING = "https://datasets-server.huggingface.co/parquet?dataset={repo}"
CONVERTED = "refs/convert/parquet"
SCHEMA = pa.schema([("id", pa.int64()), ("text", pa.string()), ("row", pa.string())])
NGRAM_SIZES = (8, 13)


def text_of(value) -> str:
    """Every string in a row, in field order, so the n-grams see questions, choices and answers alike."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [text_of(part) for part in value.values()]
    elif isinstance(value, list | tuple):
        parts = [text_of(part) for part in value]
    else:
        return ""
    return "\n".join(part for part in parts if part)


def rows_of(payload: bytes, filename: str) -> list[dict]:
    # split on "\n" only: splitlines() also breaks on U+2028 and U+0085, which occur inside JSON strings
    lines = payload.decode("utf-8").split("\n")
    if not filename.endswith((".json", ".jsonl")):
        return [{"text": line.rstrip("\r")} for line in lines if line.strip()]
    try:
        data = json.loads("\n".join(lines))
    except json.JSONDecodeError:
        return [json.loads(line) for line in lines if line.strip()]
    if isinstance(data, dict):
        lists = [value for value in data.values() if isinstance(value, list)]
        data = max(lists, key=len) if lists else [data]
    return [row if isinstance(row, dict) else {"value": row} for row in data]


def converted_files(eval_set: EvalSet) -> list[str]:
    response = get_session().get(PARQUET_LISTING.format(repo=eval_set.repo), headers=build_hf_headers(), timeout=60)
    response.raise_for_status()
    files = [
        entry["url"].split(f"/resolve/{CONVERTED.replace('/', '%2F')}/", 1)[1]
        for entry in response.json()["parquet_files"]
        if entry["split"] == eval_set.split and eval_set.config in (None, entry["config"])
    ]
    if not files:
        raise ValueError(f"{eval_set.name}: no parquet for config={eval_set.config} split={eval_set.split}")
    return files


def load_rows(eval_set: EvalSet, api: HfApi) -> tuple[list[dict], str]:
    """The set's rows and the exact revision they came from (commit sha, or sha256 for an archive)."""
    if eval_set.repo.startswith("https://"):
        archive = cached_download(eval_set.repo, "evals/" + eval_set.repo.rsplit("/", 1)[1])
        revision = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
        with tarfile.open(archive) as tar:
            rows = [row for name in eval_set.files for row in rows_of(tar.extractfile(name).read(), name)]
        return rows, revision
    if eval_set.files:
        revision = api.dataset_info(eval_set.repo).sha
        rows = []
        for name in eval_set.files:
            path = hf_hub_download(eval_set.repo, name, repo_type="dataset", revision=revision)
            rows.extend(rows_of(Path(path).read_bytes(), name))
        return rows, revision
    revision = api.dataset_info(eval_set.repo, revision=CONVERTED).sha
    rows = []
    for name in converted_files(eval_set):
        path = hf_hub_download(eval_set.repo, name, repo_type="dataset", revision=revision)
        config = name.split("/", 1)[0]
        for row in pq.read_table(path).to_pylist():
            rows.append(row if eval_set.config else {"config": config, **row})
    return rows, revision


def freeze(eval_set: EvalSet, output: Path, api: HfApi) -> tuple[dict, list[str]]:
    rows, revision = load_rows(eval_set, api)
    if eval_set.columns:
        rows = [{column: row[column] for column in eval_set.columns} for row in rows]
    texts = [text_of(row) for row in rows]
    table = pa.table(
        {
            "id": list(range(len(rows))),
            "text": texts,
            "row": [json.dumps(row, ensure_ascii=False, default=str) for row in rows],
        },
        schema=SCHEMA,
    )
    path = output / "sets" / f"{eval_set.name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    entry = {
        "name": eval_set.name,
        "suite": eval_set.suite,
        "language": eval_set.language,
        "repo": eval_set.repo,
        "config": eval_set.config,
        "split": eval_set.split,
        "files": list(eval_set.files),
        "columns": list(eval_set.columns),
        "revision": revision,
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return entry, texts


def freeze_all(output: Path, eval_sets=EVAL_SETS) -> dict:
    api = HfApi()
    entries, texts = [], []
    for eval_set in eval_sets:
        entry, set_texts = freeze(eval_set, output, api)
        entry["first_item"] = len(texts)
        entries.append(entry)
        texts.extend(set_texts)
        print(f"{eval_set.name:<28} {entry['rows']:>8,} rows  {entry['revision'][:12]}", flush=True)
    indexes = {}
    for n in NGRAM_SIZES:
        index = NgramIndex.build(texts, n)
        index.save(output / f"ngrams-{n}.npz")
        indexes[n] = {"file": f"ngrams-{n}.npz", "ngrams": len(index), "items": len(texts)}
        print(f"{n}-gram index: {len(index):,} n-grams over {len(texts):,} items", flush=True)
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "normalization": "dataflow simplify_text (lowercase, digits to 0, punctuation to space), whitespace words",
        "sets": entries,
        "indexes": indexes,
        "not_frozen": NOT_FROZEN,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the eval sets and build their n-gram indexes")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sets", nargs="*", help="only these set names")
    parser.add_argument("--upload", help="bucket destination, e.g. hf://buckets/<namespace>/akshara-evals/frozen/v1")
    args = parser.parse_args()
    eval_sets = [eval_set for eval_set in EVAL_SETS if not args.sets or eval_set.name in args.sets]
    freeze_all(args.output, eval_sets)
    if args.upload:
        HfApi().sync_bucket(str(args.output), args.upload)


if __name__ == "__main__":
    main()
