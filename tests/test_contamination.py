import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from akshara.evals import NgramIndex, instruction_contamination, per_benchmark, source_is_contaminated

QUIZ = [
    "What is the boiling point of water at sea level in degrees Celsius on a normal day",
    "Name the longest river that flows through the continent of Africa from the south to the north",
]
STORY = ["Once upon a time a small fox lived at the edge of a very large and quiet forest"]


def frozen(tmp_path) -> dict:
    manifest = {"sets": []}
    for name, texts in [("quiz", QUIZ), ("story", STORY)]:
        first = sum(entry["rows"] for entry in manifest["sets"])
        manifest["sets"].append({"name": name, "suite": "base", "rows": len(texts), "first_item": first})
        pq.write_table(pa.table({"id": list(range(len(texts))), "text": texts}), tmp_path / f"{name}.parquet")
    return manifest


def test_per_benchmark_counts_items_inside_each_set():
    manifest = {
        "sets": [
            {"name": "a", "suite": "x", "rows": 4, "first_item": 0},
            {"name": "b", "suite": "x", "rows": 2, "first_item": 4},
        ]
    }
    report = per_benchmark(manifest, np.array([1, 3, 5]))
    assert [(row["name"], row["contaminated"], row["share"]) for row in report] == [("a", 2, 0.5), ("b", 1, 0.5)]


def test_an_item_counts_only_when_most_of_its_words_are_covered(tmp_path):
    manifest = frozen(tmp_path)
    index = NgramIndex.build(QUIZ + STORY, 8)
    prompts = [
        f"Answer this: {QUIZ[0]}? Explain briefly.",
        "Name the longest river that flows through the city",
        "Tell me a story about a dragon.",
    ]
    report = instruction_contamination(prompts, index, manifest, tmp_path)
    assert {row["name"]: row["contaminated"] for row in report} == {"quiz": 1, "story": 0}
    assert source_is_contaminated(report)


def test_coverage_is_measured_on_the_eval_prompt_not_the_answer(tmp_path):
    question = "Janet has three baskets with twelve apples in each basket and gives away five of them"
    answer = (
        "She starts with three times twelve which is thirty six apples and after giving five she has thirty one left"
    )
    row = json.dumps({"question": question, "answer": answer})
    pq.write_table(pa.table({"id": [0], "text": [f"{question}\n{answer}"], "row": [row]}), tmp_path / "gsm8k.parquet")
    manifest = {"sets": [{"name": "gsm8k", "suite": "base", "rows": 1, "first_item": 0}]}
    index = NgramIndex.build([f"{question}\n{answer}"], 8)
    report = instruction_contamination([question], index, manifest, tmp_path)
    assert report[0]["contaminated"] == 1


def test_a_clean_source_passes(tmp_path):
    manifest = frozen(tmp_path)
    index = NgramIndex.build(QUIZ + STORY, 8)
    report = instruction_contamination(["Write a haiku about rain on a tin roof."], index, manifest, tmp_path)
    assert sum(row["contaminated"] for row in report) == 0
    assert not source_is_contaminated(report)
