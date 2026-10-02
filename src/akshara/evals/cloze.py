import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


@dataclass(frozen=True)
class ClozeItem:
    """One multiple-choice question as (context, continuation) pairs, one per choice; `gold` is the right one."""

    pairs: tuple[tuple[str, str], ...]
    gold: int


def shared(context: str, choices: list[str], gold: int) -> ClozeItem:
    return ClozeItem(tuple((context, " " + choice.strip()) for choice in choices), gold)


def wikihow(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ").strip()


def hellaswag(row: dict) -> ClozeItem:
    context = wikihow(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
    return shared(context, [wikihow(ending) for ending in row["endings"]], int(row["label"]))


def question(word: str, answer: str, field: str = "question") -> Callable[[dict], str]:
    return lambda row: f"{word}: {row[field]}\n{answer}:"


def lettered(prompt: Callable[[dict], str]) -> Callable[[dict], ClozeItem]:
    """ARC, CommonsenseQA, OpenBookQA: choices with letter labels and a letter answer."""
    return lambda row: shared(prompt(row), row["choices"]["text"], row["choices"]["label"].index(row["answerKey"]))


def mmlu(prompt: Callable[[dict], str]) -> Callable[[dict], ClozeItem]:
    def item(row: dict) -> ClozeItem:
        answer = row["answer"]
        return shared(prompt(row), row["choices"], answer if isinstance(answer, int) else "ABCD".index(answer))

    return item


def piqa(row: dict) -> ClozeItem:
    return shared(f"Question: {row['goal']}\nAnswer:", [row["sol1"], row["sol2"]], int(row["label"]))


def winogrande(row: dict) -> ClozeItem:
    """The choices change the context; the rest of the sentence is the continuation scored for each."""
    blank = row["sentence"].index("_")
    rest = " " + row["sentence"][blank + 1 :].strip()
    contexts = [row["sentence"][:blank] + row[option] for option in ("option1", "option2")]
    return ClozeItem(tuple((context, rest) for context in contexts), int(row["answer"]) - 1)


def belebele(row: dict) -> ClozeItem:
    context = f"{row['flores_passage']}\nప్రశ్న: {row['question']}\nసమాధానం:"
    choices = [row[f"mc_answer{number}"] for number in range(1, 5)]
    return shared(context, choices, int(row["correct_answer_num"]) - 1)


def xstory_cloze(row: dict) -> ClozeItem:
    story = " ".join(row[f"input_sentence_{number}"] for number in range(1, 5))
    return shared(story, [row["sentence_quiz1"], row["sentence_quiz2"]], int(row["answer_right_ending"]) - 1)


def copa_te(row: dict) -> ClozeItem:
    connector = "ఎందుకంటే" if row["question"] == "cause" else "కాబట్టి"
    return shared(f"{row['premise'].rstrip('.')} {connector}", [row["choice1"], row["choice2"]], int(row["label"]))


TELUGU_QUESTION = question("ప్రశ్న", "సమాధానం")
TASKS: dict[str, Callable[[dict], ClozeItem]] = {
    "hellaswag": hellaswag,
    "arc-easy": lettered(question("Question", "Answer")),
    "arc-challenge": lettered(question("Question", "Answer")),
    "piqa": piqa,
    "winogrande": winogrande,
    "commonsense-qa": lettered(question("Question", "Answer")),
    "openbookqa": lettered(question("Question", "Answer", "question_stem")),
    "mmlu": mmlu(question("Question", "Answer")),
    "belebele-te": belebele,
    "xstory-cloze-te": xstory_cloze,
    "indiccopa-te": copa_te,
    "hellaswag-te": hellaswag,
    "arc-te": lettered(TELUGU_QUESTION),
    "mmlu-te": mmlu(TELUGU_QUESTION),
}
CORE_TASKS = ("hellaswag", "arc-easy", "arc-challenge", "piqa", "winogrande", "commonsense-qa", "openbookqa")


def load_items(name: str, sets_folder: str | Path, limit: int | None = None, seed: int = 0) -> list[ClozeItem]:
    """A task's items from the frozen set, a fixed random sample of `limit` when given."""
    rows = pq.read_table(Path(sets_folder) / f"{name}.parquet", columns=["row"]).column("row").to_pylist()
    order = np.random.default_rng(seed).permutation(len(rows)) if limit and limit < len(rows) else range(len(rows))
    return [TASKS[name](json.loads(rows[index])) for index in list(order)[:limit]]


def encode_pair(tokenizer: Tokenizer, context: str, continuation: str) -> tuple[list[int], int]:
    """Tokens of context + continuation and where the continuation starts, splitting at the context's own tokens
    when the whole string tokenizes that way."""
    context_ids = tokenizer.encode(context, add_special_tokens=False).ids
    whole = tokenizer.encode(context + continuation, add_special_tokens=False).ids
    if whole[: len(context_ids)] == context_ids and len(whole) > len(context_ids):
        return whole, len(context_ids)
    return context_ids + tokenizer.encode(continuation, add_special_tokens=False).ids, len(context_ids)


@torch.no_grad()
def continuation_scores(
    model: torch.nn.Module, tokenizer: Tokenizer, pairs: list[tuple[str, str]], device, max_len: int
) -> list[float]:
    """Log-probability of each continuation given its context, per UTF-8 byte of the continuation."""
    encoded = [encode_pair(tokenizer, context, continuation) for context, continuation in pairs]
    encoded = [(ids[-max_len:], max(1, start - max(0, len(ids) - max_len))) for ids, start in encoded]
    width = max(len(ids) for ids, _ in encoded)
    tokens = torch.zeros(len(encoded), width, dtype=torch.long)
    mask = torch.zeros(len(encoded), width - 1, dtype=torch.bool)
    for row, (ids, start) in enumerate(encoded):
        tokens[row, : len(ids)] = torch.tensor(ids)
        mask[row, start - 1 : len(ids) - 1] = True
    tokens, mask = tokens.to(device), mask.to(device)
    logprobs = F.log_softmax(model(tokens[:, :-1]).float(), dim=-1)
    picked = logprobs.gather(-1, tokens[:, 1:, None]).squeeze(-1)
    totals = (picked * mask).sum(dim=1).tolist()
    return [
        total / max(1, len(continuation.encode("utf-8")))
        for total, (_, continuation) in zip(totals, pairs, strict=True)
    ]


def accuracy(model, tokenizer: Tokenizer, items: list[ClozeItem], device, max_len: int = 2048) -> float:
    model.eval()
    correct = 0
    for item in items:
        scores = continuation_scores(model, tokenizer, list(item.pairs), device, max_len)
        correct += int(np.argmax(scores)) == item.gold
    return correct / len(items)


def run_cloze(
    model, tokenizer: Tokenizer, sets_folder: str | Path, tasks=tuple(TASKS), limit: int = 1000, device="cpu"
) -> dict[str, float]:
    return {name: accuracy(model, tokenizer, load_items(name, sets_folder, limit), device) for name in tasks}


def random_baseline(name: str, sets_folder: str | Path) -> float:
    items = load_items(name, sets_folder, limit=200)
    return float(np.mean([1 / len(item.pairs) for item in items]))


def core_subset(results: dict[str, float], baselines: dict[str, float]) -> float:
    """Mean centered accuracy over the CORE tasks we run: 0 is chance, 1 is perfect (DCLM's CORE convention)."""
    tasks = [name for name in CORE_TASKS if name in results]
    return float(np.mean([(results[name] - baselines[name]) / (1 - baselines[name]) for name in tasks]))
