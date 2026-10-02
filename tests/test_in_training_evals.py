import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from akshara.evals.bpb import bits_per_byte, token_bytes
from akshara.evals.cloze import (
    ClozeItem,
    accuracy,
    continuation_scores,
    copa_te,
    core_subset,
    hellaswag,
    load_items,
    winogrande,
)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    corpus = ["the cat sat on the mat", "yes no maybe", "తెలుగు భాష చాలా అందమైనది", "a dog ran in the park"] * 50
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


class Uniform(torch.nn.Module):
    def __init__(self, vocab: int, favourite: int | None = None):
        super().__init__()
        self.vocab, self.favourite = vocab, favourite

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*tokens.shape, self.vocab)
        if self.favourite is not None:
            logits[..., self.favourite] = 10.0
        return logits


def test_token_bytes_add_up_to_the_utf8_length(tokenizer):
    sizes = token_bytes(tokenizer)
    text = "the cat sat on తెలుగు mat"
    ids = tokenizer.encode(text).ids
    assert int(sizes[ids].sum()) == len(text.encode("utf-8"))
    assert sizes[tokenizer.token_to_id("<|endoftext|>")] == 0


def test_a_uniform_model_scores_log2_vocab_bits_per_token(tokenizer):
    sizes = token_bytes(tokenizer)
    eos = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode("the cat sat on the mat").ids + [eos]
    batch = torch.tensor([ids])
    vocab = tokenizer.get_vocab_size()
    targets = batch[0, 1:]
    expected = math.log2(vocab) * int((sizes[targets] > 0).sum()) / int(sizes[targets].sum())
    assert bits_per_byte(Uniform(vocab), [batch], sizes, "cpu") == pytest.approx(expected)


def test_continuation_scores_count_only_continuation_tokens(tokenizer):
    vocab = tokenizer.get_vocab_size()
    pairs = [("the cat", " sat on the mat"), ("a dog ran in the", " park")]
    scores = continuation_scores(Uniform(vocab), tokenizer, pairs, "cpu", max_len=64)
    for score, (context, continuation) in zip(scores, pairs, strict=True):
        tokens = len(tokenizer.encode(context + continuation).ids) - len(tokenizer.encode(context).ids)
        assert score == pytest.approx(-math.log(vocab) * tokens / len(continuation.encode()))


def test_accuracy_picks_the_choice_the_model_prefers(tokenizer):
    yes = tokenizer.encode(" yes").ids[-1]
    items = [ClozeItem(pairs=(("maybe", " yes"), ("maybe", " no")), gold=0)] * 3
    assert accuracy(Uniform(tokenizer.get_vocab_size(), favourite=yes), tokenizer, items, "cpu") == 1.0


def test_task_formats():
    row = {
        "activity_label": "Roofing",
        "ctx_a": "A man climbs.",
        "ctx_b": "he",
        "endings": ["[step] falls.", "waves."],
    }
    context = "Roofing: A man climbs. He"
    assert hellaswag({**row, "label": "1"}) == ClozeItem(((context, " falls."), (context, " waves.")), 1)

    item = winogrande(
        {"sentence": "Sam beat Al because _ was fast.", "option1": "Sam", "option2": "Al", "answer": "1"}
    )
    assert item.pairs == (("Sam beat Al because Sam", " was fast."), ("Sam beat Al because Al", " was fast."))
    assert item.gold == 0

    copa = copa_te(
        {"premise": "వర్షం పడింది.", "choice1": "నేల తడిసింది.", "choice2": "ఎండ కాసింది.", "question": "effect", "label": "0"}
    )
    assert copa.pairs[0] == ("వర్షం పడింది కాబట్టి", " నేల తడిసింది.")


def test_load_items_takes_a_fixed_sample(tmp_path):
    rows = [json.dumps({"goal": f"goal {i}", "sol1": "a", "sol2": "b", "label": i % 2}) for i in range(50)]
    pq.write_table(pa.table({"row": rows}), tmp_path / "piqa.parquet")
    first, again = load_items("piqa", tmp_path, limit=10), load_items("piqa", tmp_path, limit=10)
    assert first == again and len(first) == 10 and len(load_items("piqa", tmp_path)) == 50


def test_core_subset_is_zero_at_chance_and_one_when_perfect():
    baselines = {"piqa": 0.5, "hellaswag": 0.25}
    assert core_subset({"piqa": 0.5, "hellaswag": 0.25}, baselines) == 0.0
    assert core_subset({"piqa": 1.0, "hellaswag": 1.0}, baselines) == 1.0
    assert core_subset({"piqa": 0.75, "hellaswag": 0.25, "mmlu": 0.9}, baselines) == pytest.approx(0.25)
