import json

import numpy as np

from akshara.evals import EVAL_SETS, NgramIndex, ngram_hashes, words_of
from akshara.evals.freeze import rows_of, text_of


def test_set_names_are_unique():
    names = [eval_set.name for eval_set in EVAL_SETS]
    assert len(names) == len(set(names))


def test_ngram_hashes_ignore_case_digits_and_punctuation():
    first = ngram_hashes(words_of("The capital of France is Paris, built in 1200."), 8)
    second = ngram_hashes(words_of("the CAPITAL of france is paris built in 3456"), 8)
    assert len(first) == 2
    assert np.array_equal(first, second)


def test_ngram_hashes_depend_on_word_order():
    assert not np.array_equal(ngram_hashes("a b c".split(), 3), ngram_hashes("c b a".split(), 3))
    assert len(ngram_hashes("too short".split(), 3)) == 0


def test_telugu_words_keep_their_vowel_signs():
    assert words_of("తెలుగు భాష, చాలా అందమైనది!") == ["తెలుగు", "భాష", "చాలా", "అందమైనది"]


def test_index_finds_which_items_an_ngram_came_from():
    items = ["one two three four five", "nothing in common here at all", "zero one two three four"]
    index = NgramIndex.build(items, 3)
    probe = ngram_hashes(words_of("one two three"), 3)
    assert index.contains(probe).tolist() == [True]
    assert sorted(index.items_of(int(probe[0])).tolist()) == [0, 2]
    assert not index.contains(ngram_hashes("five six seven".split(), 3)).any()


def test_index_round_trips_through_disk(tmp_path):
    index = NgramIndex.build(["a b c d", "b c d e"], 2)
    index.save(tmp_path / "index.npz")
    loaded = NgramIndex.load(tmp_path / "index.npz")
    assert loaded.n == 2
    assert np.array_equal(loaded.hashes, index.hashes)
    assert np.array_equal(loaded.items, index.items)


def test_rows_of_reads_json_jsonl_wrapped_lists_and_plain_lines():
    assert rows_of(b'{"q": "a"}\n{"q": "b"}\n', "test.jsonl") == [{"q": "a"}, {"q": "b"}]
    wrapped = json.dumps({"version": "1", "data": [{"q": "a"}, {"q": "b"}]}).encode()
    assert rows_of(wrapped, "test.json") == [{"q": "a"}, {"q": "b"}]
    assert rows_of("ఒకటి\nరెండు\n".encode(), "tel_Telu.devtest") == [{"text": "ఒకటి"}, {"text": "రెండు"}]


def test_rows_of_keeps_unicode_line_separators_inside_jsonl_strings():
    payload = '{"q": "a\u2028b"}\n{"q": "c\u0085d"}\n'.encode()
    assert rows_of(payload, "question.jsonl") == [{"q": "a\u2028b"}, {"q": "c\u0085d"}]


def test_text_of_collects_every_string_in_field_order():
    row = {"question": "Which?", "choices": ["red", "blue"], "label": 1, "meta": {"source": "quiz"}}
    assert text_of(row) == "Which?\nred\nblue\nquiz"
