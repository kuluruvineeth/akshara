import json

from akshara.evals import EVAL_SETS
from akshara.evals.freeze import rows_of, text_of


def test_set_names_are_unique():
    names = [eval_set.name for eval_set in EVAL_SETS]
    assert len(names) == len(set(names))


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
