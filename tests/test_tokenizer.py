import io
import tarfile

import pytest
from tokenizers import Regex, pre_tokenizers

from akshara.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, load_flores, measure, telugu_characters, train_tokenizer
from akshara.tokenizer import fertility as fertility_module

TEXTS = [
    "The river runs through the old town, and people built 12 bridges in 1998.",
    "తెలుగు భాష చాలా అందమైనది. గోదావరి నది ఒడ్డున అనేక పట్టణాలు ఉన్నాయి.",
    "def add(a, b):\n    return a + b\n",
] * 50


@pytest.fixture(scope="module")
def tokenizer():
    return train_tokenizer(TEXTS, vocab_size=1200, seed_repeats=20)


def pieces(text):
    return [piece for piece, _ in pre_tokenizers.Split(Regex(SPLIT_PATTERN), "isolated").pre_tokenize_str(text)]


def test_split_keeps_telugu_words_whole_and_splits_digits():
    assert pieces("తెలుగు భాష 2026లో") == ["తెలుగు", " భాష", " ", "2", "0", "2", "6", "లో"]


def test_special_tokens_come_first_and_stay_whole(tokenizer):
    assert len(SPECIAL_TOKENS) == 64
    assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == list(range(64))
    assert tokenizer.encode("<|im_start|>hi<|im_end|>").ids[0] == 1


def test_every_telugu_character_is_one_token(tokenizer):
    assert len(telugu_characters()) > 90
    for character in telugu_characters():
        assert len(tokenizer.encode(character, add_special_tokens=False).ids) == 1, hex(ord(character))


def test_round_trip_is_exact(tokenizer):
    text = "Mixed తెలుగు and English, 3.14 and ౧౨౩!\n\tdone"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_measure_counts_words_characters_bytes_and_tokens(tokenizer):
    result = measure(tokenizer, ["తెలుగు భాష", "two words"])
    assert (result.sentences, result.words, result.characters) == (2, 4, 19)
    assert result.bytes == len("తెలుగు భాష".encode()) + len("two words")
    assert result.tokens_per_word == result.tokens / 4
    assert result.bytes_per_token == result.bytes / result.tokens


def test_load_flores_reads_one_language_from_the_archive(tmp_path, monkeypatch):
    archive = tmp_path / "flores.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = "ఒకటి\nరెండు\n".encode()
        info = tarfile.TarInfo("./flores200_dataset/devtest/tel_Telu.devtest")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(fertility_module, "cached_download", lambda url, path: archive)
    assert load_flores("tel_Telu") == ["ఒకటి", "రెండు"]
