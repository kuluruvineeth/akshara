import tarfile
from dataclasses import dataclass

from dataflow.utils.cache import cached_download
from tokenizers import Tokenizer

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def load_flores(language: str, split: str = "devtest") -> list[str]:
    archive = cached_download(FLORES_URL, "flores/flores200_dataset.tar.gz")
    with tarfile.open(archive) as tar:
        member = tar.extractfile(f"./flores200_dataset/{split}/{language}.{split}")
        return member.read().decode("utf-8").splitlines()


@dataclass(frozen=True)
class Fertility:
    sentences: int
    words: int
    characters: int
    bytes: int
    tokens: int

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words

    @property
    def tokens_per_character(self) -> float:
        return self.tokens / self.characters

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / self.tokens


def measure(tokenizer: Tokenizer, sentences: list[str]) -> Fertility:
    encodings = tokenizer.encode_batch(sentences, add_special_tokens=False)
    return Fertility(
        sentences=len(sentences),
        words=sum(len(sentence.split()) for sentence in sentences),
        characters=sum(map(len, sentences)),
        bytes=sum(len(sentence.encode("utf-8")) for sentence in sentences),
        tokens=sum(len(encoding.ids) for encoding in encodings),
    )
