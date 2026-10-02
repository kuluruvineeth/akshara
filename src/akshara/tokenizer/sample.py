from collections.abc import Iterator

from dataflow.pipeline.readers import JsonlReader, ParquetReader

SOURCES = {
    "english": lambda: ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT"),
    "code": lambda: JsonlReader(
        "hf://datasets/codeparrot/codeparrot-clean-valid", text_key="content", glob_pattern="*.json.gz"
    ),
    "telugu": lambda: ParquetReader("hf://datasets/HuggingFaceFW/fineweb-2/data/tel_Telu/train"),
}


def sample_texts(source: str, characters: int) -> Iterator[str]:
    taken = 0
    for document in SOURCES[source]().run():
        if taken >= characters:
            return
        text = document.text[: characters - taken]
        taken += len(text)
        yield text


def mixture(characters: dict[str, int]) -> Iterator[str]:
    for source, amount in characters.items():
        yield from sample_texts(source, amount)
