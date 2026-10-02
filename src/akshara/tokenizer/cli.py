import argparse
from pathlib import Path

from dataflow.pipeline.tokens.tokenizer import load_tokenizer

from akshara.tokenizer.fertility import load_flores, measure
from akshara.tokenizer.sample import mixture
from akshara.tokenizer.train import VOCAB_SIZE, train_tokenizer

LANGUAGES = {"English": "eng_Latn", "Telugu": "tel_Telu"}


def train(args: argparse.Namespace) -> None:
    characters = {"english": args.english, "code": args.code, "telugu": args.telugu}
    tokenizer = train_tokenizer(mixture(characters), vocab_size=args.vocab_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print(f"saved {tokenizer.get_vocab_size():,}-token tokenizer to {args.output}")


def fertility(args: argparse.Namespace) -> None:
    sentences = {name: load_flores(code) for name, code in LANGUAGES.items()}
    header = "".join(f"{name + ' tok/word':>18}{'bytes/tok':>11}" for name in LANGUAGES)
    print(f"{'tokenizer':<40}{'vocab':>8}{header}")
    for name in args.tokenizers:
        tokenizer = load_tokenizer(name)
        row = f"{name:<40}{tokenizer.get_vocab_size():>8,}"
        for language in LANGUAGES:
            result = measure(tokenizer, sentences[language])
            row += f"{result.tokens_per_word:>18.2f}{result.bytes_per_token:>11.2f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(prog="akshara-tokenizer")
    commands = parser.add_subparsers(required=True)
    train_parser = commands.add_parser("train", help="train the English + Telugu BPE tokenizer")
    train_parser.add_argument("output", help="path of the tokenizer.json to write")
    train_parser.add_argument("--english", type=int, default=50_000_000, help="characters of English web text")
    train_parser.add_argument("--code", type=int, default=10_000_000, help="characters of Python code")
    train_parser.add_argument("--telugu", type=int, default=40_000_000, help="characters of Telugu web text")
    train_parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    train_parser.set_defaults(handler=train)
    fertility_parser = commands.add_parser("fertility", help="tokens per word on FLORES-200 devtest")
    fertility_parser.add_argument("tokenizers", nargs="+", help="tokenizer.json paths or Hugging Face model ids")
    fertility_parser.set_defaults(handler=fertility)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
