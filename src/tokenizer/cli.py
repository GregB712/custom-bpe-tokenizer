"""Command-line interface for custom tokenizer implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Type

from .base import BaseTokenizer
from .benchmark import benchmark_tokenizers, format_benchmark_results
from .bpe import BPETokenizer
from .byte_bpe import ByteLevelBPETokenizer
from .unigram import UnigramTokenizer
from .utils import read_corpus
from .wordpiece import WordPieceTokenizer

TOKENIZER_TYPES: Dict[str, Type[BaseTokenizer]] = {
    "bpe": BPETokenizer,
    "byte_bpe": ByteLevelBPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Train and use educational tokenizers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a tokenizer.")
    train_parser.add_argument("--tokenizer-type", choices=TOKENIZER_TYPES, default="bpe")
    train_parser.add_argument("--input", required=True, help="Path to a text corpus.")
    train_parser.add_argument("--output", required=True, help="Output tokenizer JSON path.")
    train_parser.add_argument("--vocab-size", type=int, default=1000)
    train_parser.add_argument("--min-frequency", type=int, default=2)
    train_parser.add_argument("--no-lowercase", action="store_true")
    train_parser.add_argument("--no-split-punctuation", action="store_true")

    encode_parser = subparsers.add_parser("encode", help="Encode text with a saved tokenizer.")
    encode_parser.add_argument("--tokenizer-type", choices=TOKENIZER_TYPES, default="bpe")
    encode_parser.add_argument("--model", required=True, help="Path to tokenizer JSON.")
    encode_parser.add_argument("--text", required=True, help="Text to encode.")
    encode_parser.add_argument("--no-special-tokens", action="store_true")

    decode_parser = subparsers.add_parser("decode", help="Decode token IDs with a saved tokenizer.")
    decode_parser.add_argument("--tokenizer-type", choices=TOKENIZER_TYPES, default="bpe")
    decode_parser.add_argument("--model", required=True, help="Path to tokenizer JSON.")
    decode_parser.add_argument("--ids", required=True, help="Whitespace-separated token IDs.")
    decode_parser.add_argument("--keep-special-tokens", action="store_true")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark tokenizers on a corpus.")
    benchmark_parser.add_argument("--input", required=True, help="Path to a text corpus.")
    benchmark_parser.add_argument("--include-hf", action="store_true")
    benchmark_parser.add_argument("--vocab-size", type=int, default=300)
    benchmark_parser.add_argument("--min-frequency", type=int, default=1)

    return parser


def parse_ids(raw_ids: str) -> List[int]:
    """Parse a whitespace-separated ID string."""

    try:
        return [int(value) for value in raw_ids.split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ids must contain only integers") from exc


def create_tokenizer(tokenizer_type: str, vocab_size: int = 1000, min_frequency: int = 2, lowercase: bool = True) -> BaseTokenizer:
    """Create a tokenizer instance from a CLI tokenizer type."""

    tokenizer_cls = TOKENIZER_TYPES[tokenizer_type]
    kwargs = {"vocab_size": vocab_size, "min_frequency": min_frequency, "lowercase": lowercase}
    if tokenizer_type == "byte_bpe" and vocab_size <= 256:
        kwargs["vocab_size"] = 300
    return tokenizer_cls(**kwargs)  # type: ignore[call-arg]


def load_cli_tokenizer(tokenizer_type: str, model: str) -> BaseTokenizer:
    """Load a saved tokenizer for a CLI command."""

    tokenizer_cls = TOKENIZER_TYPES[tokenizer_type]
    return tokenizer_cls.load(model)  # type: ignore[attr-defined]


def main() -> None:
    """Run the tokenizer CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        tokenizer = create_tokenizer(
            args.tokenizer_type,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            lowercase=not args.no_lowercase,
        )
        if args.tokenizer_type == "bpe":
            tokenizer.split_punctuation = not args.no_split_punctuation  # type: ignore[attr-defined]
        tokenizer.fit(args.input)
        tokenizer.save(args.output)
        print(f"Saved {args.tokenizer_type} tokenizer to {Path(args.output)}")
        print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
        if hasattr(tokenizer, "merges"):
            print(f"Learned merges: {len(getattr(tokenizer, 'merges'))}")
        return

    if args.command == "benchmark":
        texts = read_corpus(args.input)
        tokenizers = {
            name: create_tokenizer(name, args.vocab_size, args.min_frequency)
            for name in TOKENIZER_TYPES
        }
        for tokenizer in tokenizers.values():
            tokenizer.fit(texts)
        hf_names = ["bert-base-uncased", "gpt2"] if args.include_hf else None
        print(format_benchmark_results(benchmark_tokenizers(tokenizers, texts, hf_names)))
        return

    tokenizer = load_cli_tokenizer(args.tokenizer_type, args.model)

    if args.command == "encode":
        ids = tokenizer.encode(args.text, add_special_tokens=not args.no_special_tokens)
        print(json.dumps(ids))
        return

    if args.command == "decode":
        ids = parse_ids(args.ids)
        print(tokenizer.decode(ids, skip_special_tokens=not args.keep_special_tokens))


if __name__ == "__main__":
    main()
