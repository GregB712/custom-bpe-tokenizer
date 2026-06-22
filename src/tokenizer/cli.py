"""Command-line interface for the custom BPE tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .bpe import BPETokenizer


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Train and use a custom BPE tokenizer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a BPE tokenizer.")
    train_parser.add_argument("--input", required=True, help="Path to a text corpus.")
    train_parser.add_argument("--output", required=True, help="Output tokenizer JSON path.")
    train_parser.add_argument("--vocab-size", type=int, default=1000)
    train_parser.add_argument("--min-frequency", type=int, default=2)
    train_parser.add_argument("--no-lowercase", action="store_true")
    train_parser.add_argument("--no-split-punctuation", action="store_true")

    encode_parser = subparsers.add_parser("encode", help="Encode text with a saved tokenizer.")
    encode_parser.add_argument("--model", required=True, help="Path to tokenizer JSON.")
    encode_parser.add_argument("--text", required=True, help="Text to encode.")
    encode_parser.add_argument("--no-special-tokens", action="store_true")

    decode_parser = subparsers.add_parser("decode", help="Decode token IDs with a saved tokenizer.")
    decode_parser.add_argument("--model", required=True, help="Path to tokenizer JSON.")
    decode_parser.add_argument("--ids", required=True, help="Whitespace-separated token IDs.")
    decode_parser.add_argument("--keep-special-tokens", action="store_true")

    return parser


def parse_ids(raw_ids: str) -> List[int]:
    """Parse a whitespace-separated ID string."""

    try:
        return [int(value) for value in raw_ids.split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ids must contain only integers") from exc


def main() -> None:
    """Run the tokenizer CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        tokenizer = BPETokenizer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            lowercase=not args.no_lowercase,
            split_punctuation=not args.no_split_punctuation,
        )
        tokenizer.fit(args.input)
        tokenizer.save(args.output)
        print(f"Saved tokenizer to {Path(args.output)}")
        print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
        print(f"Learned merges: {len(tokenizer.merges)}")
        return

    tokenizer = BPETokenizer.load(args.model)

    if args.command == "encode":
        ids = tokenizer.encode(args.text, add_special_tokens=not args.no_special_tokens)
        print(json.dumps(ids))
        return

    if args.command == "decode":
        ids = parse_ids(args.ids)
        print(tokenizer.decode(ids, skip_special_tokens=not args.keep_special_tokens))


if __name__ == "__main__":
    main()
