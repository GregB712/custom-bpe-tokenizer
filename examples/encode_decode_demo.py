"""Encode and decode example sentences with a trained tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    model_path = PROJECT_ROOT / "artifacts" / "tokenizer.json"
    if not model_path.exists():
        print("No saved tokenizer found. Run examples/train_tokenizer.py first.")
        return

    tokenizer = BPETokenizer.load(str(model_path))
    samples = [
        "Tokenization is important for NLP.",
        "Byte Pair Encoding learns reusable subword units.",
        "Unknown terminology should degrade gracefully.",
    ]

    for text in samples:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print("=" * 72)
        print(f"Text:    {text}")
        print(f"IDs:     {ids}")
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
