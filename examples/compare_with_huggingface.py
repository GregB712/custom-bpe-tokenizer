"""Optional comparison against a Hugging Face tokenizer.

This script intentionally keeps Hugging Face out of the core implementation.
Install optional dependencies with:

    pip install transformers
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install transformers to run this optional comparison.")
        return

    model_path = PROJECT_ROOT / "artifacts" / "tokenizer.json"
    if not model_path.exists():
        tokenizer = BPETokenizer(vocab_size=180, min_frequency=2)
        tokenizer.fit(str(PROJECT_ROOT / "data" / "sample_corpus.txt"))
        tokenizer.save(str(model_path))

    custom_tokenizer = BPETokenizer.load(str(model_path))
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    samples = [
        "Tokenization is a bridge between raw text and model-ready IDs.",
        "BPE creates compact representations using frequent subword patterns.",
        "A custom tokenizer makes algorithmic trade-offs visible.",
    ]

    print(f"{'Text':<68} {'Custom':>8} {'BERT':>8}")
    print("-" * 88)
    for text in samples:
        custom_count = len(custom_tokenizer.encode(text))
        hf_count = len(hf_tokenizer.encode(text))
        print(f"{text[:66]:<68} {custom_count:>8} {hf_count:>8}")


if __name__ == "__main__":
    main()
