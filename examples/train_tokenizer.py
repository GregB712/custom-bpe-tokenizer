"""Train the custom BPE tokenizer on the sample corpus."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    corpus_path = PROJECT_ROOT / "data" / "sample_corpus.txt"
    output_path = PROJECT_ROOT / "artifacts" / "bpe_tokenizer.json"

    tokenizer = BPETokenizer(vocab_size=180, min_frequency=2, lowercase=True)
    tokenizer.fit(str(corpus_path))
    tokenizer.save(str(output_path))

    print(f"Tokenizer saved to: {output_path}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    print("Sample merge rules:")
    for index, pair in enumerate(tokenizer.merges[:15], start=1):
        print(f"{index:02d}. {pair[0]!r} + {pair[1]!r} -> {''.join(pair)!r}")


if __name__ == "__main__":
    main()
