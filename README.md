# custom-bpe-tokenizer

A production-quality Byte Pair Encoding tokenizer implemented from scratch in Python. This project is designed as a GitHub portfolio piece for NLP, algorithmic engineering, and maintainable Python package design.

## Why Tokenization Matters

Tokenization is the bridge between raw text and numerical machine-learning models. A tokenizer defines how text is segmented, how tokens map to integer IDs, how unknown text is handled, and how reliably model inputs can be reproduced. Strong tokenization choices improve vocabulary coverage, reduce sequence length, and make NLP systems easier to debug.

## What Is Byte Pair Encoding?

Byte Pair Encoding (BPE) is a subword tokenization algorithm. It starts with small units, such as characters, then repeatedly merges the most frequent adjacent pair. Over many iterations, common fragments become compact tokens while rare words can still be represented as smaller subword pieces.

This implementation uses a word-aware character-level BPE variant with an explicit end-of-word marker. That keeps the algorithm readable while allowing decoded output to reconstruct whitespace and punctuation cleanly.

## Features

- From-scratch BPE training loop with deterministic merge selection
- Configurable vocabulary size
- Configurable minimum merge frequency
- Encoding text into token IDs
- Decoding token IDs back into readable text
- Batch encoding
- Save and load JSON-readable tokenizer files
- Default special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- Optional lowercasing
- Optional basic punctuation splitting
- Graceful unknown-token fallback
- Pytest test suite
- CLI built with `argparse`
- Practical examples and analysis notebook

## Installation

```bash
git clone <your-repo-url>
cd custom-bpe-tokenizer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the optional Hugging Face comparison:

```bash
pip install -e ".[hf]"
```

## Usage

```python
from tokenizer import BPETokenizer

corpus = [
    "Tokenization converts text into model-ready token IDs.",
    "Byte Pair Encoding learns frequent subword patterns.",
]

tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
tokenizer.fit(corpus)

ids = tokenizer.encode("Tokenization is important for NLP.")
text = tokenizer.decode(ids)

print(ids)
print(text)
```

## CLI Examples

Train a tokenizer:

```bash
python -m tokenizer.cli train \
  --input data/sample_corpus.txt \
  --output artifacts/tokenizer.json \
  --vocab-size 1000
```

Encode text:

```bash
python -m tokenizer.cli encode \
  --model artifacts/tokenizer.json \
  --text "Tokenization is important for NLP."
```

Decode IDs:

```bash
python -m tokenizer.cli decode \
  --model artifacts/tokenizer.json \
  --ids "2 45 98 3"
```

If you are running from a source checkout without installing the package, use:

```bash
PYTHONPATH=src python -m tokenizer.cli train --input data/sample_corpus.txt --output artifacts/tokenizer.json
```

## Training Example

```bash
python examples/train_tokenizer.py
```

Example output:

```text
Tokenizer saved to: artifacts/tokenizer.json
Vocabulary size: 180
Sample merge rules:
01. 'e' + '</w>' -> 'e</w>'
02. 't' + 'i' -> 'ti'
03. 'i' + 'n' -> 'in'
```

## Encoding and Decoding Demo

```bash
python examples/encode_decode_demo.py
```

Example output:

```text
Text:    Tokenization is important for NLP.
IDs:     [2, 58, 31, 74, 3]
Decoded: tokenization is important for nlp.
```

Token IDs will vary when you change vocabulary size, minimum frequency, corpus, or preprocessing settings.

## Project Structure

```text
custom-bpe-tokenizer/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   └── tokenizer/
│       ├── __init__.py
│       ├── bpe.py
│       ├── cli.py
│       ├── preprocessing.py
│       ├── vocabulary.py
│       └── utils.py
├── examples/
│   ├── train_tokenizer.py
│   ├── encode_decode_demo.py
│   └── compare_with_huggingface.py
├── tests/
│   ├── test_bpe.py
│   ├── test_vocabulary.py
│   └── test_preprocessing.py
├── data/
│   └── sample_corpus.txt
└── notebooks/
    └── tokenizer_analysis.ipynb
```

## Testing

Run the full test suite:

```bash
pytest
```

The tests cover training, encoding, decoding, persistence, special tokens, unknown tokens, batch encoding, and deterministic behavior on a fixed corpus.

## Implementation Notes

The core BPE algorithm lives in `src/tokenizer/bpe.py`:

1. Normalize and tokenize text.
2. Convert each token to character symbols plus an end-of-word marker.
3. Count adjacent symbol-pair frequencies.
4. Merge the most frequent pair above `min_frequency`.
5. Store the merge rule and add the merged token to the vocabulary.
6. Repeat until the vocabulary limit is reached or no eligible pairs remain.

The saved tokenizer JSON contains configuration, vocabulary, and merge rules, making artifacts inspectable and version-control friendly.

## Limitations

- This is an educational from-scratch implementation, not a drop-in replacement for highly optimized production tokenizers.
- It uses character-level initialization rather than true byte-level UTF-8 initialization.
- It does not implement dropout, normalization pipelines, offsets, truncation, or padding utilities.
- Training recomputes pair frequencies after each merge for clarity, which is slower than optimized heap-based implementations.

## Possible Future Improvements

- Byte-level initialization for full Unicode byte coverage
- Offset mapping for aligning tokens back to source text
- Padding and truncation helpers for model batching
- Streaming corpus training for large datasets
- Faster pair-stat updates during training
- Vocabulary export formats compatible with downstream ML tooling
- Richer normalization options

## Skills Demonstrated

- NLP fundamentals and subword tokenization
- Algorithm implementation from first principles
- Deterministic training behavior
- Python package architecture with `src/` layout
- Type hints and public API docstrings
- JSON model persistence
- CLI design with `argparse`
- Unit testing with `pytest`
- Practical examples and notebook-based analysis
