# custom-bpe-tokenizer

[![CI](https://github.com/<your-user>/custom-bpe-tokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-user>/custom-bpe-tokenizer/actions/workflows/ci.yml)

An advanced tokenizer engineering portfolio project implemented from scratch in Python. The repository demonstrates NLP algorithm implementation, typed package design, JSON model artifacts, benchmarking, FastAPI serving, Streamlit visualization, tests, examples, and CI/CD.

## Why Tokenizers Matter

Tokenizers define how raw text becomes model input IDs. For LLMs and NLP systems, tokenization affects sequence length, vocabulary coverage, Unicode handling, latency, cost, and debugging quality. This project implements several foundational tokenization families so their tradeoffs can be inspected directly.

## Implemented Tokenizers

- **BPE**: word-aware character/subword Byte Pair Encoding with deterministic merge rules.
- **Byte-level BPE**: GPT-style UTF-8 byte initialization with JSON-safe byte tokens such as `<0xE2>`.
- **WordPiece**: simplified BERT-style tokenizer with `##` continuation pieces and greedy longest-match-first encoding.
- **Unigram LM**: simplified SentencePiece-style tokenizer with substring probabilities and Viterbi segmentation.

The WordPiece and Unigram trainers are educational implementations, not drop-in replacements for the original Google/SentencePiece training objectives.

## Architecture

```text
src/tokenizer/
├── base.py          # Abstract tokenizer interface
├── bpe.py           # Character/subword BPE
├── byte_bpe.py      # UTF-8 byte-level BPE
├── wordpiece.py     # WordPiece-style tokenizer
├── unigram.py       # Unigram LM tokenizer
├── benchmark.py     # Custom + optional Hugging Face benchmarks
├── api.py           # FastAPI app
├── cli.py           # Multi-tokenizer CLI
├── serialization.py # Artifact inspection/loading helpers
├── preprocessing.py # Normalization and pre-tokenization
├── vocabulary.py    # Bidirectional token-ID mapping
└── utils.py         # JSON and corpus helpers
```

## Installation

```bash
git clone <your-repo-url>
cd custom-bpe-tokenizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

For Hugging Face comparisons only:

```bash
pip install "transformers>=4.40"
```

Tests do not require internet access, Hugging Face downloads, or torch.

## Quickstart

```python
from tokenizer import BPETokenizer, ByteLevelBPETokenizer, WordPieceTokenizer, UnigramTokenizer

corpus = [
    "Tokenization converts text into model-ready token IDs.",
    "Byte pair encoding learns frequent subword patterns.",
    "Unicode examples: Ελληνικά, café, 🚀.",
]

tokenizer = ByteLevelBPETokenizer(vocab_size=340, min_frequency=1)
tokenizer.fit(corpus)

ids = tokenizer.encode("Tokenization handles Ελληνικά and 🚀.")
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)
```

## CLI Usage

Train any tokenizer:

```bash
python -m tokenizer.cli train \
  --tokenizer-type bpe \
  --input data/sample_corpus.txt \
  --output artifacts/bpe_tokenizer.json \
  --vocab-size 1000
```

Encode text:

```bash
python -m tokenizer.cli encode \
  --tokenizer-type byte_bpe \
  --model artifacts/byte_bpe_tokenizer.json \
  --text "Tokenization is important."
```

Decode IDs:

```bash
python -m tokenizer.cli decode \
  --tokenizer-type bpe \
  --model artifacts/bpe_tokenizer.json \
  --ids "2 45 982 3"
```

Benchmark:

```bash
python -m tokenizer.cli benchmark \
  --input data/sample_corpus.txt \
  --include-hf
```

Supported `--tokenizer-type` values: `bpe`, `byte_bpe`, `wordpiece`, `unigram`.

## Examples

```bash
python examples/train_tokenizer.py
python examples/train_byte_bpe.py
python examples/train_wordpiece.py
python examples/train_unigram.py
python examples/encode_decode_demo.py
python examples/benchmark_tokenizers.py
```

Artifacts are saved under `artifacts/` and ignored by git except for `.gitkeep`.

## FastAPI

Run the API:

```bash
uvicorn tokenizer.api:app --reload
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Tokenize:

```bash
curl -X POST http://127.0.0.1:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text":"Tokenization is useful.","tokenizer_type":"bpe","add_special_tokens":true}'
```

If an artifact is missing, the API trains a small default tokenizer from `data/sample_corpus.txt` and caches it in memory.

## Streamlit Merge Visualization

```bash
streamlit run app/streamlit_app.py
```

The app visualizes learned BPE and byte-level BPE merge rules, step-by-step encoding traces, encoded IDs, decoded text, and a compact token frequency chart.

## Benchmarking

`tokenizer.benchmark.benchmark_tokenizers` reports:

- average tokens per text
- total tokens
- texts/sec
- tokens/sec
- average encode latency in milliseconds
- approximate vocabulary size

Example table:

```text
name      | avg_tok/text | total_tokens | texts/sec | tokens/sec | latency_ms | vocab
----------+--------------+--------------+-----------+------------+------------+------
bpe       | 18.40        | 184          | 8200.5    | 150889.2   | 0.122      | 180
byte_bpe  | 31.70        | 317          | 5900.1    | 187033.1   | 0.169      | 340
wordpiece | 15.20        | 152          | 9100.4    | 138326.0   | 0.110      | 180
unigram   | 16.80        | 168          | 7600.8    | 127693.4   | 0.132      | 180
```

Hugging Face tokenizers are optional. If `transformers` is missing or model download fails, the benchmark continues with custom tokenizers and emits a warning.

## Algorithm Notes

**BPE** starts with character tokens plus an end-of-word marker, repeatedly merges the most frequent adjacent pair above `min_frequency`, and applies learned merges in rank order during encoding.

**Byte-level BPE** starts from all 256 UTF-8 byte values, learns merges over byte-token sequences, and decodes merged byte tokens back through UTF-8 with `errors="replace"`. This makes arbitrary Unicode text representable without a character vocabulary.

**WordPiece** builds a vocabulary from frequent full words, characters, and substrings. Encoding uses greedy longest-match-first segmentation with a `##` continuation prefix.

**Unigram LM** builds substring candidates, estimates negative log probabilities from frequencies, and uses Viterbi dynamic programming to choose a minimum-cost segmentation.

## Testing

```bash
pytest
```

Coverage includes existing BPE behavior, byte-level Unicode handling, WordPiece continuation pieces, Unigram Viterbi segmentation, JSON serialization, benchmark schema/formatting, FastAPI endpoints, BPE trace output, vocabulary utilities, preprocessing, and CLI factory behavior.

## Project Structure

```text
custom-bpe-tokenizer/
├── .github/workflows/ci.yml
├── app/streamlit_app.py
├── artifacts/.gitkeep
├── data/sample_corpus.txt
├── examples/
├── notebooks/tokenizer_analysis.ipynb
├── src/tokenizer/
├── tests/
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Limitations

- Implementations prioritize readability and educational value over production throughput.
- WordPiece training is a simplified frequency-based approximation.
- Unigram training does not perform iterative EM pruning.
- No offset mapping, truncation, padding strategies, or streaming large-corpus trainer are included.
- Benchmark results on tiny corpora are illustrative, not statistically rigorous.

## Future Work

- Offset maps for token-to-text alignment
- Padding/truncation utilities for model batches
- Streaming trainers for large corpora
- Faster BPE pair-stat updates
- More normalization options
- Export adapters for downstream ML frameworks
- Richer benchmark datasets and profiler integration

## Skills Demonstrated

- NLP tokenizer algorithm implementation
- Typed Python package architecture
- JSON model serialization
- API development with FastAPI
- Visualization with Streamlit
- Benchmarking and graceful optional integrations
- Test design with pytest
- CI/CD with GitHub Actions
