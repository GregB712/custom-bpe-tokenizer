import sys

from tokenizer.benchmark import benchmark_tokenizers, format_benchmark_results
from tokenizer.bpe import BPETokenizer


def test_benchmark_runs_with_custom_tokenizers():
    tokenizer = BPETokenizer(vocab_size=60, min_frequency=1)
    texts = ["Tokenization matters.", "BPE learns merges."]
    tokenizer.fit(texts)
    results = benchmark_tokenizers({"bpe": tokenizer}, texts, benchmark_runs=1, warmup_runs=0)
    assert results[0]["name"] == "bpe"
    assert "average_tokens_per_text" in results[0]
    assert "tokens_per_second" in results[0]


def test_formatter_returns_string():
    formatted = format_benchmark_results(
        [
            {
                "name": "demo",
                "average_tokens_per_text": 2.0,
                "total_tokens": 4,
                "texts_per_second": 10.0,
                "tokens_per_second": 20.0,
                "average_encode_latency_ms": 1.0,
                "vocab_size": 50,
            }
        ]
    )
    assert "demo" in formatted
    assert "tokens/sec" in formatted


def test_huggingface_absence_is_handled(monkeypatch):
    tokenizer = BPETokenizer(vocab_size=60, min_frequency=1)
    texts = ["Tokenization matters."]
    tokenizer.fit(texts)
    monkeypatch.setitem(sys.modules, "transformers", None)
    results = benchmark_tokenizers(
        {"bpe": tokenizer},
        texts,
        hf_tokenizer_names=["bert-base-uncased"],
        benchmark_runs=1,
        warmup_runs=0,
    )
    assert [result["name"] for result in results] == ["bpe"]
