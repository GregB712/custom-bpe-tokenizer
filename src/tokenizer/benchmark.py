"""Benchmark helpers for custom and Hugging Face tokenizers."""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional


def _encode(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return list(encoded)


def _vocab_size(tokenizer: Any) -> int:
    if hasattr(tokenizer, "get_vocab"):
        return len(tokenizer.get_vocab())
    vocab = getattr(tokenizer, "vocab", None)
    if vocab is not None:
        return len(vocab)
    return 0


def _measure(name: str, tokenizer: Any, texts: List[str], warmup_runs: int, benchmark_runs: int) -> Dict[str, Any]:
    for _ in range(warmup_runs):
        for text in texts:
            _encode(tokenizer, text)

    total_elapsed = 0.0
    total_tokens = 0
    total_texts = len(texts) * benchmark_runs
    for _ in range(benchmark_runs):
        started = time.perf_counter()
        for text in texts:
            total_tokens += len(_encode(tokenizer, text))
        total_elapsed += time.perf_counter() - started

    total_elapsed = max(total_elapsed, 1e-12)
    return {
        "name": name,
        "average_tokens_per_text": total_tokens / max(total_texts, 1),
        "total_tokens": total_tokens,
        "texts_per_second": total_texts / total_elapsed,
        "tokens_per_second": total_tokens / total_elapsed,
        "average_encode_latency_ms": (total_elapsed / max(total_texts, 1)) * 1000,
        "vocab_size": _vocab_size(tokenizer),
    }


def benchmark_tokenizers(
    tokenizers: Dict[str, Any],
    texts: List[str],
    hf_tokenizer_names: Optional[List[str]] = None,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> List[Dict[str, Any]]:
    """Benchmark custom tokenizers and optional Hugging Face tokenizers."""

    if not texts:
        raise ValueError("texts must not be empty")
    results = [
        _measure(name, tokenizer, texts, warmup_runs, benchmark_runs)
        for name, tokenizer in tokenizers.items()
    ]

    if hf_tokenizer_names:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            warnings.warn(
                "transformers is not installed; skipping Hugging Face tokenizer benchmarks.",
                RuntimeWarning,
                stacklevel=2,
            )
            return results
        for hf_name in hf_tokenizer_names:
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(hf_name)
            except Exception as exc:  # pragma: no cover - depends on network/cache
                warnings.warn(
                    f"Could not load Hugging Face tokenizer {hf_name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            results.append(_measure(f"hf:{hf_name}", hf_tokenizer, texts, warmup_runs, benchmark_runs))
    return results


def format_benchmark_results(results: List[Dict[str, Any]]) -> str:
    """Format benchmark dictionaries as a readable fixed-width table."""

    if not results:
        return "No benchmark results."
    headers = [
        "name",
        "avg_tok/text",
        "total_tokens",
        "texts/sec",
        "tokens/sec",
        "latency_ms",
        "vocab",
    ]
    rows = [
        [
            str(row["name"]),
            f"{row['average_tokens_per_text']:.2f}",
            str(int(row["total_tokens"])),
            f"{row['texts_per_second']:.1f}",
            f"{row['tokens_per_second']:.1f}",
            f"{row['average_encode_latency_ms']:.3f}",
            str(int(row["vocab_size"])),
        ]
        for row in results
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    output = [" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    output.append("-+-".join("-" * width for width in widths))
    output.extend(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
    return "\n".join(output)
