"""Benchmark custom tokenizers and optional Hugging Face baselines."""

from tokenizer import BPETokenizer, ByteLevelBPETokenizer, UnigramTokenizer, WordPieceTokenizer
from tokenizer.benchmark import benchmark_tokenizers, format_benchmark_results
from tokenizer.utils import read_corpus


def main() -> None:
    texts = read_corpus("data/sample_corpus.txt")
    tokenizers = {
        "bpe": BPETokenizer(vocab_size=180, min_frequency=1),
        "byte_bpe": ByteLevelBPETokenizer(vocab_size=340, min_frequency=1),
        "wordpiece": WordPieceTokenizer(vocab_size=180, min_frequency=1),
        "unigram": UnigramTokenizer(vocab_size=180, min_frequency=1),
    }
    for tokenizer in tokenizers.values():
        tokenizer.fit(texts)
    results = benchmark_tokenizers(
        tokenizers,
        texts,
        hf_tokenizer_names=["bert-base-uncased", "gpt2"],
        benchmark_runs=3,
    )
    print(format_benchmark_results(results))


if __name__ == "__main__":
    main()
