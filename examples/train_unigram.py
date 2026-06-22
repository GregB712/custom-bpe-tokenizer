"""Train a simplified SentencePiece-style unigram tokenizer."""

from pathlib import Path

from tokenizer import UnigramTokenizer


def main() -> None:
    tokenizer = UnigramTokenizer(vocab_size=180, min_frequency=1)
    tokenizer.fit("data/sample_corpus.txt")
    print("Lowest-cost candidate pieces:")
    for token, score in sorted(tokenizer.token_scores.items(), key=lambda item: item[1])[:15]:
        print(f"{token!r}: {score:.3f}")
    text = "Tokenization finds useful pieces."
    ids = tokenizer.encode(text)
    print(f"ids={ids}")
    print(f"decoded={tokenizer.decode(ids)!r}")
    output = Path("artifacts/unigram_tokenizer.json")
    tokenizer.save(str(output))
    print(f"Saved tokenizer to {output}")


if __name__ == "__main__":
    main()
