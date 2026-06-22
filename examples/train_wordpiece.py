"""Train a simplified WordPiece tokenizer."""

from pathlib import Path

from tokenizer import WordPieceTokenizer


def main() -> None:
    tokenizer = WordPieceTokenizer(vocab_size=180, min_frequency=1)
    tokenizer.fit("data/sample_corpus.txt")
    for word in ["tokenization", "subword", "embeddings"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        tokens = [tokenizer.id_to_token(token_id) for token_id in ids]
        print(f"{word}: {tokens} -> {ids}")
    output = Path("artifacts/wordpiece_tokenizer.json")
    tokenizer.save(str(output))
    print(f"Saved tokenizer to {output}")


if __name__ == "__main__":
    main()
