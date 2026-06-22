"""Train a byte-level BPE tokenizer and save it to artifacts."""

from pathlib import Path

from tokenizer import ByteLevelBPETokenizer


def main() -> None:
    tokenizer = ByteLevelBPETokenizer(vocab_size=340, min_frequency=1)
    tokenizer.fit("data/sample_corpus.txt")
    for text in ["Tokenization is useful.", "Καλημέρα κόσμε.", "Emoji 🚀 tokenization."]:
        ids = tokenizer.encode(text)
        print(f"text={text!r}")
        print(f"ids={ids}")
        print(f"decoded={tokenizer.decode(ids)!r}\n")
    output = Path("artifacts/byte_bpe_tokenizer.json")
    tokenizer.save(str(output))
    print(f"Saved tokenizer to {output}")


if __name__ == "__main__":
    main()
