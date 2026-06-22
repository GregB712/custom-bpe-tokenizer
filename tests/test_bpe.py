from pathlib import Path

import pytest

from tokenizer import BPETokenizer


FIXED_CORPUS = [
    "Tokenization matters for natural language processing.",
    "Tokenization converts text into model friendly token ids.",
    "Byte pair encoding learns frequent subword patterns.",
    "Frequent pairs become merged tokens.",
]


def test_training_builds_vocabulary_and_merges():
    tokenizer = BPETokenizer(vocab_size=80, min_frequency=2)
    tokenizer.fit(FIXED_CORPUS)
    vocab = tokenizer.get_vocab()
    assert "<UNK>" in vocab
    assert len(vocab) <= 80
    assert tokenizer.merges


def test_encode_returns_integer_ids():
    tokenizer = BPETokenizer(vocab_size=80, min_frequency=2)
    tokenizer.fit(FIXED_CORPUS)
    ids = tokenizer.encode("Tokenization matters.")
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_decode_returns_readable_text():
    tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
    tokenizer.fit(FIXED_CORPUS)
    ids = tokenizer.encode("Tokenization matters.")
    decoded = tokenizer.decode(ids)
    assert "tokenization" in decoded
    assert decoded.endswith(".")


def test_save_and_load_preserves_vocabulary_and_merges(tmp_path: Path):
    tokenizer = BPETokenizer(vocab_size=90, min_frequency=2)
    tokenizer.fit(FIXED_CORPUS)
    output = tmp_path / "tokenizer.json"
    tokenizer.save(str(output))

    restored = BPETokenizer.load(str(output))
    assert restored.get_vocab() == tokenizer.get_vocab()
    assert restored.merges == tokenizer.merges
    assert restored.encode("Byte pair encoding.") == tokenizer.encode("Byte pair encoding.")


def test_special_token_handling():
    tokenizer = BPETokenizer(vocab_size=80, min_frequency=2)
    tokenizer.fit(FIXED_CORPUS)
    ids = tokenizer.encode("hello", add_special_tokens=True)
    assert ids[0] == tokenizer.token_to_id("<BOS>")
    assert ids[-1] == tokenizer.token_to_id("<EOS>")
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert "<BOS>" not in decoded
    assert "<EOS>" not in decoded


def test_unknown_token_handling():
    tokenizer = BPETokenizer(vocab_size=25, min_frequency=2)
    tokenizer.fit(FIXED_CORPUS)
    ids = tokenizer.encode("zzzzzz")
    assert tokenizer.token_to_id("<UNK>") in ids
    assert tokenizer.id_to_token(123456) == "<UNK>"


def test_batch_encoding():
    tokenizer = BPETokenizer(vocab_size=80, min_frequency=2)
    tokenizer.fit(FIXED_CORPUS)
    batch = tokenizer.batch_encode(["Tokenization matters.", "BPE learns merges."])
    assert len(batch) == 2
    assert all(isinstance(sequence, list) for sequence in batch)
    assert all(all(isinstance(token_id, int) for token_id in sequence) for sequence in batch)


def test_deterministic_behavior_on_fixed_corpus():
    first = BPETokenizer(vocab_size=80, min_frequency=2)
    second = BPETokenizer(vocab_size=80, min_frequency=2)
    first.fit(FIXED_CORPUS)
    second.fit(FIXED_CORPUS)
    assert first.get_vocab() == second.get_vocab()
    assert first.merges == second.merges


def test_encode_before_fit_raises_error():
    tokenizer = BPETokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.encode("not fitted")
