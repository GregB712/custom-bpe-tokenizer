from pathlib import Path

from tokenizer.unigram import UnigramTokenizer


CORPUS = [
    "tokenization token tokenized tokens",
    "unigram tokenization finds likely pieces",
    "natural language processing",
]


def test_vocabulary_training():
    tokenizer = UnigramTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    assert "<UNK>" in tokenizer.get_vocab()
    assert tokenizer.token_scores


def test_viterbi_segmentation_returns_valid_ids():
    tokenizer = UnigramTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    ids = tokenizer.encode("tokenization")
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_unknown_handling():
    tokenizer = UnigramTokenizer(vocab_size=8, min_frequency=2)
    tokenizer.fit(CORPUS)
    ids = tokenizer.encode("zzzzzz", add_special_tokens=False)
    assert tokenizer.token_to_id("<UNK>") in ids


def test_save_load_preserves_probabilities(tmp_path: Path):
    tokenizer = UnigramTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    output = tmp_path / "unigram.json"
    tokenizer.save(str(output))
    restored = UnigramTokenizer.load(str(output))
    assert restored.token_scores == tokenizer.token_scores
    assert restored.encode("natural language") == tokenizer.encode("natural language")


def test_decode_returns_readable_text():
    tokenizer = UnigramTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    decoded = tokenizer.decode(tokenizer.encode("natural language"))
    assert "natural" in decoded
