from pathlib import Path

from tokenizer.wordpiece import WordPieceTokenizer


CORPUS = [
    "tokenization token tokenized tokens",
    "embedding embeddings embedded",
    "wordpiece uses continuation pieces",
]


def test_training_produces_vocabulary():
    tokenizer = WordPieceTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    assert "[UNK]" in tokenizer.get_vocab()
    assert len(tokenizer.get_vocab()) <= 80


def test_known_words_encode_to_valid_ids():
    tokenizer = WordPieceTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    ids = tokenizer.encode("tokenization")
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_unknown_words_use_unk_when_vocab_is_tiny():
    tokenizer = WordPieceTokenizer(vocab_size=8, min_frequency=2)
    tokenizer.fit(CORPUS)
    ids = tokenizer.encode("zzzzzz", add_special_tokens=False)
    assert tokenizer.token_to_id("[UNK]") in ids


def test_continuation_tokens_are_decoded():
    tokenizer = WordPieceTokenizer(vocab_size=100, min_frequency=1)
    tokenizer.fit(CORPUS)
    token_ids = [tokenizer.token_to_id("token"), tokenizer.token_to_id("##s")]
    assert tokenizer.decode(token_ids) == "tokens"


def test_save_load_roundtrip(tmp_path: Path):
    tokenizer = WordPieceTokenizer(vocab_size=80, min_frequency=1)
    tokenizer.fit(CORPUS)
    output = tmp_path / "wordpiece.json"
    tokenizer.save(str(output))
    restored = WordPieceTokenizer.load(str(output))
    assert restored.encode("tokenized words") == tokenizer.encode("tokenized words")
