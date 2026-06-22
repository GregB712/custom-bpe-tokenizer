from pathlib import Path

from tokenizer.byte_bpe import ByteLevelBPETokenizer, byte_token


CORPUS = [
    "Tokenization handles English text.",
    "Byte level tokenization handles Ελληνικά and café.",
    "Emoji like 🚀 and 😊 are UTF-8 bytes.",
]


def test_byte_vocab_includes_all_base_bytes():
    tokenizer = ByteLevelBPETokenizer(vocab_size=300, min_frequency=1)
    tokenizer.fit(CORPUS)
    vocab = tokenizer.get_vocab()
    assert byte_token(0) in vocab
    assert byte_token(255) in vocab


def test_encode_decode_english_text():
    tokenizer = ByteLevelBPETokenizer(vocab_size=320, min_frequency=1)
    tokenizer.fit(CORPUS)
    text = "Tokenization handles English text."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_decode_unicode_text():
    tokenizer = ByteLevelBPETokenizer(vocab_size=340, min_frequency=1)
    tokenizer.fit(CORPUS)
    text = "Καλημέρα café 🚀"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_load_preserves_behavior(tmp_path: Path):
    tokenizer = ByteLevelBPETokenizer(vocab_size=320, min_frequency=1)
    tokenizer.fit(CORPUS)
    output = tmp_path / "byte_bpe.json"
    tokenizer.save(str(output))
    restored = ByteLevelBPETokenizer.load(str(output))
    assert restored.get_vocab() == tokenizer.get_vocab()
    assert restored.encode("Emoji 😊") == tokenizer.encode("Emoji 😊")


def test_deterministic_training_on_small_corpus():
    first = ByteLevelBPETokenizer(vocab_size=310, min_frequency=1)
    second = ByteLevelBPETokenizer(vocab_size=310, min_frequency=1)
    first.fit(CORPUS)
    second.fit(CORPUS)
    assert first.get_vocab() == second.get_vocab()
    assert first.merges == second.merges


def test_trace_encode_has_initial_and_final_steps():
    tokenizer = ByteLevelBPETokenizer(vocab_size=310, min_frequency=1)
    tokenizer.fit(CORPUS)
    trace = tokenizer.trace_encode("hello")
    stages = [step["stage"] for step in trace[0]["steps"]]
    assert "initial" in stages
    assert "final" in stages
