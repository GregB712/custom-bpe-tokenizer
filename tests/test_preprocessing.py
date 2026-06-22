from tokenizer.preprocessing import basic_tokenize, detokenize, normalize_text


def test_normalize_text_lowercases_and_collapses_whitespace():
    assert normalize_text("  Tokenization   Matters  ", lowercase=True) == "tokenization matters"


def test_basic_tokenize_splits_punctuation():
    assert basic_tokenize("Hello, NLP!", lowercase=True) == ["hello", ",", "nlp", "!"]


def test_basic_tokenize_can_keep_punctuation_with_words():
    assert basic_tokenize("Hello, NLP!", split_punctuation=False) == ["hello,", "nlp!"]


def test_detokenize_removes_spaces_before_punctuation():
    assert detokenize(["hello", ",", "nlp", "!"]) == "hello, nlp!"
