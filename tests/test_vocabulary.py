from tokenizer.vocabulary import DEFAULT_SPECIAL_TOKENS, Vocabulary


def test_vocabulary_initializes_special_tokens_first():
    vocab = Vocabulary()
    assert vocab.token_to_id("<PAD>") == 0
    assert vocab.token_to_id("<UNK>") == 1
    assert vocab.id_to_token(2) == "<BOS>"


def test_add_token_is_idempotent():
    vocab = Vocabulary()
    first_id = vocab.add_token("token")
    second_id = vocab.add_token("token")
    assert first_id == second_id
    assert len(vocab) == len(DEFAULT_SPECIAL_TOKENS) + 1


def test_unknown_token_and_id_fallback_to_unk():
    vocab = Vocabulary()
    assert vocab.token_to_id("missing") == vocab.token_to_id("<UNK>")
    assert vocab.id_to_token(9999) == "<UNK>"


def test_vocabulary_round_trip_serialization():
    vocab = Vocabulary()
    vocab.add_token("nlp")
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.token_to_id("nlp") == vocab.token_to_id("nlp")
    assert restored.id_to_token(restored.token_to_id("nlp")) == "nlp"
