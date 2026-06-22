from tokenizer.cli import create_tokenizer, parse_ids
from tokenizer import BPETokenizer, ByteLevelBPETokenizer, UnigramTokenizer, WordPieceTokenizer


def test_parse_ids():
    assert parse_ids("1 2 3") == [1, 2, 3]


def test_cli_factory_creates_supported_tokenizers():
    assert isinstance(create_tokenizer("bpe", 50, 1), BPETokenizer)
    assert isinstance(create_tokenizer("byte_bpe", 260, 1), ByteLevelBPETokenizer)
    assert isinstance(create_tokenizer("wordpiece", 50, 1), WordPieceTokenizer)
    assert isinstance(create_tokenizer("unigram", 50, 1), UnigramTokenizer)
