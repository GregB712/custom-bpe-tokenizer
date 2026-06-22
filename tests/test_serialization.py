from tokenizer.bpe import BPETokenizer
from tokenizer.serialization import artifact_metadata, load_tokenizer_artifact


def test_artifact_metadata_and_generic_load(tmp_path):
    tokenizer = BPETokenizer(vocab_size=50, min_frequency=1)
    tokenizer.fit(["serialization keeps artifacts inspectable"])
    output = tmp_path / "bpe.json"
    tokenizer.save(str(output))
    metadata = artifact_metadata(str(output))
    restored = load_tokenizer_artifact(str(output), BPETokenizer)
    assert metadata["tokenizer_type"] == "bpe"
    assert restored.encode("serialization") == tokenizer.encode("serialization")
