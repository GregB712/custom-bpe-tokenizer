import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tokenizer.api import TOKENIZER_CACHE, app


def test_health_endpoint_works():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_tokenize_endpoint_returns_token_ids():
    TOKENIZER_CACHE.clear()
    client = TestClient(app)
    response = client.post(
        "/tokenize",
        json={"text": "Tokenization is useful.", "tokenizer_type": "bpe", "add_special_tokens": True},
    )
    assert response.status_code == 200
    assert response.json()["token_ids"]


def test_detokenize_endpoint_returns_text():
    TOKENIZER_CACHE.clear()
    client = TestClient(app)
    tokenized = client.post(
        "/tokenize",
        json={"text": "Tokenization is useful.", "tokenizer_type": "bpe", "add_special_tokens": True},
    ).json()
    response = client.post(
        "/detokenize",
        json={"token_ids": tokenized["token_ids"], "tokenizer_type": "bpe", "skip_special_tokens": True},
    )
    assert response.status_code == 200
    assert "tokenization" in response.json()["text"]


def test_unsupported_tokenizer_type_returns_error():
    client = TestClient(app)
    response = client.post(
        "/tokenize",
        json={"text": "test", "tokenizer_type": "missing", "add_special_tokens": True},
    )
    assert response.status_code == 400
