"""FastAPI app exposing tokenizer encode/decode endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError("Install fastapi and pydantic to run tokenizer.api") from exc

from .base import BaseTokenizer
from .bpe import BPETokenizer
from .byte_bpe import ByteLevelBPETokenizer
from .unigram import UnigramTokenizer
from .wordpiece import WordPieceTokenizer

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
SAMPLE_CORPUS = ROOT / "data" / "sample_corpus.txt"

TOKENIZER_CONFIG = {
    "bpe": (BPETokenizer, ARTIFACTS / "bpe_tokenizer.json", {"vocab_size": 120, "min_frequency": 1}),
    "byte_bpe": (
        ByteLevelBPETokenizer,
        ARTIFACTS / "byte_bpe_tokenizer.json",
        {"vocab_size": 320, "min_frequency": 1},
    ),
    "wordpiece": (
        WordPieceTokenizer,
        ARTIFACTS / "wordpiece_tokenizer.json",
        {"vocab_size": 120, "min_frequency": 1},
    ),
    "unigram": (
        UnigramTokenizer,
        ARTIFACTS / "unigram_tokenizer.json",
        {"vocab_size": 120, "min_frequency": 1},
    ),
}
TOKENIZER_CACHE: Dict[str, BaseTokenizer] = {}

app = FastAPI(title="Custom Tokenizer API", version="0.2.0")


class TokenizeRequest(BaseModel):
    """Request body for tokenization."""

    text: str
    tokenizer_type: str = "bpe"
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    """Response body for tokenization."""

    tokens: List[str]
    token_ids: List[int]
    tokenizer_type: str


class DetokenizeRequest(BaseModel):
    """Request body for detokenization."""

    token_ids: List[int]
    tokenizer_type: str = "bpe"
    skip_special_tokens: bool = True


class DetokenizeResponse(BaseModel):
    """Response body for detokenization."""

    text: str
    tokenizer_type: str


@app.get("/health")
def health() -> Dict[str, str]:
    """Return service health."""

    return {"status": "ok"}


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    """Encode text with a selected tokenizer."""

    tokenizer = get_tokenizer(request.tokenizer_type)
    token_ids = tokenizer.encode(request.text, add_special_tokens=request.add_special_tokens)
    return TokenizeResponse(
        tokens=[tokenizer.id_to_token(token_id) for token_id in token_ids],
        token_ids=token_ids,
        tokenizer_type=request.tokenizer_type,
    )


@app.post("/detokenize", response_model=DetokenizeResponse)
def detokenize(request: DetokenizeRequest) -> DetokenizeResponse:
    """Decode token IDs with a selected tokenizer."""

    tokenizer = get_tokenizer(request.tokenizer_type)
    return DetokenizeResponse(
        text=tokenizer.decode(request.token_ids, skip_special_tokens=request.skip_special_tokens),
        tokenizer_type=request.tokenizer_type,
    )


def get_tokenizer(tokenizer_type: str) -> BaseTokenizer:
    """Load, train, and cache the requested tokenizer."""

    if tokenizer_type not in TOKENIZER_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported tokenizer_type: {tokenizer_type}")
    if tokenizer_type in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[tokenizer_type]

    tokenizer_cls, artifact_path, kwargs = TOKENIZER_CONFIG[tokenizer_type]
    if artifact_path.exists():
        tokenizer = tokenizer_cls.load(str(artifact_path))  # type: ignore[attr-defined]
    else:
        tokenizer = tokenizer_cls(**kwargs)
        corpus = str(SAMPLE_CORPUS) if SAMPLE_CORPUS.exists() else "Tokenization is useful for NLP."
        tokenizer.fit(corpus)
    TOKENIZER_CACHE[tokenizer_type] = tokenizer
    return tokenizer
