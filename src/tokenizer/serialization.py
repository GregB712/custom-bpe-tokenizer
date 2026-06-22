"""Serialization helpers for tokenizer artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

from .base import BaseTokenizer
from .utils import read_json


def load_tokenizer_artifact(path: str, tokenizer_cls: Type[BaseTokenizer]) -> BaseTokenizer:
    """Load a tokenizer artifact using the class-level ``load`` method."""

    load = getattr(tokenizer_cls, "load", None)
    if load is None:
        raise TypeError(f"{tokenizer_cls.__name__} does not provide a load() method")
    return load(path)


def artifact_metadata(path: str) -> Dict[str, Any]:
    """Return top-level JSON artifact metadata for inspection."""

    data = read_json(Path(path))
    return {
        "tokenizer_type": data.get("tokenizer_type", "unknown"),
        "config": data.get("config", {}),
        "vocab_size": len(data.get("vocabulary", {}).get("token_to_id", {})),
    }
