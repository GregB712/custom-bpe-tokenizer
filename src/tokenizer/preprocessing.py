"""Text normalization and tokenization helpers."""

from __future__ import annotations

import re
from typing import List


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize whitespace and optionally lowercase text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def basic_tokenize(
    text: str,
    lowercase: bool = True,
    split_punctuation: bool = True,
) -> List[str]:
    """Split text into word and punctuation tokens.

    When ``split_punctuation`` is false, the function keeps whitespace-delimited
    chunks intact after normalization.
    """

    text = normalize_text(text, lowercase=lowercase)
    if not text:
        return []
    if split_punctuation:
        return TOKEN_PATTERN.findall(text)
    return text.split()


def detokenize(tokens: List[str]) -> str:
    """Join decoded word pieces into readable text."""

    text = " ".join(token for token in tokens if token)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return text.strip()
