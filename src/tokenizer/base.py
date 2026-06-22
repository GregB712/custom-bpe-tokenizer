"""Common tokenizer interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union


class BaseTokenizer(ABC):
    """Abstract interface shared by educational tokenizer implementations."""

    @abstractmethod
    def fit(self, corpus: Union[List[str], str]) -> None:
        """Train tokenizer state from a corpus list, raw text, or file path."""

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text into token IDs."""

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into readable text."""

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Encode multiple texts with the same options."""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist tokenizer state to disk."""

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Return token-to-ID vocabulary."""

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """Return an integer ID for a token."""

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """Return the token string for an integer ID."""
