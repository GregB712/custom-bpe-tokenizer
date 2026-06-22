"""Vocabulary utilities for the custom BPE tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


DEFAULT_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


@dataclass
class Vocabulary:
    """Bidirectional token to integer-id mapping.

    The class preserves insertion order so token IDs are deterministic when
    tokens are added in a deterministic order.
    """

    special_tokens: List[str] = field(default_factory=lambda: DEFAULT_SPECIAL_TOKENS.copy())
    token_to_id_map: Dict[str, int] = field(default_factory=dict)
    id_to_token_map: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_to_id_map:
            self.add_tokens(self.special_tokens)
        elif not self.id_to_token_map:
            self.id_to_token_map = {idx: token for token, idx in self.token_to_id_map.items()}

    def __len__(self) -> int:
        return len(self.token_to_id_map)

    @property
    def unk_token(self) -> str:
        """Return the unknown-token string."""

        return "<UNK>"

    @property
    def unk_id(self) -> int:
        """Return the unknown-token ID."""

        return self.token_to_id(self.unk_token)

    def add_token(self, token: str) -> int:
        """Add a token if missing and return its ID."""

        if token not in self.token_to_id_map:
            token_id = len(self.token_to_id_map)
            self.token_to_id_map[token] = token_id
            self.id_to_token_map[token_id] = token
        return self.token_to_id_map[token]

    def add_tokens(self, tokens: Iterable[str]) -> None:
        """Add multiple tokens, preserving iteration order."""

        for token in tokens:
            self.add_token(token)

    def token_to_id(self, token: str) -> int:
        """Return the ID for a token, falling back to ``<UNK>``."""

        if token in self.token_to_id_map:
            return self.token_to_id_map[token]
        if self.unk_token in self.token_to_id_map:
            return self.token_to_id_map[self.unk_token]
        raise KeyError("Vocabulary does not contain an <UNK> token.")

    def id_to_token(self, token_id: int) -> str:
        """Return the token for an ID, falling back to ``<UNK>``."""

        return self.id_to_token_map.get(token_id, self.unk_token)

    def to_dict(self) -> Dict[str, object]:
        """Serialize the vocabulary to a JSON-compatible dictionary."""

        return {
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        """Create a vocabulary from serialized data."""

        special_tokens = list(data.get("special_tokens", DEFAULT_SPECIAL_TOKENS))
        token_to_id = {
            str(token): int(token_id)
            for token, token_id in dict(data["token_to_id"]).items()
        }
        return cls(special_tokens=special_tokens, token_to_id_map=token_to_id)
