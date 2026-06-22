"""Simplified educational WordPiece tokenizer."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Union

from .base import BaseTokenizer
from .preprocessing import basic_tokenize, detokenize
from .utils import read_corpus, read_json, write_json
from .vocabulary import Vocabulary


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece-style tokenizer using continuation prefixes and greedy encoding.

    Training is intentionally simplified for readability: it counts full words,
    characters, and frequent substrings, then keeps the highest-frequency pieces.
    Encoding follows the important BERT-like behavior: greedy longest-match-first
    segmentation with ``##`` continuation pieces after the first subword.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        lowercase: bool = True,
        unk_token: str = "[UNK]",
        continuation_prefix: str = "##",
        special_tokens: Optional[List[str]] = None,
        max_subword_length: int = 12,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.continuation_prefix = continuation_prefix
        self.special_tokens = special_tokens or ["[PAD]", unk_token, "[CLS]", "[SEP]"]
        if unk_token not in self.special_tokens:
            self.special_tokens.insert(1, unk_token)
        self.max_subword_length = max_subword_length
        self.vocab = Vocabulary(self.special_tokens)
        self.is_fitted = False

    def fit(self, corpus: Union[List[str], str]) -> None:
        """Build a WordPiece vocabulary from frequent words and subwords."""

        words = [
            token
            for text in read_corpus(corpus)
            for token in basic_tokenize(text, lowercase=self.lowercase, split_punctuation=True)
        ]
        if not words:
            raise ValueError("Cannot train tokenizer on an empty corpus.")
        word_counts = Counter(words)
        candidate_counts: Counter[str] = Counter()

        for word, frequency in word_counts.items():
            if frequency >= self.min_frequency:
                candidate_counts[word] += frequency * max(len(word), 1)
            for index, char in enumerate(word):
                candidate_counts[char if index == 0 else self.continuation_prefix + char] += frequency
            for start in range(len(word)):
                max_end = min(len(word), start + self.max_subword_length)
                for end in range(start + 2, max_end + 1):
                    piece = word[start:end]
                    token = piece if start == 0 else self.continuation_prefix + piece
                    candidate_counts[token] += frequency

        self.vocab = Vocabulary(self.special_tokens)
        for token, _ in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0])):
            if len(self.vocab) >= self.vocab_size:
                break
            self.vocab.add_token(token)
        self.is_fitted = True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text using greedy longest-match-first segmentation."""

        self._require_fitted()
        tokens: List[str] = []
        if add_special_tokens and "[CLS]" in self.special_tokens:
            tokens.append("[CLS]")
        for word in basic_tokenize(text, lowercase=self.lowercase, split_punctuation=True):
            tokens.extend(self._encode_word(word))
        if add_special_tokens and "[SEP]" in self.special_tokens:
            tokens.append("[SEP]")
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode WordPiece token IDs into readable text."""

        self._require_fitted()
        words: List[str] = []
        current = ""
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in self.special_tokens:
                if not skip_special_tokens:
                    if current:
                        words.append(current)
                        current = ""
                    words.append(token)
                continue
            if token == self.unk_token:
                if current:
                    words.append(current)
                words.append(token)
                current = ""
                continue
            if token.startswith(self.continuation_prefix):
                current += token[len(self.continuation_prefix) :]
            else:
                if current:
                    words.append(current)
                current = token
        if current:
            words.append(current)
        return detokenize(words)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""

        return super().batch_encode(texts, add_special_tokens=add_special_tokens)

    def save(self, path: str) -> None:
        """Save tokenizer state as JSON."""

        self._require_fitted()
        write_json(
            path,
            {
                "tokenizer_type": "wordpiece",
                "config": {
                    "vocab_size": self.vocab_size,
                    "min_frequency": self.min_frequency,
                    "lowercase": self.lowercase,
                    "unk_token": self.unk_token,
                    "continuation_prefix": self.continuation_prefix,
                    "special_tokens": self.special_tokens,
                    "max_subword_length": self.max_subword_length,
                },
                "vocabulary": self.vocab.to_dict(),
            },
        )

    @classmethod
    def load(cls, path: str) -> "WordPieceTokenizer":
        """Load tokenizer state from JSON."""

        data = read_json(path)
        config = data["config"]
        tokenizer = cls(
            vocab_size=int(config["vocab_size"]),
            min_frequency=int(config["min_frequency"]),
            lowercase=bool(config["lowercase"]),
            unk_token=str(config["unk_token"]),
            continuation_prefix=str(config["continuation_prefix"]),
            special_tokens=list(config["special_tokens"]),
            max_subword_length=int(config.get("max_subword_length", 12)),
        )
        tokenizer.vocab = Vocabulary.from_dict(dict(data["vocabulary"]))
        tokenizer.is_fitted = True
        return tokenizer

    def get_vocab(self) -> Dict[str, int]:
        """Return token-to-ID vocabulary."""

        return dict(self.vocab.token_to_id_map)

    def token_to_id(self, token: str) -> int:
        """Return token ID, falling back to configured unknown token."""

        if token in self.vocab.token_to_id_map:
            return self.vocab.token_to_id_map[token]
        return self.vocab.token_to_id_map[self.unk_token]

    def id_to_token(self, token_id: int) -> str:
        """Return token string, falling back to configured unknown token."""

        return self.vocab.id_to_token_map.get(token_id, self.unk_token)

    def _encode_word(self, word: str) -> List[str]:
        if word in self.vocab.token_to_id_map:
            return [word]
        pieces: List[str] = []
        start = 0
        while start < len(word):
            match = None
            end = min(len(word), start + self.max_subword_length)
            while end > start:
                piece = word[start:end]
                token = piece if start == 0 else self.continuation_prefix + piece
                if token in self.vocab.token_to_id_map:
                    match = token
                    break
                end -= 1
            if match is None:
                return [self.unk_token]
            pieces.append(match)
            start = end
        return pieces

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before use.")
