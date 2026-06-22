"""Simplified SentencePiece-style unigram tokenizer."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from .base import BaseTokenizer
from .preprocessing import basic_tokenize, detokenize
from .utils import read_corpus, read_json, write_json
from .vocabulary import DEFAULT_SPECIAL_TOKENS, Vocabulary


class UnigramTokenizer(BaseTokenizer):
    """Educational unigram language-model tokenizer with Viterbi segmentation."""

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        lowercase: bool = True,
        unk_token: str = "<UNK>",
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
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS.copy()
        if unk_token not in self.special_tokens:
            self.special_tokens.insert(1, unk_token)
        self.max_subword_length = max_subword_length
        self.vocab = Vocabulary(self.special_tokens)
        self.token_scores: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, corpus: Union[List[str], str]) -> None:
        """Train candidate subword probabilities from substring frequencies."""

        words = [
            token
            for text in read_corpus(corpus)
            for token in basic_tokenize(text, lowercase=self.lowercase, split_punctuation=True)
        ]
        if not words:
            raise ValueError("Cannot train tokenizer on an empty corpus.")
        counts: Counter[str] = Counter()
        for word in words:
            for char in word:
                counts[char] += 1
            for start in range(len(word)):
                max_end = min(len(word), start + self.max_subword_length)
                for end in range(start + 2, max_end + 1):
                    counts[word[start:end]] += 1

        self.vocab = Vocabulary(self.special_tokens)
        candidates = [
            (token, count)
            for token, count in counts.items()
            if count >= self.min_frequency or len(token) == 1
        ]
        candidates.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        kept = candidates[: max(0, self.vocab_size - len(self.special_tokens))]
        total = sum(count for _, count in kept) or 1
        self.token_scores = {}
        for token, count in kept:
            self.vocab.add_token(token)
            self.token_scores[token] = -math.log(count / total)
        self.token_scores[self.unk_token] = max(self.token_scores.values(), default=10.0) + 5.0
        self.is_fitted = True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text by selecting the minimum-cost subword segmentation."""

        self._require_fitted()
        tokens: List[str] = []
        if add_special_tokens and "<BOS>" in self.special_tokens:
            tokens.append("<BOS>")
        for word in basic_tokenize(text, lowercase=self.lowercase, split_punctuation=True):
            tokens.extend(self.segment_word(word))
        if add_special_tokens and "<EOS>" in self.special_tokens:
            tokens.append("<EOS>")
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode unigram token IDs into readable text."""

        self._require_fitted()
        pieces: List[str] = []
        current = ""
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in self.special_tokens:
                if not skip_special_tokens:
                    if current:
                        pieces.append(current)
                        current = ""
                    pieces.append(token)
                continue
            if token == self.unk_token or len(token) == 1 and not token.isalnum():
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(token)
            else:
                current += token
        if current:
            pieces.append(current)
        return detokenize(pieces)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""

        return super().batch_encode(texts, add_special_tokens=add_special_tokens)

    def save(self, path: str) -> None:
        """Save tokenizer state as JSON."""

        self._require_fitted()
        write_json(
            path,
            {
                "tokenizer_type": "unigram",
                "config": {
                    "vocab_size": self.vocab_size,
                    "min_frequency": self.min_frequency,
                    "lowercase": self.lowercase,
                    "unk_token": self.unk_token,
                    "special_tokens": self.special_tokens,
                    "max_subword_length": self.max_subword_length,
                },
                "vocabulary": self.vocab.to_dict(),
                "token_scores": self.token_scores,
            },
        )

    @classmethod
    def load(cls, path: str) -> "UnigramTokenizer":
        """Load tokenizer state from JSON."""

        data = read_json(path)
        config = data["config"]
        tokenizer = cls(
            vocab_size=int(config["vocab_size"]),
            min_frequency=int(config["min_frequency"]),
            lowercase=bool(config["lowercase"]),
            unk_token=str(config["unk_token"]),
            special_tokens=list(config["special_tokens"]),
            max_subword_length=int(config.get("max_subword_length", 12)),
        )
        tokenizer.vocab = Vocabulary.from_dict(dict(data["vocabulary"]))
        tokenizer.token_scores = {str(token): float(score) for token, score in dict(data["token_scores"]).items()}
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

    def segment_word(self, word: str) -> List[str]:
        """Segment one pre-tokenized word with Viterbi dynamic programming."""

        n = len(word)
        best: List[Tuple[float, List[str]]] = [(math.inf, []) for _ in range(n + 1)]
        best[0] = (0.0, [])
        for start in range(n):
            if math.isinf(best[start][0]):
                continue
            max_end = min(n, start + self.max_subword_length)
            for end in range(start + 1, max_end + 1):
                piece = word[start:end]
                if piece not in self.token_scores:
                    continue
                cost = best[start][0] + self.token_scores[piece]
                if cost < best[end][0]:
                    best[end] = (cost, best[start][1] + [piece])
        if math.isinf(best[n][0]):
            return [self.unk_token]
        return best[n][1]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before use.")
