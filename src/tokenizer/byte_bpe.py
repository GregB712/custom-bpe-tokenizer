"""Byte-level Byte Pair Encoding tokenizer."""

from __future__ import annotations

from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Optional, Tuple, Union

from .base import BaseTokenizer
from .utils import read_corpus, read_json, write_json
from .vocabulary import DEFAULT_SPECIAL_TOKENS, Vocabulary

Pair = Tuple[str, str]
Sequence = Tuple[str, ...]


def byte_token(value: int) -> str:
    """Return a JSON-safe token for one byte value."""

    return f"<0x{value:02X}>"


class ByteLevelBPETokenizer(BaseTokenizer):
    """GPT-style educational BPE tokenizer trained over UTF-8 byte tokens."""

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        lowercase: bool = False,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        if vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256 byte tokens")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS.copy()
        if "<UNK>" not in self.special_tokens:
            raise ValueError("special_tokens must include <UNK>")
        self.vocab = Vocabulary(self.special_tokens)
        self.merges: List[Pair] = []
        self.merge_ranks: Dict[Pair, int] = {}
        self.is_fitted = False

    def fit(self, corpus: Union[List[str], str]) -> None:
        """Train byte-level BPE merge rules from UTF-8 byte sequences."""

        sequences = Counter(self._text_to_base_tokens(text) for text in read_corpus(corpus))
        sequences = Counter({sequence: frequency for sequence, frequency in sequences.items() if sequence})
        if not sequences:
            raise ValueError("Cannot train tokenizer on an empty corpus.")

        self.vocab = Vocabulary(self.special_tokens)
        self.merges = []
        self.merge_ranks = {}
        for value in range(256):
            self.vocab.add_token(byte_token(value))

        while len(self.vocab) < self.vocab_size:
            pair_counts = self._get_pair_counts(sequences)
            best_pair = self._select_best_pair(pair_counts)
            if best_pair is None:
                break
            merged = "".join(best_pair)
            if merged not in self.vocab.token_to_id_map:
                self.vocab.add_token(merged)
            self._register_merge(best_pair)
            sequences = self._merge_corpus(sequences, best_pair)
        self.is_fitted = True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode arbitrary Unicode text into byte-level BPE token IDs."""

        self._require_fitted()
        tokens = list(self._apply_merges(self._text_to_base_tokens(text)))
        if add_special_tokens and "<BOS>" in self.special_tokens:
            tokens.insert(0, "<BOS>")
        if add_special_tokens and "<EOS>" in self.special_tokens:
            tokens.append("<EOS>")
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode byte-level token IDs using UTF-8 with replacement fallback."""

        self._require_fitted()
        raw_bytes = bytearray()
        specials: List[str] = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in self.special_tokens:
                if not skip_special_tokens:
                    specials.append(token)
                continue
            for part in self._split_merged_token(token):
                try:
                    raw_bytes.append(int(part[3:5], 16))
                except ValueError:
                    if not skip_special_tokens:
                        specials.append("<UNK>")
        decoded = bytes(raw_bytes).decode("utf-8", errors="replace")
        return " ".join(specials + [decoded]).strip() if specials else decoded

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""

        return super().batch_encode(texts, add_special_tokens=add_special_tokens)

    def save(self, path: str) -> None:
        """Save tokenizer state as inspectable JSON."""

        self._require_fitted()
        write_json(
            path,
            {
                "tokenizer_type": "byte_bpe",
                "config": {
                    "vocab_size": self.vocab_size,
                    "min_frequency": self.min_frequency,
                    "lowercase": self.lowercase,
                    "special_tokens": self.special_tokens,
                },
                "vocabulary": self.vocab.to_dict(),
                "merges": [list(pair) for pair in self.merges],
            },
        )

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        """Load tokenizer state from JSON."""

        data = read_json(path)
        config = data["config"]
        tokenizer = cls(
            vocab_size=int(config["vocab_size"]),
            min_frequency=int(config["min_frequency"]),
            lowercase=bool(config["lowercase"]),
            special_tokens=list(config["special_tokens"]),
        )
        tokenizer.vocab = Vocabulary.from_dict(dict(data["vocabulary"]))
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.merge_ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
        tokenizer.is_fitted = True
        return tokenizer

    def get_vocab(self) -> Dict[str, int]:
        """Return token-to-ID vocabulary."""

        return dict(self.vocab.token_to_id_map)

    def token_to_id(self, token: str) -> int:
        """Return token ID, falling back to ``<UNK>``."""

        return self.vocab.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Return token string, falling back to ``<UNK>``."""

        return self.vocab.id_to_token(token_id)

    def trace_encode(self, text: str) -> List[Dict[str, object]]:
        """Return byte-token merge steps for visualization."""

        self._require_fitted()
        symbols = self._text_to_base_tokens(text)
        steps: List[Dict[str, object]] = [{"stage": "initial", "tokens": list(symbols), "merge": None}]
        while len(symbols) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            symbols = self._merge_sequence(symbols, best_pair)
            steps.append({"stage": "merge", "merge": list(best_pair), "tokens": list(symbols)})
        steps.append({"stage": "final", "tokens": list(symbols), "merge": None})
        return [{"text": text, "steps": steps, "final_tokens": list(symbols)}]

    def _text_to_base_tokens(self, text: str) -> Sequence:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = text.lower() if self.lowercase else text
        return tuple(byte_token(value) for value in normalized.encode("utf-8"))

    def _get_pair_counts(self, sequences: CounterType[Sequence]) -> CounterType[Pair]:
        counts: CounterType[Pair] = Counter()
        for sequence, frequency in sequences.items():
            for pair in zip(sequence, sequence[1:]):
                counts[pair] += frequency
        return counts

    def _select_best_pair(self, pair_counts: CounterType[Pair]) -> Optional[Pair]:
        candidates = [(pair, count) for pair, count in pair_counts.items() if count >= self.min_frequency]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (-item[1], item[0][0], item[0][1]))[0][0]

    def _register_merge(self, pair: Pair) -> None:
        if pair not in self.merge_ranks:
            self.merge_ranks[pair] = len(self.merges)
            self.merges.append(pair)

    def _merge_corpus(self, sequences: CounterType[Sequence], pair: Pair) -> CounterType[Sequence]:
        merged: CounterType[Sequence] = Counter()
        for sequence, frequency in sequences.items():
            merged[self._merge_sequence(sequence, pair)] += frequency
        return merged

    def _merge_sequence(self, sequence: Sequence, pair: Pair) -> Sequence:
        output: List[str] = []
        index = 0
        while index < len(sequence):
            if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
                output.append(sequence[index] + sequence[index + 1])
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return tuple(output)

    def _apply_merges(self, sequence: Sequence) -> Sequence:
        symbols = sequence
        while len(symbols) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            symbols = self._merge_sequence(symbols, best_pair)
        return symbols

    def _split_merged_token(self, token: str) -> List[str]:
        if len(token) % 6 != 0:
            return []
        return [token[index : index + 6] for index in range(0, len(token), 6)]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before use.")
