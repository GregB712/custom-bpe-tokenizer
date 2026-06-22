"""Byte Pair Encoding tokenizer implemented from scratch."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .base import BaseTokenizer
from .preprocessing import basic_tokenize, detokenize
from .utils import read_corpus, read_json, write_json
from .vocabulary import DEFAULT_SPECIAL_TOKENS, Vocabulary


Pair = Tuple[str, str]
Word = Tuple[str, ...]


class BPETokenizer(BaseTokenizer):
    """A compact, educational Byte Pair Encoding tokenizer.

    The implementation starts from character-level word tokens and learns merge
    rules by repeatedly combining the most frequent adjacent pair. An internal
    end-of-word marker lets decoding reconstruct readable whitespace-separated
    text while still keeping the learned BPE units explicit.
    """

    end_of_word = "</w>"

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        lowercase: bool = True,
        special_tokens: Optional[List[str]] = None,
        split_punctuation: bool = True,
    ) -> None:
        """Initialize a BPE tokenizer.

        Args:
            vocab_size: Maximum vocabulary size, including special tokens.
            min_frequency: Minimum adjacent-pair count required for a merge.
            lowercase: Whether to lowercase text before training and encoding.
            special_tokens: Optional replacement for the default special tokens.
            split_punctuation: Whether punctuation is split into separate tokens.
        """

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS.copy()
        self.split_punctuation = split_punctuation

        if "<UNK>" not in self.special_tokens:
            raise ValueError("special_tokens must include <UNK>")
        if self.vocab_size <= len(self.special_tokens):
            raise ValueError("vocab_size must be greater than the number of special tokens")

        self.vocab = Vocabulary(self.special_tokens)
        self.merges: List[Pair] = []
        self.merge_ranks: Dict[Pair, int] = {}
        self.is_fitted = False

    def fit(self, corpus: Union[List[str], str]) -> None:
        """Train the tokenizer on a list of texts or a text-file path."""

        texts = read_corpus(corpus)
        tokenized_words = Counter(self._iter_initial_words(texts))
        if not tokenized_words:
            raise ValueError("Cannot train tokenizer on an empty corpus.")

        self.vocab = Vocabulary(self.special_tokens)
        self.merges = []
        self.merge_ranks = {}

        symbol_counts: CounterType[str] = Counter()
        for word, frequency in tokenized_words.items():
            for symbol in word:
                symbol_counts[symbol] += frequency

        for symbol, _ in sorted(symbol_counts.items(), key=lambda item: (-item[1], item[0])):
            if len(self.vocab) >= self.vocab_size:
                break
            self.vocab.add_token(symbol)

        # Core BPE training loop:
        # 1. Count all adjacent symbol pairs in the current corpus representation.
        # 2. Select the most frequent pair above the frequency threshold.
        # 3. Replace that pair everywhere with its concatenated token.
        # 4. Add the new token and repeat until the vocabulary budget is exhausted.
        while len(self.vocab) < self.vocab_size:
            pair_counts = self._get_pair_counts(tokenized_words)
            best_pair = self._select_best_pair(pair_counts)
            if best_pair is None:
                break

            merged_token = "".join(best_pair)
            if merged_token in self.vocab.token_to_id_map:
                tokenized_words = self._merge_corpus(tokenized_words, best_pair)
                self._register_merge(best_pair)
                continue

            self.vocab.add_token(merged_token)
            self._register_merge(best_pair)
            tokenized_words = self._merge_corpus(tokenized_words, best_pair)

        self.is_fitted = True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs."""

        self._require_fitted()
        tokens: List[str] = []
        if add_special_tokens and "<BOS>" in self.special_tokens:
            tokens.append("<BOS>")

        for raw_token in basic_tokenize(
            text,
            lowercase=self.lowercase,
            split_punctuation=self.split_punctuation,
        ):
            tokens.extend(self._encode_word(raw_token))

        if add_special_tokens and "<EOS>" in self.special_tokens:
            tokens.append("<EOS>")
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into readable text."""

        self._require_fitted()
        decoded_units: List[str] = []
        current = ""

        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if skip_special_tokens and token in self.special_tokens:
                continue
            if token in self.special_tokens:
                decoded_units.append(token)
                continue
            if token == "<UNK>":
                current += token
                continue
            if token.endswith(self.end_of_word):
                current += token[: -len(self.end_of_word)]
                decoded_units.append(current)
                current = ""
            else:
                current += token

        if current:
            decoded_units.append(current)
        return detokenize(decoded_units)

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Encode a batch of texts."""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def save(self, path: str) -> None:
        """Save the tokenizer to a JSON file."""

        self._require_fitted()
        data = {
            "tokenizer_type": "bpe",
            "config": {
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "lowercase": self.lowercase,
                "special_tokens": self.special_tokens,
                "split_punctuation": self.split_punctuation,
                "end_of_word": self.end_of_word,
            },
            "vocabulary": self.vocab.to_dict(),
            "merges": [list(pair) for pair in self.merges],
        }
        write_json(path, data)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from a JSON file."""

        data = read_json(path)
        config = dict(data["config"])
        tokenizer = cls(
            vocab_size=int(config["vocab_size"]),
            min_frequency=int(config["min_frequency"]),
            lowercase=bool(config["lowercase"]),
            special_tokens=list(config["special_tokens"]),
            split_punctuation=bool(config.get("split_punctuation", True)),
        )
        tokenizer.vocab = Vocabulary.from_dict(dict(data["vocabulary"]))
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.merge_ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
        tokenizer.is_fitted = True
        return tokenizer

    def get_vocab(self) -> Dict[str, int]:
        """Return a copy of the token-to-ID vocabulary."""

        return dict(self.vocab.token_to_id_map)

    def token_to_id(self, token: str) -> int:
        """Return the ID for a token, using ``<UNK>`` for missing tokens."""

        return self.vocab.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Return the token string for an ID, using ``<UNK>`` for missing IDs."""

        return self.vocab.id_to_token(token_id)

    def trace_encode(self, text: str) -> List[Dict[str, object]]:
        """Return merge steps used while encoding text for visualization."""

        self._require_fitted()
        traces: List[Dict[str, object]] = []
        for raw_token in basic_tokenize(
            text,
            lowercase=self.lowercase,
            split_punctuation=self.split_punctuation,
        ):
            symbols = self._word_to_symbols(raw_token)
            steps: List[Dict[str, object]] = [
                {"stage": "initial", "tokens": list(symbols), "merge": None}
            ]
            while len(symbols) > 1:
                ranked_pairs = [
                    (self.merge_ranks[pair], pair)
                    for pair in zip(symbols, symbols[1:])
                    if pair in self.merge_ranks
                ]
                if not ranked_pairs:
                    break
                _, best_pair = min(ranked_pairs, key=lambda item: item[0])
                symbols = self._merge_word(symbols, best_pair)
                steps.append(
                    {"stage": "merge", "merge": list(best_pair), "tokens": list(symbols)}
                )
            steps.append({"stage": "final", "tokens": list(symbols), "merge": None})
            traces.append({"text": raw_token, "steps": steps, "final_tokens": list(symbols)})
        if not traces:
            return [
                {
                    "text": "",
                    "steps": [
                        {"stage": "initial", "tokens": [], "merge": None},
                        {"stage": "final", "tokens": [], "merge": None},
                    ],
                    "final_tokens": [],
                }
            ]
        return traces

    def _iter_initial_words(self, texts: Iterable[str]) -> Iterable[Word]:
        for text in texts:
            for token in basic_tokenize(
                text,
                lowercase=self.lowercase,
                split_punctuation=self.split_punctuation,
            ):
                yield self._word_to_symbols(token)

    def _word_to_symbols(self, word: str) -> Word:
        return tuple(word) + (self.end_of_word,)

    def _get_pair_counts(self, corpus_words: CounterType[Word]) -> CounterType[Pair]:
        pair_counts: CounterType[Pair] = Counter()
        for word, frequency in corpus_words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += frequency
        return pair_counts

    def _select_best_pair(self, pair_counts: CounterType[Pair]) -> Optional[Pair]:
        candidates = [
            (pair, frequency)
            for pair, frequency in pair_counts.items()
            if frequency >= self.min_frequency
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (-item[1], item[0][0], item[0][1]))[0][0]

    def _register_merge(self, pair: Pair) -> None:
        if pair not in self.merge_ranks:
            self.merge_ranks[pair] = len(self.merges)
            self.merges.append(pair)

    def _merge_corpus(
        self,
        corpus_words: CounterType[Word],
        pair: Pair,
    ) -> CounterType[Word]:
        merged: CounterType[Word] = Counter()
        for word, frequency in corpus_words.items():
            merged[self._merge_word(word, pair)] += frequency
        return merged

    def _merge_word(self, word: Word, pair: Pair) -> Word:
        output: List[str] = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
                output.append(word[index] + word[index + 1])
                index += 2
            else:
                output.append(word[index])
                index += 1
        return tuple(output)

    def _encode_word(self, word: str) -> List[str]:
        symbols = self._word_to_symbols(word)
        if not self.merges:
            return list(symbols)

        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in pairs
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            symbols = self._merge_word(symbols, best_pair)

        return [
            token if token in self.vocab.token_to_id_map else "<UNK>"
            for token in symbols
        ]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before use.")


def load_tokenizer(path: Union[str, Path]) -> BPETokenizer:
    """Convenience function for loading a tokenizer."""

    return BPETokenizer.load(str(path))
