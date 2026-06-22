"""Utility helpers for file IO and corpus handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def read_corpus(corpus: Union[List[str], str]) -> List[str]:
    """Read a corpus from a list of texts, a path, or a raw text string."""

    if isinstance(corpus, list):
        if not all(isinstance(item, str) for item in corpus):
            raise TypeError("All corpus items must be strings.")
        return corpus

    if isinstance(corpus, str):
        path = Path(corpus)
        if path.exists():
            return path.read_text(encoding="utf-8").splitlines()
        return [corpus]

    raise TypeError("corpus must be a list of strings or a string path/text")


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """Create the parent directory for a file path and return the path."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Write JSON with deterministic formatting."""

    output_path = ensure_parent_dir(path)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON file into a dictionary."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten(iterables: Iterable[Iterable[str]]) -> List[str]:
    """Flatten one level of nested string iterables."""

    return [item for iterable in iterables for item in iterable]
