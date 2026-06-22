"""Streamlit app for visualizing BPE merge behavior."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from tokenizer.bpe import BPETokenizer
from tokenizer.byte_bpe import ByteLevelBPETokenizer


DEFAULT_CORPUS = """Tokenization converts text into model-friendly token IDs.
Byte pair encoding learns frequent adjacent symbol pairs.
Byte-level BPE handles Unicode text such as Ελληνικά, café, and 🚀."""


def build_tokenizer(tokenizer_type: str, vocab_size: int, min_frequency: int, lowercase: bool):
    """Create a tokenizer instance for the selected app mode."""

    if tokenizer_type == "Byte-Level BPE":
        return ByteLevelBPETokenizer(
            vocab_size=max(vocab_size, 257),
            min_frequency=min_frequency,
            lowercase=lowercase,
        )
    return BPETokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        lowercase=lowercase,
    )


def flatten_trace(trace: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert nested trace output into a display table."""

    rows = []
    for item in trace:
        for index, step in enumerate(item["steps"]):
            rows.append(
                {
                    "segment": item["text"],
                    "step": index,
                    "stage": step["stage"],
                    "merge": " + ".join(step.get("merge") or []),
                    "tokens": " | ".join(step.get("tokens") or []),
                }
            )
    return pd.DataFrame(rows)


def token_frequency_frame(tokenizer) -> pd.DataFrame:
    """Create a simple frequency-like table from learned merge outputs."""

    counter = Counter()
    for pair in getattr(tokenizer, "merges", []):
        counter["".join(pair)] += 1
    if not counter:
        for token in list(tokenizer.get_vocab())[:20]:
            counter[token] += 1
    return pd.DataFrame(counter.most_common(20), columns=["token", "count"])


st.set_page_config(page_title="Tokenizer Merge Lab", layout="wide")
st.title("Tokenizer Merge Lab")
st.caption("Inspect how character-level BPE and byte-level BPE compress text into learned subword units.")

with st.sidebar:
    tokenizer_type = st.selectbox("Tokenizer type", ["BPE", "Byte-Level BPE"])
    minimum_vocab = 260 if tokenizer_type == "Byte-Level BPE" else 20
    vocab_size = st.slider("Vocabulary size", minimum_vocab, 1000, 320 if tokenizer_type == "Byte-Level BPE" else 120)
    min_frequency = st.slider("Minimum merge frequency", 1, 10, 1)
    lowercase = st.checkbox("Lowercase", value=tokenizer_type == "BPE")
    sample_text = st.text_input("Sample text", "Tokenization handles Ελληνικά and 🚀.")
    corpus = st.text_area("Training corpus", DEFAULT_CORPUS, height=220)
    train_clicked = st.button("Train tokenizer", type="primary")

if train_clicked or "tokenizer" not in st.session_state:
    tokenizer = build_tokenizer(tokenizer_type, vocab_size, min_frequency, lowercase)
    tokenizer.fit(corpus.splitlines())
    st.session_state["tokenizer"] = tokenizer
    st.session_state["tokenizer_type"] = tokenizer_type

tokenizer = st.session_state["tokenizer"]
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
trace = tokenizer.trace_encode(sample_text)

metric_left, metric_mid, metric_right = st.columns(3)
metric_left.metric("Vocabulary", len(tokenizer.get_vocab()))
metric_mid.metric("Merge rules", len(getattr(tokenizer, "merges", [])))
metric_right.metric("Encoded length", len(encoded))

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Merge Trace")
    st.dataframe(flatten_trace(trace), use_container_width=True, hide_index=True)
with right:
    st.subheader("Top Merge Rules")
    merge_rows = [
        {"rank": index + 1, "left": pair[0], "right": pair[1], "merged": "".join(pair)}
        for index, pair in enumerate(getattr(tokenizer, "merges", [])[:25])
    ]
    st.dataframe(pd.DataFrame(merge_rows), use_container_width=True, hide_index=True)

st.subheader("Encode / Decode")
st.code(f"IDs: {encoded}\nDecoded: {decoded}", language="text")

st.subheader("Token Frequency View")
st.bar_chart(token_frequency_frame(tokenizer), x="token", y="count")
