"""
Data preparation for next-word prediction (Task 14).

Corpus: the `title` column of medium_data.csv (~6,500 Medium article
titles). Each title is cleaned, tokenized, and turned into a set of
"n-gram" training examples in the classic language-modelling style:

    "the fall of graph neural networks" (tokenized)
    -> [the]                     -> fall
    -> [the, fall]                -> of
    -> [the, fall, of]            -> graph
    -> [the, fall, of, graph]     -> neural
    -> ...

i.e. every prefix of a title is one training example, and the label is
the word that comes right after that prefix. This is the standard setup
used for RNN/LSTM next-word-prediction demos, and it mirrors the N-gram
idea from the blog-post part of this task: an N-gram model estimates
P(next word | last N-1 words) from counts, while our LSTM learns the
same conditional distribution with a neural network instead of counting.

Run:
    python preprocess.py
Produces (in ./data/):
    vocab.json              word -> id mapping (+ id -> word)
    train_sequences.npz     padded X_train, y_train
    val_sequences.npz       padded X_val,   y_val
    stats.json              basic corpus/vocab statistics
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
MIN_WORD_FREQ = 2      # words appearing fewer times than this become <UNK>
MAX_SEQ_LEN = 20        # covers ~99.5% of titles (max title length is 23 words)
VAL_FRACTION = 0.2

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    """Lowercase, drop everything except letters/digits/apostrophes/spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return clean_text(text).split()


def build_vocab(token_lists: list, min_freq: int = MIN_WORD_FREQ) -> dict:
    from collections import Counter
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    words = [PAD_TOKEN, UNK_TOKEN] + sorted(
        [w for w, c in counter.items() if c >= min_freq]
    )
    word2id = {w: i for i, w in enumerate(words)}
    return word2id


def encode(tokens: list, word2id: dict) -> list:
    unk = word2id[UNK_TOKEN]
    return [word2id.get(t, unk) for t in tokens]


def make_ngram_examples(encoded_titles: list, max_len: int = MAX_SEQ_LEN):
    """
    For every title, emit one (prefix, next_word) example per position
    (skipping the trivial 1-token prefix -> nothing case). Prefixes are
    pre-padded with <PAD> (id 0) up to `max_len - 1` tokens, Keras-style,
    so every X row has the same fixed length.
    """
    X, y = [], []
    width = max_len - 1
    for ids in encoded_titles:
        ids = ids[:max_len]  # truncate very long titles
        for i in range(1, len(ids)):
            prefix = ids[:i][-width:]           # keep the most recent `width` tokens
            pad_len = width - len(prefix)
            padded = [0] * pad_len + prefix       # pre-padding
            X.append(padded)
            y.append(ids[i])
    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


def main():
    df = pd.read_csv("medium_data.csv")
    titles = df["title"].dropna().tolist()
    print(f"Loaded {len(titles)} titles")

    tokenized = [tokenize(t) for t in titles]
    tokenized = [t for t in tokenized if len(t) >= 2]  # need at least 2 tokens for one example
    print(f"{len(tokenized)} titles have >= 2 tokens")

    # Split at the TITLE level first (not the n-gram-example level) so that
    # fragments of the same title never leak across train/val.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(tokenized))
    n_val = int(len(tokenized) * VAL_FRACTION)
    val_idx, train_idx = set(idx[:n_val].tolist()), set(idx[n_val:].tolist())
    train_titles = [tokenized[i] for i in sorted(train_idx)]
    val_titles = [tokenized[i] for i in sorted(val_idx)]
    print(f"Train titles: {len(train_titles)}, Val titles: {len(val_titles)}")

    # Build vocab from TRAINING titles only, to keep the val set a fair
    # measure of generalization (unseen val-only words become <UNK>).
    word2id = build_vocab(train_titles)
    id2word = {i: w for w, i in word2id.items()}
    print(f"Vocabulary size: {len(word2id)} (min_freq={MIN_WORD_FREQ})")

    train_encoded = [encode(t, word2id) for t in train_titles]
    val_encoded = [encode(t, word2id) for t in val_titles]

    X_train, y_train = make_ngram_examples(train_encoded)
    X_val, y_val = make_ngram_examples(val_encoded)
    print(f"Train examples: {X_train.shape}, Val examples: {X_val.shape}")

    np.savez(DATA_DIR / "train_sequences.npz", X=X_train, y=y_train)
    np.savez(DATA_DIR / "val_sequences.npz", X=X_val, y=y_val)
    with open(DATA_DIR / "vocab.json", "w") as f:
        json.dump({"word2id": word2id, "id2word": id2word,
                    "max_len": MAX_SEQ_LEN}, f, indent=2)
    with open(DATA_DIR / "stats.json", "w") as f:
        json.dump({
            "n_titles_total": len(tokenized),
            "n_titles_train": len(train_titles),
            "n_titles_val": len(val_titles),
            "vocab_size": len(word2id),
            "n_train_examples": int(X_train.shape[0]),
            "n_val_examples": int(X_val.shape[0]),
            "max_seq_len": MAX_SEQ_LEN,
            "min_word_freq": MIN_WORD_FREQ,
        }, f, indent=2)

    print("Saved vocab.json, train_sequences.npz, val_sequences.npz, stats.json -> ./data/")


if __name__ == "__main__":
    main()
