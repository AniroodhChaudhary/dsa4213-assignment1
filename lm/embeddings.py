"""Embedding tables aligned to the WikiText-2 vocabulary: skip-gram trained on
the train split, public GloVe-6B vectors, coverage statistics and neighbours.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import EOS, PAD, Vocab


def train_skipgram(vocab: Vocab, train_stream: np.ndarray, dim: int, seed: int,
                   window=5, negative=10, sample=1e-4, epochs=10, workers=1, min_count=1):
    """Skip-gram with negative sampling (gensim) on the train stream, one
    sentence per WikiText line (lines end with <eos>, which therefore gets a
    vector too). Returns (KeyedVectors, seconds, settings)."""
    from gensim.models import Word2Vec

    itos = vocab.itos
    sentences, cur = [], []
    for t in train_stream[1:].tolist():  # skip the leading <eos>
        cur.append(itos[t])
        if t == vocab.eos_id:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    settings = dict(vector_size=dim, sg=1, window=window, negative=negative, sample=sample,
                    epochs=epochs, workers=workers, min_count=min_count, seed=seed)
    t0 = time.perf_counter()
    model = Word2Vec(sentences, **settings)
    seconds = time.perf_counter() - t0
    settings["n_sentences"] = len(sentences)
    settings["n_tokens"] = int(len(train_stream) - 1)
    return model.wv, seconds, settings


def load_glove(path: str | Path, wanted: set[str], dim: int) -> dict[str, np.ndarray]:
    """Read only the rows of a GloVe text file whose word is in `wanted`."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            w, rest = line.split(" ", 1)
            if w in wanted:
                v = np.asarray(rest.split(), dtype=np.float32)
                assert len(v) == dim, (w, len(v))
                vectors[w] = v
    return vectors


def build_matrix(vocab: Vocab, get_vector, dim: int, seed: int) -> tuple[np.ndarray, dict]:
    """Assemble a (V, dim) float32 table. Missing words get N(0, sigma_d) rows
    where sigma_d is the per-dimension std of the found rows."""
    rng = np.random.default_rng(seed)
    found = np.zeros(len(vocab), dtype=bool)
    M = np.zeros((len(vocab), dim), dtype=np.float32)
    for i, w in enumerate(vocab.itos):
        v = get_vector(w)
        if v is not None:
            M[i] = v
            found[i] = True
    sigma = M[found].std(axis=0)
    M[~found] = rng.normal(0.0, 1.0, size=((~found).sum(), dim)).astype(np.float32) * sigma
    counts = np.asarray(vocab.counts)
    train_tokens = counts.sum()
    missing = [vocab.itos[i] for i in np.flatnonzero(~found)]
    missing_sorted = sorted(missing, key=lambda w: -vocab.counts[vocab.stoi[w]])
    norms = np.linalg.norm(M, axis=1)
    stats = {
        "dim": dim,
        "vocab_size": len(vocab),
        "found_types": int(found.sum()),
        "coverage_types": float(found.mean()),
        "coverage_train_tokens": float(counts[found].sum() / train_tokens),
        "missing_types": int((~found).sum()),
        "missing_examples_most_frequent": missing_sorted[:25],
        "missing_specials": [w for w in (EOS, PAD, "<unk>") if w in missing],
        "init_sigma_mean": float(sigma.mean()),
        "mean_row_norm_found": float(norms[found].mean()),
        "mean_row_norm_missing": float(norms[~found].mean()) if (~found).any() else None,
        "mean_row_norm_all": float(norms.mean()),
    }
    return M, stats


def nearest_neighbours(M: np.ndarray, vocab: Vocab, words: list[str], k: int = 5) -> dict:
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    out = {}
    for w in words:
        if w not in vocab.stoi:
            continue
        i = vocab.stoi[w]
        sims = Mn @ Mn[i]
        sims[i] = -np.inf
        sims[vocab.pad_id] = -np.inf
        top = np.argsort(-sims)[:k]
        out[w] = [(vocab.itos[j], round(float(sims[j]), 3)) for j in top]
    return out


PROBE_WORDS = ["king", "city", "war", "album", "1990", "north", "played", "@-@", "<eos>", "<unk>"]
