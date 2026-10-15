"""Interpolated Kneser-Ney n-gram language model over integer token streams.

For a model of order N:
    P_N(w | h)   uses raw counts of N-grams (h has N-1 tokens),
    P_n(w | h')  for n < N uses continuation counts N1+(. h' w),
    P_0(w) = 1 / V.
    P_n(w | h) = [max(c_n(h w) - D_n, 0) + D_n * N1+(h .) * P_{n-1}(w | h[1:])] / c_n(h .)
with one absolute discount per order, D_n = n1 / (n1 + 2 n2) estimated from the
counts-of-counts of that order's table. Unseen contexts back off fully.
`<unk>` is an ordinary vocabulary token, exactly as for the neural models.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


class KneserNeyLM:
    def __init__(self, order: int, vocab_size: int):
        assert order >= 1
        self.order = order
        self.V = vocab_size
        # per level n (1..order): table[n], ctx_total[n], followers[n], discount[n]
        self.table: dict[int, dict] = {}
        self.ctx_total: dict[int, dict] = {}
        self.followers: dict[int, dict] = {}
        self.discount: dict[int, float] = {}
        self.n_train_tokens = 0

    # ------------------------------------------------------------------ fit
    def fit(self, stream: np.ndarray) -> "KneserNeyLM":
        toks = stream.tolist()
        self.n_train_tokens = len(toks)
        N = self.order
        # raw counts are only needed at the top order; every lower order uses
        # continuation counts N1+(. w_1..w_n), i.e. the number of distinct left
        # extensions, which is derived from the *types* of the level above.
        self.table[N] = Counter(zip(*[toks[k:] for k in range(N)]))
        for n in range(N - 1, 0, -1):
            cont: Counter = Counter()
            for key in self.table[n + 1]:
                cont[key[1:]] += 1
            self.table[n] = cont
        for n in range(1, N + 1):
            tot: Counter = Counter()
            fol: Counter = Counter()
            for key, c in self.table[n].items():
                ctx = key[:-1]
                tot[ctx] += c
                fol[ctx] += 1
            self.ctx_total[n] = dict(tot)
            self.followers[n] = dict(fol)
            cc = Counter(self.table[n].values())
            n1, n2 = cc.get(1, 0), cc.get(2, 0)
            self.discount[n] = n1 / (n1 + 2.0 * n2) if (n1 + 2 * n2) > 0 else 0.5
        return self

    # ------------------------------------------------------------- scoring
    def prob(self, hist: tuple, w: int) -> float:
        """P(w | hist) with hist a tuple of up to order-1 preceding tokens."""
        p = 1.0 / self.V
        h = hist[-(self.order - 1):] if self.order > 1 else ()
        # start at the highest level for which we have enough history
        start = min(len(h) + 1, self.order)
        for n in range(1, start + 1):
            ctx = h[len(h) - (n - 1):] if n > 1 else ()
            tot = self.ctx_total[n].get(ctx)
            if not tot:
                continue  # unseen context: keep the lower-order estimate
            D = self.discount[n]
            c = self.table[n].get(ctx + (w,), 0)
            p = (max(c - D, 0.0) + D * self.followers[n][ctx] * p) / tot
        return p

    def logprob_stream(self, stream: np.ndarray) -> np.ndarray:
        """Per-target natural log-probabilities for targets stream[1:]."""
        toks = stream.tolist()
        n_ctx = self.order - 1
        out = np.empty(len(toks) - 1, dtype=np.float32)
        for i in range(1, len(toks)):
            hist = tuple(toks[max(0, i - n_ctx):i])
            out[i - 1] = math.log(self.prob(hist, toks[i]))
        return out

    def logprob_stream_blocks(self, stream: np.ndarray, block: int) -> np.ndarray:
        """Per-target log-probabilities with the history reset every `block`
        tokens, matching the neural models' context-reset protocol."""
        toks = stream.tolist()
        n_ctx = self.order - 1
        out = np.empty(len(toks) - 1, dtype=np.float32)
        for i in range(1, len(toks)):
            start = ((i - 1) // block) * block
            hist = tuple(toks[max(start, i - n_ctx):i])
            out[i - 1] = math.log(self.prob(hist, toks[i]))
        return out

    def next_token_distribution(self, hist) -> np.ndarray:
        """Full P(. | hist) as a length-V vector (used for generation)."""
        h = tuple(int(t) for t in hist)
        h = h[-(self.order - 1):] if self.order > 1 else ()
        p = np.full(self.V, 1.0 / self.V, dtype=np.float64)
        start = min(len(h) + 1, self.order)
        for n in range(1, start + 1):
            ctx = h[len(h) - (n - 1):] if n > 1 else ()
            tot = self.ctx_total[n].get(ctx)
            if not tot:
                continue
            D = self.discount[n]
            table = self.table[n]
            new = p * (D * self.followers[n][ctx] / tot)
            if n == 1:
                for key, c in table.items():
                    new[key[0]] += max(c - D, 0.0) / tot
            else:
                # only words that follow ctx have a discounted-count term
                for w in range(self.V):
                    c = table.get(ctx + (w,))
                    if c:
                        new[w] += max(c - D, 0.0) / tot
            p = new
        return p / p.sum()

    # --------------------------------------------------------------- size
    def n_entries(self) -> int:
        return sum(len(t) for t in self.table.values())

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "KneserNeyLM":
        with open(path, "rb") as f:
            return pickle.load(f)
