import math

import numpy as np

from lm.ngram import KneserNeyLM


def _stream(V=30, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    # Zipf-ish stream with some structure so higher orders matter
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    s = rng.choice(V, size=n, p=p)
    s[1::3] = (s[::3][: len(s[1::3])] + 1) % V  # token after every 3rd copies its predecessor + 1
    return s.astype(np.int32)


def test_distribution_sums_to_one_and_matches_prob():
    s = _stream()
    for order in (1, 2, 3, 4):
        lm = KneserNeyLM(order, 30).fit(s[:5000])
        rng = np.random.default_rng(1)
        for _ in range(20):
            hist = tuple(rng.integers(0, 30, size=rng.integers(0, 5)).tolist())
            dist = lm.next_token_distribution(hist)
            assert abs(dist.sum() - 1.0) < 1e-9
            assert (dist > 0).all()
            w = int(rng.integers(0, 30))
            assert math.isclose(dist[w], lm.prob(hist, w), rel_tol=1e-9)


def test_held_out_perplexity_is_finite_and_better_than_uniform():
    s = _stream()
    lm = KneserNeyLM(3, 30).fit(s[:5000])
    lp = lm.logprob_stream(s[5000:])
    ppl = math.exp(-lp.mean())
    assert np.isfinite(lp).all()
    assert ppl < 30
