import numpy as np
import torch

from lm.evaluate import score_full, score_windows
from lm.generate import generate_neural, repetition_metrics
from lm.models import build_model

BASE = {"emb_dim": 16, "dropout": 0.0, "bptt": 8}
V = 40


def _stream(n=203, seed=0):
    return np.random.default_rng(seed).integers(0, V, size=n).astype(np.int32)


def test_full_equals_single_window_for_recurrent():
    torch.manual_seed(0)
    model = build_model({**BASE, "model": {"type": "lstm", "hidden": 16, "n_layers": 1}}, V)
    s = _stream(50)
    a = score_full(model, s, torch.device("cpu"), chunk=7)
    b = score_windows(model, s, torch.device("cpu"), context=49, stride=49)
    assert a.shape == b.shape == (49,)
    assert np.allclose(a, b, atol=1e-5)


def test_sliding_window_scores_every_target_with_enough_context():
    torch.manual_seed(0)
    model = build_model({**BASE, "model": {"type": "transformer", "n_layers": 1, "n_heads": 2,
                                           "ff_dim": 32, "max_len": 16}}, V)
    s = _stream(203)
    lp = score_windows(model, s, torch.device("cpu"), context=16, stride=4, batch=5)
    assert lp.shape == (202,) and np.isfinite(lp).all()
    # target t (>= 15) must equal the last-position logprob of the window ending at t
    model.eval()
    x = torch.as_tensor(s, dtype=torch.long)
    n_targets = len(s) - 1
    for t in (3, 15, 40, 41, 100, 199, 201):
        # window whose scored block (its last `stride` positions) contains target t
        start = 0 if t < 12 else 4 * ((t - 12) // 4)
        L = min(16, n_targets - start)
        inp = x[start : start + L].unsqueeze(0)
        with torch.no_grad():
            logits = model(inp)[0][0]
        ref = torch.log_softmax(logits, -1)[t - start, s[t + 1]].item()
        assert abs(lp[t] - ref) < 1e-5, t


def test_generation_lengths_and_determinism():
    torch.manual_seed(0)
    for spec in ({"type": "rnn", "hidden": 16, "n_layers": 1},
                 {"type": "transformer", "n_layers": 1, "n_heads": 2, "ff_dim": 32, "max_len": 16}):
        model = build_model({**BASE, "model": spec}, V)
        prompt = [1, 2, 3]
        g1 = generate_neural(model, prompt, 10, torch.device("cpu"), np.random.default_rng(5), 5, 1.0, False, context=16)
        g2 = generate_neural(model, prompt, 10, torch.device("cpu"), np.random.default_rng(5), 5, 1.0, False, context=16)
        gr = generate_neural(model, prompt, 10, torch.device("cpu"), np.random.default_rng(9), 5, 1.0, True, context=16)
        assert len(g1) == 13 and g1[:3] == prompt and g1 == g2 and len(gr) == 13
    m = repetition_metrics(list("abcabcabcabc"))
    assert m["distinct_1"] == 3 / 12 and 0 < m["repeated_4gram_frac"] <= 1
