import numpy as np
import torch

from lm.features import embedding_features, lm_features
from lm.models import build_model

BASE = {"emb_dim": 16, "dropout": 0.0, "bptt": 8}
V, PAD = 40, 39


def _docs():
    ids = np.full((3, 12), PAD, dtype=np.int32)
    lens = np.array([12, 5, 1], dtype=np.int32)
    rng = np.random.default_rng(0)
    for i, L in enumerate(lens):
        ids[i, :L] = rng.integers(0, PAD, size=L)
    return ids, lens


def test_pooling_ignores_padding_and_is_length_invariant():
    torch.manual_seed(0)
    ids, lens = _docs()
    for spec in ({"type": "lstm", "hidden": 16, "n_layers": 1},
                 {"type": "transformer", "n_layers": 1, "n_heads": 2, "ff_dim": 32, "max_len": 16}):
        model = build_model({**BASE, "model": spec}, V).eval()
        mean = lm_features(model, ids, lens, torch.device("cpu"), "mean", batch=2)
        last = lm_features(model, ids, lens, torch.device("cpu"), "last", batch=2)
        with torch.no_grad():
            for i, L in enumerate(lens):
                h = model.hidden_states(torch.as_tensor(ids[i:i + 1, :L], dtype=torch.long))[0][0]
                assert np.allclose(mean[i], h.mean(0).numpy(), atol=1e-5)
                assert np.allclose(last[i], h[-1].numpy(), atol=1e-5)
        # more padding must not change the features
        ids_wide = np.concatenate([ids, np.full((3, 4), PAD, np.int32)], axis=1)
        assert np.allclose(mean, lm_features(model, ids_wide, lens, torch.device("cpu"), "mean"), atol=1e-5)


def test_embedding_mean_excludes_padding():
    ids, lens = _docs()
    M = np.random.default_rng(1).normal(size=(V, 8)).astype(np.float32)
    f = embedding_features(M, ids, lens)
    assert np.allclose(f[1], M[ids[1, :5]].mean(0))
    assert np.allclose(f[2], M[ids[2, 0]])
