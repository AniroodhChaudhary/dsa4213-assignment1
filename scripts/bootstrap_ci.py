"""Bootstrap confidence intervals for the Part III test metrics (no retraining).

For every representation's saved test predictions (full 20k-label fit) resample the
7,600 test documents with replacement B times and report 95% percentile intervals for
accuracy and macro-F1, plus paired intervals for selected representation differences.

    uv run scripts/bootstrap_ci.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.data import load_agnews_split  # noqa: E402
from lm.utils import dump_json, load_config, load_json  # noqa: E402

PAIRS = [("transformer1l_rand_mean", "transformer_rand_mean"), ("transformer_rand_mean", "lstm_rand_mean"),
         ("transformer_rand_mean", "rnn_rand_mean"), ("rnn_rand_mean", "lstm_rand_mean"),
         ("emb_glove_mean", "lstm_glove_frozen_mean"), ("tfidf", "emb_glove_mean"),
         ("lstm_rand_mean", "lstm_rand_last"), ("transformer_rand_mean", "transformer_rand_last")]


def main(B: int = 1000, seed: int = 0) -> None:
    common = load_config(ROOT / "configs" / "common.yaml")
    res_dir = ROOT / common["results_dir"] / "classification"
    y = load_agnews_split(ROOT / common["data_dir"], "test")[2]
    info = load_json(res_dir / "features_info.json")
    preds = {n: np.load(res_dir / f"{n}_test_pred.npy") for n in info if (res_dir / f"{n}_test_pred.npy").exists()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(B, len(y)))
    out = {"B": B, "seed": seed, "n_test": int(len(y)), "per_representation": {}, "paired": {}}
    f1s = {}
    for n, p in preds.items():
        acc = np.array([(p[i] == y[i]).mean() for i in idx])
        f1 = np.array([f1_score(y[i], p[i], average="macro") for i in idx])
        f1s[n] = f1
        out["per_representation"][n] = {
            "accuracy": float((p == y).mean()), "accuracy_ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "macro_f1": float(f1_score(y, p, average="macro")), "macro_f1_ci": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
            "macro_f1_halfwidth": float((np.percentile(f1, 97.5) - np.percentile(f1, 2.5)) / 2),
        }
    for a, b in PAIRS:
        if a in f1s and b in f1s:
            d = f1s[a] - f1s[b]
            out["paired"][f"{a}__minus__{b}"] = {
                "point": float(f1_score(y, preds[a], average="macro") - f1_score(y, preds[b], average="macro")),
                "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "separable": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
            }
    dump_json(out, res_dir / "bootstrap.json")
    for n, r in out["per_representation"].items():
        print(f"{n:34s} macro-F1 {100*r['macro_f1']:.1f} +- {100*r['macro_f1_halfwidth']:.1f}")
    for k, r in out["paired"].items():
        print(f"{k:60s} {100*r['point']:+.1f} [{100*r['ci'][0]:+.1f}, {100*r['ci'][1]:+.1f}] {'separable' if r['separable'] else 'not separable'}")


if __name__ == "__main__":
    main()
