"""Train and evaluate the same logistic-regression classifier on every
representation, over a grid of labelled-set sizes, with C chosen on the
validation split and the official AG News test split used once per setting.

    uv run scripts/train_classifier.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.data import AG_LABELS, Vocab, load_agnews, load_agnews_split  # noqa: E402
from lm.utils import dump_json, ensure_dir, load_config, load_json  # noqa: E402

SPLITS = ("train", "val", "test")


def load_feats(feat_dir: Path, name: str, sparse: bool):
    if sparse:
        return {s: sp.load_npz(feat_dir / f"{name}_{s}.npz") for s in SPLITS}
    return {s: np.load(feat_dir / f"{name}_{s}.npy") for s in SPLITS}


def fit_eval(Xtr, ytr, Xva, yva, C, max_iter, seed):
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    return clf, f1_score(yva, pred, average="macro")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    common = load_config(ROOT / "configs" / "common.yaml")
    cfg = load_config(ROOT / "configs" / "classify.yaml")
    seed = common["seed"]
    feat_dir = ROOT / "data" / "features"
    res_dir = ensure_dir(ROOT / common["results_dir"] / "classification")
    info = load_json(res_dir / "features_info.json")
    vocab = Vocab.load(ROOT / common["data_dir"] / "wikitext" / "vocab.json")
    y = {s: load_agnews_split(ROOT / common["data_dir"], s)[2] for s in SPLITS}
    ids_te, lens_te, _ = load_agnews_split(ROOT / common["data_dir"], "test")
    unk_rate_te = np.array([(ids_te[i, 1:lens_te[i]] == vocab.unk_id).mean() for i in range(len(ids_te))])
    test_texts, _ = load_agnews("test")
    only = set(args.only.split(",")) if args.only else None

    summary_rows = []
    for name, meta in info.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        sparse = meta.get("sparse", False)
        X = load_feats(feat_dir, name, sparse)
        if not sparse:
            scaler = StandardScaler().fit(X["train"])
            X = {s: scaler.transform(v) for s, v in X.items()}
        out = {"name": name, "meta": meta, "by_n_train": {}}
        for n in cfg["n_train_grid"]:
            Xtr, ytr = X["train"][:n], y["train"][:n]
            best = None
            for C in cfg["C_grid"]:
                clf, f1v = fit_eval(Xtr, ytr, X["val"], y["val"], C, cfg["max_iter"], seed)
                if best is None or f1v > best[1]:
                    best = (C, f1v, clf)
            C, f1v, clf = best
            pred = clf.predict(X["test"])
            rec = {
                "n_train": n, "C": C, "val_macro_f1": f1v,
                "test_accuracy": accuracy_score(y["test"], pred),
                "test_macro_f1": f1_score(y["test"], pred, average="macro"),
                "test_f1_per_class": dict(zip(AG_LABELS, f1_score(y["test"], pred, average=None).tolist())),
                "confusion": confusion_matrix(y["test"], pred).tolist(),
                "class_counts_train": np.bincount(ytr, minlength=4).tolist(),
            }
            out["by_n_train"][str(n)] = rec
            summary_rows.append({"representation": name, "n_train": n, "C": C,
                                 "val_macro_f1": round(f1v, 4),
                                 "test_accuracy": round(rec["test_accuracy"], 4),
                                 "test_macro_f1": round(rec["test_macro_f1"], 4)})
            if n == max(cfg["n_train_grid"]):
                proba = clf.predict_proba(X["test"])
                wrong = np.flatnonzero(pred != y["test"])
                conf = proba[wrong, pred[wrong]]
                top = wrong[np.argsort(-conf)[:20]]
                errors = [{
                    "index": int(i), "true": AG_LABELS[y["test"][i]], "pred": AG_LABELS[pred[i]],
                    "confidence": float(proba[i, pred[i]]), "unk_rate": float(unk_rate_te[i]),
                    "text": test_texts[i][:300],
                } for i in top]
                out["errors_top20"] = errors
                out["unk_rate_correct"] = float(unk_rate_te[pred == y["test"]].mean())
                out["unk_rate_wrong"] = float(unk_rate_te[pred != y["test"]].mean())
                np.save(res_dir / f"{name}_test_pred.npy", pred)
        out["seconds"] = time.perf_counter() - t0
        dump_json(out, res_dir / f"{name}.json")
        full = out["by_n_train"][str(max(cfg["n_train_grid"]))]
        print(f"{name:28s} acc {full['test_accuracy']:.4f}  macro-F1 {full['test_macro_f1']:.4f}  "
              f"(C={full['C']}, {out['seconds']:.0f}s)")
    summary_path = res_dir / "summary.csv"
    if only and summary_path.exists():  # keep the rows of representations not re-run
        with open(summary_path) as f:
            old_rows = [r for r in csv.DictReader(f) if r["representation"] not in only]
        summary_rows = old_rows + summary_rows
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)


if __name__ == "__main__":
    main()
