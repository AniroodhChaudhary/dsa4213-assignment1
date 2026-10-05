"""Extract fixed-length AG News features for every representation listed in
configs/classify.yaml (frozen LMs, raw embedding tables, TF-IDF).

    uv run scripts/extract_features.py [--only name1,name2]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.data import load_agnews_split  # noqa: E402
from lm.features import embedding_features, lm_features, tfidf_features  # noqa: E402
from lm.models import load_checkpoint  # noqa: E402
from lm.utils import dump_json, ensure_dir, get_device, load_config, load_json  # noqa: E402

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    common = load_config(ROOT / "configs" / "common.yaml")
    cfg = load_config(ROOT / "configs" / "classify.yaml")
    device = get_device(args.device or common["device"])
    out_dir = ensure_dir(ROOT / "data" / "features")
    res_dir = ensure_dir(ROOT / common["results_dir"] / "classification")
    data = {s: load_agnews_split(ROOT / common["data_dir"], s) for s in SPLITS}
    only = set(args.only.split(",")) if args.only else None
    info_path = res_dir / "features_info.json"
    info = load_json(info_path) if info_path.exists() else {}

    for spec in cfg["representations"]:
        name = spec["name"]
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        if spec.get("type") == "tfidf":
            mats, n_feat = tfidf_features((data["train"][0], data["train"][1]),
                                          [(data[s][0], data[s][1]) for s in SPLITS[1:]])
            for s, m in zip(SPLITS, mats):
                sp.save_npz(out_dir / f"{name}_{s}.npz", m.tocsr())
            info[name] = {"kind": "tfidf", "dim": n_feat, "sparse": True}
        else:
            if "run" in spec:
                ck = ROOT / common["checkpoint_dir"] / f"{spec['run']}.pt"
                if not ck.exists():
                    print(f"skip {name}: missing checkpoint {ck.name}")
                    continue
                model, _, _ = load_checkpoint(ck, device)
                feats = {s: lm_features(model, data[s][0], data[s][1], device, spec["pooling"])
                         for s in SPLITS}
                info[name] = {"kind": "lm", "run": spec["run"], "pooling": spec["pooling"],
                              "dim": int(feats["train"].shape[1])}
                del model
            else:
                if "embedding_from_run" in spec:
                    ck = ROOT / common["checkpoint_dir"] / f"{spec['embedding_from_run']}.pt"
                    if not ck.exists():
                        print(f"skip {name}: missing checkpoint {ck.name}")
                        continue
                    model, _, _ = load_checkpoint(ck, torch.device("cpu"))
                    matrix = model.embed.weight.detach().numpy()
                    src = f"input embedding of {spec['embedding_from_run']}"
                else:
                    matrix = np.load(ROOT / spec["matrix"])
                    src = spec["matrix"]
                feats = {s: embedding_features(matrix, data[s][0], data[s][1]) for s in SPLITS}
                info[name] = {"kind": "embedding_only", "source": src, "pooling": "mean",
                              "dim": int(matrix.shape[1])}
            for s, f in feats.items():
                np.save(out_dir / f"{name}_{s}.npy", f)
        info[name]["seconds"] = time.perf_counter() - t0
        print(f"{name}: {info[name]}")
    dump_json(info, info_path)


if __name__ == "__main__":
    main()
