"""Align public GloVe-6B 300d vectors to the WikiText-2 vocabulary (Part II
setting c) and record coverage plus how missing words are initialised.

    uv run scripts/build_embedding_matrix.py --glove data/raw/glove.6B.300d.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.data import Vocab  # noqa: E402
from lm.embeddings import PROBE_WORDS, build_matrix, load_glove, nearest_neighbours  # noqa: E402
from lm.utils import dump_json, ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default="data/raw/glove.6B.300d.txt")
    ap.add_argument("--rescale_mean_norm", type=float, default=None,
                    help="also write a copy whose mean row norm equals this value (row-norm control)")
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "common.yaml")
    vocab = Vocab.load(ROOT / cfg["data_dir"] / "wikitext" / "vocab.json")
    out_dir = ensure_dir(ROOT / "data" / "embeddings")
    res_dir = ensure_dir(ROOT / cfg["results_dir"] / "embeddings")

    t0 = time.perf_counter()
    vectors = load_glove(ROOT / args.glove, set(vocab.itos), cfg["emb_dim"])
    M, stats = build_matrix(vocab, vectors.get, cfg["emb_dim"], cfg["seed"])
    np.save(out_dir / "glove300.npy", M)
    stats.update({
        "source": "GloVe 6B 300d (Wikipedia 2014 + Gigaword 5, uncased), Pennington et al. 2014",
        "file": args.glove, "load_seconds": time.perf_counter() - t0,
        "missing_init": "N(0, sigma_d) per dimension, sigma_d = std of found rows, seed %d" % cfg["seed"],
        "nearest_neighbours": nearest_neighbours(M, vocab, PROBE_WORDS),
    })
    dump_json(stats, res_dir / "glove_coverage.json")
    if args.rescale_mean_norm:
        factor = args.rescale_mean_norm / float(np.linalg.norm(M, axis=1).mean())
        Ms = (M * factor).astype(np.float32)
        np.save(out_dir / "glove300_scaled.npy", Ms)
        dump_json({**stats, "rescale_factor": factor, "mean_row_norm_all": float(np.linalg.norm(Ms, axis=1).mean()),
                   "source": stats["source"] + f"; rows rescaled by {factor:.4f} so the mean row norm is {args.rescale_mean_norm}"},
                  res_dir / "glove_scaled_coverage.json")
        print(f"rescaled copy: factor {factor:.4f}, mean row norm {np.linalg.norm(Ms, axis=1).mean():.3f}")
    print(f"GloVe: coverage {stats['coverage_types']:.4f} of types, "
          f"{stats['coverage_train_tokens']:.4f} of train tokens; missing {stats['missing_types']} "
          f"e.g. {stats['missing_examples_most_frequent'][:8]}; mean row norm {stats['mean_row_norm_all']:.3f}")


if __name__ == "__main__":
    main()
