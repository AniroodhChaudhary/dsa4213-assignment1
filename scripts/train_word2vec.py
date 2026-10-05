"""Train skip-gram embeddings on the WikiText-2 train stream (Part II setting b).
Its cost is recorded separately from language-model training.

    PYTHONHASHSEED=0 uv run scripts/train_word2vec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gensim  # noqa: E402

from lm.data import Vocab, load_stream  # noqa: E402
from lm.embeddings import PROBE_WORDS, build_matrix, nearest_neighbours, train_skipgram  # noqa: E402
from lm.timing import hardware_info  # noqa: E402
from lm.utils import dump_json, ensure_dir, load_config, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "common.yaml")
    set_seed(cfg["seed"])
    vocab = Vocab.load(ROOT / cfg["data_dir"] / "wikitext" / "vocab.json")
    train = load_stream(ROOT / cfg["data_dir"], "train")
    out_dir = ensure_dir(ROOT / "data" / "embeddings")
    res_dir = ensure_dir(ROOT / cfg["results_dir"] / "embeddings")

    kv, seconds, settings = train_skipgram(vocab, train, cfg["emb_dim"], cfg["seed"])
    M, stats = build_matrix(vocab, lambda w: kv[w] if w in kv.key_to_index else None,
                            cfg["emb_dim"], cfg["seed"])
    np.save(out_dir / "skipgram300.npy", M)
    kv.save(str(out_dir / "skipgram300.kv"))
    stats.update({
        "source": "skip-gram (gensim Word2Vec, sg=1) on WikiText-2 train only",
        "gensim_version": gensim.__version__, "settings": settings,
        "train_seconds": seconds, "hardware": hardware_info(),
        "nearest_neighbours": nearest_neighbours(M, vocab, PROBE_WORDS),
    })
    dump_json(stats, res_dir / "skipgram_coverage.json")
    print(f"skip-gram: {seconds:.0f}s, coverage {stats['coverage_types']:.4f} types, "
          f"mean row norm {stats['mean_row_norm_all']:.3f}; missing {stats['missing_types']}")


if __name__ == "__main__":
    main()
