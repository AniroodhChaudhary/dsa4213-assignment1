"""Fit interpolated Kneser-Ney models of several orders on the WikiText-2 train
stream, select the order on validation perplexity and keep that model.

    uv run scripts/train_ngram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.data import Vocab, load_stream  # noqa: E402
from lm.evaluate import summarize  # noqa: E402
from lm.ngram import KneserNeyLM  # noqa: E402
from lm.timing import Timer, file_bytes, hardware_info  # noqa: E402
from lm.utils import dump_json, ensure_dir, load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "ngram.yaml")
    vocab = Vocab.load(ROOT / cfg["data_dir"] / "wikitext" / "vocab.json")
    train = load_stream(ROOT / cfg["data_dir"], "train")
    val = load_stream(ROOT / cfg["data_dir"], "validation")
    run_dir = ensure_dir(ROOT / cfg["results_dir"] / "runs" / "ngram")
    ckpt_dir = ensure_dir(ROOT / cfg["checkpoint_dir"])

    per_order = {}
    best_order, best_ppl, best_model = None, float("inf"), None
    for order in cfg["orders"]:
        with Timer() as t_fit:
            lm = KneserNeyLM(order, len(vocab)).fit(train)
        with Timer() as t_val:
            val_lp = lm.logprob_stream(val)
        s = summarize(val_lp, val, vocab)
        per_order[order] = {
            "fit_seconds": t_fit.elapsed, "val_score_seconds": t_val.elapsed,
            "val_ppl": s["perplexity"], "val_ce": s["cross_entropy"],
            "n_entries": lm.n_entries(),
            "entries_per_level": {n: len(t) for n, t in lm.table.items()},
            "discounts": lm.discount,
        }
        print(f"order {order}: val ppl {s['perplexity']:.2f}  fit {t_fit.elapsed:.1f}s  "
              f"entries {lm.n_entries():,}  D={ {n: round(d, 3) for n, d in lm.discount.items()} }")
        if s["perplexity"] < best_ppl:
            best_order, best_ppl, best_model = order, s["perplexity"], lm
        else:
            del lm
    path = ckpt_dir / "ngram.pkl"
    best_model.save(path)
    metrics = {
        "run_name": "ngram", "model": {"type": "kneser_ney", "order": best_order},
        "selected_order": best_order, "best_val_ppl": best_ppl,
        "per_order": per_order,
        "train_seconds_total": per_order[best_order]["fit_seconds"],
        "n_entries": best_model.n_entries(), "checkpoint_bytes": file_bytes(path),
        "n_train_tokens": best_model.n_train_tokens, "hardware": hardware_info(),
    }
    dump_json(metrics, run_dir / "metrics.json")
    print(f"selected order {best_order} (val ppl {best_ppl:.2f}); saved {path} "
          f"({file_bytes(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
