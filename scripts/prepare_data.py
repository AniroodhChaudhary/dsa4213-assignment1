"""Download WikiText-2 and AG News, build the vocabulary, write token streams,
padded AG News arrays, coverage statistics and the fixed generation prompts.

Everything derived from text uses the WikiText-2 *train* split only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.data import (  # noqa: E402
    AG_LABELS,
    BUCKET_NAMES,
    Vocab,
    encode_agnews,
    encode_stream,
    load_agnews,
    load_wikitext_lines,
)
from lm.tokenize import WikiTextTokenizer  # noqa: E402
from lm.utils import dump_json, ensure_dir, load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "common.yaml")
    out_wt = ensure_dir(ROOT / cfg["data_dir"] / "wikitext")
    out_ag = ensure_dir(ROOT / cfg["data_dir"] / "agnews")
    res = ensure_dir(ROOT / cfg["results_dir"] / "data")
    rng = np.random.default_rng(cfg["seed"])

    # ---------------- WikiText-2 ----------------
    t0 = time.perf_counter()
    lines = {s: load_wikitext_lines(s) for s in ("train", "validation", "test")}
    vocab = Vocab.build(lines["train"])
    vocab.save(out_wt / "vocab.json")
    coverage = {"wikitext": {"vocab_size": len(vocab), "splits": {}}}
    for split, ls in lines.items():
        stream, n_oov = encode_stream(ls, vocab)
        np.save(out_wt / f"{split}.npy", stream)
        n_unk_native = sum(t == "<unk>" for toks in ls for t in toks)
        coverage["wikitext"]["splits"][split] = {
            "lines": len(ls),
            "tokens_in_stream": int(len(stream)),
            "targets": int(len(stream) - 1),
            "native_unk_tokens": int(n_unk_native),
            "oov_mapped_to_unk": int(n_oov),
        }
    bucket_hist = {name: int((vocab.buckets == i).sum()) for i, name in enumerate(BUCKET_NAMES)}
    coverage["wikitext"]["types_per_frequency_bucket"] = bucket_hist
    coverage["wikitext"]["seconds"] = round(time.perf_counter() - t0, 1)
    print("WikiText-2:", coverage["wikitext"])

    # fixed generation prompts: 5 validation lines, >= 30 tokens, not headings
    cand = [i for i, toks in enumerate(lines["validation"])
            if len(toks) >= 30 and toks[0] != "=" and "<unk>" not in toks[:10]]
    chosen = sorted(rng.choice(cand, size=5, replace=False).tolist())
    prompts = [{"validation_line": i, "tokens": lines["validation"][i][:10]} for i in chosen]
    dump_json({"source": "wikitext-2 validation split, seed %d" % cfg["seed"], "prompts": prompts},
              res / "prompts.json")
    print("prompts:", [" ".join(p["tokens"]) for p in prompts])

    # ---------------- AG News ----------------
    t0 = time.perf_counter()
    tok = WikiTextTokenizer()
    max_len = cfg["agnews"]["max_len"]
    texts_tr, y_tr = load_agnews("train")
    texts_te, y_te = load_agnews("test")

    # stratified, disjoint train / val subsets of the official train split
    n_train, n_val = cfg["agnews"]["n_train"], cfg["agnews"]["n_val"]
    per_class_tr, per_class_val = n_train // 4, n_val // 4
    tr_idx, val_idx = [], []
    for c in range(4):
        idx = np.flatnonzero(y_tr == c)
        idx = rng.permutation(idx)
        tr_idx.extend(idx[:per_class_tr].tolist())
        val_idx.extend(idx[per_class_tr : per_class_tr + per_class_val].tolist())
    tr_idx = rng.permutation(np.array(tr_idx))  # shuffle so nested prefixes are stratified-ish
    val_idx = np.array(sorted(val_idx))
    np.save(out_ag / "train_source_indices.npy", tr_idx)
    np.save(out_ag / "val_source_indices.npy", val_idx)

    coverage["agnews"] = {"max_len": max_len, "splits": {}}
    for split, (texts, y) in {
        "train": ([texts_tr[i] for i in tr_idx], y_tr[tr_idx]),
        "val": ([texts_tr[i] for i in val_idx], y_tr[val_idx]),
        "test": (texts_te, y_te),
    }.items():
        ids, lens, stats, unk_doc = encode_agnews(texts, vocab, tok, max_len)
        np.save(out_ag / f"{split}_ids.npy", ids)
        np.save(out_ag / f"{split}_lens.npy", lens)
        np.save(out_ag / f"{split}_labels.npy", y)
        stats["unk_rate_per_class"] = {
            AG_LABELS[c]: float(unk_doc[y == c].mean()) for c in range(4)
        }
        stats["class_counts"] = {AG_LABELS[c]: int((y == c).sum()) for c in range(4)}
        coverage["agnews"]["splits"][split] = stats
        print(f"AG News {split}:", {k: v for k, v in stats.items() if not isinstance(v, dict)})
    coverage["agnews"]["seconds"] = round(time.perf_counter() - t0, 1)
    coverage["agnews"]["example_tokenisation"] = {
        "raw": texts_te[0],
        "tokens": " ".join(tok(texts_te[0])),
    }
    dump_json(coverage, res / "coverage.json")
    print("wrote", res / "coverage.json")


if __name__ == "__main__":
    main()
