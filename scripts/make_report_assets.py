"""Turn everything under results/ into the report's tables, figures and number
macros. Nothing in the report is typed by hand: re-run this after any experiment.

    uv run scripts/make_report_assets.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.data import AG_LABELS, BUCKET_NAMES  # noqa: E402
from lm.utils import dump_json, ensure_dir, load_config, load_json  # noqa: E402

RES = ROOT / "results"
TAB = ensure_dir(ROOT / "report" / "tables")
FIG = ensure_dir(ROOT / "report" / "figures")
SUM = ensure_dir(RES / "summary")
BUDGET_SECONDS = 480          # matched wall-clock budget for the Part I control column
MATCHED_EPOCH = 3             # matched number of updates

# fixed entity -> colour assignment (validated categorical order; never cycled)
C = {"rnn": "#2a78d6", "lstm": "#eb6834", "transformer": "#1baf7a", "transformer_1l": "#eda100",
     "ngram": "#4a3aa7", "random": "#2a78d6", "skipgram": "#eb6834", "glove": "#1baf7a",
     "tfidf": "#4a3aa7", "emb": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
})

PART1 = [("ngram", "KN $n$-gram"), ("rnn_s0", "RNN"), ("lstm_s0", "LSTM"),
         ("transformer_s0", "Transformer (4L)"), ("transformer_1layer_s0", "Transformer (1L, control)")]
PART2 = [  # (run stem, source, trainable, label)
    ("lstm", "random", True, "random, trainable (Part I LSTM)"),
    ("lstm_random_frozen", "random", False, "random, frozen"),
    ("lstm_skipgram_train", "skipgram", True, "skip-gram, trainable"),
    ("lstm_skipgram_frozen", "skipgram", False, "skip-gram, frozen"),
    ("lstm_glove_train", "glove", True, "GloVe, trainable"),
    ("lstm_glove_frozen", "glove", False, "GloVe, frozen"),
]
REP_COLOR = {"lstm_rand_last": "#eb6834", "lstm_rand_mean": "#eb6834", "transformer_rand_last": "#1baf7a",
             "transformer_rand_mean": "#1baf7a", "rnn_rand_mean": "#2a78d6", "transformer1l_rand_mean": "#eda100",
             "lstm_skipgram_frozen_mean": "#e87ba4", "lstm_glove_frozen_mean": "#008300",
             "emb_rand_learned_mean": "#eb6834", "emb_skipgram_mean": "#e87ba4", "emb_glove_mean": "#008300",
             "tfidf": "#4a3aa7"}
PART3 = [  # (name, label, source run for PPL)
    ("lstm_rand_last", "LSTM, last state", "lstm_s0"),
    ("lstm_rand_mean", "LSTM, mean", "lstm_s0"),
    ("transformer_rand_last", "Transformer, last state", "transformer_s0"),
    ("transformer_rand_mean", "Transformer, mean", "transformer_s0"),
    ("rnn_rand_mean", "RNN, mean", "rnn_s0"),
    ("transformer1l_rand_mean", "Transformer 1L, mean", "transformer_1layer_s0"),
    ("lstm_skipgram_frozen_mean", "LSTM + frozen skip-gram, mean", "lstm_skipgram_frozen_s0"),
    ("lstm_glove_frozen_mean", "LSTM + frozen GloVe, mean", "lstm_glove_frozen_s0"),
    ("emb_rand_learned_mean", "embedding only: LSTM-learned table", None),
    ("emb_skipgram_mean", "embedding only: skip-gram", None),
    ("emb_glove_mean", "embedding only: GloVe", None),
    ("tfidf", "TF-IDF uni+bigrams (reference)", None),
]


# ----------------------------------------------------------------- helpers
def tex(s) -> str:
    s = "".join(ch if ord(ch) < 0x250 or ch in "—–‘’“”…" else "?" for ch in str(s))
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("$", r"\$").replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
            .replace("}", r"\}").replace("~", r"\textasciitilde{}").replace("^", r"\^{}")
            .replace("<", r"\textless{}").replace(">", r"\textgreater{}"))


def fmt(x, nd=1, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:,.{nd}f}" if nd else f"{x:,.0f}"


def run_dir(name):
    return RES / "runs" / name


def load_run(name):
    d = run_dir(name)
    if not (d / "metrics.json").exists():
        return None
    r = {"metrics": load_json(d / "metrics.json")}
    for k in ("eval", "timing", "generations"):
        r[k] = load_json(d / f"{k}.json") if (d / f"{k}.json").exists() else None
    return r


def history(run):
    return run["metrics"].get("history", [])


def val_at_epoch(run, epoch):
    h = history(run)
    return h[epoch - 1]["val_ppl"] if len(h) >= epoch else None


def val_within_budget(run, seconds):
    ok = [r["val_ppl"] for r in history(run) if r["cum_train_seconds"] <= seconds]
    return min(ok) if ok else None


def epochs_to_within(run, frac=0.02):
    h = history(run)
    best = min(r["val_ppl"] for r in h)
    for r in h:
        if r["val_ppl"] <= best * (1 + frac):
            return r["epoch"]
    return None


def test_ppl(run, protocol="default"):
    if run and run["eval"] and protocol in run["eval"]["protocols"]:
        return run["eval"]["protocols"][protocol]["perplexity"]
    return None


def write(path: Path, text: str):
    path.write_text(text)
    print("wrote", path.relative_to(ROOT))


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ Part I
def part1(numbers: dict):
    runs = {n: load_run(n) for n, _ in PART1}
    ng = runs["ngram"]
    ng_order = ng["metrics"]["selected_order"] if ng else None
    labels = dict(PART1)
    if ng_order:
        labels["ngram"] = f"KN {ng_order}-gram"
    rows = []
    for n, _ in PART1:
        r = runs[n]
        if r is None:
            continue
        m = r["metrics"]
        neural = n != "ngram"
        row = {
            "model": labels[n],
            "test_ppl": test_ppl(r),
            "test_ppl_blocks": test_ppl(r, "blocks128"),
            "val_ppl_best": m["best_val_ppl"],
            "val_ppl_ep3": val_at_epoch(r, MATCHED_EPOCH) if neural else None,
            "val_ppl_budget": val_within_budget(r, BUDGET_SECONDS) if neural else None,
            "best_epoch": m.get("best_epoch") if neural else None,
            "val_ppl_ep24": (load_run(n.replace("_s0", "_e24_s0")) or {}).get("metrics", {}).get("best_val_ppl") if neural else None,
            "test_ppl_ep24": test_ppl(load_run(n.replace("_s0", "_e24_s0"))) if neural else None,
            "params_total_M": m["params"]["total"] / 1e6 if neural else None,
            "params_core_M": m["params"]["core"] / 1e6 if neural else None,
            "n_entries_M": m["n_entries"] / 1e6 if not neural else None,
            "size_MB": (r["timing"]["size"]["checkpoint_bytes"] if r["timing"] else m.get("checkpoint_bytes")) / 1e6,
            "train_min_to_best": (m["train_seconds_to_best"] if neural else m["train_seconds_total"]) / 60,
            "train_min_total": m["train_seconds_total"] / 60,
            "epoch_s": m.get("epoch_seconds_mean_after_first") if neural else None,
            "score_tok_s": r["timing"]["score_test"]["tokens_per_second"] if r["timing"] else None,
            "gen_ms_tok": r["timing"]["generate"]["ms_per_token"] if r["timing"] and "generate" in r["timing"] else None,
        }
        rows.append(row)
        key = {"ngram": "Ngram", "rnn_s0": "RNN", "lstm_s0": "LSTM", "transformer_s0": "Transformer",
               "transformer_1layer_s0": "TransformerOne"}[n]
        numbers[f"ppl{key}"] = fmt(row["test_ppl"], 1)
        numbers[f"pplBlocks{key}"] = fmt(row["test_ppl_blocks"], 1)
        numbers[f"pplLong{key}"] = fmt(row["test_ppl_ep24"], 1)
        long = load_run(n.replace("_s0", "_e24_s0")) if neural else None
        if long:
            numbers[f"pplLongBlocks{key}"] = fmt(test_ppl(long, "blocks128"), 1)
            numbers[f"bestEpochLong{key}"] = fmt(long["metrics"]["best_epoch"], 0)
            numbers[f"epochsRunLong{key}"] = fmt(long["metrics"]["epochs_run"], 0)
            numbers[f"trainMinLong{key}"] = fmt(long["metrics"]["train_seconds_total"] / 60, 0)
        numbers[f"trainMin{key}"] = fmt(row["train_min_to_best"], 1)
        numbers[f"valEpThree{key}"] = fmt(row["val_ppl_ep3"], 1)
        numbers[f"valBudget{key}"] = fmt(row["val_ppl_budget"], 1)
        numbers[f"scoreTokS{key}"] = fmt(row["score_tok_s"], 0)
        numbers[f"genMs{key}"] = fmt(row["gen_ms_tok"], 1)
        numbers[f"coreM{key}"] = fmt(row["params_core_M"], 2)
        if r["eval"]:
            e = r["eval"]["protocols"]["default"]
            numbers[f"ceRare{key}"] = fmt(e["bucket_ce"]["1-10"], 2)
            numbers[f"ceMid{key}"] = fmt(e["bucket_ce"]["11-100"], 2)
            numbers[f"ceTop{key}"] = fmt(e["bucket_ce"][">10k"], 2)
            numbers[f"ceEos{key}"] = fmt(e["eos_ce"], 2)
            numbers[f"ceUnk{key}"] = fmt(e["unk_ce"], 2)
        if r["generations"]:
            g = r["generations"]["generations"]
            numbers[f"repGreedy{key}"] = fmt(100 * float(np.mean([x["greedy"]["repeated_4gram_frac"] for x in g])), 0)
            numbers[f"repTopk{key}"] = fmt(100 * float(np.mean([x["topk"]["repeated_4gram_frac"] for x in g])), 1)
            numbers[f"distinctTopk{key}"] = fmt(float(np.mean([x["topk"]["distinct_2"] for x in g])), 2)
        if neural and history(r):
            h = history(r)[-1]
            numbers[f"trainPplLast{key}"] = fmt(h["train_ppl"], 1)
            numbers[f"valPplLast{key}"] = fmt(h["val_ppl"], 1)
            numbers[f"epochSec{key}"] = fmt(m.get("epoch_seconds_mean_after_first"), 0)
    save_csv(SUM / "part1.csv", rows)
    if not rows:
        return
    byname = {r["model"]: r for r in rows}
    def _d(a, b, key):
        ra, rb = byname.get(labels[a]), byname.get(labels[b])
        return fmt(ra[key] - rb[key], 1) if ra and rb and ra[key] and rb[key] else "--"
    for n in ("lstm_s0", "transformer_s0", "rnn_s0"):
        r = byname.get(labels[n])
        if r and r["test_ppl"] and r["test_ppl_blocks"]:
            key = {"lstm_s0": "LSTM", "transformer_s0": "Transformer", "rnn_s0": "RNN"}[n]
            numbers[f"deltaBlocks{key}"] = fmt(r["test_ppl_blocks"] - r["test_ppl"], 1)
    numbers["deltaOneLayerVsLSTM"] = _d("transformer_1layer_s0", "lstm_s0", "test_ppl")
    numbers["deltaTransformerVsLSTM"] = _d("transformer_s0", "lstm_s0", "test_ppl")
    # quality table
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Model & Test PPL & Test, ctx$\le$128 & Val, ep.\ 3 & Val, 8\,min & Val, ep.\ 12 & Test, 24 ep. \\",
         r"\midrule"]
    for r in rows:
        L.append(f"{r['model']} & {fmt(r['test_ppl'])} & {fmt(r['test_ppl_blocks'])} & "
                 f"{fmt(r['val_ppl_ep3'])} & {fmt(r['val_ppl_budget'])} & {fmt(r['val_ppl_best']) if r['best_epoch'] else '--'} & {fmt(r['test_ppl_ep24'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "part1_quality.tex", "\n".join(L))
    # cost table
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Model & Params (M) & Core (M) & Size (MB) & Train to best (min) & Score (tok/s) & Generate (ms/tok) \\",
         r"\midrule"]
    for r in rows:
        p = fmt(r["params_total_M"], 1) if r["params_total_M"] else f"{fmt(r['n_entries_M'],1)} entries"
        L.append(f"{r['model']} & {p} & {fmt(r['params_core_M'], 2)} & {fmt(r['size_MB'], 0)} & "
                 f"{fmt(r['train_min_to_best'], 1)} & {fmt(r['score_tok_s'], 0)} & {fmt(r['gen_ms_tok'], 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "part1_cost.tex", "\n".join(L))

    # learning curves: val PPL vs epoch (train dotted) and vs wall-clock
    neural = [(n, labels[n]) for n, _ in PART1 if n != "ngram" and runs[n]]
    if neural:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1))
        for n, lab in neural:
            h = history(runs[n])
            col = C["transformer_1l" if "1layer" in n else n.split("_")[0]]
            ep = [r["epoch"] for r in h]
            axes[0].plot(ep, [r["val_ppl"] for r in h], color=col, lw=1.6, label=lab)
            axes[0].plot(ep, [r["train_ppl"] for r in h], color=col, lw=1.0, ls=":")
            axes[1].plot([r["cum_train_seconds"] / 60 for r in h], [r["val_ppl"] for r in h],
                         color=col, lw=1.6, label=lab)
            long = load_run(n.replace("_s0", "_e24_s0"))
            if long:
                hl = history(long)
                axes[0].plot([r["epoch"] for r in hl], [r["val_ppl"] for r in hl], color=col, lw=0.9, ls="--",
                             label=f"{lab}, 24-epoch run" if n == "lstm_s0" else None)
                axes[1].plot([r["cum_train_seconds"] / 60 for r in hl], [r["val_ppl"] for r in hl], color=col, lw=0.9, ls="--")
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        for ax in axes:
            ax.set_yscale("log")
            ax.set_ylim(100, 1000)
            ax.yaxis.set_major_locator(FixedLocator([100, 150, 200, 300, 500, 1000]))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(FixedLocator([]))
        axes[0].set_ylabel("perplexity (log scale)")
        axes[0].set_xlabel("epoch")
        axes[1].set_xlabel("training wall-clock (minutes)")
        axes[1].axvline(BUDGET_SECONDS / 60, color=MUTED, lw=0.8, ls="--")
        axes[1].text(BUDGET_SECONDS / 60 + 0.4, 110, "matched\n8-min budget", color=MUTED, fontsize=6, va="bottom")
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.16), fontsize=6.5)
        fig.subplots_adjust(wspace=0.22, bottom=0.2)
        fig.savefig(FIG / "part1_curves.pdf")
        plt.close(fig)
        print("wrote report/figures/part1_curves.pdf")

    # cross-entropy by train-frequency bucket
    with_eval = [(n, labels[n]) for n, _ in PART1 if runs[n] and runs[n]["eval"]]
    if with_eval:
        fig, ax = plt.subplots(figsize=(6.4, 1.9))
        cats = list(BUCKET_NAMES) + ["<eos>", "<unk>"]
        width = 0.8 / len(with_eval)
        for i, (n, lab) in enumerate(with_eval):
            s = runs[n]["eval"]["protocols"]["default"]
            vals = [s["bucket_ce"][b] for b in BUCKET_NAMES] + [s["eos_ce"], s["unk_ce"]]
            col = C["transformer_1l" if "1layer" in n else n.split("_")[0]]
            x = np.arange(len(cats)) + (i - (len(with_eval) - 1) / 2) * width
            ax.bar(x, vals, width * 0.9, color=col, label=lab)
        fracs = runs[with_eval[0][0]]["eval"]["protocols"]["default"]["bucket_frac"]
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels([f"{b}\n({100*fracs[b]:.0f}%)" for b in BUCKET_NAMES] + ["<eos>", "<unk>"], fontsize=6.5)
        ax.set_xlabel("target's frequency in the train split (share of test targets)")
        ax.set_ylabel("test cross-entropy (nats)")
        ax.set_ylim(0, 15.5)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=6.5)
        fig.savefig(FIG / "part1_buckets.pdf")
        plt.close(fig)
        print("wrote report/figures/part1_buckets.pdf")

    # generations: one prompt for the main text, everything for the appendix
    with_gen = [(n, labels[n]) for n, _ in PART1 if runs[n] and runs[n]["generations"]]
    if with_gen:
        gens = {n: runs[n]["generations"]["generations"] for n, _ in with_gen}
        prompts = gens[with_gen[0][0]]
        pi = 0
        L = [r"\begin{tabular}{@{}p{2.3cm}>{\raggedright\arraybackslash}p{10.4cm}rr@{}}", r"\toprule",
             r"Model & Continuation (top-$k$=40, $T$=1, 50 tokens; first 24 shown) & distinct-2 & rep.\ 4-gram \\",
             r"\midrule", rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{Prompt:}} \texttt{{{tex(prompts[pi]['prompt'])}}}}} \\", r"\midrule"]
        for n, lab in with_gen:
            g = gens[n][pi]["topk"]
            words = g["text"].split()[:24]
            L.append(f"{lab} & \\texttt{{{tex(' '.join(words))}}} \\ldots & {g['distinct_2']:.2f} & {g['repeated_4gram_frac']:.2f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(TAB / "part1_generations.tex", "\n".join(L))
        # repetition summary averaged over prompts
        rep_rows = []
        for n, lab in with_gen:
            for mode in ("greedy", "topk"):
                rep_rows.append({"model": lab, "decoding": mode,
                                 **{k: float(np.mean([g[mode][k] for g in gens[n]]))
                                    for k in ("distinct_1", "distinct_2", "distinct_3", "repeated_4gram_frac")}})
        save_csv(SUM / "part1_repetition.csv", rep_rows)
        L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Model & Decoding & distinct-1 & distinct-2 & distinct-3 & repeated 4-grams \\", r"\midrule"]
        for r in rep_rows:
            L.append(f"{r['model']} & {r['decoding']} & {r['distinct_1']:.2f} & {r['distinct_2']:.2f} & "
                     f"{r['distinct_3']:.2f} & {r['repeated_4gram_frac']:.2f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(TAB / "part1_repetition.tex", "\n".join(L))
        # appendix: all prompts, both decodings
        A = []
        for pi, p in enumerate(prompts):
            A.append(rf"\paragraph{{Prompt {pi+1}}} \texttt{{{tex(p['prompt'])}}}")
            A.append(r"\begin{description}")
            for n, lab in with_gen:
                for mode in ("greedy", "topk"):
                    A.append(rf"\item[{lab}, {mode}] \texttt{{\small {tex(gens[n][pi][mode]['text'])}}}")
            A.append(r"\end{description}")
        write(TAB / "appendix_generations.tex", "\n".join(A))


# ----------------------------------------------------------------- Part II
def part2(numbers: dict, part3_f1: dict):
    emb = {}
    for src in ("skipgram", "glove"):
        p = RES / "embeddings" / f"{src}_coverage.json"
        emb[src] = load_json(p) if p.exists() else None
    if emb["glove"]:
        numbers["gloveCoverageTypes"] = fmt(100 * emb["glove"]["coverage_types"], 1)
        numbers["gloveCoverageTokens"] = fmt(100 * emb["glove"]["coverage_train_tokens"], 2)
        numbers["gloveMissing"] = fmt(emb["glove"]["missing_types"], 0)
    if emb["skipgram"]:
        numbers["skipgramMinutes"] = fmt(emb["skipgram"]["train_seconds"] / 60, 1)
        numbers["rowNormSkipgramTable"] = fmt(emb["skipgram"]["mean_row_norm_all"], 2)
    if emb["glove"]:
        numbers["rowNormGloveTable"] = fmt(emb["glove"]["mean_row_norm_all"], 2)
        numbers["gloveMissingExamples"] = tex(", ".join(emb["glove"]["missing_examples_most_frequent"][:6]))
    rows = []
    curves = {}
    for stem, source, trainable, label in PART2:
        seeds = {s: load_run(f"{stem}_s{s}") for s in (0, 1)}
        seeds = {s: r for s, r in seeds.items() if r}
        if not seeds:
            continue
        r0 = seeds.get(0) or next(iter(seeds.values()))
        m0 = r0["metrics"]
        ppls = [test_ppl(r) for r in seeds.values() if test_ppl(r)]
        cov = emb[source]["coverage_types"] if source in emb and emb[source] else 1.0
        row = {
            "setting": label, "source": source, "trainable": trainable, "runs": ",".join(f"s{s}" for s in seeds),
            "coverage_types": cov,
            "emb_train_seconds": emb["skipgram"]["train_seconds"] if source == "skipgram" and emb["skipgram"] else 0.0,
            "test_ppl_mean": float(np.mean(ppls)) if ppls else None,
            "test_ppl_halfrange": float((max(ppls) - min(ppls)) / 2) if len(ppls) > 1 else None,
            "test_ppl_s0": test_ppl(seeds[0]) if 0 in seeds else None,
            "test_ppl_s1": test_ppl(seeds[1]) if 1 in seeds else None,
            "best_epoch": m0["best_epoch"],
            "epochs_to_2pct": epochs_to_within(r0),
            "val_ppl_ep3": val_at_epoch(r0, 3), "val_ppl_ep6": val_at_epoch(r0, 6), "val_ppl_ep12": val_at_epoch(r0, 12),
            "grad_norm_mean": float(np.mean([h["grad_norm_mean"] for h in history(r0)])),
            "clipped_frac": float(np.mean([h["clipped_frac"] for h in history(r0)])),
            "step_loss_std": float(np.mean([h["step_loss_std"] for h in history(r0)])),
            "row_norm": m0["embedding_mean_row_norm"],
            "train_min_to_best": m0["train_seconds_to_best"] / 60,
            "frozen_verified": m0["embedding_hash_before"] == m0["embedding_hash_after"] if m0["embedding_frozen"] else None,
            "agnews_macro_f1": part3_f1.get(f"{stem}_s0"),
        }
        rows.append(row)
        curves[label] = {s: history(r) for s, r in seeds.items()}
        key = {"lstm": "LSTMRand", "lstm_random_frozen": "LSTMRandFrozen", "lstm_skipgram_train": "LSTMSkipTrain",
               "lstm_skipgram_frozen": "LSTMSkipFrozen", "lstm_glove_train": "LSTMGloveTrain",
               "lstm_glove_frozen": "LSTMGloveFrozen"}[stem]
        numbers[f"ppl{key}"] = fmt(row["test_ppl_mean"], 1)
        if row["test_ppl_halfrange"] is not None:
            numbers[f"pplRange{key}"] = fmt(row["test_ppl_halfrange"], 1)
        h0 = history(r0)
        numbers[f"valEpThree{key}"] = fmt(row["val_ppl_ep3"], 1)
        numbers[f"valEpTwelve{key}"] = fmt(row["val_ppl_ep12"], 1)
        numbers[f"gap{key}"] = fmt(h0[-1]["val_ppl"] - h0[-1]["train_ppl"], 1)
        numbers[f"gnMax{key}"] = fmt(max(x["grad_norm_max"] for x in h0), 1)
        numbers[f"gnMean{key}"] = fmt(row["grad_norm_mean"], 2)
        if 1 in seeds:
            h1 = history(seeds[1])
            numbers[f"gnMax{key}SeedOne"] = fmt(max(x["grad_norm_max"] for x in h1), 1)
            numbers[f"valEpTwelve{key}SeedOne"] = fmt(h1[-1]["val_ppl"], 1)
            numbers[f"pplSeedOne{key}"] = fmt(row["test_ppl_s1"], 1)
        base_run = load_run("lstm_s0")
        if base_run and base_run["eval"] and r0["eval"]:
            bb = base_run["eval"]["protocols"]["default"]["bucket_ce"]
            cb = r0["eval"]["protocols"]["default"]["bucket_ce"]
            numbers[f"gainRare{key}"] = fmt(bb["1-10"] - cb["1-10"], 2)
            numbers[f"gainMid{key}"] = fmt(bb["11-100"] - cb["11-100"], 2)
            top = round(bb[">10k"] - cb[">10k"], 2) + 0.0
            numbers[f"gainTop{key}"] = f"{top:+.2f}" if abs(top) >= 0.005 else "0.00"
        if r0["generations"]:
            g = r0["generations"]["generations"]
            numbers[f"repGreedy{key}"] = fmt(100 * float(np.mean([x["greedy"]["repeated_4gram_frac"] for x in g])), 0)
            numbers[f"distinctTopk{key}"] = fmt(float(np.mean([x["topk"]["distinct_2"] for x in g])), 2)
        if base_run and stem != "lstm":
            target = history(base_run)[-1]["val_ppl"]
            reach = next((x["epoch"] for x in h0 if x["val_ppl"] <= target), None)
            numbers[f"epochReach{key}"] = fmt(reach, 0) if reach else "never"
    save_csv(SUM / "part2.csv", rows)
    # factorial contrasts on test PPL (means over seeds where available)
    cell = {r["setting"]: r["test_ppl_mean"] for r in rows}
    def _c(a, b):
        return fmt(cell[a] - cell[b], 1) if cell.get(a) and cell.get(b) else "--"
    numbers["cRandFrozenVsTrain"] = _c("random, frozen", "random, trainable (Part I LSTM)")
    numbers["cSkipFrozenVsTrain"] = _c("skip-gram, frozen", "skip-gram, trainable")
    numbers["cGloveFrozenVsTrain"] = _c("GloVe, frozen", "GloVe, trainable")
    numbers["cRandVsSkipFrozen"] = _c("random, trainable (Part I LSTM)", "skip-gram, frozen")
    numbers["cRandVsGloveFrozen"] = _c("random, trainable (Part I LSTM)", "GloVe, frozen")
    numbers["cSkipVsGloveFrozen"] = _c("skip-gram, frozen", "GloVe, frozen")
    numbers["cSkipVsGloveTrain"] = _c("skip-gram, trainable", "GloVe, trainable")
    numbers["cRandVsGloveTrain"] = _c("random, trainable (Part I LSTM)", "GloVe, trainable")
    if not rows:
        return
    L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r"Embedding source & Trainable & Cov.\ (\%) & Emb.\ cost & Test PPL & Val ep.\ 3 & Val ep.\ 6 & $\|g\|$ mean & Clipped & Row norm \\",
         r"\midrule"]
    for r in rows:
        ppl = fmt(r["test_ppl_mean"]) + (f" $\\pm$ {fmt(r['test_ppl_halfrange'])}" if r["test_ppl_halfrange"] is not None else "")
        cost = f"{r['emb_train_seconds']/60:.1f} min" if r["emb_train_seconds"] else ("--" if r["source"] != "glove" else "public")
        L.append(f"{r['source'].replace('skipgram', 'skip-gram').replace('glove', 'GloVe')} & "
                 f"{'yes' if r['trainable'] else 'no'} & {fmt(100*r['coverage_types'])} & {cost} & {ppl} & "
                 f"{fmt(r['val_ppl_ep3'])} & {fmt(r['val_ppl_ep6'])} & {r['grad_norm_mean']:.2f} & "
                 f"{100*r['clipped_frac']:.1f}\\% & {r['row_norm']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "part2_factorial.tex", "\n".join(L))

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for label, by_seed in curves.items():
        source = [s for _, s, _, l in PART2 if l == label][0]
        trainable = [t for _, _, t, l in PART2 if l == label][0]
        for s, h in by_seed.items():
            ax.plot([r["epoch"] for r in h], [r["val_ppl"] for r in h], color=C[source],
                    ls="-" if trainable else "--", lw=1.6 if s == 0 else 0.9, alpha=1 if s == 0 else 0.6,
                    label=label if s == 0 else None)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.set_yscale("log")
    ax.set_ylim(110, 700)
    ax.yaxis.set_major_locator(FixedLocator([120, 150, 200, 300, 400, 600]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation perplexity (log scale)")
    ax.legend(loc="upper right", fontsize=5.5)
    fig.savefig(FIG / "part2_curves.pdf")
    plt.close(fig)
    print("wrote report/figures/part2_curves.pdf")

    base = load_run("lstm_s0")
    others = [(stem, source, trainable, label) for stem, source, trainable, label in PART2
              if stem not in ("lstm", "lstm_random_frozen") and load_run(f"{stem}_s0") and load_run(f"{stem}_s0")["eval"]]
    if base and base["eval"] and others:
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        b = base["eval"]["protocols"]["default"]["bucket_ce"]
        width = 0.8 / len(others)
        for i, (stem, source, trainable, label) in enumerate(others):
            s = load_run(f"{stem}_s0")["eval"]["protocols"]["default"]["bucket_ce"]
            d = [s[k] - b[k] for k in BUCKET_NAMES]
            x = np.arange(len(BUCKET_NAMES)) + (i - (len(others) - 1) / 2) * width
            ax.bar(x, d, width * 0.9, color=C[source], hatch=None if trainable else "////",
                   edgecolor="white", lw=0.4, label=label)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(np.arange(len(BUCKET_NAMES)))
        ax.set_xticklabels(BUCKET_NAMES, fontsize=6.5)
        ax.set_xlabel("target's train frequency")
        ax.set_ylabel("$\\Delta$ test CE vs. random-trainable (nats)")
        ax.legend(fontsize=5.5, loc="lower right")
        fig.savefig(FIG / "part2_buckets.pdf")
        plt.close(fig)
        print("wrote report/figures/part2_buckets.pdf")

    # appendix: nearest neighbours
    if emb["glove"] and emb["skipgram"]:
        words = [w for w in emb["glove"]["nearest_neighbours"] if w in emb["skipgram"]["nearest_neighbours"]]
        L = [r"\begin{tabular}{@{}lp{6.2cm}p{6.2cm}@{}}", r"\toprule",
             r"Word & Skip-gram (own, WikiText-2 train) & GloVe-6B \\", r"\midrule"]
        for w in words[:8]:
            sg = ", ".join(x[0] for x in emb["skipgram"]["nearest_neighbours"][w])
            gl = ", ".join(x[0] for x in emb["glove"]["nearest_neighbours"][w])
            L.append(f"\\texttt{{{tex(w)}}} & \\texttt{{{tex(sg)}}} & \\texttt{{{tex(gl)}}} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(TAB / "appendix_neighbours.tex", "\n".join(L))


# ---------------------------------------------------------------- Part III
def part3(numbers: dict) -> dict:
    """Returns {source_run: macro_f1 at full n_train} for mean-pooled reps."""
    cdir = RES / "classification"
    cfg = load_config(ROOT / "configs" / "classify.yaml")
    n_full = max(cfg["n_train_grid"])
    reps = {}
    for name, label, src in PART3:
        p = cdir / f"{name}.json"
        if p.exists():
            reps[name] = load_json(p)
    if not reps:
        return {}
    rows, f1_by_run = [], {}
    for name, label, src in PART3:
        if name not in reps:
            continue
        by_n = reps[name]["by_n_train"]
        full = by_n[str(n_full)]
        src_run = load_run(src) if src else None
        row = {"representation": label, "name": name, "dim": reps[name]["meta"]["dim"],
               "source_test_ppl": test_ppl(src_run) if src_run else None,
               "test_accuracy": full["test_accuracy"], "test_macro_f1": full["test_macro_f1"],
               "C": full["C"], "unk_rate_correct": reps[name].get("unk_rate_correct"),
               "unk_rate_wrong": reps[name].get("unk_rate_wrong")}
        for n in cfg["n_train_grid"]:
            row[f"macro_f1_n{n}"] = by_n[str(n)]["test_macro_f1"] if str(n) in by_n else None
        rows.append(row)
        if src and name.endswith("_mean"):
            f1_by_run[src] = full["test_macro_f1"]
    save_csv(SUM / "part3.csv", rows)
    grid = cfg["n_train_grid"]
    L = [r"\begin{tabular}{@{}lrrrr" + "r" * (len(grid) - 1) + "@{}}", r"\toprule",
         r"Representation & Dim & Source LM test PPL & Acc.\ & " +
         " & ".join(f"F1 ($n$={n:,})" for n in sorted(grid, reverse=True)) + r" \\", r"\midrule"]
    for r in rows:
        L.append(f"{r['representation']} & {r['dim']:,} & {fmt(r['source_test_ppl'])} & "
                 f"{100*r['test_accuracy']:.1f} & " +
                 " & ".join(fmt(100 * r[f'macro_f1_n{n}'], 1) for n in sorted(grid, reverse=True)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "part3_main.tex", "\n".join(L))

    lm_rows = [r for r in rows if r["source_test_ppl"] and r["name"].endswith("_mean")]
    best = max(rows, key=lambda r: r["test_macro_f1"])
    best_lm = max((r for r in rows if r["source_test_ppl"]), key=lambda r: r["test_macro_f1"])
    numbers["bestRep"] = tex(best["representation"])
    numbers["bestRepF1"] = fmt(100 * best["test_macro_f1"], 1)
    numbers["bestLMRep"] = tex(best_lm["representation"])
    numbers["bestLMRepF1"] = fmt(100 * best_lm["test_macro_f1"], 1)
    numbers["bestLMRepAcc"] = fmt(100 * best_lm["test_accuracy"], 1)
    for r in rows:
        key = "".join(w.capitalize() for w in r["name"].split("_"))
        numbers[f"fOne{key}"] = fmt(100 * r["test_macro_f1"], 1)
        numbers[f"acc{key}"] = fmt(100 * r["test_accuracy"], 1)
        numbers[f"fOneSmall{key}"] = fmt(100 * r[f"macro_f1_n{min(grid)}"], 1)
    numbers["nLabelsSmall"] = f"{min(grid):,}"
    numbers["nLabelsFull"] = f"{n_full:,}"
    if len(lm_rows) >= 3:
        from scipy.stats import spearmanr
        rho, pval = spearmanr([r["source_test_ppl"] for r in lm_rows], [r["test_macro_f1"] for r in lm_rows])
        numbers["spearmanRho"] = f"{rho:.2f}"
        numbers["spearmanN"] = str(len(lm_rows))
        dump_json({"rho": rho, "p": pval, "n": len(lm_rows),
                   "points": [(r["name"], r["source_test_ppl"], r["test_macro_f1"]) for r in lm_rows]},
                  SUM / "part3_rank_agreement.json")
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        for r in lm_rows:
            col = REP_COLOR.get(r["name"], INK)
            ax.scatter(r["source_test_ppl"], 100 * r["test_macro_f1"], s=28, color=col, zorder=3)
            ax.annotate(r["representation"].replace(", mean", ""), (r["source_test_ppl"], 100 * r["test_macro_f1"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=5.5, color=INK)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([120, 130, 140, 150, 160, 170, 180]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel("source LM test perplexity (log scale)")
        ax.set_ylabel("AG News macro-F1 (%), mean pooling")
        ax.set_title(f"Spearman $\\rho$ = {rho:.2f} (n = {len(lm_rows)})", fontsize=7, color=MUTED)
        fig.savefig(FIG / "part3_ppl_vs_f1.pdf")
        plt.close(fig)
        print("wrote report/figures/part3_ppl_vs_f1.pdf")

    # learning curve over labelled-set size
    show = ["lstm_rand_mean", "transformer_rand_mean", "rnn_rand_mean", "lstm_glove_frozen_mean",
            "emb_glove_mean", "tfidf"]
    show = [s for s in show if s in reps]
    if show:
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        for name in show:
            label = [l for n, l, _ in PART3 if n == name][0]
            ys = [100 * reps[name]["by_n_train"][str(n)]["test_macro_f1"] for n in grid]
            ax.plot(grid, ys, marker="o", ms=3.5, lw=1.5, color=REP_COLOR.get(name, INK),
                    ls="--" if name.startswith("emb_") else "-", label=label)
        ax.set_xscale("log")
        ax.set_xticks(grid)
        ax.set_xticklabels([f"{n:,}" for n in grid])
        ax.set_xlabel("labelled training examples")
        ax.set_ylabel("AG News macro-F1 (%)")
        ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        fig.savefig(FIG / "part3_learning_curve.pdf")
        plt.close(fig)
        print("wrote report/figures/part3_learning_curve.pdf")

    # confusion matrix and confident errors for the best LM-based representation
    bl = reps[best_lm["name"]]
    cm = np.array(bl["by_n_train"][str(n_full)]["confusion"])
    numbers["confBusinessSciTech"] = f"{cm[2, 3] + cm[3, 2]:,}"
    numbers["confWorldBusiness"] = f"{cm[0, 2] + cm[2, 0]:,}"
    numbers["confWorldSports"] = f"{cm[0, 1] + cm[1, 0]:,}"
    numbers["nTestDocs"] = f"{int(cm.sum()):,}"
    L = [r"\begin{tabular}{@{}l" + "r" * 4 + "@{}}", r"\toprule",
         "true $\\backslash$ pred & " + " & ".join(AG_LABELS) + r" \\", r"\midrule"]
    for i, lab in enumerate(AG_LABELS):
        L.append(f"{lab} & " + " & ".join(f"\\textbf{{{cm[i, j]}}}" if i == j else str(cm[i, j]) for j in range(4)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "part3_confusion.tex", "\n".join(L))
    numbers["confusionRep"] = tex(best_lm["representation"])
    numbers["unkRateCorrect"] = fmt(100 * bl["unk_rate_correct"], 1)
    numbers["unkRateWrong"] = fmt(100 * bl["unk_rate_wrong"], 1)
    L = [r"\begin{tabular}{@{}llr>{\raggedright\arraybackslash}p{5.6cm}@{}}", r"\toprule", r"True & Pred. & Conf. & Text (truncated) \\", r"\midrule"]
    for e in bl["errors_top20"][:4]:
        L.append(f"{e['true']} & {e['pred']} & {e['confidence']:.2f} & {tex(e['text'].replace(chr(92), '')[:95])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "part3_errors.tex", "\n".join(L))
    return f1_by_run


# ------------------------------------------------------------- appendix etc
def hyperparams():
    common = load_config(ROOT / "configs" / "common.yaml")
    L = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{11.3cm}@{}}", r"\toprule", r"Setting & Value \\", r"\midrule"]
    items = [
        ("Embedding dim / Transformer $d_\\mathrm{model}$", common["emb_dim"]),
        ("RNN / LSTM", "2 layers, hidden 300, tanh / LSTM cells"),
        ("Transformer", "4 layers (control: 1), 6 heads, FFN 1200, pre-LN, GELU, learned positions, context 128"),
        ("Dropout", f"{common['dropout']} after embedding, between layers, before output"),
        ("Batch $\\times$ BPTT", f"{common['batch_size']} $\\times$ {common['bptt']} tokens (continuous stream, state detached between chunks)"),
        ("Optimiser", f"Adam, lr {common['lr']}, $\\beta$=({common['betas'][0]}, {common['betas'][1]}), {common['warmup_steps']}-step linear warm-up"),
        ("Schedule", f"halve lr after {common['plateau_patience']} epoch(s) without val.\\ improvement; early stop after {common['early_stop_patience']}; max {common['max_epochs']} epochs"),
        ("Gradient clipping", f"global norm {common['grad_clip']}"),
        ("Output layer", "untied (frozen input tables would otherwise freeze the softmax)"),
        ("Validation / test protocol", f"recurrent: full history, batch 1; Transformer: sliding window {common['eval']['transformer_context']} / stride {common['eval']['transformer_stride']}"),
        ("Decoding", f"greedy and top-$k$={common['generation']['top_k']}, $T$={common['generation']['temperature']}, {common['generation']['max_new_tokens']} new tokens, seed {common['generation']['gen_seed']}+prompt index"),
        ("$n$-gram", "interpolated Kneser--Ney, orders 2--5 fitted, order chosen on validation, $D_n = n_1/(n_1+2n_2)$"),
        ("Skip-gram", "gensim, sg=1, window 5, 10 negatives, subsampling $10^{-4}$, 10 epochs, min\\_count 1, 1 worker"),
        ("GloVe", "6B tokens, 300d, uncased; missing rows $\\mathcal{N}(0, \\sigma_d)$ from the found rows"),
        ("Part III classifier", "scikit-learn multinomial logistic regression (lbfgs), standardised features, $C\\in\\{0.01,0.1,1,10\\}$ on validation"),
        ("Seeds", "0 everywhere; Part II required settings also seed 1"),
    ]
    for k, v in items:
        L.append(f"{k} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(TAB / "appendix_hyperparams.tex", "\n".join(L))


def data_numbers(numbers: dict):
    p = RES / "data" / "coverage.json"
    if not p.exists():
        return
    c = load_json(p)
    w = c["wikitext"]
    numbers["nVocab"] = f"{w['vocab_size']:,}"
    numbers["nTrainTok"] = f"{w['splits']['train']['tokens_in_stream']:,}"
    numbers["nValTok"] = f"{w['splits']['validation']['tokens_in_stream']:,}"
    numbers["nTestTargets"] = f"{w['splits']['test']['targets']:,}"
    numbers["unkFracTest"] = fmt(100 * w["splits"]["test"]["native_unk_tokens"] / w["splits"]["test"]["tokens_in_stream"], 1)
    a = c["agnews"]["splits"]
    numbers["agInVocab"] = fmt(100 * a["test"]["in_vocab_rate"], 1)
    numbers["agMeanLen"] = fmt(a["test"]["mean_len"], 0)
    numbers["agTruncated"] = fmt(100 * a["test"]["truncated_frac"], 1)
    numbers["agUnkBusiness"] = fmt(100 * a["test"]["unk_rate_per_class"]["Business"], 1)
    numbers["agUnkSports"] = fmt(100 * a["test"]["unk_rate_per_class"]["Sports"], 1)
    numbers["agUnkWorld"] = fmt(100 * a["test"]["unk_rate_per_class"]["World"], 1)
    numbers["agUnkSciTech"] = fmt(100 * a["test"]["unk_rate_per_class"]["Sci/Tech"], 1)


def main():
    numbers = {}
    data_numbers(numbers)
    part1(numbers)
    f1_by_run = part3(numbers)
    part2(numbers, f1_by_run)
    hyperparams()
    L = ["% generated by scripts/make_report_assets.py -- do not edit"]
    _digits = {str(i): w for i, w in enumerate(["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"])}
    numbers = {"".join(_digits.get(ch, ch) for ch in k): v for k, v in numbers.items()}  # TeX names: letters only
    for k, v in sorted(numbers.items()):
        L.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    # every macro-like token used in main.tex that no experiment has produced yet
    # gets a visible [pending] default (providecommand never overrides LaTeX's own)
    import re
    src = (ROOT / "report" / "main.tex").read_text()
    defined_in_tex = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", src))
    used = set(re.findall(r"\\([a-z][A-Za-z]{3,})", src))
    for name in sorted(used - set(numbers) - defined_in_tex):
        L.append(f"\\providecommand{{\\{name}}}{{\\textit{{[pending]}}}}")
    write(TAB / "numbers.tex", "\n".join(L) + "\n")
    dump_json(numbers, SUM / "numbers.json")


if __name__ == "__main__":
    main()
