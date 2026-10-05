"""Evaluate one trained language model on the WikiText-2 test stream with the
common protocol, measure the common inference workload, and generate the
matched continuations.

    uv run scripts/evaluate_lm.py --run lstm_s0
    uv run scripts/evaluate_lm.py --ngram

Writes results/runs/<run>/{eval.json,timing.json,generations.json}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.data import Vocab, load_stream  # noqa: E402
from lm.evaluate import score_model, score_windows, summarize  # noqa: E402
from lm.generate import generate_neural, generate_ngram, repetition_metrics  # noqa: E402
from lm.models import load_checkpoint  # noqa: E402
from lm.ngram import KneserNeyLM  # noqa: E402
from lm.timing import Timer, count_params, file_bytes, hardware_info  # noqa: E402
from lm.utils import dump_json, ensure_dir, get_device, load_config, load_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run name of a neural model (checkpoints/<run>.pt)")
    ap.add_argument("--ngram", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip_generation", action="store_true")
    ap.add_argument("--limit_tokens", type=int, default=None,
                    help="debug: score only the first N test tokens and write to <run>_debug/")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    assert bool(args.run) != args.ngram, "give --run <name> or --ngram"

    common = load_config(ROOT / "configs" / "common.yaml")
    device = get_device(args.device or common["device"])
    vocab = Vocab.load(ROOT / common["data_dir"] / "wikitext" / "vocab.json")
    test = load_stream(ROOT / common["data_dir"], "test")
    if args.limit_tokens:
        test = test[: args.limit_tokens]
    gen_cfg = common["generation"]
    prompts = load_json(ROOT / gen_cfg["prompts_file"])["prompts"]

    if args.ngram:
        run_name = "ngram"
        model = KneserNeyLM.load(ROOT / common["checkpoint_dir"] / "ngram.pkl")
        cfg = common
        protocols = {"default": lambda s: model.logprob_stream(s)}
        default_name = f"kneser-ney order {model.order}, full history"
        size = {"n_entries": model.n_entries(),
                "checkpoint_bytes": file_bytes(ROOT / common["checkpoint_dir"] / "ngram.pkl")}
        timing_device = None
    else:
        run_name = args.run
        model, cfg, extra = load_checkpoint(ROOT / common["checkpoint_dir"] / f"{run_name}.pt", device)
        ev = cfg["eval"]
        ctx = ev["transformer_context"]
        protocols = {"default": lambda s: score_model(model, s, device, cfg)}
        if model.is_recurrent:
            default_name = "full history, state carried, batch 1"
            protocols[f"blocks{ctx}"] = lambda s: score_windows(model, s, device, ctx, ctx)
        else:
            default_name = f"sliding window context {ctx} stride {ev['transformer_stride']}"
            protocols[f"blocks{ctx}"] = lambda s: score_windows(model, s, device, ctx, ctx)
        size = {**count_params(model),
                "checkpoint_bytes": file_bytes(ROOT / common["checkpoint_dir"] / f"{run_name}.pt")}
        timing_device = device
    run_dir = ensure_dir(ROOT / common["results_dir"] / "runs" /
                         (run_name + "_debug" if args.limit_tokens else run_name))

    # ---- test perplexity under each protocol (default protocol timed after a warm-up)
    warm = test[: min(len(test), 20_000)]
    protocols["default"](warm)  # warm-up (shader compilation, caches)
    results = {"run_name": run_name, "default_protocol": default_name, "protocols": {}}
    timing = {"run_name": run_name, "device": str(device) if not args.ngram else "cpu",
              "hardware": hardware_info(), "size": size}
    for name, fn in protocols.items():
        with Timer(timing_device) as t:
            lp = fn(test)
        results["protocols"][name] = summarize(lp, test, vocab)
        results["protocols"][name]["seconds"] = t.elapsed
        if name == "default":
            timing["score_test"] = {"tokens": int(len(test) - 1), "seconds": t.elapsed,
                                    "tokens_per_second": (len(test) - 1) / t.elapsed}
        print(f"[{run_name}] {name}: ppl {results['protocols'][name]['perplexity']:.2f} "
              f"({t.elapsed:.1f}s)")
    dump_json(results, run_dir / "eval.json")

    # ---- generation: same prompts, same decoding, same lengths for every model
    if not args.skip_generation:
        gens = []
        gen_seconds, gen_tokens = 0.0, 0
        for pi, p in enumerate(prompts):
            prompt_ids = vocab.encode(p["tokens"])
            entry = {"prompt": " ".join(p["tokens"]), "validation_line": p["validation_line"]}
            for mode in ("greedy", "topk"):
                greedy = mode == "greedy"
                rng = np.random.default_rng(gen_cfg["gen_seed"] + pi)
                with Timer(timing_device) as t:
                    if args.ngram:
                        ids = generate_ngram(model, prompt_ids, gen_cfg["max_new_tokens"], rng,
                                             gen_cfg["top_k"], gen_cfg["temperature"], greedy)
                    else:
                        ids = generate_neural(model, prompt_ids, gen_cfg["max_new_tokens"], device, rng,
                                              gen_cfg["top_k"], gen_cfg["temperature"], greedy,
                                              context=cfg["eval"]["transformer_context"])
                cont = ids[len(prompt_ids):]
                entry[mode] = {"text": vocab.decode(cont), "seconds": t.elapsed,
                               **repetition_metrics([vocab.itos[i] for i in cont])}
                if mode == "topk":
                    gen_seconds += t.elapsed
                    gen_tokens += len(cont)
            gens.append(entry)
            print(f"[{run_name}] {entry['prompt']} ||| {entry['topk']['text'][:120]}")
        timing["generate"] = {"prompts": len(prompts), "new_tokens_per_prompt": gen_cfg["max_new_tokens"],
                              "decoding": f"top-k {gen_cfg['top_k']}, T={gen_cfg['temperature']}",
                              "seconds": gen_seconds, "ms_per_token": 1000 * gen_seconds / gen_tokens,
                              "tokens_per_second": gen_tokens / gen_seconds}
        dump_json({"run_name": run_name, "decoding": gen_cfg, "generations": gens},
                  run_dir / "generations.json")
    dump_json(timing, run_dir / "timing.json")
    print(f"[{run_name}] score {timing['score_test']['tokens_per_second']:.0f} tok/s; "
          + (f"generate {timing['generate']['ms_per_token']:.1f} ms/token" if "generate" in timing else ""))


if __name__ == "__main__":
    main()
