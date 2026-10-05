"""Training loop shared by every neural language-model run.

Recipe (identical across architectures and embedding settings): Adam with a
linear warm-up, halve the learning rate when validation perplexity does not
improve for `plateau_patience` epochs, stop after `early_stop_patience` epochs
without improvement, clip the global gradient norm and log the pre-clip norm at
every step. Validation uses the same scoring protocol as the final test
evaluation (full history for recurrent models, sliding window for the
Transformer), so the curves and the reported numbers are directly comparable.
"""
from __future__ import annotations

import csv
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Vocab, batchify, iter_bptt, load_stream, n_bptt_steps
from .evaluate import score_model, summarize
from .models import build_model, save_checkpoint
from .timing import Timer, count_params, file_bytes, hardware_info
from .utils import dump_json, ensure_dir, set_seed


def embedding_hash(model) -> str:
    return hashlib.sha256(model.embed.weight.detach().cpu().numpy().tobytes()).hexdigest()


def prepare_model(cfg: dict, vocab_size: int, root: Path):
    """Build the model and apply the run's embedding setting."""
    model = build_model(cfg, vocab_size)
    emb = cfg.get("embedding", {"source": "random", "freeze": False})
    if emb["source"] != "random":
        matrix = torch.from_numpy(np.load(root / emb["matrix_path"]))
        model.set_embedding(matrix, freeze=emb["freeze"])
    elif emb.get("freeze", False):
        model.freeze_embedding()
    return model


def run_training(cfg: dict, device: torch.device, root: Path, max_steps: int | None = None) -> dict:
    set_seed(cfg["seed"])
    run_dir = ensure_dir(root / cfg["results_dir"] / "runs" / cfg["run_name"])
    ckpt_dir = ensure_dir(root / cfg["checkpoint_dir"])
    ckpt_path = ckpt_dir / f"{cfg['run_name']}.pt"
    dump_json(cfg, run_dir / "config.json")

    vocab = Vocab.load(root / cfg["data_dir"] / "wikitext" / "vocab.json")
    train_stream = load_stream(root / cfg["data_dir"], "train")
    val_stream = load_stream(root / cfg["data_dir"], "validation")
    V = len(vocab)

    model = prepare_model(cfg, V, root).to(device)
    emb_frozen = not model.embed.weight.requires_grad
    emb_hash_before = embedding_hash(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"], betas=tuple(cfg["betas"]))
    pcount = count_params(model)
    print(f"[{cfg['run_name']}] params: {pcount}  embedding frozen: {emb_frozen}  device: {device}")

    data = batchify(train_stream, cfg["batch_size"], device)
    steps_per_epoch = n_bptt_steps(data.size(1), cfg["bptt"])
    base_lr, warmup = cfg["lr"], cfg["warmup_steps"]

    step_f = open(run_dir / "steps.csv", "w", newline="")
    step_w = csv.writer(step_f)
    step_w.writerow(["step", "epoch", "loss", "grad_norm", "lr", "seconds"])
    history = []
    best_val, best_epoch, bad_epochs, since_lr_drop = math.inf, 0, 0, 0
    lr = base_lr
    step = 0
    t_start = time.perf_counter()
    cum_train_seconds = 0.0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        state = model.init_state(cfg["batch_size"], device)
        losses, gnorms = [], []
        with Timer(device) as t_epoch:
            for x, y in iter_bptt(data, cfg["bptt"]):
                step += 1
                cur_lr = lr * min(1.0, step / warmup) if warmup > 0 else lr
                for g in opt.param_groups:
                    g["lr"] = cur_lr
                state = model.detach_state(state)
                logits, state = model(x, state)
                loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]).item()
                opt.step()
                li = loss.item()
                losses.append(li)
                gnorms.append(gn)
                step_w.writerow([step, epoch, f"{li:.5f}", f"{gn:.4f}", f"{cur_lr:.6f}",
                                 f"{time.perf_counter() - t_start:.2f}"])
                if max_steps is not None and step >= max_steps:
                    break
        cum_train_seconds += t_epoch.elapsed
        train_loss = float(np.mean(losses))
        with Timer(device) as t_val:
            val_lp = score_model(model, val_stream, device, cfg)
        val = summarize(val_lp, val_stream, vocab)
        gn_arr = np.asarray(gnorms)
        row = {
            "epoch": epoch, "steps": step, "train_loss": train_loss,
            "train_ppl": math.exp(train_loss), "val_loss": val["cross_entropy"],
            "val_ppl": val["perplexity"], "lr": cur_lr,
            "epoch_seconds": t_epoch.elapsed, "val_seconds": t_val.elapsed,
            "cum_train_seconds": cum_train_seconds,
            "grad_norm_mean": float(gn_arr.mean()), "grad_norm_max": float(gn_arr.max()),
            "clipped_frac": float((gn_arr > cfg["grad_clip"]).mean()),
            "step_loss_std": float(np.std(losses)),
        }
        history.append(row)
        print(f"[{cfg['run_name']}] epoch {epoch:2d} train {math.exp(train_loss):8.2f} "
              f"val {val['perplexity']:8.2f} lr {cur_lr:.2e} "
              f"({t_epoch.elapsed:.0f}s train, {t_val.elapsed:.0f}s val, gn {gn_arr.mean():.2f})")
        improved = val["perplexity"] < best_val
        if improved:
            best_val, best_epoch, bad_epochs, since_lr_drop = val["perplexity"], epoch, 0, 0
            save_checkpoint(ckpt_path, model, cfg, {"best_epoch": epoch, "val_ppl": best_val,
                                                    "val_summary": val})
        else:
            bad_epochs += 1
            since_lr_drop += 1
            if since_lr_drop > cfg["plateau_patience"]:
                lr *= cfg["plateau_factor"]
                since_lr_drop = 0
                print(f"[{cfg['run_name']}] lr -> {lr:.2e}")
        if bad_epochs >= cfg["early_stop_patience"]:
            print(f"[{cfg['run_name']}] early stop at epoch {epoch}")
            break
        if max_steps is not None and step >= max_steps:
            break
    step_f.close()

    emb_hash_after = embedding_hash(model)
    if emb_frozen:
        assert emb_hash_after == emb_hash_before, "frozen embedding table changed!"
    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    epoch_secs = [h["epoch_seconds"] for h in history]
    metrics = {
        "run_name": cfg["run_name"], "seed": cfg["seed"], "device": str(device),
        "model": cfg["model"], "embedding": cfg.get("embedding"),
        "params": pcount, "embedding_frozen": emb_frozen,
        "embedding_hash_before": emb_hash_before, "embedding_hash_after": emb_hash_after,
        "embedding_mean_row_norm": float(model.embed.weight.detach().norm(dim=1).mean()),
        "steps_per_epoch": steps_per_epoch, "total_steps": step,
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_ppl": best_val,
        "train_seconds_total": cum_train_seconds,
        "train_seconds_to_best": float(sum(epoch_secs[:best_epoch])),
        "epoch1_seconds": epoch_secs[0],
        "epoch_seconds_mean_after_first": float(np.mean(epoch_secs[1:])) if len(epoch_secs) > 1 else None,
        "wall_seconds_total": time.perf_counter() - t_start,
        "checkpoint_bytes": file_bytes(ckpt_path) if ckpt_path.exists() else None,
        "hardware": hardware_info(),
        "history": history,
    }
    dump_json(metrics, run_dir / "metrics.json")
    return metrics
