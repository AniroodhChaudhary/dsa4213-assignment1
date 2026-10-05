"""Common scoring protocols producing one log-probability per stream target.

Every protocol scores exactly the N-1 targets stream[1:] of a split, so
perplexities are comparable across n-gram, RNN, LSTM and Transformer:

* full      recurrent models, batch 1, hidden state carried over the whole
            stream (every target sees its complete history);
* windows   context window of `context` tokens moved by `stride`; window 0
            scores all its targets, later windows only their last `stride`
            targets. stride == context gives non-overlapping blocks (context
            reset), stride < context gives the Transformer's sliding window.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import BUCKET_NAMES, Vocab


@torch.no_grad()
def _target_logprobs(model, x: torch.Tensor, y: torch.Tensor, state=None):
    logits, state = model(x, state)
    lp = F.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return lp, state


@torch.no_grad()
def score_full(model, stream: np.ndarray, device: torch.device, chunk: int = 512) -> np.ndarray:
    model.eval()
    s = torch.as_tensor(stream, dtype=torch.long, device=device).unsqueeze(0)
    n = s.size(1)
    state = model.init_state(1, device)
    out = []
    for i in range(0, n - 1, chunk):
        y = s[:, i + 1 : i + 1 + chunk]
        x = s[:, i : i + y.size(1)]
        lp, state = _target_logprobs(model, x, y, state)
        out.append(lp.squeeze(0).cpu())
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def score_windows(model, stream: np.ndarray, device: torch.device, context: int,
                  stride: int, batch: int = 16) -> np.ndarray:
    assert 1 <= stride <= context
    model.eval()
    s = torch.as_tensor(stream, dtype=torch.long)
    n_targets = len(stream) - 1
    out = np.full(n_targets, np.nan, dtype=np.float32)
    starts = list(range(0, n_targets, stride))
    # keep only windows that contribute unscored targets
    starts = [st for st in starts if st == 0 or st + (context - stride) < n_targets]
    full = [st for st in starts if st + context <= n_targets]
    tail = [st for st in starts if st + context > n_targets]
    for b in range(0, len(full), batch):
        chunk_starts = full[b : b + batch]
        idx = torch.as_tensor(chunk_starts).unsqueeze(1) + torch.arange(context).unsqueeze(0)
        x = s[idx].to(device)
        y = s[idx + 1].to(device)
        lp, _ = _target_logprobs(model, x, y)
        lp = lp.cpu().numpy()
        for row, st in enumerate(chunk_starts):
            first = 0 if st == 0 else context - stride
            out[st + first : st + context] = lp[row, first:]
    for st in tail:
        L = n_targets - st
        x = s[st : st + L].unsqueeze(0).to(device)
        y = s[st + 1 : st + 1 + L].unsqueeze(0).to(device)
        lp, _ = _target_logprobs(model, x, y)
        first = 0 if st == 0 else context - stride
        if first < L:
            out[st + first : st + L] = lp[0, first:].cpu().numpy()
    assert not np.isnan(out).any(), "window scoring left targets unscored"
    return out


def score_model(model, stream: np.ndarray, device: torch.device, cfg: dict,
                protocol: str | None = None) -> np.ndarray:
    """Default protocol per model family: `full` for recurrent models and the
    sliding window for the Transformer. `protocol` can force `blocks`."""
    ev = cfg["eval"]
    if protocol is None:
        protocol = "full" if model.is_recurrent else "sliding"
    if protocol == "full":
        return score_full(model, stream, device, chunk=ev["eval_chunk"])
    if protocol == "sliding":
        return score_windows(model, stream, device, ev["transformer_context"], ev["transformer_stride"])
    if protocol == "blocks":
        return score_windows(model, stream, device, ev["transformer_context"], ev["transformer_context"])
    raise ValueError(protocol)


def summarize(logprobs: np.ndarray, stream: np.ndarray, vocab: Vocab) -> dict:
    """Cross-entropy / perplexity overall, per train-frequency bucket and on
    the special tokens, for per-target log-probs aligned with stream[1:]."""
    targets = stream[1:]
    assert len(logprobs) == len(targets), (len(logprobs), len(targets))
    nll = -logprobs.astype(np.float64)
    ce = float(nll.mean())
    res = {"n_targets": int(len(targets)), "cross_entropy": ce, "perplexity": math.exp(ce)}
    b = vocab.buckets[targets]
    res["bucket_ce"] = {}
    res["bucket_frac"] = {}
    for i, name in enumerate(BUCKET_NAMES):
        m = b == i
        res["bucket_frac"][name] = float(m.mean())
        res["bucket_ce"][name] = float(nll[m].mean()) if m.any() else None
    for name, tid in (("eos", vocab.eos_id), ("unk", vocab.unk_id)):
        m = targets == tid
        res[f"{name}_ce"] = float(nll[m].mean()) if m.any() else None
        res[f"{name}_frac"] = float(m.mean())
    return res
