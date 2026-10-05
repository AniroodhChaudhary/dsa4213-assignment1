"""Matched decoding for every model family plus repetition statistics.

Sampling is done on the CPU with one numpy Generator so that the decoding
procedure is identical for the n-gram model and the neural models.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def sample_next(probs: np.ndarray, rng: np.random.Generator, top_k: int,
                temperature: float, greedy: bool) -> int:
    if greedy:
        return int(np.argmax(probs))
    logits = np.log(np.clip(probs, 1e-12, None)) / max(temperature, 1e-6)
    if top_k and top_k < len(logits):
        keep = np.argpartition(-logits, top_k)[:top_k]
        mask = np.full_like(logits, -np.inf)
        mask[keep] = logits[keep]
        logits = mask
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


@torch.no_grad()
def generate_neural(model, prompt: list[int], max_new_tokens: int, device: torch.device,
                    rng: np.random.Generator, top_k: int, temperature: float,
                    greedy: bool, context: int = 128) -> list[int]:
    model.eval()
    ids = list(prompt)
    if model.is_recurrent:
        state = model.init_state(1, device)
        x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, state = model(x, state)
        for _ in range(max_new_tokens):
            probs = F.softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
            nxt = sample_next(probs, rng, top_k, temperature, greedy)
            ids.append(nxt)
            x = torch.as_tensor([[nxt]], dtype=torch.long, device=device)
            logits, state = model(x, state)
    else:  # transformer: re-encode the (truncated) sequence each step, no KV cache
        for _ in range(max_new_tokens):
            x = torch.as_tensor(ids[-context:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            probs = F.softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
            ids.append(sample_next(probs, rng, top_k, temperature, greedy))
    return ids


def generate_ngram(model, prompt: list[int], max_new_tokens: int, rng: np.random.Generator,
                   top_k: int, temperature: float, greedy: bool) -> list[int]:
    ids = list(prompt)
    for _ in range(max_new_tokens):
        probs = model.next_token_distribution(ids)
        ids.append(sample_next(probs, rng, top_k, temperature, greedy))
    return ids


def repetition_metrics(tokens: list) -> dict:
    """distinct-n for n=1..3 and the fraction of 4-grams already seen earlier
    in the same continuation."""
    out = {}
    for n in (1, 2, 3):
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        out[f"distinct_{n}"] = len(set(grams)) / max(len(grams), 1)
    grams4 = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    seen: Counter = Counter()
    rep = 0
    for g in grams4:
        if seen[g]:
            rep += 1
        seen[g] += 1
    out["repeated_4gram_frac"] = rep / max(len(grams4), 1)
    return out
