"""Fixed-length document representations for Part III.

* `lm_features`        top-layer hidden states of a frozen LM, pooled either as
                        the mean over non-pad positions or the state at the last
                        real token. Inputs are right-padded; because every model
                        is causal, real positions never see the padding.
* `embedding_features` mean of the raw input-embedding rows (no LM), a cheap
                        baseline isolating what the sequence model adds.
* `tfidf_features`     TF-IDF over the same vocabulary ids (uni+bigrams), the
                        classic non-neural reference point.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def lm_features(model, ids: np.ndarray, lens: np.ndarray, device: torch.device,
                pooling: str, batch: int = 128) -> np.ndarray:
    assert pooling in ("mean", "last")
    model.eval()
    n = len(ids)
    out = None
    for b in range(0, n, batch):
        ln_np = lens[b : b + batch]
        Lb = int(ln_np.max())
        x = torch.as_tensor(ids[b : b + batch, :Lb], dtype=torch.long, device=device)
        ln = torch.as_tensor(ln_np, dtype=torch.long, device=device)
        h, _ = model.hidden_states(x)  # (B, Lb, d)
        if pooling == "mean":
            mask = (torch.arange(Lb, device=device)[None, :] < ln[:, None]).to(h.dtype)
            f = (h * mask[..., None]).sum(1) / ln[:, None].to(h.dtype)
        else:
            f = h[torch.arange(h.size(0), device=device), ln - 1]
        f = f.float().cpu().numpy()
        if out is None:
            out = np.zeros((n, f.shape[1]), dtype=np.float32)
        out[b : b + batch] = f
    return out


def embedding_features(matrix: np.ndarray, ids: np.ndarray, lens: np.ndarray) -> np.ndarray:
    out = np.zeros((len(ids), matrix.shape[1]), dtype=np.float32)
    for i in range(len(ids)):
        out[i] = matrix[ids[i, : lens[i]]].mean(axis=0)
    return out


def _docs_as_strings(ids: np.ndarray, lens: np.ndarray) -> list[str]:
    # drop the leading <eos> (position 0) and the padding
    return [" ".join(f"t{t}" for t in ids[i, 1 : lens[i]]) for i in range(len(ids))]


def tfidf_features(train: tuple, others: list[tuple]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, lowercase=False)
    Xtr = vec.fit_transform(_docs_as_strings(*train))
    return [Xtr] + [vec.transform(_docs_as_strings(*o)) for o in others], len(vec.vocabulary_)
