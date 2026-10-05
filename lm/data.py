"""WikiText-2 and AG News loading, vocabulary, token streams and batching.

Design choices (documented in the report):
* Everything is lowercased once at preprocessing so that the uncased GloVe-6B
  vectors in Part II are compared on equal footing with the other settings.
* The vocabulary is built from the WikiText-2 *train* split only. `<eos>` is
  appended to every non-empty line, `<unk>` is the dataset's own token, and
  `<pad>` (last id) exists only for the padded batches of Part III.
* Language-model training and evaluation use one continuous token stream per
  split, so no padding ever enters the LM loss.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

EOS, UNK, PAD = "<eos>", "<unk>", "<pad>"
BUCKET_EDGES = (10, 100, 1000, 10000)
BUCKET_NAMES = ("1-10", "11-100", "101-1k", "1k-10k", ">10k")
AG_LABELS = ("World", "Sports", "Business", "Sci/Tech")


def bucket_index(count: int) -> int:
    """Train-frequency bucket of a vocabulary item (-1 for never-seen `<pad>`)."""
    if count <= 0:
        return -1
    for i, edge in enumerate(BUCKET_EDGES):
        if count <= edge:
            return i
    return len(BUCKET_EDGES)


class Vocab:
    def __init__(self, itos: list[str], counts: list[int]):
        self.itos = list(itos)
        self.counts = [int(c) for c in counts]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]
        self.pad_id = self.stoi[PAD]
        self.buckets = np.array([bucket_index(c) for c in self.counts], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, train_lines: list[list[str]]) -> "Vocab":
        c: Counter = Counter()
        for toks in train_lines:
            c.update(toks)
        n_eos = len(train_lines) + 1  # one per line plus the stream prefix
        c.pop(EOS, None)
        unk_count = c.pop(UNK, 0)
        rest = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        itos = [EOS, UNK] + [t for t, _ in rest] + [PAD]
        counts = [n_eos, unk_count] + [n for _, n in rest] + [0]
        return cls(itos, counts)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"itos": self.itos, "counts": self.counts}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path) as f:
            d = json.load(f)
        return cls(d["itos"], d["counts"])

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


# ----------------------------------------------------------------------------
# WikiText-2
# ----------------------------------------------------------------------------

def load_wikitext_lines(split: str) -> list[list[str]]:
    """Non-empty, lowercased, whitespace-tokenised lines of a split
    (split in {"train", "validation", "test"})."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    lines = []
    for text in ds["text"]:
        toks = text.strip().lower().split()
        if toks:
            lines.append(toks)
    return lines


def encode_stream(lines: list[list[str]], vocab: Vocab) -> tuple[np.ndarray, int]:
    """Concatenate lines into one int32 stream: <eos> tok ... tok <eos> ... .
    Returns the stream and the number of tokens that fell back to <unk>."""
    ids = [vocab.eos_id]
    n_oov = 0
    stoi, unk = vocab.stoi, vocab.unk_id
    for toks in lines:
        for t in toks:
            i = stoi.get(t)
            if i is None:
                i = unk
                n_oov += 1
            ids.append(i)
        ids.append(vocab.eos_id)
    return np.asarray(ids, dtype=np.int32), n_oov


def load_stream(processed_dir: str | Path, split: str) -> np.ndarray:
    return np.load(Path(processed_dir) / "wikitext" / f"{split}.npy")


def batchify(stream: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    """Reshape a stream into (batch_size, n) where each row is a contiguous
    segment of the stream, so hidden state can be carried across bptt chunks."""
    n = len(stream) // batch_size
    data = torch.as_tensor(stream[: n * batch_size], dtype=torch.long).view(batch_size, n)
    return data.to(device)


def iter_bptt(data: torch.Tensor, bptt: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (inputs, targets) chunks of length <= bptt along the time axis."""
    n = data.size(1)
    for i in range(0, n - 1, bptt):
        length = min(bptt, n - 1 - i)
        yield data[:, i : i + length], data[:, i + 1 : i + 1 + length]


def n_bptt_steps(n_tokens_per_row: int, bptt: int) -> int:
    return (n_tokens_per_row - 2) // bptt + 1


# ----------------------------------------------------------------------------
# AG News
# ----------------------------------------------------------------------------

def load_agnews(split: str) -> tuple[list[str], np.ndarray]:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split=split)
    return list(ds["text"]), np.asarray(ds["label"], dtype=np.int64)


def encode_agnews(texts: list[str], vocab: Vocab, tokenizer, max_len: int):
    """Tokenise, prepend <eos>, map to the WikiText vocab, truncate and right-pad.
    Returns ids (N, max_len) int32, lengths (N,) int32 and coverage stats."""
    n = len(texts)
    ids = np.full((n, max_len), vocab.pad_id, dtype=np.int32)
    lens = np.zeros(n, dtype=np.int32)
    n_tok = n_unk = n_trunc = 0
    unk_per_doc = np.zeros(n, dtype=np.float32)
    for i, text in enumerate(texts):
        enc = [vocab.eos_id] + vocab.encode(tokenizer(text))
        if len(enc) > max_len:
            n_trunc += 1
            enc = enc[:max_len]
        body = enc[1:]
        unk_here = sum(1 for t in body if t == vocab.unk_id)
        n_tok += len(body)
        n_unk += unk_here
        unk_per_doc[i] = unk_here / max(len(body), 1)
        ids[i, : len(enc)] = enc
        lens[i] = len(enc)
    stats = {
        "n_docs": n,
        "n_tokens": n_tok,
        "n_unk": n_unk,
        "unk_rate": n_unk / max(n_tok, 1),
        "in_vocab_rate": 1.0 - n_unk / max(n_tok, 1),
        "truncated_frac": n_trunc / n,
        "mean_len": float(lens.mean()),
    }
    return ids, lens, stats, unk_per_doc


def load_agnews_split(processed_dir: str | Path, split: str):
    d = Path(processed_dir) / "agnews"
    return (
        np.load(d / f"{split}_ids.npy"),
        np.load(d / f"{split}_lens.npy"),
        np.load(d / f"{split}_labels.npy"),
    )
