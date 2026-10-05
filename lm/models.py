"""Neural next-token language models behind one interface.

All three models share the same input embedding (V x d, randomly initialised
U(-0.1, 0.1) unless replaced by `set_embedding`) and the same output layer
(d x V, untied), so their parameter counts differ only in the sequence core.

    logits, state = model(x, state)     # x: (B, T) long -> logits (B, T, V)
    h = model.hidden_states(x)          # (B, T, d) top-layer states (Part III)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LMBase(nn.Module):
    is_recurrent = False

    def __init__(self, vocab_size: int, emb_dim: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        self.drop = nn.Dropout(dropout)

    def set_embedding(self, matrix: torch.Tensor, freeze: bool) -> None:
        """Replace the input embedding table (e.g. skip-gram or GloVe vectors)."""
        assert tuple(matrix.shape) == (self.vocab_size, self.emb_dim), matrix.shape
        self.embed = nn.Embedding.from_pretrained(matrix.clone().float(), freeze=freeze)

    def freeze_embedding(self) -> None:
        self.embed.weight.requires_grad_(False)

    def init_state(self, batch_size: int, device: torch.device):
        return None

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(s.detach() for s in state)
        return state.detach()

    def _init_output(self, hidden: int) -> None:
        self.out = nn.Linear(hidden, self.vocab_size)
        nn.init.uniform_(self.out.weight, -0.1, 0.1)
        nn.init.zeros_(self.out.bias)


class RecurrentLM(LMBase):
    """Vanilla tanh RNN or LSTM language model."""

    is_recurrent = True

    def __init__(self, vocab_size, emb_dim, hidden, n_layers, dropout, cell: str):
        super().__init__(vocab_size, emb_dim, dropout)
        assert cell in ("rnn", "lstm")
        kwargs = dict(batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        if cell == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.rnn = (nn.RNN if cell == "rnn" else nn.LSTM)(emb_dim, hidden, n_layers, **kwargs)
        self.cell, self.hidden, self.n_layers = cell, hidden, n_layers
        self._init_output(hidden)

    def init_state(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden, device=device)
        return (h, torch.zeros_like(h)) if self.cell == "lstm" else h

    def hidden_states(self, x, state=None):
        h, state = self.rnn(self.drop(self.embed(x)), state)
        return h, state

    def forward(self, x, state=None):
        h, state = self.hidden_states(x, state)
        return self.out(self.drop(h)), state


class TransformerLM(LMBase):
    """Pre-LN Transformer decoder (causal self-attention, learned positions)."""

    def __init__(self, vocab_size, emb_dim, n_layers, n_heads, ff_dim, max_len, dropout):
        super().__init__(vocab_size, emb_dim, dropout)
        self.max_len = max_len
        self.pos = nn.Embedding(max_len, emb_dim)
        nn.init.uniform_(self.pos.weight, -0.1, 0.1)
        layer = nn.TransformerEncoderLayer(
            emb_dim, n_heads, ff_dim, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(emb_dim)
        self._init_output(emb_dim)

    def hidden_states(self, x, state=None):
        B, T = x.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.embed(x) + self.pos(pos))
        # float mask: -inf above the diagonal so position t attends to <= t only
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.blocks(h, mask=mask)
        return self.ln_f(h), None

    def forward(self, x, state=None):
        h, _ = self.hidden_states(x)
        return self.out(self.drop(h)), None


def build_model(cfg: dict, vocab_size: int) -> LMBase:
    m = cfg["model"]
    t = m["type"]
    if t in ("rnn", "lstm"):
        return RecurrentLM(vocab_size, cfg["emb_dim"], m["hidden"], m["n_layers"], cfg["dropout"], cell=t)
    if t == "transformer":
        return TransformerLM(
            vocab_size, cfg["emb_dim"], m["n_layers"], m["n_heads"], m["ff_dim"],
            m.get("max_len", cfg["bptt"]), cfg["dropout"],
        )
    raise ValueError(f"unknown model type {t}")


def save_checkpoint(path, model: LMBase, cfg: dict, extra: dict) -> None:
    torch.save({"config": cfg, "state_dict": model.state_dict(),
                "vocab_size": model.vocab_size, **extra}, path)


def load_checkpoint(path, device: torch.device) -> tuple[LMBase, dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"], ck["vocab_size"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    extra = {k: v for k, v in ck.items() if k not in ("config", "state_dict", "vocab_size")}
    return model, ck["config"], extra
