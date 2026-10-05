"""Every neural LM must predict each position from preceding tokens only."""
import torch

from lm.models import build_model

BASE = {"emb_dim": 32, "dropout": 0.0, "bptt": 16}
CONFIGS = {
    "rnn": {"model": {"type": "rnn", "hidden": 32, "n_layers": 2}},
    "lstm": {"model": {"type": "lstm", "hidden": 32, "n_layers": 2}},
    "transformer": {"model": {"type": "transformer", "n_layers": 2, "n_heads": 4, "ff_dim": 64, "max_len": 16}},
}


def _logits(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)[0]


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    V, B, T, cut = 50, 3, 16, 7
    for name, c in CONFIGS.items():
        model = build_model({**BASE, **c}, V)
        x = torch.randint(0, V, (B, T))
        x2 = x.clone()
        x2[:, cut:] = torch.randint(0, V, (B, T - cut))
        a, b = _logits(model, x), _logits(model, x2)
        assert torch.allclose(a[:, :cut], b[:, :cut], atol=1e-5), f"{name} leaks future tokens"
        # and changing the future must change at least one future logit (sanity)
        assert not torch.allclose(a[:, cut:], b[:, cut:]), f"{name} ignores its input"


def test_output_and_embedding_are_not_tied():
    for name, c in CONFIGS.items():
        model = build_model({**BASE, **c}, 50)
        assert model.out.weight.data_ptr() != model.embed.weight.data_ptr(), name
