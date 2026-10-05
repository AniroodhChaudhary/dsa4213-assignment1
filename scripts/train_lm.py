"""Train one neural language model from a YAML config.

    uv run scripts/train_lm.py --config configs/lstm.yaml [--seed 1] [--max_steps 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.train import run_training  # noqa: E402
from lm.utils import get_device, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test: stop after N steps")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    cfg["run_name"] = args.run_name or f"{Path(args.config).stem}_s{cfg['seed']}"
    if args.max_steps is not None:
        cfg["run_name"] = f"smoke_{cfg['run_name']}"
    device = get_device(cfg["device"])
    m = run_training(cfg, device, ROOT, max_steps=args.max_steps)
    print(f"done: best val ppl {m['best_val_ppl']:.2f} at epoch {m['best_epoch']} "
          f"in {m['train_seconds_total']:.0f}s training")


if __name__ == "__main__":
    main()
