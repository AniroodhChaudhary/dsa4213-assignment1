#!/usr/bin/env bash
# Reproduce every number in results/ from scratch (Apple M4, ~7 h wall-clock incl. the 24-epoch runs).
# Each step is idempotent; comment out what you have already run.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0

uv sync                                                    # environment (pyproject + uv.lock)
bash scripts/download_glove.sh                             # data/raw/glove.6B.300d.txt (822 MB zip)
uv run scripts/prepare_data.py                             # vocab, streams, AG News arrays, prompts
uv run pytest -q                                           # causality / KN / pooling / scoring tests

# ---- Part I: four language models on WikiText-2 (+ parameter-count control)
uv run scripts/train_ngram.py
for c in rnn lstm transformer transformer_1layer; do uv run scripts/train_lm.py --config configs/$c.yaml; done

# ---- Part II: embedding tables and the 3 x 2 (source x trainability) LSTM factorial
uv run scripts/train_word2vec.py                           # own skip-gram vectors (timed separately)
uv run scripts/build_embedding_matrix.py                   # GloVe-6B aligned to the vocab
for c in lstm_skipgram_frozen lstm_glove_frozen lstm_glove_train lstm_random_frozen lstm_skipgram_train; do
  uv run scripts/train_lm.py --config configs/$c.yaml
done
for c in lstm lstm_skipgram_frozen lstm_glove_frozen; do uv run scripts/train_lm.py --config configs/$c.yaml --seed 1; done

# ---- Evaluation: common test protocol, inference workload, matched generations
uv run scripts/evaluate_lm.py --ngram
for r in rnn_s0 lstm_s0 transformer_s0 transformer_1layer_s0 \
         lstm_skipgram_frozen_s0 lstm_glove_frozen_s0 lstm_glove_train_s0 lstm_random_frozen_s0 lstm_skipgram_train_s0 \
         lstm_s1 lstm_skipgram_frozen_s1 lstm_glove_frozen_s1; do
  uv run scripts/evaluate_lm.py --run $r
done

# ---- Convergence check for Part I (optional, ~2 h): the LSTM and Transformer for 24 epochs
for c in lstm transformer; do uv run scripts/train_lm.py --config configs/$c.yaml --max_epochs 24 --run_name ${c}_e24_s0; done
for r in lstm_e24_s0 transformer_e24_s0; do uv run scripts/evaluate_lm.py --run $r --skip_generation; done

# ---- Part III: representations -> logistic regression on AG News
uv run scripts/extract_features.py
uv run scripts/train_classifier.py

# ---- Report assets and PDF
uv run scripts/make_report_assets.py
bash report/build.sh
