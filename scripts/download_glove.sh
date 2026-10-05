#!/usr/bin/env bash
# Fetch GloVe 6B (Pennington et al., 2014) and extract the 300d table.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ ! -f data/raw/glove.6B.300d.txt ]; then
  curl -L --retry 3 -o data/raw/glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
  unzip -o -q data/raw/glove.6B.zip glove.6B.300d.txt -d data/raw
fi
ls -la data/raw/glove.6B.300d.txt
