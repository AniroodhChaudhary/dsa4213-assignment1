#!/usr/bin/env bash
# Compile the report with tectonic (https://tectonic-typesetting.github.io).
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-$(command -v tectonic || echo /opt/homebrew/bin/tectonic)}
"$TECTONIC" -X compile main.tex --keep-logs 2>&1 | grep -vE "^(warning: |note: )" || true
if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo main.pdf | grep -E "^Pages"
else
  ls -la main.pdf
fi
