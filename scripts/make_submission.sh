#!/usr/bin/env bash
# Build the submission ZIP from the committed tree (code, configs, tests, results,
# report source and PDF). Data and checkpoints are excluded; README explains how
# to regenerate them.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=DSA4213_A1_Aniroodh_Chaudhary_A0284770W
rm -rf "dist" && mkdir -p dist
git archive --format=zip --prefix="$NAME/" -o "dist/$NAME.zip" HEAD
cp report/main.pdf "dist/${NAME}_report.pdf"
( cd dist && mkdir -p "$NAME" && cp "${NAME}_report.pdf" "$NAME/" && zip -q "$NAME.zip" "$NAME/${NAME}_report.pdf" && rm -r "$NAME" )
ls -la dist
unzip -l "dist/$NAME.zip" | tail -1
