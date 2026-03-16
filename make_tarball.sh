#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_TAR="${SCRIPT_DIR}/final_submission.tar.gz"

cd "${SCRIPT_DIR}"
rm -f "${OUTPUT_TAR}"

tar -czf "${OUTPUT_TAR}" \
  --exclude='.venv' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='*.egg-info' \
  --exclude='__pycache__' \
  --exclude='attention_ext/*.so' \
  --exclude='attention_ext/_C*.so' \
  468\ Project\ Handout.pdf \
  Proposal.pdf \
  codex_instructions.md \
  baseline_attention.py \
  official_attention.py \
  bench.py \
  plot.py \
  README.md \
  workflow.md \
  requirements.txt \
  setup.py \
  make_tarball.sh \
  attention_ext \
  results

echo "Wrote ${OUTPUT_TAR}"
