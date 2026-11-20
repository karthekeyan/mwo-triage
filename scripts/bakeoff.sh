#!/usr/bin/env bash
# Bake-off: for each of the six candidate base models in ~/models/, run
# two fine-tuning learning rates plus one frozen-encoder baseline (lr
# 1e-3, heads only). 6 models x 3 runs = 18 runs, each up to 10 epochs,
# evaluated on data/processed/val.jsonl; data/processed/test.jsonl is
# never touched by scripts/train.py.
#
# The two H-128 ("tiny") models use lr 1e-4 and 3e-4 for fine-tuning; the
# three H-256 ("small") models use lr 5e-5 and 1e-4. See CLAUDE.md
# section 2 for the Tiny/Small split rationale.
#
# No API calls. Each training run inside this script is launched with
# `python -u` for unbuffered, line-by-line logging; run the script itself
# in the background with nohup so a flaky network/session doesn't kill it:
#   nohup bash scripts/bakeoff.sh > bakeoff.log 2>&1 &
#
# Usage:
#   bash scripts/bakeoff.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="$HOME/models"
OUTPUT_DIR="outputs/bakeoff"

# model_id|cache_dir_name|short_name|group (tiny=H-128, small=H-256)
MODELS=(
  "google/bert_uncased_L-2_H-128_A-2|google__bert_uncased_L-2_H-128_A-2|L2|tiny"
  "google/bert_uncased_L-4_H-128_A-2|google__bert_uncased_L-4_H-128_A-2|L4|tiny"
  "google/bert_uncased_L-6_H-128_A-2|google__bert_uncased_L-6_H-128_A-2|L6|tiny"
  "google/electra-small-discriminator|google__electra-small-discriminator|electra|small"
  "google/bert_uncased_L-8_H-256_A-4|google__bert_uncased_L-8_H-256_A-4|L8|small"
  "google/bert_uncased_L-12_H-256_A-4|google__bert_uncased_L-12_H-256_A-4|L12|small"
)

TINY_LRS=(1e-4 3e-4)
SMALL_LRS=(5e-5 1e-4)
FROZEN_LR=1e-3

RUN_START=$(date +%s)
echo "=== Bake-off started at $(date) ==="

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_id cache_name short_name group <<< "$entry"
  cache_dir="$MODELS_DIR/$cache_name"

  if [[ "$group" == "tiny" ]]; then
    lrs=("${TINY_LRS[@]}")
  else
    lrs=("${SMALL_LRS[@]}")
  fi

  for lr in "${lrs[@]}"; do
    run_name="${short_name}_lr${lr}"
    echo "--- run: $run_name ($model_id, lr=$lr) ---"
    python -u scripts/train.py \
      --model-id "$model_id" \
      --cache-dir "$cache_dir" \
      --lr "$lr" \
      --run-name "$run_name" \
      --output-dir "$OUTPUT_DIR"
  done

  run_name="${short_name}_frozen"
  echo "--- run: $run_name ($model_id, frozen, lr=$FROZEN_LR) ---"
  python -u scripts/train.py \
    --model-id "$model_id" \
    --cache-dir "$cache_dir" \
    --lr "$FROZEN_LR" \
    --freeze-encoder \
    --run-name "$run_name" \
    --output-dir "$OUTPUT_DIR"
done

RUN_END=$(date +%s)
echo "=== Bake-off finished at $(date), $((RUN_END - RUN_START))s total ==="

echo
echo "=== Summary ==="
python -u scripts/bakeoff_report.py
