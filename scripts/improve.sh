#!/usr/bin/env bash
# Improvement round on the two bake-off winners only (val only, test
# never touched):
#   (a) same model/lr as the winning bake-off run, up to 25 epochs with
#       early stopping (patience 4 on val mean macro-F1).
#   (b) same as (a), plus class-weighted cross-entropy on the priority
#       head (1/sqrt(train class frequency), mean-1 normalised).
#   (c) scripts/stop_now_threshold.py: for whichever of (a)/(b) scored
#       higher, a post-hoc STOP_NOW probability threshold tuned on val
#       for STOP_NOW recall >= 0.85, then prints the baseline/(a)/(b)/(c)
#       comparison table and the priority confusion matrix.
#
# Winners (docs/experiment-log.md bake-off section):
#   Small = google/electra-small-discriminator, lr 1e-4
#   Tiny  = google/bert_uncased_L-4_H-128_A-2,  lr 3e-4
#
# No API calls. Run in the background:
#   nohup bash scripts/improve.sh > outputs/bakeoff/improve.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/bakeoff"
MAX_EPOCHS=25
PATIENCE=4

ELECTRA_ID="google/electra-small-discriminator"
ELECTRA_CACHE="$HOME/models/google__electra-small-discriminator"
ELECTRA_LR="1e-4"

L4_ID="google/bert_uncased_L-4_H-128_A-2"
L4_CACHE="$HOME/models/google__bert_uncased_L-4_H-128_A-2"
L4_LR="3e-4"

RUN_START=$(date +%s)
echo "=== Improvement round started at $(date) ==="

echo "--- (a) electra_a_earlystop ---"
python -u scripts/train.py --model-id "$ELECTRA_ID" --cache-dir "$ELECTRA_CACHE" --lr "$ELECTRA_LR" \
  --epochs "$MAX_EPOCHS" --early-stopping-patience "$PATIENCE" \
  --run-name electra_a_earlystop --output-dir "$OUTPUT_DIR"

echo "--- (b) electra_b_classweighted ---"
python -u scripts/train.py --model-id "$ELECTRA_ID" --cache-dir "$ELECTRA_CACHE" --lr "$ELECTRA_LR" \
  --epochs "$MAX_EPOCHS" --early-stopping-patience "$PATIENCE" --class-weight-priority \
  --run-name electra_b_classweighted --output-dir "$OUTPUT_DIR"

echo "--- (a) L4_a_earlystop ---"
python -u scripts/train.py --model-id "$L4_ID" --cache-dir "$L4_CACHE" --lr "$L4_LR" \
  --epochs "$MAX_EPOCHS" --early-stopping-patience "$PATIENCE" \
  --run-name L4_a_earlystop --output-dir "$OUTPUT_DIR"

echo "--- (b) L4_b_classweighted ---"
python -u scripts/train.py --model-id "$L4_ID" --cache-dir "$L4_CACHE" --lr "$L4_LR" \
  --epochs "$MAX_EPOCHS" --early-stopping-patience "$PATIENCE" --class-weight-priority \
  --run-name L4_b_classweighted --output-dir "$OUTPUT_DIR"

RUN_END=$(date +%s)
echo "=== Improvement training finished at $(date), $((RUN_END - RUN_START))s total ==="

echo
echo "--- (c) post-hoc STOP_NOW threshold ---"
python -u scripts/stop_now_threshold.py
