#!/usr/bin/env bash
# Final-evaluation step (1): retrain the frozen-encoder (heads-only)
# baseline for each of the two final base models, on the current train
# set (data/processed/train.jsonl, i.e. after the section-4
# passB-default rebuild), lr 1e-3, same early stopping as the section-2
# improvement round (up to 25 epochs, patience 4 on val mean macro-F1).
# val only; data/processed/test.jsonl is never loaded by this script.
#
# Final models (docs/experiment-log.md section 6):
#   Small = google/electra-small-discriminator  -> mwo-triage-13M (electra_distilled)
#   Tiny  = google/bert_uncased_L-4_H-128_A-2    -> mwo-triage-5M  (L4_a_earlystop_passb)
#
# These frozen baselines are the comparison point for each final model
# against its own base model, frozen, with a simple head (CLAUDE.md
# locked decision), and feed into scripts/evaluate.py.
#
# No API calls. Run in the background:
#   nohup bash scripts/train_frozen_final.sh > outputs/bakeoff/frozen_final.log 2>&1 &
# (each underlying train.py invocation still runs with python -u)

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/bakeoff"
MAX_EPOCHS=25
PATIENCE=4
LR="1e-3"

ELECTRA_ID="google/electra-small-discriminator"
ELECTRA_CACHE="$HOME/models/google__electra-small-discriminator"

L4_ID="google/bert_uncased_L-4_H-128_A-2"
L4_CACHE="$HOME/models/google__bert_uncased_L-4_H-128_A-2"

RUN_START=$(date +%s)
echo "=== Frozen-baseline final retrain started at $(date) ==="

echo "--- electra_frozen_final ---"
python -u scripts/train.py --model-id "$ELECTRA_ID" --cache-dir "$ELECTRA_CACHE" --lr "$LR" \
  --freeze-encoder --epochs "$MAX_EPOCHS" --early-stopping-patience "$PATIENCE" \
  --run-name electra_frozen_final --output-dir "$OUTPUT_DIR"

echo "--- L4_frozen_final ---"
python -u scripts/train.py --model-id "$L4_ID" --cache-dir "$L4_CACHE" --lr "$LR" \
  --freeze-encoder --epochs "$MAX_EPOCHS" --early-stopping-patience "$PATIENCE" \
  --run-name L4_frozen_final --output-dir "$OUTPUT_DIR"

RUN_END=$(date +%s)
echo "=== Frozen-baseline final retrain finished at $(date), $((RUN_END - RUN_START))s total ==="
