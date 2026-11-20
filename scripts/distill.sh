#!/usr/bin/env bash
# Knowledge distillation round: val only, test never touched.
#   (1) fine-tune microsoft/deberta-v3-base as the teacher (two-head
#       setup, lr 3e-5, batch 32, max_length 32, up to 10 epochs, early
#       stopping patience 3 on val mean macro-F1).
#   (2) export the teacher's temperature-2 soft probabilities for both
#       heads on every train text.
#   (3) retrain ELECTRA-small (lr 1e-4) and BERT L-4/H-128 (lr 3e-4)
#       with a distillation loss per head: 0.5 x hard-label CE + 0.5 x
#       KL(student || teacher) at temperature 2, scaled by T^2; same
#       early stopping (patience 4, up to 25 epochs) as the (a) runs.
#   (4) print the comparison table.
#
# --amp-dtype fp16 (the default) needs fp32 master weights, which
# src/mwo_triage/classifier.py now forces regardless of a checkpoint's
# stored dtype (microsoft/deberta-v3-base ships in fp16, which otherwise
# conflicts with the fp32 classification heads).
#
# No API calls. Run in the background:
#   nohup bash scripts/distill.sh > outputs/bakeoff/distill.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/bakeoff"
TEACHER_ID="microsoft/deberta-v3-base"
TEACHER_CACHE="$HOME/models/microsoft__deberta-v3-base"
TEACHER_PROBS="data/interim/teacher_train_probs.jsonl"

ELECTRA_ID="google/electra-small-discriminator"
ELECTRA_CACHE="$HOME/models/google__electra-small-discriminator"
ELECTRA_LR="1e-4"

L4_ID="google/bert_uncased_L-4_H-128_A-2"
L4_CACHE="$HOME/models/google__bert_uncased_L-4_H-128_A-2"
L4_LR="3e-4"

RUN_START=$(date +%s)
echo "=== Distillation round started at $(date) ==="

echo "--- (1) teacher: teacher_deberta ---"
python -u scripts/train.py --model-id "$TEACHER_ID" --cache-dir "$TEACHER_CACHE" \
  --lr 3e-5 --batch-size 32 --max-length 32 --epochs 10 --early-stopping-patience 3 \
  --run-name teacher_deberta --output-dir "$OUTPUT_DIR"

echo "--- (2) export teacher soft probabilities (T=2) on train ---"
python -u scripts/export_teacher_probs.py --model-id "$TEACHER_ID" --cache-dir "$TEACHER_CACHE" \
  --checkpoint "$OUTPUT_DIR/teacher_deberta/best_model.pt" --output "$TEACHER_PROBS" --temperature 2.0

echo "--- (3a) distill ELECTRA-small: electra_distilled ---"
python -u scripts/train.py --model-id "$ELECTRA_ID" --cache-dir "$ELECTRA_CACHE" --lr "$ELECTRA_LR" \
  --epochs 25 --early-stopping-patience 4 \
  --distill-teacher-probs "$TEACHER_PROBS" --distill-alpha 0.5 --distill-temperature 2.0 \
  --run-name electra_distilled --output-dir "$OUTPUT_DIR"

echo "--- (3b) distill BERT L-4/H-128: L4_distilled ---"
python -u scripts/train.py --model-id "$L4_ID" --cache-dir "$L4_CACHE" --lr "$L4_LR" \
  --epochs 25 --early-stopping-patience 4 \
  --distill-teacher-probs "$TEACHER_PROBS" --distill-alpha 0.5 --distill-temperature 2.0 \
  --run-name L4_distilled --output-dir "$OUTPUT_DIR"

RUN_END=$(date +%s)
echo "=== Distillation round finished at $(date), $((RUN_END - RUN_START))s total ==="

echo
echo "=== Summary ==="
python -u scripts/distill_report.py
