"""Print the knowledge-distillation comparison table: teacher, each
winner's current best non-distilled run, and its distilled run, with
val priority/route/mean macro-F1, STOP_NOW recall and precision.

Usage:
  python scripts/distill_report.py
"""

import json
from pathlib import Path

OUTPUT_ROOT = Path("outputs/bakeoff")

ROWS = [
    ("teacher (microsoft/deberta-v3-base)", "teacher_deberta"),
    ("ELECTRA-small (a) passB-default", "electra_a_earlystop_passb"),
    ("ELECTRA-small distilled", "electra_distilled"),
    ("BERT L-4/H-128 (a) passB-default", "L4_a_earlystop_passb"),
    ("BERT L-4/H-128 distilled", "L4_distilled"),
]

FIELDS = ["val_priority_macro_f1", "val_route_macro_f1", "val_mean_macro_f1",
          "val_stop_now_recall", "val_stop_now_precision"]


def main():
    header = ["label", "run_name", "best_epoch"] + FIELDS
    widths = [36, 28, 11, 12, 10, 10, 15, 18]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for label, run_name in ROWS:
        path = OUTPUT_ROOT / run_name / "metrics.json"
        if not path.exists():
            print(f"{label.ljust(widths[0])} | {run_name.ljust(widths[1])} | MISSING ({path})")
            continue
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        best = m["best_epoch_metrics"]
        values = [label, run_name, str(m["best_epoch"])] + [f"{best[k]:.4f}" for k in FIELDS]
        print(" | ".join(v.ljust(w) for v, w in zip(values, widths)))


if __name__ == "__main__":
    main()
