"""Print the bake-off summary table: for every run under
outputs/bakeoff/<run_name>/metrics.json, show model, size (encoder
params), mode (finetune/frozen), lr, best epoch, val macro-F1 (priority,
route, mean), STOP_NOW recall at the best epoch, and total training time.

Usage:
  python scripts/bakeoff_report.py
"""

import json
from pathlib import Path

OUTPUT_ROOT = Path("outputs/bakeoff")
SURVEY_PATH = Path.home() / "models" / "survey_results.json"


def load_survey():
    if not SURVEY_PATH.exists():
        return {}
    with open(SURVEY_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["model_id"]: r for r in rows}


def main():
    survey = load_survey()
    run_dirs = sorted(OUTPUT_ROOT.glob("*/metrics.json"))
    if not run_dirs:
        print(f"No runs found under {OUTPUT_ROOT}/")
        return

    rows = []
    for metrics_path in run_dirs:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        best = m["best_epoch_metrics"]
        info = survey.get(m["model_id"], {})
        rows.append({
            "run_name": m["run_name"],
            "model_id": m["model_id"],
            "encoder_params": info.get("encoder_params", "?"),
            "mode": "frozen" if m["freeze_encoder"] else "finetune",
            "lr": m["lr"],
            "best_epoch": m["best_epoch"],
            "val_priority_f1": best["val_priority_macro_f1"],
            "val_route_f1": best["val_route_macro_f1"],
            "val_mean_f1": best["val_mean_macro_f1"],
            "stop_now_recall": best["val_stop_now_recall"],
            "train_time_sec": m["total_train_time_sec"],
        })

    rows.sort(key=lambda r: (r["encoder_params"] if isinstance(r["encoder_params"], int) else 0, r["run_name"]))

    header = ["model_id", "encoder_params", "mode", "lr", "best_epoch",
              "val_priority_f1", "val_route_f1", "val_mean_f1", "stop_now_recall", "train_time_sec"]
    widths = [34, 14, 9, 8, 10, 16, 14, 12, 15, 15]

    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        values = [
            r["model_id"],
            str(r["encoder_params"]),
            r["mode"],
            f"{r['lr']:g}",
            str(r["best_epoch"]),
            f"{r['val_priority_f1']:.4f}",
            f"{r['val_route_f1']:.4f}",
            f"{r['val_mean_f1']:.4f}",
            f"{r['stop_now_recall']:.4f}",
            f"{r['train_time_sec']:.1f}",
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(values, widths)))


if __name__ == "__main__":
    main()
