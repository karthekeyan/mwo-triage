"""Post-hoc STOP_NOW thresholding (improvement round step (c)).

For each winner model, re-evaluates the baseline bake-off run and the
(a) early-stopping / (b) class-weighted improvement runs (fresh forward
pass on val, so every row reports the same metric set including
STOP_NOW precision, which the original bake-off runs predate). Then,
for whichever of (a)/(b) has the higher val mean macro-F1, searches val
for the highest STOP_NOW probability threshold t such that predicting
STOP_NOW whenever P(STOP_NOW) >= t still reaches STOP_NOW recall >=
0.85 (the highest t, i.e. the most selective threshold, that clears the
recall floor -- this maximises STOP_NOW precision subject to the recall
constraint). Reports STOP_NOW precision at t, the resulting priority
macro-F1 (route is untouched by this -- only the priority head's
decision rule changes), and the full priority confusion matrix.

Saves {"post_hoc_stop_now": {...}} into that winning (a)/(b) run's
metrics.json, and prints the baseline/(a)/(b)/(c) comparison table.

data/processed/test.jsonl is never loaded.

No API calls.

Usage:
  python -u scripts/stop_now_threshold.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mwo_triage.classifier import MultiHeadClassifier, MwoDataset  # noqa: E402

LABELS_PATH = Path("configs/labels.yaml")
VAL_PATH = Path("data/processed/val.jsonl")
OUTPUT_ROOT = Path("outputs/bakeoff")
TARGET_STOP_NOW_RECALL = 0.85

WINNERS = [
    {
        "short": "electra",
        "model_id": "google/electra-small-discriminator",
        "cache_dir": str(Path.home() / "models" / "google__electra-small-discriminator"),
        "baseline_run": "electra_lr1e-4",
        "a_run": "electra_a_earlystop",
        "b_run": "electra_b_classweighted",
    },
    {
        "short": "L4",
        "model_id": "google/bert_uncased_L-4_H-128_A-2",
        "cache_dir": str(Path.home() / "models" / "google__bert_uncased_L-4_H-128_A-2"),
        "baseline_run": "L4_lr3e-4",
        "a_run": "L4_a_earlystop",
        "b_run": "L4_b_classweighted",
    },
]


def load_priority_order():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    priority_ids = cfg["priority"]
    stop_now_id = priority_ids["STOP_NOW"]
    order = sorted(priority_ids, key=lambda name: priority_ids[name])
    return stop_now_id, order


def load_model(model_id, cache_dir, run_dir, device):
    model = MultiHeadClassifier(model_id, freeze_encoder=False, cache_dir=cache_dir, local_files_only=True)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader, device):
    priority_true, priority_probs = [], []
    route_true, route_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            priority_logits, route_logits = model(input_ids, attention_mask, token_type_ids)

            priority_true.extend(batch["priority_id"].tolist())
            priority_probs.extend(torch.softmax(priority_logits, dim=-1).cpu().tolist())
            route_true.extend(batch["route_id"].tolist())
            route_pred.extend(route_logits.argmax(dim=-1).cpu().tolist())

    return (
        np.array(priority_true),
        np.array(priority_probs),
        np.array(route_true),
        np.array(route_pred),
    )


def evaluate_run(model_id, cache_dir, run_name, loader, device, stop_now_id):
    run_dir = OUTPUT_ROOT / run_name
    model = load_model(model_id, cache_dir, run_dir, device)
    priority_true, priority_probs, route_true, route_pred = collect_predictions(model, loader, device)
    priority_pred = priority_probs.argmax(axis=-1)

    priority_f1 = f1_score(priority_true, priority_pred, average="macro", zero_division=0)
    route_f1 = f1_score(route_true, route_pred, average="macro", zero_division=0)

    tp = int(((priority_pred == stop_now_id) & (priority_true == stop_now_id)).sum())
    actual_pos = int((priority_true == stop_now_id).sum())
    predicted_pos = int((priority_pred == stop_now_id).sum())
    recall = tp / actual_pos if actual_pos else 0.0
    precision = tp / predicted_pos if predicted_pos else 0.0

    return {
        "run_name": run_name,
        "val_priority_macro_f1": priority_f1,
        "val_route_macro_f1": route_f1,
        "val_mean_macro_f1": (priority_f1 + route_f1) / 2,
        "val_stop_now_recall": recall,
        "val_stop_now_precision": precision,
    }, (priority_true, priority_probs, route_true)


def find_stop_now_threshold(priority_true, priority_probs, stop_now_id, target_recall):
    stop_now_probs = priority_probs[:, stop_now_id]
    actual_pos = int((priority_true == stop_now_id).sum())

    candidates = sorted(set(stop_now_probs.tolist()) | {0.0, 1.0}, reverse=True)
    best_t = 0.0
    best_recall = 1.0  # t=0.0 always gives recall 1.0 (every example predicted STOP_NOW)
    for t in candidates:
        predicted_pos_mask = stop_now_probs >= t
        tp = int((predicted_pos_mask & (priority_true == stop_now_id)).sum())
        recall = tp / actual_pos if actual_pos else 0.0
        if recall >= target_recall:
            best_t = t
            best_recall = recall
            break  # candidates sorted descending: first hit is the highest (most selective) t
    return best_t, best_recall


def apply_threshold(priority_probs, stop_now_id, t):
    stop_now_probs = priority_probs[:, stop_now_id]
    non_stop_now_ids = [i for i in range(priority_probs.shape[1]) if i != stop_now_id]
    fallback_pred = np.array(non_stop_now_ids)[priority_probs[:, non_stop_now_ids].argmax(axis=-1)]
    return np.where(stop_now_probs >= t, stop_now_id, fallback_pred)


def run_post_hoc(model_id, cache_dir, a_metrics, b_metrics, priority_order, stop_now_id, device):
    a_better = a_metrics["recomputed"]["val_mean_macro_f1"] >= b_metrics["recomputed"]["val_mean_macro_f1"]
    chosen = a_metrics if a_better else b_metrics
    run_name = chosen["recomputed"]["run_name"]
    priority_true, priority_probs, _ = chosen["predictions"]

    t, achieved_recall = find_stop_now_threshold(priority_true, priority_probs, stop_now_id, TARGET_STOP_NOW_RECALL)
    thresholded_pred = apply_threshold(priority_probs, stop_now_id, t)

    predicted_pos = int((thresholded_pred == stop_now_id).sum())
    tp = int(((thresholded_pred == stop_now_id) & (priority_true == stop_now_id)).sum())
    precision = tp / predicted_pos if predicted_pos else 0.0
    priority_f1_post = f1_score(priority_true, thresholded_pred, average="macro", zero_division=0)
    cm = confusion_matrix(priority_true, thresholded_pred, labels=list(range(len(priority_order))))

    post_hoc = {
        "chosen_run": run_name,
        "target_stop_now_recall": TARGET_STOP_NOW_RECALL,
        "stop_now_threshold": t,
        "achieved_stop_now_recall": achieved_recall,
        "achieved_stop_now_precision": precision,
        "priority_macro_f1_after_threshold": priority_f1_post,
        "priority_confusion_matrix": {
            "label_order": priority_order,
            "matrix": cm.tolist(),
        },
    }

    run_dir = OUTPUT_ROOT / run_name
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing["post_hoc_stop_now"] = post_hoc
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    route_f1_unchanged = chosen["recomputed"]["val_route_macro_f1"]
    c_row = {
        "run_name": f"{run_name}+threshold",
        "val_priority_macro_f1": priority_f1_post,
        "val_route_macro_f1": route_f1_unchanged,
        "val_mean_macro_f1": (priority_f1_post + route_f1_unchanged) / 2,
        "val_stop_now_recall": achieved_recall,
        "val_stop_now_precision": precision,
    }
    return c_row, post_hoc, a_better


def print_row(label, m):
    print(f"  {label:28s} priority_f1={m['val_priority_macro_f1']:.4f}  route_f1={m['val_route_macro_f1']:.4f}  "
          f"mean_f1={m['val_mean_macro_f1']:.4f}  stop_now_recall={m['val_stop_now_recall']:.4f}  "
          f"stop_now_precision={m['val_stop_now_precision']:.4f}")


def main():
    stop_now_id, priority_order = load_priority_order()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for winner in WINNERS:
        print(f"\n=== {winner['short']} ({winner['model_id']}) ===")
        tokenizer = AutoTokenizer.from_pretrained(winner["model_id"], cache_dir=winner["cache_dir"], local_files_only=True)
        val_dataset = MwoDataset(VAL_PATH, tokenizer, max_length=32)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        baseline_metrics, _ = evaluate_run(winner["model_id"], winner["cache_dir"], winner["baseline_run"],
                                            val_loader, device, stop_now_id)
        a_recomputed, a_preds = evaluate_run(winner["model_id"], winner["cache_dir"], winner["a_run"],
                                              val_loader, device, stop_now_id)
        b_recomputed, b_preds = evaluate_run(winner["model_id"], winner["cache_dir"], winner["b_run"],
                                              val_loader, device, stop_now_id)

        a_metrics = {"recomputed": a_recomputed, "predictions": a_preds}
        b_metrics = {"recomputed": b_recomputed, "predictions": b_preds}

        c_row, post_hoc, a_better = run_post_hoc(
            winner["model_id"], winner["cache_dir"], a_metrics, b_metrics, priority_order, stop_now_id, device
        )

        print_row("baseline (" + winner["baseline_run"] + ")", baseline_metrics)
        print_row("(a) early-stop (" + winner["a_run"] + ")", a_recomputed)
        print_row("(b) class-weighted (" + winner["b_run"] + ")", b_recomputed)
        print_row("(c) " + post_hoc["chosen_run"] + "+threshold", c_row)
        print(f"  (c) chose {'(a)' if a_better else '(b)'} as the better of a/b; "
              f"STOP_NOW threshold t={post_hoc['stop_now_threshold']:.4f}, "
              f"achieved recall={post_hoc['achieved_stop_now_recall']:.4f}")
        print(f"  (c) priority confusion matrix (rows=actual, cols=predicted, order={priority_order}):")
        for label, row in zip(priority_order, post_hoc["priority_confusion_matrix"]["matrix"]):
            print(f"    {label:10s} {row}")


if __name__ == "__main__":
    main()
