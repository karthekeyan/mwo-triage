"""Final-evaluation step (2): per-head temperature scaling and a
STOP_NOW safety threshold, fit on val, for each of the two final models
(docs/experiment-log.md section 6):

  mwo-triage-13M (Small) <- outputs/bakeoff/electra_distilled
  mwo-triage-5M  (Tiny)  <- outputs/bakeoff/L4_a_earlystop_passb

For each model:
  1. Collects priority and route logits on val (data/processed/val.jsonl).
  2. Fits one scalar temperature per head (Guo et al. 2017 style: find
     T > 0 minimising NLL of logits/T against the true labels) with
     scipy.optimize.minimize_scalar, bounded to [0.05, 10].
  3. Using the temperature-scaled priority probabilities, searches val
     for the highest (most selective) STOP_NOW probability threshold
     that still reaches STOP_NOW recall >= 0.85 (same rule as
     scripts/stop_now_threshold.py, reused here so the threshold is
     fit on calibrated rather than raw probabilities).
  4. Writes outputs/bakeoff/<run_name>/calibration.json next to that
     model's checkpoint.

data/processed/test.jsonl is never loaded by this script.

No API calls.

Usage:
  python -u scripts/calibrate.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mwo_triage.classifier import MultiHeadClassifier, MwoDataset  # noqa: E402

LABELS_PATH = Path("configs/labels.yaml")
VAL_PATH = Path("data/processed/val.jsonl")
OUTPUT_ROOT = Path("outputs/bakeoff")
TARGET_STOP_NOW_RECALL = 0.85

FINAL_MODELS = [
    {
        "published_name": "mwo-triage-13M",
        "model_id": "google/electra-small-discriminator",
        "cache_dir": str(Path.home() / "models" / "google__electra-small-discriminator"),
        "run_name": "electra_distilled",
    },
    {
        "published_name": "mwo-triage-5M",
        "model_id": "google/bert_uncased_L-4_H-128_A-2",
        "cache_dir": str(Path.home() / "models" / "google__bert_uncased_L-4_H-128_A-2"),
        "run_name": "L4_a_earlystop_passb",
    },
]


def collect_logits(model, loader, device):
    priority_true, priority_logits = [], []
    route_true, route_logits = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            p_logits, r_logits = model(input_ids, attention_mask, token_type_ids)

            priority_true.extend(batch["priority_id"].tolist())
            priority_logits.append(p_logits.cpu())
            route_true.extend(batch["route_id"].tolist())
            route_logits.append(r_logits.cpu())

    return (
        torch.tensor(priority_true),
        torch.cat(priority_logits, dim=0),
        torch.tensor(route_true),
        torch.cat(route_logits, dim=0),
    )


def fit_temperature(logits, true_labels):
    """Scalar T > 0 minimising NLL of softmax(logits / T) against true_labels."""

    def nll(t):
        if t <= 0:
            return float("inf")
        scaled = logits / t
        return F.cross_entropy(scaled, true_labels).item()

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def ece(probs, true_labels, n_bins=10):
    """Standard top-1-confidence expected calibration error."""
    confidences, preds = probs.max(dim=-1)
    correct = (preds == true_labels).float()
    confidences = confidences.numpy()
    correct = correct.numpy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    err = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        err += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(err)


def find_stop_now_threshold(priority_true, priority_probs, stop_now_id, target_recall):
    stop_now_probs = priority_probs[:, stop_now_id]
    actual_pos = int((priority_true == stop_now_id).sum())

    candidates = sorted(set(stop_now_probs.tolist()) | {0.0, 1.0}, reverse=True)
    best_t, best_recall = 0.0, 1.0
    for t in candidates:
        predicted_pos_mask = stop_now_probs >= t
        tp = int((predicted_pos_mask & (priority_true == stop_now_id)).sum())
        recall = tp / actual_pos if actual_pos else 0.0
        if recall >= target_recall:
            best_t, best_recall = t, recall
            break
    return best_t, best_recall


def main():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_cfg = yaml.safe_load(f)
    stop_now_id = labels_cfg["priority"]["STOP_NOW"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg in FINAL_MODELS:
        run_dir = OUTPUT_ROOT / cfg["run_name"]
        print(f"\n=== {cfg['published_name']} ({cfg['run_name']}, {cfg['model_id']}) ===")

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], cache_dir=cfg["cache_dir"], local_files_only=True)
        val_dataset = MwoDataset(VAL_PATH, tokenizer, max_length=32)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = MultiHeadClassifier(cfg["model_id"], freeze_encoder=False, cache_dir=cfg["cache_dir"], local_files_only=True)
        state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        priority_true, priority_logits, route_true, route_logits = collect_logits(model, val_loader, device)

        raw_priority_probs = F.softmax(priority_logits, dim=-1)
        raw_route_probs = F.softmax(route_logits, dim=-1)
        ece_priority_before = ece(raw_priority_probs, priority_true)
        ece_route_before = ece(raw_route_probs, route_true)

        priority_temperature = fit_temperature(priority_logits, priority_true)
        route_temperature = fit_temperature(route_logits, route_true)

        scaled_priority_probs = F.softmax(priority_logits / priority_temperature, dim=-1)
        scaled_route_probs = F.softmax(route_logits / route_temperature, dim=-1)
        ece_priority_after = ece(scaled_priority_probs, priority_true)
        ece_route_after = ece(scaled_route_probs, route_true)

        threshold, achieved_recall = find_stop_now_threshold(
            priority_true, scaled_priority_probs, stop_now_id, TARGET_STOP_NOW_RECALL
        )
        predicted_pos = int((scaled_priority_probs[:, stop_now_id] >= threshold).sum())
        tp = int(
            ((scaled_priority_probs[:, stop_now_id] >= threshold) & (priority_true == stop_now_id)).sum()
        )
        achieved_precision = tp / predicted_pos if predicted_pos else 0.0

        calibration = {
            "published_name": cfg["published_name"],
            "run_name": cfg["run_name"],
            "model_id": cfg["model_id"],
            "priority_temperature": priority_temperature,
            "route_temperature": route_temperature,
            "val_ece_priority_before": ece_priority_before,
            "val_ece_priority_after": ece_priority_after,
            "val_ece_route_before": ece_route_before,
            "val_ece_route_after": ece_route_after,
            "target_stop_now_recall": TARGET_STOP_NOW_RECALL,
            "stop_now_threshold": threshold,
            "val_achieved_stop_now_recall": achieved_recall,
            "val_achieved_stop_now_precision": achieved_precision,
        }
        with open(run_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)

        print(f"  priority_temperature={priority_temperature:.4f}  route_temperature={route_temperature:.4f}")
        print(f"  val ECE priority before/after: {ece_priority_before:.4f} -> {ece_priority_after:.4f}")
        print(f"  val ECE route before/after:    {ece_route_before:.4f} -> {ece_route_after:.4f}")
        print(f"  STOP_NOW threshold t={threshold:.4f}  recall={achieved_recall:.4f}  precision={achieved_precision:.4f}")
        print(f"  -> {run_dir}/calibration.json")


if __name__ == "__main__":
    main()
