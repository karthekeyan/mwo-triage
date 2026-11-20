"""Fine-tune (or train heads only, with --freeze-encoder) one candidate
base model as the mwo-triage two-head classifier, and evaluate on val
after every epoch.

Encoder + two linear heads (priority: 4 classes, route: 6 classes) on the
[CLS] token (src/mwo_triage/classifier.py). Loss is the sum of the two
heads' cross-entropy losses. AdamW with a linear warmup (10% of steps)
then linear decay to 0. fp16 autocast on CUDA. Seed 42 by default.

Evaluates on data/processed/val.jsonl every epoch and keeps the
checkpoint with the best val macro-F1 (mean of priority macro-F1 and
route macro-F1). data/processed/test.jsonl is never loaded by this
script.

Writes outputs/bakeoff/<run_name>/best_model.pt and metrics.json.

No API calls. Uses only locally cached Hugging Face models (set
HF_HUB_OFFLINE so a flaky network can't interrupt a run mid-epoch).

Usage:
  python -u scripts/train.py --model-id google/bert_uncased_L-2_H-128_A-2 \
      --cache-dir ~/models/google__bert_uncased_L-2_H-128_A-2 \
      --lr 1e-4 --run-name tiny_L2_lr1e-4
  python -u scripts/train.py --model-id ... --cache-dir ... \
      --lr 1e-3 --freeze-encoder --run-name tiny_L2_frozen
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mwo_triage.classifier import MultiHeadClassifier, MwoDataset  # noqa: E402

LABELS_PATH = Path("configs/labels.yaml")
TRAIN_PATH = Path("data/processed/train.jsonl")
VAL_PATH = Path("data/processed/val.jsonl")
OUTPUT_ROOT = Path("outputs/bakeoff")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="Hugging Face model id, e.g. google/bert_uncased_L-2_H-128_A-2")
    p.add_argument("--cache-dir", required=True, help="Local HF cache dir this model was downloaded into")
    p.add_argument("--run-name", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return p.parse_args()


def evaluate(model, loader, device, stop_now_id):
    model.eval()
    priority_true, priority_pred = [], []
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
            priority_pred.extend(priority_logits.argmax(dim=-1).cpu().tolist())
            route_true.extend(batch["route_id"].tolist())
            route_pred.extend(route_logits.argmax(dim=-1).cpu().tolist())

    priority_macro_f1 = f1_score(priority_true, priority_pred, average="macro", zero_division=0)
    route_macro_f1 = f1_score(route_true, route_pred, average="macro", zero_division=0)
    priority_acc = accuracy_score(priority_true, priority_pred)
    route_acc = accuracy_score(route_true, route_pred)
    stop_now_recall = recall_score(
        priority_true, priority_pred, labels=[stop_now_id], average="macro", zero_division=0
    )

    return {
        "val_priority_macro_f1": priority_macro_f1,
        "val_route_macro_f1": route_macro_f1,
        "val_mean_macro_f1": (priority_macro_f1 + route_macro_f1) / 2,
        "val_priority_acc": priority_acc,
        "val_route_acc": route_acc,
        "val_stop_now_recall": stop_now_recall,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_cfg = yaml.safe_load(f)
    stop_now_id = labels_cfg["priority"]["STOP_NOW"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=True)
    train_dataset = MwoDataset(TRAIN_PATH, tokenizer, max_length=args.max_length)
    val_dataset = MwoDataset(VAL_PATH, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiHeadClassifier(
        args.model_id, freeze_encoder=args.freeze_encoder, cache_dir=args.cache_dir, local_files_only=True
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    loss_fn = torch.nn.CrossEntropyLoss()

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    epoch_logs = []
    best_mean_f1 = -1.0
    best_epoch = None
    run_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            priority_id = batch["priority_id"].to(device)
            route_id = batch["route_id"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_fp16):
                priority_logits, route_logits = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(priority_logits, priority_id) + loss_fn(route_logits, route_id)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * input_ids.size(0)

        train_loss = total_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, device, stop_now_id)
        epoch_time = time.time() - epoch_start

        log_entry = {"epoch": epoch, "train_loss": train_loss, "epoch_time_sec": epoch_time, **val_metrics}
        epoch_logs.append(log_entry)
        print(
            f"[{args.run_name}] epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.4f} "
            f"priority_f1={val_metrics['val_priority_macro_f1']:.4f} "
            f"route_f1={val_metrics['val_route_macro_f1']:.4f} "
            f"priority_acc={val_metrics['val_priority_acc']:.4f} "
            f"route_acc={val_metrics['val_route_acc']:.4f} "
            f"stop_now_recall={val_metrics['val_stop_now_recall']:.4f} "
            f"({epoch_time:.1f}s)",
            flush=True,
        )

        if val_metrics["val_mean_macro_f1"] > best_mean_f1:
            best_mean_f1 = val_metrics["val_mean_macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    total_time = time.time() - run_start

    metrics = {
        "model_id": args.model_id,
        "run_name": args.run_name,
        "lr": args.lr,
        "freeze_encoder": args.freeze_encoder,
        "epochs_run": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "epochs": epoch_logs,
        "best_epoch": best_epoch,
        "best_val_mean_macro_f1": best_mean_f1,
        "best_epoch_metrics": epoch_logs[best_epoch - 1],
        "total_train_time_sec": total_time,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{args.run_name}] best epoch {best_epoch}, val mean macro-F1 {best_mean_f1:.4f}, "
          f"total time {total_time:.1f}s -> {run_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
