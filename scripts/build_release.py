"""Package the two final checkpoints (docs/experiment-log.md section 6)
into self-contained release/<published_name>/ folders: model.safetensors
(encoder + both heads), config.json (base model id, max_length, label
lists, per-head calibration temperatures, the STOP_NOW safety threshold,
the parameter count, the rulebook version), the base encoder's own HF
config (for offline architecture rebuild), the tokenizer files, and a
copy of scripts/inference.py.

No training, no API calls; reads outputs/bakeoff/<run>/best_model.pt and
calibration.json (scripts/calibrate.py) and configs/labels.yaml.

Usage:
  python scripts/build_release.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mwo_triage.classifier import MultiHeadClassifier  # noqa: E402

LABELS_PATH = Path("configs/labels.yaml")
OUTPUT_ROOT = Path("outputs/bakeoff")
RELEASE_ROOT = Path("release")
INFERENCE_SRC = Path("scripts/inference.py")
MAX_LENGTH = 32

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


def ordered_labels(label_ids):
    """{'STOP_NOW': 0, 'TODAY': 1, ...} -> ['STOP_NOW', 'TODAY', ...]."""
    return [name for name, _ in sorted(label_ids.items(), key=lambda kv: kv[1])]


def build_one(cfg, priority_labels, route_labels, rulebook_version):
    published_name = cfg["published_name"]
    run_dir = OUTPUT_ROOT / cfg["run_name"]
    release_dir = RELEASE_ROOT / published_name
    release_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {published_name} ({cfg['run_name']}, {cfg['model_id']}) ===")

    with open(run_dir / "calibration.json", "r", encoding="utf-8") as f:
        calibration = json.load(f)

    model = MultiHeadClassifier(
        cfg["model_id"], freeze_encoder=False, cache_dir=cfg["cache_dir"], local_files_only=True
    )
    state_dict = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    num_parameters = sum(p.numel() for p in model.parameters())

    # safetensors needs contiguous tensors; the loaded state dict already
    # is, but be defensive since this is the published artifact.
    flat_state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(flat_state_dict, release_dir / "model.safetensors")

    encoder_config_dir = release_dir / "encoder_config"
    encoder_config_dir.mkdir(exist_ok=True)
    model.encoder.config.save_pretrained(encoder_config_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], cache_dir=cfg["cache_dir"], local_files_only=True)
    tokenizer.save_pretrained(release_dir)

    config = {
        "published_name": published_name,
        "base_model_id": cfg["model_id"],
        "max_length": MAX_LENGTH,
        "priority_labels": priority_labels,
        "route_labels": route_labels,
        "priority_temperature": calibration["priority_temperature"],
        "route_temperature": calibration["route_temperature"],
        "stop_now_threshold": calibration["stop_now_threshold"],
        "num_parameters": num_parameters,
        "rulebook_version": rulebook_version,
    }
    with open(release_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    shutil.copyfile(INFERENCE_SRC, release_dir / "inference.py")

    print(f"  {num_parameters:,} parameters")
    print(f"  priority_temperature={calibration['priority_temperature']:.4f}  "
          f"route_temperature={calibration['route_temperature']:.4f}  "
          f"stop_now_threshold={calibration['stop_now_threshold']:.4f}")
    print(f"  -> {release_dir}/")


def main():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_cfg = yaml.safe_load(f)
    priority_labels = ordered_labels(labels_cfg["priority"])
    route_labels = ordered_labels(labels_cfg["route"])
    rulebook_version = labels_cfg["rulebook_version"]

    for cfg in FINAL_MODELS:
        build_one(cfg, priority_labels, route_labels, rulebook_version)


if __name__ == "__main__":
    main()
