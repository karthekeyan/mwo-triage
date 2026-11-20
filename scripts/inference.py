"""Self-contained inference wrapper for a packaged mwo-triage model
(release/mwo-triage-13M/ or release/mwo-triage-5M/, or their published
Hugging Face repo once pushed).

Depends only on torch, transformers, safetensors and huggingface_hub.

A packaged model directory holds:
  model.safetensors   encoder + priority_head + route_head state dict
  config.json         base_model_id, max_length, label lists (in
                       configs/labels.yaml order), per-head calibration
                       temperatures, the STOP_NOW safety threshold, the
                       total parameter count, the rulebook version, and
                       the base encoder's own HF config (so the encoder
                       architecture can be rebuilt offline, without
                       contacting the Hub for anything but a repo id)
  encoder_config/      the base encoder's HF config.json, as its own
                       directory so AutoConfig.from_pretrained can read
                       it directly
  tokenizer files      vocab.txt / tokenizer.json / tokenizer_config.json
                       / special_tokens_map.json, as saved by
                       tokenizer.save_pretrained

Usage:
  from inference import TriageModel
  model = TriageModel.from_pretrained("release/mwo-triage-13M")
  # or, once published: TriageModel.from_pretrained("karthe05/mwo-triage-13M")
  model.predict(["Hyd hose burst on press 3, oil spraying across walkway"], mode="safety")
"""

import json
import os

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
ENCODER_CONFIG_DIRNAME = "encoder_config"


class TriageModel:
    """A fine-tuned encoder with two linear classification heads
    (priority, route) on the [CLS] token of the last hidden state, plus
    the calibration (per-head temperature, STOP_NOW safety threshold)
    fit for this specific checkpoint.
    """

    def __init__(self, encoder, priority_head, route_head, tokenizer, config):
        self.encoder = encoder
        self.priority_head = priority_head
        self.route_head = route_head
        self.tokenizer = tokenizer
        self.config = config

        self.encoder.eval()
        self.priority_head.eval()
        self.route_head.eval()

        self.max_length = config["max_length"]
        self.priority_labels = config["priority_labels"]
        self.route_labels = config["route_labels"]
        self.priority_temperature = config["priority_temperature"]
        self.route_temperature = config["route_temperature"]
        self.stop_now_threshold = config["stop_now_threshold"]
        self.stop_now_id = self.priority_labels.index("STOP_NOW")

    @classmethod
    def from_pretrained(cls, path_or_repo_id, device="cpu"):
        """`path_or_repo_id` is either a local packaged-model directory
        (as written by scripts/build_release.py) or a Hugging Face repo
        id, which is fetched with huggingface_hub.snapshot_download.
        """
        if os.path.isdir(path_or_repo_id):
            local_dir = path_or_repo_id
        else:
            local_dir = snapshot_download(repo_id=path_or_repo_id)

        with open(os.path.join(local_dir, CONFIG_FILENAME), "r", encoding="utf-8") as f:
            config = json.load(f)

        encoder_config = AutoConfig.from_pretrained(os.path.join(local_dir, ENCODER_CONFIG_DIRNAME))
        encoder = AutoModel.from_config(encoder_config)
        hidden_size = encoder_config.hidden_size
        priority_head = torch.nn.Linear(hidden_size, len(config["priority_labels"]))
        route_head = torch.nn.Linear(hidden_size, len(config["route_labels"]))

        state_dict = load_file(os.path.join(local_dir, WEIGHTS_FILENAME))
        encoder.load_state_dict(_sub_state_dict(state_dict, "encoder."))
        priority_head.load_state_dict(_sub_state_dict(state_dict, "priority_head."))
        route_head.load_state_dict(_sub_state_dict(state_dict, "route_head."))

        encoder.to(device)
        priority_head.to(device)
        route_head.to(device)

        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        return cls(encoder, priority_head, route_head, tokenizer, config)

    @torch.no_grad()
    def predict(self, texts, mode="balanced"):
        """`texts` is a list of strings. `mode` is "balanced" (plain
        argmax on calibrated priority probabilities) or "safety"
        (STOP_NOW whenever its calibrated probability is at or above
        this checkpoint's fitted threshold, else argmax).

        Returns one dict per text: priority, route,
        priority_confidence, route_confidence (both calibrated), mode.
        """
        if mode not in ("balanced", "safety"):
            raise ValueError(f"mode must be 'balanced' or 'safety', got {mode!r}")

        device = next(self.encoder.parameters()).device
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in encoded:
            encoder_kwargs["token_type_ids"] = encoded["token_type_ids"].to(device)

        outputs = self.encoder(**encoder_kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0]
        priority_logits = self.priority_head(cls_hidden)
        route_logits = self.route_head(cls_hidden)

        priority_probs = F.softmax(priority_logits / self.priority_temperature, dim=-1)
        route_probs = F.softmax(route_logits / self.route_temperature, dim=-1)

        route_conf, route_idx = route_probs.max(dim=-1)
        balanced_conf, balanced_idx = priority_probs.max(dim=-1)
        stop_now_prob = priority_probs[:, self.stop_now_id]

        results = []
        for i in range(len(texts)):
            if mode == "safety" and stop_now_prob[i].item() >= self.stop_now_threshold:
                priority_id = self.stop_now_id
                priority_confidence = stop_now_prob[i].item()
            else:
                priority_id = int(balanced_idx[i].item())
                priority_confidence = float(balanced_conf[i].item())

            results.append({
                "priority": self.priority_labels[priority_id],
                "route": self.route_labels[int(route_idx[i].item())],
                "priority_confidence": priority_confidence,
                "route_confidence": float(route_conf[i].item()),
                "mode": mode,
            })
        return results


def _sub_state_dict(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
