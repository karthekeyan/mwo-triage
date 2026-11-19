# CLAUDE.md: mwo-triage project brief

Read this whole file before doing anything in this repo.

## 1. Goal

Build a small "System 1" decision model that reads one short maintenance work-order text and instantly predicts two labels:

- **Priority**: `STOP_NOW`, `TODAY`, `PLAN`, `MONITOR`
- **Route**: `MECHANICAL`, `ELECTRICAL`, `HYDRAULIC_PNEUMATIC`, `INSTRUMENTATION_CONTROLS`, `LUBRICATION`, `ESCALATE`

It is a portfolio piece, published publicly on Hugging Face (account `karthe05`, later moved to the `CobuildX-AI` organisation) with a public GitHub repo (`karthekeyan/mwo-triage`).

## 2. Locked decisions (do not change without the user's explicit approval)

- **Model type:** encoder classifier (BERT family) with two classification heads (priority and route) on a shared encoder. No generative LLM.
- **Sizes:** two models, both under 30M parameters: Tiny (about 4–5M) and Small (about 15–17M). Planned names: `mwo-triage-5M` and `mwo-triage-17M` (exact suffix follows the real parameter count).
- **Base models:** Apache-2.0 or MIT licensed only. Exact choice is made in the baselines phase.
- **Training:** full fine-tune. No LoRA, no QLoRA.
- **Comparison:** each fine-tuned model is compared against its own base model, frozen, with a simple classification head, on the same test set.
- **Labels:** defined only by `docs/labelling-rulebook.md` (v1.0, approved). Never invent, rename or merge labels. If the rulebook is unclear, stop and ask the user; changes require a version bump.
- **Data licences:** only commercially usable data (MIT, Apache, CC0, CC BY). Nothing marked non-commercial. No vision data.
- **Real data:** MaintNorm and MaintIE (both MIT, NLP-TLP Group, University of Western Australia). They must be credited, with their licence notices, in the README, the dataset card and the model cards.
- **Synthetic data:** generated with Claude through the Anthropic API, labelled with the same rulebook.
- **Test set:** real text only, never seen in training.
- **No human domain reviewer.** Quality comes from the rulebook, double labelling with agreement scoring, and documented limits.

## 3. Environment rules (critical)

- Machine: `srk-ubuntu`, NVIDIA Titan RTX, driver 535 (supports CUDA up to 12.2). Shared machine: never change system packages, drivers or system CUDA.
- Conda environment: `mwo-triage` (Python 3.11, conda-forge). Always run `conda activate mwo-triage` first.
- **Always install with `python -m pip`, never plain `pip`.** Plain `pip` resolves to an old copy in `~/.local/bin`.
- **Torch is pinned to `torch==2.6.0+cu118`.** Include `"torch==2.6.0"` in every `pip install` command so pip cannot replace it. Transformers 5.x needs torch 2.6 or newer, and driver 535 cannot safely run CUDA 12.4+ builds.
- After installing anything, update `requirements.txt` (its first line is `--extra-index-url https://download.pytorch.org/whl/cu118`, followed by `python -m pip freeze` output).
- Do not uninstall packages without checking for shared files first. (cuDNN for cu11 and cu12 share `libcudnn.so.9`; removing one broke the other once.)
- The Wi-Fi and SSH link are unreliable: run anything longer than a couple of minutes (downloads, labelling, training) in the background with `nohup ... > <log> 2>&1 &` and check it with `tail`.
- Git 2.25: no `git init -b`. Push to `origin main`.
- **No script may call the paid Anthropic API without the user's explicit approval for that specific run, with an estimated cost.** The API credit is $0; in-session work under the Pro subscription is the default.

## 4. Data rules

- `data/raw/` holds the downloaded datasets. It is git-ignored. Never commit it, and never republish raw files as they are.
- Asset IDs (`<id>` in the sources) are replaced by made-up codes of the same shape. `<sensitive>` tokens become `[name]`. Never restore or guess real IDs.
- Never commit `.env`, API keys, model weights (`outputs/`, `models/`) or logs.
- Keep every random step seeded (seed 42) so results are reproducible.

## 5. Working style (the user's preference)

- **Work one step at a time.** Before each step, state in one simple line *why* it is being done, then do only that step and wait for the user's confirmation before the next.
- Show commands as single lines; don't bundle multi-line paste blocks for the user.
- Keep explanations short and plain; the user is a senior AI program lead, not a hands-on ML engineer.
- Commit small, meaningful changes with clear messages, and push after each completed step.

## 6. Repo layout

- `configs/`: label definitions and training settings
- `data/raw/`: downloaded sources (git-ignored)
- `data/interim/`: work in progress (cleaned texts, draft labels)
- `data/processed/`: final train, validation and test files
- `docs/`: labelling rulebook, dataset card, experiment log
- `scripts/`: runnable steps (extract, label, generate, train, evaluate)
- `src/mwo_triage/`: reusable model code
- `tests/`: quick checks

## 7. Progress so far

- Done: repo, environment, folder structure, pinned `requirements.txt`, `.gitignore`, README.
- Done: `docs/labelling-rulebook.md` v1.0 approved and committed.
- Done: MaintNorm and MaintIE cloned into `data/raw/`. Licences checked (MIT).
- Done: `scripts/extract_real_texts.py` produced `data/interim/real_texts.jsonl`: 20,076 texts read, 11,633 kept after exact de-duplication (MaintNorm 8,975; MaintIE 2,658), 10 dropped as too short. **Not yet committed.**

## 8. Roadmap (remaining phases)

1. Commit the extraction script.
2. Clean and split: remove near-duplicates and junk, and set aside the real-text test pool first.
3. Labelling pipeline: two independent labelling passes with the Anthropic API following the rulebook, adjudication of disagreements, agreement rate reported. Give the user a cost estimate before any large run.
4. Synthetic data: factory equipment (CNC, presses, robots, conveyors, compressors, paint shops), rare classes, borderline cases, written in realistic messy style; labelled the same way.
5. Final dataset: merge, balance, leakage checks, train/validation/test files.
6. Baselines: choose the two base models; score them frozen with a simple head.
7. Training: full fine-tune of Tiny and Small.
8. Evaluation: compare against baselines, calibrate confidence, error analysis.
9. Release: dataset card and model cards with credits and honest limits; push to Hugging Face; polish the README.
10. If time permits: Gradio demo, then a CPU-only inference Dockerfile.
