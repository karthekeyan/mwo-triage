"""Extract real maintenance work-order texts from MaintNorm and MaintIE.

Output: data/interim/real_texts.jsonl, one JSON object per line:
    {"uid", "source", "text", "text_norm"}

- text      : what the model will see (original technician wording for
              MaintNorm, cleaned wording for MaintIE), with asset IDs replaced
              by made-up codes of the same shape and sensitive tokens masked.
- text_norm : lowercase normalised form, used only for de-duplication.

Run from the repo root:
    python scripts/extract_real_texts.py
"""

import json
import random
import re
import string
from collections import Counter
from pathlib import Path

SEED = 42
RAW = Path("data/raw")
OUT = Path("data/interim/real_texts.jsonl")

rng = random.Random(SEED)


def fake_id_like(token: str) -> str:
    """Return a made-up code with the same letter/digit shape as `token`."""
    out = []
    for ch in token:
        if ch.isdigit():
            out.append(rng.choice(string.digits))
        elif ch.isalpha():
            letter = rng.choice(string.ascii_uppercase)
            out.append(letter if ch.isupper() else letter.lower())
        else:
            out.append(ch)
    return "".join(out)


def fake_id() -> str:
    """Return a made-up asset code such as 'KT402'."""
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
    digits = "".join(rng.choice(string.digits) for _ in range(3))
    return letters + digits


def norm_key(text: str) -> str:
    """Lowercase, strip placeholders and punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r"<[a-z]+>", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- MaintNorm
def read_norm_file(path: Path):
    """Yield one work order at a time as a list of (raw, normalised) pairs."""
    block = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if block:
                    yield block
                    block = []
                continue
            parts = line.split("\t")
            raw = parts[0].strip()
            norm = parts[1].strip() if len(parts) > 1 else raw
            block.append((raw, norm))
    if block:
        yield block


def maintnorm_records():
    folder = RAW / "maintnorm" / "data" / "combined"
    for split in ("train", "val", "test"):
        for block in read_norm_file(folder / f"{split}.norm"):
            raw_tokens, norm_tokens = [], []
            for raw, norm in block:
                if norm == "<id>":
                    raw_tokens.append(fake_id_like(raw))
                elif norm == "<sensitive>":
                    raw_tokens.append("[name]")
                else:
                    raw_tokens.append(raw)
                norm_tokens.append(norm)
            yield {
                "source": f"maintnorm/{split}",
                "text": " ".join(raw_tokens),
                "text_norm": norm_key(" ".join(norm_tokens)),
            }


# ------------------------------------------------------------------ MaintIE
def maintie_text(text: str) -> str:
    text = re.sub(r"<id>", lambda _: fake_id(), text)
    text = re.sub(r"<sensitive>", "[name]", text)
    text = re.sub(r"<num>", lambda _: str(rng.randint(1, 12)), text)
    text = re.sub(r"<date>", lambda _: f"{rng.randint(1, 28)}/{rng.randint(1, 12)}", text)
    return re.sub(r"\s+", " ", text).strip()


def maintie_records():
    folder = RAW / "maintie" / "data"
    for name in ("gold", "silver"):
        data = json.loads((folder / f"{name}_release.json").read_text(encoding="utf-8"))
        for item in data:
            yield {
                "source": f"maintie/{name}",
                "text": maintie_text(item["text"]),
                "text_norm": norm_key(item["text"]),
            }


# --------------------------------------------------------------------- main
def main():
    seen = set()
    kept = []
    counts_in, counts_kept = Counter(), Counter()
    dropped_empty = 0

    # MaintNorm first: its original wording is the more valuable version.
    for rec in list(maintnorm_records()) + list(maintie_records()):
        counts_in[rec["source"]] += 1
        if len(rec["text_norm"]) < 3:
            dropped_empty += 1
            continue
        if rec["text_norm"] in seen:
            continue
        seen.add(rec["text_norm"])
        counts_kept[rec["source"]] += 1
        kept.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(kept):
            rec = {"uid": f"real-{i:05d}", **rec}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{'source':<20}{'read':>8}{'kept':>8}")
    for src in counts_in:
        print(f"{src:<20}{counts_in[src]:>8}{counts_kept[src]:>8}")
    print(f"{'TOTAL':<20}{sum(counts_in.values()):>8}{len(kept):>8}")
    print(f"dropped as empty/too short: {dropped_empty}")
    print(f"written to {OUT}")
    print("\n5 random examples:")
    for rec in rng.sample(kept, min(5, len(kept))):
        print(f"  [{rec['source']}] {rec['text']}")


if __name__ == "__main__":
    main()
