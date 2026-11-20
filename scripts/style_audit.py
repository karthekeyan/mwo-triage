"""Compare the style of synthetic texts against real train-pool texts.

Checks, for each corpus:
  - mean / median word count
  - share of texts containing an asset code (two letters + three digits,
    the shape used in docs/labelling-rulebook.md section 7's examples)
  - share of texts containing a rulebook section 7 abbreviation
  - share of texts that are entirely upper case
  - the 30 most frequent 2- and 3-word phrases in the synthetic corpus
    that are rare (or absent) in the real corpus

No API calls. Reads data/interim/real_labelled.jsonl (pool == "train")
and data/interim/synthetic_raw.jsonl.

Usage: python scripts/style_audit.py
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

REAL_PATH = Path("data/interim/real_labelled.jsonl")
SYNTHETIC_PATH = Path("data/interim/synthetic_raw.jsonl")

ASSET_CODE_RE = re.compile(r"\b[A-Za-z]{2}\d{3}\b")

# Rulebook section 7 abbreviations, matched case-insensitively on word
# boundaries. Slash forms (c/o, u/s, o/s, lhs/rhs, fr/rr, p/m) are matched
# literally; "pm" alone is also accepted since real texts often drop the
# slash.
ABBREV_PATTERNS = [
    r"c/o", r"u/s", r"o/s", r"lhs", r"rhs", r"fr", r"rr", r"brg", r"eng",
    r"p/m", r"\bpm\b", r"insp", r"repl", r"rplc", r"lkg", r"temp", r"vib",
    r"\bad\b", r"elec", r"hyd", r"chk",
]
ABBREV_RE = re.compile(r"\b(" + "|".join(ABBREV_PATTERNS) + r")\b", re.IGNORECASE)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_real_train_texts():
    records = load_jsonl(REAL_PATH)
    return [r["text"] for r in records if r.get("pool") == "train"]


def load_synthetic_texts():
    records = load_jsonl(SYNTHETIC_PATH)
    return [r["text"] for r in records]


def word_counts(texts):
    return [len(t.split()) for t in texts]


def share(texts, pattern):
    hits = sum(1 for t in texts if pattern.search(t))
    return hits / len(texts) if texts else 0.0


def share_all_caps(texts):
    hits = 0
    for t in texts:
        letters = [c for c in t if c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            hits += 1
    return hits / len(texts) if texts else 0.0


NONALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def normalize(text):
    text = text.lower()
    text = NONALNUM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def ngram_counts(texts, n):
    counts = Counter()
    for t in texts:
        words = normalize(t).split()
        for i in range(len(words) - n + 1):
            counts[" ".join(words[i : i + n])] += 1
    return counts


def report_corpus(name, texts):
    wc = word_counts(texts)
    print(f"\n{name}: {len(texts)} texts")
    print(f"  mean word count:   {statistics.mean(wc):.2f}")
    print(f"  median word count: {statistics.median(wc):.1f}")
    print(f"  asset code share:  {share(texts, ASSET_CODE_RE) * 100:.1f}%")
    print(f"  abbreviation share:{share(texts, ABBREV_RE) * 100:.1f}%")
    print(f"  ALL CAPS share:    {share_all_caps(texts) * 100:.1f}%")


def report_rare_phrases(real_texts, synth_texts, top_n=30):
    print(f"\nTop {top_n} 2/3-word phrases common in synthetic, rare in real:")
    rows = []
    for n in (2, 3):
        synth_counts = ngram_counts(synth_texts, n)
        real_counts = ngram_counts(real_texts, n)
        real_total = sum(real_counts.values()) or 1
        synth_total = sum(synth_counts.values()) or 1
        for phrase, synth_n in synth_counts.items():
            if synth_n < 3:
                continue
            real_n = real_counts.get(phrase, 0)
            real_rate = real_n / real_total
            synth_rate = synth_n / synth_total
            # "rare in real": real rate at most a fifth of the synthetic
            # rate (or absent entirely), and appears meaningfully in synthetic.
            if real_rate <= synth_rate / 5:
                rows.append((n, phrase, synth_n, real_n))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'n':>2} {'synth_count':>11} {'real_count':>10}  phrase")
    for n, phrase, synth_n, real_n in rows[:top_n]:
        print(f"  {n:>2} {synth_n:>11} {real_n:>10}  {phrase}")


def main():
    real_texts = load_real_train_texts()
    synth_texts = load_synthetic_texts()

    report_corpus("Real (train pool)", real_texts)
    report_corpus("Synthetic", synth_texts)
    report_rare_phrases(real_texts, synth_texts)


if __name__ == "__main__":
    main()
