"""Apply the brake/steering P1-SAFETY ruling to the agreed records in
full_pass_a.jsonl and full_pass_b.jsonl, and check two already-adjudicated
output files for the same overreach.

Ruling: a brake or steering fault, u/s, not working or failure on mobile
equipment is STOP_NOW with rule P1-SAFETY. Wear, adjustment or inspection
stays as labelled. Excluded from the override even when the fault/u-s
wording is present, because these describe something other than the brake
or steering mechanism itself failing:
  - a warning/fault light only (the indicator, not the brake/steering)
  - a brake cooling-hose/circuit (secondary cooling, not primary function)
  - a steering-wheel comfort/adjustment feature (telescoping, tilt)
  - brake lights, i.e. the vehicle's signal/stop lights (electrical, not
    the brake mechanism)
These keep their existing priority/route/rule.

"Agreed records" are uids common to both passes where pass A and pass B
already agree on priority and route (disagreements are handled separately
by scripts/adjudicate.py).

This does not edit full_pass_a.jsonl / full_pass_b.jsonl. It writes
data/interim/rulings_overrides.jsonl, one line per record that needs a
priority and/or route change, with the uid, the new value(s), the old
labels from both passes, and the reason.

It also scans data/interim/adj_inline.jsonl and
data/interim/adjudicated_testval.jsonl (outputs of earlier adjudication
runs) for any record that: names a brake or steering component, falls into
one of the excluded categories above, and is currently STOP_NOW/P1-SAFETY
-- i.e. was set by this same ruling before the exclusions existed. Any such
record is flagged and reported (none were found at the time this was
written: see the printed report).
"""

import json
import re
from pathlib import Path

PASS_A_PATH = Path("data/interim/full_pass_a.jsonl")
PASS_B_PATH = Path("data/interim/full_pass_b.jsonl")
OUTPUT_PATH = Path("data/interim/rulings_overrides.jsonl")
TEXTS_PATH = Path("data/interim/real_all_pools.jsonl")
CHECK_PATHS = [
    Path("data/interim/adj_inline.jsonl"),
    Path("data/interim/adjudicated_testval.jsonl"),
]

NEW_PRIORITY = "STOP_NOW"
NEW_RULE = "P1-SAFETY"
PRIORITY_REASON = "Brake/steering ruling: fault, u/s, not working or failure on mobile equipment is STOP_NOW/P1-SAFETY"

COMPONENT_RE = re.compile(r"\b(brakes?|steering|steer)\b", re.I)
CONDITION_RE = re.compile(
    r"\bfault\w*\b|\bu/?s\b|\bunserviceable\b|\bnot\s+working\b|\bn/w\b|\bfailure\b|\bfailed\b",
    re.I,
)
INSPECT_FOR_FAULT_RE = re.compile(r"insp\w*.{0,25}fault|fault.{0,25}insp\w*", re.I)

# Excluded categories: warning/fault light only, brake cooling-hose/circuit,
# steering-wheel comfort/adjustment feature, brake (signal) lights.
EXCLUDE_RE = re.compile(
    r"(warning|fault)\s+light|\bcool\w*|telescop\w*|\bbrake\s+lights?\b",
    re.I,
)


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["uid"]] = record
    return records


def matches_ruling(text):
    if not COMPONENT_RE.search(text):
        return False
    if not CONDITION_RE.search(text):
        return False
    if INSPECT_FOR_FAULT_RE.search(text):
        return False
    if EXCLUDE_RE.search(text):
        return False
    return True


def build_overrides():
    pass_a = load_jsonl(PASS_A_PATH)
    pass_b = load_jsonl(PASS_B_PATH)

    common_uids = sorted(set(pass_a) & set(pass_b))
    agreed_uids = [
        u for u in common_uids
        if pass_a[u]["priority"] == pass_b[u]["priority"]
        and pass_a[u]["route"] == pass_b[u]["route"]
    ]

    overrides = []
    for uid in agreed_uids:
        a, b = pass_a[uid], pass_b[uid]

        if a["priority"] == NEW_PRIORITY or not matches_ruling(a["text"]):
            continue

        overrides.append({
            "uid": uid,
            "text": a["text"],
            "pool": a["pool"],
            "old_priority": a["priority"],
            "old_route": a["route"],
            "old_rule_a": a["rule"],
            "old_rule_b": b["rule"],
            "new_priority": NEW_PRIORITY,
            "new_rule": NEW_RULE,
            "reason": PRIORITY_REASON,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in overrides:
            f.write(json.dumps(record) + "\n")

    print(f"{len(overrides)} overrides written to {OUTPUT_PATH}")
    return overrides


def load_texts():
    texts = {}
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                texts[record["uid"]] = record["text"]
    return texts


def find_excluded_category_overreach():
    """Find records in CHECK_PATHS that are currently STOP_NOW/P1-SAFETY,
    name a brake or steering component, and fall into one of the excluded
    categories. These files carry no record of their pre-ruling label, so
    there is nothing principled to revert *to* automatically -- any match
    is reported for a manual decision rather than silently rewritten."""
    texts = load_texts()
    flagged = []

    for path in CHECK_PATHS:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("priority") != NEW_PRIORITY or record.get("rule") != NEW_RULE:
                    continue
                text = record.get("text") or texts.get(record["uid"], "")
                if not (COMPONENT_RE.search(text) and EXCLUDE_RE.search(text)):
                    continue
                flagged.append({
                    "file": str(path),
                    "uid": record["uid"],
                    "text": text,
                    "old_priority": record["priority"],
                    "old_rule": record["rule"],
                })

    return flagged


def main():
    overrides = build_overrides()
    flagged = find_excluded_category_overreach()
    if flagged:
        print(f"{len(flagged)} excluded-category STOP_NOW/P1-SAFETY records found -- needs a manual decision:")
        for r in flagged:
            print(f"  {r['file']} {r['uid']}: {r['text']!r}")
    else:
        print("No excluded-category STOP_NOW/P1-SAFETY records found in adj_inline.jsonl or adjudicated_testval.jsonl.")
    return overrides, flagged


if __name__ == "__main__":
    main()
