"""Validate data/interim/adj_inline.jsonl (or any adjudication output) against
its worksheet, then report agreement with the anonymised labellers.

Checks:
- Every uid in the worksheet has exactly one decision in the adjudication
  file, and no unexpected uids are present.
- Every KEEP decision has a valid priority, route, and rule code (no DROP
  leaves these fields non-null).

Report (uses the key file to de-anonymise labeller_1/labeller_2 back to A/B):
- How often the final decision's (priority, route) matches labeller A only,
  labeller B only, both, or neither.
- The DROP count.
- --sample N random decisions (default 10, seed 42) with text, A's and B's
  original labels, and the final decision.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from mwo_triage.rulebook_labels import PRIORITIES, ROUTES, RULE_CODES


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def matched_labeller(decision_priority, decision_route, a, b):
    a_match = decision_priority == a["priority"] and decision_route == a["route"]
    b_match = decision_priority == b["priority"] and decision_route == b["route"]
    if a_match and b_match:
        return "both"
    if a_match:
        return "A"
    if b_match:
        return "B"
    return "neither"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worksheet", default="data/interim/adj_worksheet.jsonl")
    parser.add_argument("--key-file", default="data/interim/adj_worksheet_key.json")
    parser.add_argument("--decisions", default="data/interim/adj_inline.jsonl")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    worksheet = {item["uid"]: item for item in load_jsonl(args.worksheet)}
    decisions = load_jsonl(args.decisions)
    key = json.loads(Path(args.key_file).read_text(encoding="utf-8"))

    decisions_by_uid = {}
    for d in decisions:
        uid = d["uid"]
        if uid not in worksheet:
            raise ValueError(f"decision for uid {uid} is not in the worksheet")
        if uid in decisions_by_uid:
            raise ValueError(f"duplicate decision for uid {uid}")
        decisions_by_uid[uid] = d

    missing = set(worksheet) - set(decisions_by_uid)
    if missing:
        raise ValueError(f"{len(missing)} worksheet uids have no decision: {sorted(missing)[:10]}...")

    for uid, d in decisions_by_uid.items():
        if d["decision"] not in ("KEEP", "DROP"):
            raise ValueError(f"uid {uid}: invalid decision {d['decision']!r}")
        if d["decision"] == "KEEP":
            if d.get("priority") not in PRIORITIES:
                raise ValueError(f"uid {uid}: invalid priority {d.get('priority')!r}")
            if d.get("route") not in ROUTES:
                raise ValueError(f"uid {uid}: invalid route {d.get('route')!r}")
            if d.get("rule") not in RULE_CODES:
                raise ValueError(f"uid {uid}: invalid rule code {d.get('rule')!r}")
        else:
            if d.get("priority") is not None or d.get("route") is not None or d.get("rule") is not None:
                raise ValueError(f"uid {uid}: DROP decision must not carry priority/route/rule")

    print(f"Validated {len(decisions_by_uid)} decisions against {len(worksheet)} worksheet items: OK")

    match_counts = Counter()
    drop_count = 0
    enriched = {}
    for uid, item in worksheet.items():
        d = decisions_by_uid[uid]
        a_is_1 = key[uid] == "1"
        a = item["labeller_1"] if a_is_1 else item["labeller_2"]
        b = item["labeller_2"] if a_is_1 else item["labeller_1"]
        if d["decision"] == "DROP":
            drop_count += 1
            matched = None
        else:
            matched = matched_labeller(d["priority"], d["route"], a, b)
            match_counts[matched] += 1
        enriched[uid] = {"text": item["text"], "a": a, "b": b, "decision": d, "matched": matched}

    print("\n=== Agreement with original labellers (de-anonymised) ===")
    for key_name in ("A", "B", "both", "neither"):
        print(f"  matched {key_name}: {match_counts.get(key_name, 0)}")
    print(f"  DROP: {drop_count}")
    print(f"  total: {len(worksheet)}")

    if args.sample:
        print(f"\n=== {min(args.sample, len(enriched))} random decisions (seed {args.seed}) ===")
        sample_uids = random.Random(args.seed).sample(list(enriched), min(args.sample, len(enriched)))
        for uid in sample_uids:
            e = enriched[uid]
            d = e["decision"]
            print(f"\nuid={uid}")
            print(f"  text: {e['text']!r}")
            print(f"  A: priority={e['a']['priority']} route={e['a']['route']} rule={e['a']['rule']}")
            print(f"  B: priority={e['b']['priority']} route={e['b']['route']} rule={e['b']['rule']}")
            if d["decision"] == "DROP":
                print("  Final: DROP")
            else:
                print(f"  Final: priority={d['priority']} route={d['route']} rule={d['rule']} "
                      f"matched={e['matched']}")


if __name__ == "__main__":
    main()
