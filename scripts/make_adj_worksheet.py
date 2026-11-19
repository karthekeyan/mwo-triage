"""Build a blind adjudication worksheet for the test+val labelling-pass
disagreements that scripts/adjudicate.py did not finish (API credit ran out
after only 11 items).

Reads data/interim/full_disagreements.jsonl, keeps pool in {test, val}
(--pools to change), and excludes any uid already present in
data/interim/adjudicated_testval.jsonl (--done-file) so the earlier partial
run isn't repeated.

Writes:
- data/interim/adj_worksheet.jsonl: one line per item, fields uid, text,
  labeller_1, labeller_2 (each {priority, route, rule}). No model, pass, or
  pool name -- the adjudicator (a human or an in-session Claude Code
  adjudicator) should see nothing but the text and the two anonymised
  answers.
- data/interim/adj_worksheet_key.json: {uid: "1"|"2"}, which labeller slot
  (1 or 2) holds labeller A's answers. Kept separate so the adjudicator can
  avoid reading it until after adjudication.

The labeller-1/labeller-2 order is randomised per uid with seed 42, drawn
after sorting by uid so it does not depend on input file order.
"""

import argparse
import json
import random
from pathlib import Path

ANON_SEED = 42


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/full_disagreements.jsonl")
    parser.add_argument("--done-file", default="data/interim/adjudicated_testval.jsonl")
    parser.add_argument("--pools", default="test,val")
    parser.add_argument("--worksheet-out", default="data/interim/adj_worksheet.jsonl")
    parser.add_argument("--key-out", default="data/interim/adj_worksheet_key.json")
    args = parser.parse_args()

    pools = {p.strip() for p in args.pools.split(",") if p.strip()}

    records = load_jsonl(args.input)
    records = [r for r in records if r["pool"] in pools]

    done_uids = set()
    if Path(args.done_file).exists():
        done_uids = {r["uid"] for r in load_jsonl(args.done_file)}
    records = [r for r in records if r["uid"] not in done_uids]
    records.sort(key=lambda r: r["uid"])

    rng = random.Random(ANON_SEED)
    worksheet = []
    key = {}
    for r in records:
        a = {"priority": r["a_priority"], "route": r["a_route"], "rule": r["a_rule"]}
        b = {"priority": r["b_priority"], "route": r["b_route"], "rule": r["b_rule"]}
        a_is_1 = rng.random() < 0.5
        worksheet.append({
            "uid": r["uid"],
            "text": r["text"],
            "labeller_1": a if a_is_1 else b,
            "labeller_2": b if a_is_1 else a,
        })
        key[r["uid"]] = "1" if a_is_1 else "2"

    worksheet_out = Path(args.worksheet_out)
    worksheet_out.parent.mkdir(parents=True, exist_ok=True)
    with open(worksheet_out, "w", encoding="utf-8") as f:
        for item in worksheet:
            f.write(json.dumps(item) + "\n")

    key_out = Path(args.key_out)
    key_out.write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(f"Loaded {len(load_jsonl(args.input))} disagreements, {len(done_uids)} already adjudicated, "
          f"{len(records)} pools={sorted(pools)} to worksheet.")
    print(f"Wrote {len(worksheet)} items to {worksheet_out}")
    print(f"Wrote key to {key_out}")


if __name__ == "__main__":
    main()
