"""Adjudicate labelling-pass disagreements with Claude, following
docs/labelling-rulebook.md.

Input is data/interim/full_disagreements.jsonl (from compare_passes.py),
filtered to the pools given by --pools (default: test and val -- the pools
that end up in the real-text test set, so these are the disagreements that
matter most). For each disagreement, the adjudicator sees the text and both
labellers' priority/route/rule, anonymised as "Labeller 1" and "Labeller 2"
in an order randomised per item (seed 42) so position never signals which
labeller is which. The adjudicator may KEEP (choosing any valid label, not
only the two offered) or DROP (only if the text cannot be labelled under the
rulebook at all).

Batch mode submits every batch through the Message Batches API with
tool_choice "auto", polls until it ends, validates and appends the results,
and resends any failed or invalid batches in sync mode. Sync mode sends one
tool-forced-or-auto request per batch directly. Both modes resume: already
adjudicated uids in --output are skipped.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mwo_triage.claude_batch import (  # noqa: E402
    PRICING_BATCH,
    PRICING_SYNC,
    build_batches,
    load_jsonl,
    print_usage_and_cost,
    run_batch,
    run_sync,
    sample_limit,
    state_path_for,
)
from mwo_triage.rulebook_labels import (  # noqa: E402
    PRIORITIES,
    ROUTES,
    RULE_CODES,
    RULE_CODE_TABLE,
    RULEBOOK_VERSION,
)

RULEBOOK_PATH = Path("docs/labelling-rulebook.md")

ANON_SEED = 42

DECISIONS_FIELD = "decisions"

SUBMIT_DECISIONS_TOOL = {
    "name": "submit_decisions",
    "description": "Submit the adjudication decision for every disagreement in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": ["KEEP", "DROP"],
                            "description": "KEEP to give a final label, DROP only if the text cannot be "
                                           "labelled under the rulebook at all.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": PRIORITIES,
                            "description": "Required when decision is KEEP. Any valid priority, not only "
                                           "the two offered.",
                        },
                        "route": {
                            "type": "string",
                            "enum": ROUTES,
                            "description": "Required when decision is KEEP. Any valid route, not only the "
                                           "two offered.",
                        },
                        "rule": {
                            "type": "string",
                            "maxLength": 12,
                            "enum": list(RULE_CODES.keys()),
                            "description": "Required when decision is KEEP: the rule code that decided this "
                                           "label. Code only, no explanation.",
                        },
                    },
                    "required": ["uid", "decision"],
                },
            }
        },
        "required": ["decisions"],
    },
}

INSTRUCTIONS = f"""
You are adjudicating disagreements between two independent labellers of
maintenance work orders for the mwo-triage dataset, each of whom labelled
every text with a priority, a route, and the rule code that decided it.

Decide strictly by the rulebook above. Use only what is written in the text;
do not assume facts that are not there. You are not limited to the two
labellers' answers -- choose whichever priority and route the rulebook
actually supports, even if neither labeller gave it.

For every disagreement, call the submit_decisions tool with exactly one
entry per uid, in any order:
- decision=KEEP with a priority, a route, and the rule code (from the list
  below, code only, no explanation) that decided them.
- decision=DROP, with no priority/route/rule, only if the text cannot be
  labelled under the rulebook at all (e.g. too vague, unreadable, not a
  maintenance job at all with no salvageable component/symptom).

The rule code must be exactly one of the following:

{RULE_CODE_TABLE}
""".strip()


def build_system_blocks(rulebook_text):
    return [
        {"type": "text", "text": rulebook_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": INSTRUCTIONS},
    ]


def assign_anon_order(records):
    """Decide, per uid and independent of batching/filtering order, whether
    Labeller 1 shown to the adjudicator is A or B. Sorting by uid before
    drawing keeps this reproducible regardless of input file order."""
    rng = random.Random(ANON_SEED)
    for record in sorted(records, key=lambda r: r["uid"]):
        record["_label1_is_a"] = rng.random() < 0.5
    return records


def build_user_message(batch):
    lines = []
    for i, r in enumerate(batch):
        if r["_label1_is_a"]:
            l1 = (r["a_priority"], r["a_route"], r["a_rule"])
            l2 = (r["b_priority"], r["b_route"], r["b_rule"])
        else:
            l1 = (r["b_priority"], r["b_route"], r["b_rule"])
            l2 = (r["a_priority"], r["a_route"], r["a_rule"])
        lines.append(
            f"{i + 1}. uid={r['uid']} text={r['text']!r}\n"
            f"   Labeller 1: priority={l1[0]} route={l1[1]} rule={l1[2]}\n"
            f"   Labeller 2: priority={l2[0]} route={l2[1]} rule={l2[2]}"
        )
    return "Adjudicate each of the following disagreements:\n\n" + "\n".join(lines)


def validate_decisions(batch, decisions):
    expected_uids = {r["uid"] for r in batch}
    seen_uids = set()
    by_uid = {}
    for item in decisions:
        uid = item.get("uid")
        if uid is None or uid in seen_uids:
            return None, f"duplicate or missing uid: {uid!r}"
        seen_uids.add(uid)
        decision = item.get("decision")
        if decision not in ("KEEP", "DROP"):
            return None, f"invalid decision for uid {uid}: {decision!r}"
        if decision == "KEEP":
            if item.get("priority") not in PRIORITIES:
                return None, f"invalid priority for uid {uid}: {item.get('priority')!r}"
            if item.get("route") not in ROUTES:
                return None, f"invalid route for uid {uid}: {item.get('route')!r}"
            if item.get("rule") not in RULE_CODES:
                return None, f"invalid rule code for uid {uid}: {item.get('rule')!r}"
        by_uid[uid] = item
    if seen_uids != expected_uids:
        missing = expected_uids - seen_uids
        extra = seen_uids - expected_uids
        return None, f"uid mismatch: missing={missing} extra={extra}"
    return by_uid, None


def matched_labeller(decision, record):
    """Which original labeller (de-anonymised) the final KEEP label matches:
    "A", "B", "both", or "neither". None for a DROP decision."""
    if decision["decision"] != "KEEP":
        return None
    priority, route = decision["priority"], decision["route"]
    a_match = priority == record["a_priority"] and route == record["a_route"]
    b_match = priority == record["b_priority"] and route == record["b_route"]
    if a_match and b_match:
        return "both"
    if a_match:
        return "A"
    if b_match:
        return "B"
    return "neither"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/full_disagreements.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pools", default="test,val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mode", choices=["sync", "batch"], default="sync")
    parser.add_argument("--resume-batch", action="store_true")
    args = parser.parse_args()

    pools = {p.strip() for p in args.pools.split(",") if p.strip()}

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    rulebook_text = RULEBOOK_PATH.read_text(encoding="utf-8")
    system_blocks = build_system_blocks(rulebook_text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def append_fn(batch, by_uid):
        with open(out_path, "a", encoding="utf-8") as f:
            for record in batch:
                decision = by_uid[record["uid"]]
                is_keep = decision["decision"] == "KEEP"
                out_record = {
                    "uid": record["uid"],
                    "pool": record["pool"],
                    "text": record["text"],
                    "decision": decision["decision"],
                    "priority": decision["priority"] if is_keep else None,
                    "route": decision["route"] if is_keep else None,
                    "rule": decision["rule"] if is_keep else None,
                    "matched": matched_labeller(decision, record),
                    "model": args.model,
                    "rulebook_version": RULEBOOK_VERSION,
                }
                f.write(json.dumps(out_record) + "\n")

    records = load_jsonl(args.input)
    records = [r for r in records if r["pool"] in pools]
    records.sort(key=lambda r: r["uid"])
    records = assign_anon_order(records)

    if args.mode == "sync":
        records = sample_limit(records, args.limit)
        batches = build_batches(records, out_path, args.batch_size)
        usage_log = run_sync(client, args.model, system_blocks, batches, SUBMIT_DECISIONS_TOOL, DECISIONS_FIELD,
                              build_user_message, validate_decisions, append_fn)
        print_usage_and_cost(args.model, (usage_log, PRICING_SYNC, "sync"))
    else:
        state_path = state_path_for(out_path)
        if args.resume_batch:
            batches = None
        else:
            records = sample_limit(records, args.limit)
            batches = build_batches(records, out_path, args.batch_size)
        batch_usage_log, sync_retry_usage_log = run_batch(
            client, args.model, system_blocks, batches, SUBMIT_DECISIONS_TOOL, DECISIONS_FIELD, build_user_message,
            validate_decisions, append_fn, state_path, args.resume_batch,
        )
        print_usage_and_cost(
            args.model,
            (batch_usage_log, PRICING_BATCH, "batch"),
            (sync_retry_usage_log, PRICING_SYNC, "sync retry"),
        )


if __name__ == "__main__":
    main()
