"""Label work-order texts with Claude, following docs/labelling-rulebook.md.

Sync mode sends one tool-forced request per batch, validates the response
against the rulebook's label enums, and appends results to --output so a run
can resume. Batch mode submits every batch through the Message Batches API,
polls until it ends, validates and appends the results the same way, and
resends any failed or invalid batches in sync mode.
"""

import argparse
import json
import os
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

PASS_SEEDS = {"A": 1001, "B": 1002}

LABELS_FIELD = "labels"

SUBMIT_LABELS_TOOL = {
    "name": "submit_labels",
    "description": "Submit priority and route labels for every work order in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string"},
                        "priority": {"type": "string", "enum": PRIORITIES},
                        "route": {"type": "string", "enum": ROUTES},
                        "rule": {
                            "type": "string",
                            "maxLength": 12,
                            "enum": list(RULE_CODES.keys()),
                            "description": "The rule code (from the system prompt's code list) that decided this label. Code only, no explanation.",
                        },
                    },
                    "required": ["uid", "priority", "route", "rule"],
                },
            }
        },
        "required": ["labels"],
    },
}

INSTRUCTIONS = f"""
You are labelling maintenance work orders for the mwo-triage dataset.

Label each work order strictly by the rulebook above. Use only what is
written in the text; do not assume facts that are not there. Ignore urgency
words ("urgent", "ASAP", "critical", "!!!") -- they do not change priority by
themselves.

For every work order, call the submit_labels tool with exactly one entry per
uid, in any order, giving priority, route, and a rule code. The rule code
must be exactly one of the following (code only, no explanation text):

{RULE_CODE_TABLE}
""".strip()


def build_system_blocks(rulebook_text):
    return [
        {"type": "text", "text": rulebook_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": INSTRUCTIONS},
    ]


def build_user_message(batch):
    lines = [f"{i + 1}. uid={r['uid']} text={r['text']!r}" for i, r in enumerate(batch)]
    return "Label each of the following work orders:\n\n" + "\n".join(lines)


def validate_labels(batch, labels):
    expected_uids = {r["uid"] for r in batch}
    seen_uids = set()
    by_uid = {}
    for item in labels:
        uid = item.get("uid")
        if uid is None or uid in seen_uids:
            return None, f"duplicate or missing uid: {uid!r}"
        seen_uids.add(uid)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pass-name", required=True, choices=["A", "B"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mode", choices=["sync", "batch"], default="sync")
    parser.add_argument("--resume-batch", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    rulebook_text = RULEBOOK_PATH.read_text(encoding="utf-8")
    system_blocks = build_system_blocks(rulebook_text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def append_fn(batch, by_uid):
        with open(out_path, "a", encoding="utf-8") as f:
            for record in batch:
                label = by_uid[record["uid"]]
                out_record = dict(record)
                out_record["priority"] = label["priority"]
                out_record["route"] = label["route"]
                out_record["rule"] = label["rule"]
                out_record["model"] = args.model
                out_record["pass"] = args.pass_name
                out_record["rulebook_version"] = RULEBOOK_VERSION
                f.write(json.dumps(out_record) + "\n")

    records = load_jsonl(args.input)

    if args.mode == "sync":
        records = sample_limit(records, args.limit)
        batches = build_batches(records, out_path, args.batch_size, shuffle_seed=PASS_SEEDS[args.pass_name])
        usage_log = run_sync(client, args.model, system_blocks, batches, SUBMIT_LABELS_TOOL, LABELS_FIELD,
                              build_user_message, validate_labels, append_fn)
        print_usage_and_cost(args.model, (usage_log, PRICING_SYNC, "sync"))
    else:
        state_path = state_path_for(out_path)
        if args.resume_batch:
            batches = None
        else:
            records = sample_limit(records, args.limit)
            batches = build_batches(records, out_path, args.batch_size, shuffle_seed=PASS_SEEDS[args.pass_name])
        batch_usage_log, sync_retry_usage_log = run_batch(
            client, args.model, system_blocks, batches, SUBMIT_LABELS_TOOL, LABELS_FIELD, build_user_message,
            validate_labels, append_fn, state_path, args.resume_batch,
        )
        print_usage_and_cost(
            args.model,
            (batch_usage_log, PRICING_BATCH, "batch"),
            (sync_retry_usage_log, PRICING_SYNC, "sync retry"),
        )


if __name__ == "__main__":
    main()
