"""Shared engine for sending batches of work orders to Claude: sync and
Message-Batches-API submission, polling, validation, resume, sync-mode
retry of failed/invalid batches, and cost accounting.

Callers (scripts/label_texts.py, scripts/adjudicate.py) supply the
task-specific pieces: the tool schema, how to build the user message for a
batch, and how to validate + merge a model's response into output records.
"""

import json
import random
import time
from pathlib import Path

from anthropic import BadRequestError

# Models known to reject a forced tool_choice via the sync Messages API (they
# only support "auto"). Discovered via BadRequestError; listed here so sync
# mode can pick the right tool_choice up front instead of a guaranteed error.
AUTO_ONLY_MODELS = {"claude-sonnet-5-5"}

MAX_RETRIES = 3
BATCH_POLL_SECONDS = 60
SAMPLE_SEED = 42

# $ per million tokens, (input, output). Cache reads are 5% of the input price.
PRICING_SYNC = {
    "sonnet": (2.0, 10.0),
    "opus": (4.0, 20.0),
}

# Message Batches API is priced at half the sync rate (plus Fable, which is
# currently batch-only for our purposes). Cache reads are still 5% of input.
PRICING_BATCH = {
    "sonnet": (1.0, 5.0),
    "opus": (2.0, 10.0),
    "fable": (4.0, 20.0),
}


def load_jsonl(path):
    records = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pricing_for_model(model, rates):
    model_lower = model.lower()
    for key, prices in rates.items():
        if key in model_lower:
            return prices
    raise ValueError(f"No pricing known for model '{model}'. Add it to the pricing table.")


def sync_initial_tool_choice(model, tool_name):
    """Sync (Messages API): most models accept a forced tool_choice; models
    in AUTO_ONLY_MODELS are known to reject it and need "auto" from the
    start."""
    if model in AUTO_ONLY_MODELS:
        return {"type": "auto"}
    return {"type": "tool", "name": tool_name}


def batch_tool_choice():
    """Message Batches API: forced tool_choice ("tool"/"any") is rejected
    for every model we've tried (sonnet and opus alike), even though sync
    accepts it for opus. Always submit batch requests with "auto"."""
    return {"type": "auto"}


def request_kwargs(model, system_blocks, message, tool, tool_choice):
    """Build request parameters identically for sync calls and batch requests."""
    return {
        "model": model,
        "max_tokens": 8192,
        "system": system_blocks,
        "tools": [tool],
        "tool_choice": tool_choice,
        "messages": [{"role": "user", "content": message}],
    }


def sample_limit(records, limit, seed=SAMPLE_SEED):
    if limit is not None and limit < len(records):
        return random.Random(seed).sample(records, limit)
    return records


def build_batches(records, output_path, batch_size, already_done_key="uid", shuffle_seed=None):
    already_done = {r[already_done_key] for r in load_jsonl(output_path)}
    todo = [r for r in records if r[already_done_key] not in already_done]
    print(f"Loaded {len(records)} records, {len(already_done)} already done, {len(todo)} to do.")

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(batches)
    return batches


def call_sync(client, model, system_blocks, batch, tool, items_field, build_message_fn, validate_fn, usage_log):
    """One request per batch using the model's known-good tool_choice,
    retried up to MAX_RETRIES times. Still falls back from a forced
    tool_choice to "auto" on a BadRequestError, in case a model not yet
    listed in AUTO_ONLY_MODELS turns out to need it too."""
    message = build_message_fn(batch)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                **request_kwargs(model, system_blocks, message, tool, sync_initial_tool_choice(model, tool["name"]))
            )
        except BadRequestError as exc:
            if "tool_choice" not in str(exc):
                raise
            # Some models (e.g. extended-thinking models) reject forced
            # tool_choice and only support "auto"; instructions already
            # direct the model to call the tool. These models also spend
            # output budget on a thinking block before the tool call, so
            # max_tokens=8192 gives enough headroom or the tool-call JSON
            # gets truncated mid-array.
            response = client.messages.create(
                **request_kwargs(model, system_blocks, message, tool, {"type": "auto"})
            )
        usage_log.append(response.usage)

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            last_error = "no tool_use block in response"
            continue

        items = tool_use.input.get(items_field, [])
        by_uid, error = validate_fn(batch, items)
        if error is None:
            return by_uid
        last_error = error
        print(f"    attempt {attempt}/{MAX_RETRIES} failed validation: {last_error}")

    raise RuntimeError(f"batch failed after {MAX_RETRIES} attempts: {last_error}")


def run_sync(client, model, system_blocks, batches, tool, items_field, build_message_fn, validate_fn, append_fn):
    usage_log = []
    for batch_idx, batch in enumerate(batches, start=1):
        print(f"Batch {batch_idx}/{len(batches)} ({len(batch)} items)...")
        try:
            by_uid = call_sync(client, model, system_blocks, batch, tool, items_field, build_message_fn,
                                validate_fn, usage_log)
        except RuntimeError as exc:
            print(f"  SKIPPING batch: {exc}")
            continue
        append_fn(batch, by_uid)
    return usage_log


def state_path_for(output_path):
    return Path(str(output_path) + ".batch_state.json")


def submit_batch(client, model, system_blocks, batches, tool, build_message_fn, state_path):
    requests = []
    batch_map = {}
    for idx, batch in enumerate(batches):
        custom_id = f"batch-{idx:04d}"
        batch_map[custom_id] = batch
        requests.append({
            "custom_id": custom_id,
            "params": request_kwargs(model, system_blocks, build_message_fn(batch), tool, batch_tool_choice()),
        })

    message_batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {message_batch.id} with {len(requests)} requests.")

    state = {"batch_id": message_batch.id, "model": model, "batches": batch_map}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    print(f"Saved batch state to {state_path}")
    return state


def poll_batch(client, batch_id):
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        counts = message_batch.request_counts
        print(
            f"  status={message_batch.processing_status} "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}"
        )
        if message_batch.processing_status == "ended":
            return message_batch
        time.sleep(BATCH_POLL_SECONDS)


def collect_batch_results(client, batch_id, batch_map, tool, items_field, validate_fn, append_fn, usage_log):
    """Validate each batch's result; return the list of batches (as record
    lists) that failed or came back invalid, for sync retry."""
    results_by_custom_id = {}
    for line in client.messages.batches.results(batch_id):
        results_by_custom_id[line.custom_id] = line.result

    failed_batches = []
    for custom_id, batch in batch_map.items():
        result = results_by_custom_id.get(custom_id)
        if result is None or result.type != "succeeded":
            reason = result.type if result is not None else "missing from results"
            print(f"  {custom_id}: {reason}, will retry in sync mode")
            failed_batches.append(batch)
            continue

        message = result.message
        usage_log.append(message.usage)
        tool_use = next((b for b in message.content if b.type == "tool_use"), None)
        if tool_use is None:
            print(f"  {custom_id}: no tool_use block, will retry in sync mode")
            failed_batches.append(batch)
            continue

        items = tool_use.input.get(items_field, [])
        by_uid, error = validate_fn(batch, items)
        if error is not None:
            print(f"  {custom_id}: invalid ({error}), will retry in sync mode")
            failed_batches.append(batch)
            continue

        append_fn(batch, by_uid)

    return failed_batches


def run_batch(client, model, system_blocks, batches, tool, items_field, build_message_fn, validate_fn, append_fn,
              state_path, resume_only):
    if resume_only:
        if not state_path.exists():
            raise FileNotFoundError(f"--resume-batch given but no state file at {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch_id = state["batch_id"]
        batch_map = state["batches"]
        print(f"Resuming batch {batch_id} from {state_path}")
    else:
        state = submit_batch(client, model, system_blocks, batches, tool, build_message_fn, state_path)
        batch_id = state["batch_id"]
        batch_map = state["batches"]

    poll_batch(client, batch_id)

    batch_usage_log = []
    failed_batches = collect_batch_results(client, batch_id, batch_map, tool, items_field, validate_fn, append_fn,
                                            batch_usage_log)

    sync_retry_usage_log = []
    if failed_batches:
        print(f"\nRetrying {len(failed_batches)} failed/invalid batch(es) in sync mode...")
        sync_retry_usage_log = run_sync(client, model, system_blocks, failed_batches, tool, items_field,
                                         build_message_fn, validate_fn, append_fn)

    return batch_usage_log, sync_retry_usage_log


def usage_totals(usage_log):
    return {
        "input": sum(u.input_tokens for u in usage_log),
        "output": sum(u.output_tokens for u in usage_log),
        "cache_read": sum(getattr(u, "cache_read_input_tokens", 0) or 0 for u in usage_log),
        "cache_write": sum(getattr(u, "cache_creation_input_tokens", 0) or 0 for u in usage_log),
    }


def cost_for_usage(totals, model, rates):
    input_price, output_price = pricing_for_model(model, rates)
    cache_read_price = input_price * 0.05
    return (
        (totals["input"] + totals["cache_write"]) / 1_000_000 * input_price
        + totals["cache_read"] / 1_000_000 * cache_read_price
        + totals["output"] / 1_000_000 * output_price
    )


def print_usage_and_cost(model, *priced_usage_logs):
    """priced_usage_logs: one or more (usage_log, rates, label) groups, each
    priced at its own rates (e.g. batch-priced requests vs. sync-mode
    retries), so mixing modes never bills sync tokens at batch rates or
    vice versa."""
    grand_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    total_requests = 0
    total_cost = 0.0

    print("\n=== Token usage ===")
    for usage_log, rates, label in priced_usage_logs:
        if not usage_log:
            continue
        totals = usage_totals(usage_log)
        cost = cost_for_usage(totals, model, rates)
        total_requests += len(usage_log)
        total_cost += cost
        for key in grand_totals:
            grand_totals[key] += totals[key]

        print(f"  [{label}] requests: {len(usage_log)}")
        print(f"  [{label}] input tokens (non-cached): {totals['input']}")
        print(f"  [{label}] cache write tokens: {totals['cache_write']}")
        print(f"  [{label}] cache read tokens: {totals['cache_read']}")
        print(f"  [{label}] output tokens: {totals['output']}")
        print(f"  [{label}] cost: ${cost:.4f}")

    print(f"  requests: {total_requests}")
    print(f"  input tokens (non-cached): {grand_totals['input']}")
    print(f"  cache write tokens: {grand_totals['cache_write']}")
    print(f"  cache read tokens: {grand_totals['cache_read']}")
    print(f"  output tokens: {grand_totals['output']}")
    print(f"  estimated cost: ${total_cost:.4f}")
