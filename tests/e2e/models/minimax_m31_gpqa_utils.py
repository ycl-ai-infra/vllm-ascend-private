# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helpers for the native MiniMax GPQA evaluation protocol."""

MAX_OUTPUT_TOKENS = 81920
FIRST_PASS_TOKENS = 36000
GPQA_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    """

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""
)


def render_prompt(row):
    # A single formatting pass preserves literal LaTeX {A}/{B} in field values.
    return GPQA_PROMPT.format_map(row)


def validate_output_limit(value):
    if type(value) is not int or not 0 < value <= MAX_OUTPUT_TOKENS:
        raise ValueError(f"Output length must be between 1 and {MAX_OUTPUT_TOKENS}")
    return value


def index_records(records):
    indexed = {}
    for row in records:
        sample_id = row["original_id"]
        if type(sample_id) is not int or sample_id < 0 or sample_id in indexed:
            raise ValueError("Sample IDs must be unique nonnegative integers")
        if row.get("finish_reason") not in ("length", "stop"):
            raise ValueError("A missing/error finish reason cannot be counted as truncation")
        indexed[sample_id] = row
    return indexed


def truncated_ids(records):
    return sorted(sample_id for sample_id, row in index_records(records).items() if row["finish_reason"] == "length")


def replace_truncated_results(first_pass, retries):
    first, retry = index_records(first_pass), index_records(retries)
    expected = set(truncated_ids(first_pass))
    if set(retry) != expected:
        raise ValueError("Retry records must cover exactly the first-pass length-ended samples")
    # Replace every retry, including correct -> wrong; never select the better answer.
    return [retry.get(sample_id, first[sample_id]) for sample_id in sorted(first)]


def score_records(records):
    rows = list(index_records(records).values())
    nontruncated = [row for row in rows if row["finish_reason"] != "length"]
    correct = sum(bool(row["correct"]) for row in rows)
    nontruncated_correct = sum(bool(row["correct"]) for row in nontruncated)
    return dict(
        completed=len(rows),
        correct=correct,
        truncated=len(rows) - len(nontruncated),
        full_accuracy=100 * correct / len(rows) if rows else None,
        nontruncated=len(nontruncated),
        nontruncated_correct=nontruncated_correct,
        nontruncated_accuracy=100 * nontruncated_correct / len(nontruncated) if nontruncated else None,
        passed=bool(nontruncated) and 100 * nontruncated_correct > 91 * len(nontruncated),
    )
