# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[4] / "tests/e2e/models/minimax_m31_gpqa_utils.py"
SPEC = importlib.util.spec_from_file_location("minimax_m31_gpqa_utils", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def test_prompt_substitution_does_not_reinterpret_latex_braces():
    row = dict(question=r"Find \vec{A} + \vec{B} and {answer}.", A="option {B}", B="second", C="third", D="fourth")
    prompt = utils.render_prompt(row)
    assert row["question"] in prompt
    assert "A) option {B}\nB) second" in prompt


@pytest.mark.parametrize("value", [0, -1, 81921, True, 36000.0])
def test_output_limit_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        utils.validate_output_limit(value)


@pytest.mark.parametrize("value", [36000, 81920])
def test_output_limit_accepts_requested_protocol(value):
    assert utils.validate_output_limit(value) == value


def test_retry_is_only_length_and_replaces_correct_with_wrong():
    first = [
        dict(original_id=77, finish_reason="length", correct=True),
        dict(original_id=107, finish_reason="stop", correct=False),
    ]
    retry = [dict(original_id=77, finish_reason="stop", correct=False)]
    assert utils.truncated_ids(first) == [77]
    combined = utils.replace_truncated_results(first, retry)
    assert not any(row["correct"] for row in combined)
    assert utils.score_records(first)["nontruncated_correct"] == 0
    assert utils.score_records(combined)["correct"] == 0


@pytest.mark.parametrize("reason", [None, "error", "timeout"])
def test_errors_cannot_be_excluded_as_truncation(reason):
    with pytest.raises(ValueError):
        utils.truncated_ids([dict(original_id=0, finish_reason=reason)])


def test_retry_requires_exact_original_ids():
    first = [dict(original_id=7, finish_reason="length", correct=False)]
    for retry in [[], [dict(original_id=0, finish_reason="stop")], first + first]:
        with pytest.raises(ValueError):
            utils.replace_truncated_results(first, retry)


def test_truncated_correct_is_removed_from_both_nontruncated_counts():
    rows = [
        dict(original_id=0, finish_reason="length", correct=True),
        dict(original_id=1, finish_reason="stop", correct=True),
        dict(original_id=2, finish_reason="stop", correct=False),
    ]
    result = utils.score_records(rows)
    assert result["correct"] == 2 and result["completed"] == 3
    assert result["nontruncated_correct"] == 1 and result["nontruncated"] == 2
    assert result["nontruncated_accuracy"] == 50
