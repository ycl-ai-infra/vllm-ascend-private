# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one full AISBench cohort, then one batch retry of length-ended requests.

Run in the AISBench Python environment with this directory on PYTHONPATH.
The work directory must be new. Original and post-retry metrics remain separate.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from minimax_m31_gpqa import (
    GPQA_Simple_Eval_postprocess,
    LiteralSafeGPQADataset,
    LiteralSafePromptTemplate,
    extract_non_reasoning_content,
)
from minimax_m31_gpqa_utils import (
    FIRST_PASS_TOKENS,
    MAX_OUTPUT_TOKENS,
    replace_truncated_results,
    score_records,
    truncated_ids,
)


def _rows(path):
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.split(b"\n") if line]


def audit_stage(stage, manifest, expected_ids, max_out_len):
    requests, responses = _rows(stage / "requests.jsonl"), _rows(stage / "responses.jsonl")
    request_map = {row["uuid"]: row for row in requests}
    response_map = {row["uuid"]: row for row in responses}
    if (
        len(requests) != len(responses)
        or len(request_map) != len(requests)
        or request_map.keys() != response_map.keys()
    ):
        raise ValueError("Missing or duplicate request/response UUIDs")
    if sorted(row["original_id"] for row in requests) != sorted(expected_ids):
        raise ValueError("Stage does not contain exactly the declared original sample IDs")
    predictions = {}
    for path in (stage / "results").glob("*/predictions/**/*.jsonl"):
        for row in _rows(path):
            predictions[row["uuid"]] = row
    if predictions.keys() != request_map.keys():
        raise ValueError("AISBench prediction UUIDs do not match responses")
    records = []
    for uuid, request in request_map.items():
        body = request["body"]
        if not (
            body["max_tokens"] == max_out_len
            and body["temperature"] == 0.6
            and body["top_p"] == 0.95
            and body["reasoning_effort"] == body["chat_template_kwargs"]["reasoning_effort"] == "max"
            and body["model"] == "minimax-m3"
            and body["stream"] is True
        ):
            raise ValueError("Actual request parameters differ from the declared evaluation protocol")
        sample_id = request["original_id"]
        reference = manifest[sample_id]
        digest = hashlib.sha256(body["messages"][0]["content"].encode()).hexdigest()
        if digest != reference["prompt_sha256"]:
            raise ValueError("Request prompt differs from the source manifest")
        prediction, response = predictions[uuid], response_map[uuid]
        if not prediction.get("success") or prediction["gold"] != reference["answer"]:
            raise ValueError("Failed prediction or mismatched gold mapping")
        # Confirm AISBench's local retry index still maps to the original global ID.
        if expected_ids[int(prediction["id"])] != sample_id or response["original_id"] != sample_id:
            raise ValueError("Retry prediction was mapped to the wrong original sample")
        answer = GPQA_Simple_Eval_postprocess(extract_non_reasoning_content(prediction.get("prediction") or ""))
        response_answer = GPQA_Simple_Eval_postprocess(
            response.get("content") or response.get("reasoning_content") or ""
        )
        if answer != response_answer:
            raise ValueError("Independent response and AISBench answer extraction disagree")
        records.append(
            dict(
                original_id=sample_id,
                uuid=uuid,
                gold=reference["answer"],
                answer=answer,
                correct=answer == reference["answer"],
                finish_reason=response["finish_reason"],
                output_tokens=response["output_tokens"],
            )
        )
    records.sort(key=lambda row: row["original_id"])
    result = score_records(records)
    (stage / "scored-records.json").write_text(json.dumps(records, indent=2))
    (stage / "final-audit.json").write_text(json.dumps(result, indent=2))
    return records


def main():
    """Declare a new cohort, audit the full pass, then replace every length retry.

    The lock and existing-plan check prevent duplicate benchmark controllers.
    Original IDs are assigned before filtering so retry gold rotation is stable.
    Metrics sampling covers both stages; neither stage can overwrite the other.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--aisbench-entry", type=Path, required=True)
    parser.add_argument("--experiment-note", default="")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11223)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "plan.json").exists():
        raise RuntimeError("This cohort was already declared; use a new work directory")
    dataset = LiteralSafeGPQADataset.load(str(args.dataset_dir), "gpqa_diamond.csv")
    if len(dataset) != 198:
        raise ValueError("Expected the complete 198-question GPQA Diamond split")
    template = LiteralSafePromptTemplate(template=dict(round=[dict(role="HUMAN", prompt="{gpqa_prompt}")]))
    manifest = []
    for row in dataset:
        actual = [item for item in template.generate_item(row, output_field="answer") if "prompt" in item]
        if len(actual) != 1 or actual[0]["prompt"] != row["gpqa_prompt"]:
            raise ValueError("AISBench prompt rendering changed a literal field")
        manifest.append(
            dict(
                original_id=row["original_id"],
                answer=row["answer"],
                prompt_sha256=hashlib.sha256(row["gpqa_prompt"].encode()).hexdigest(),
            )
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    plan = dict(
        declared_at=time.time(),
        batch_size=32,
        first_pass_max_out_len=FIRST_PASS_TOKENS,
        retry_max_out_len=MAX_OUTPUT_TOKENS,
        temperature=0.6,
        top_p=0.95,
        reasoning_effort="max",
        retry_policy="Once, only finish_reason=length; replace every retried result",
        prompt_protocol="One-pass rendering; preserve LaTeX braces; original full-split option rotation",
    )
    plan["experiment_note"] = args.experiment_note
    plan["dataset_sha256"] = hashlib.sha256((args.dataset_dir / "gpqa_diamond.csv").read_bytes()).hexdigest()
    plan["harness_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob("*minimax_m31_gpqa*.py")
    }
    if args.source_manifest:
        plan["serving_source_manifest"] = json.loads(args.source_manifest.read_text())
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{args.host}:{args.port}/v1/models", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("Service is not ready")
    stop = threading.Event()
    load_samples = []

    def monitor():
        while not stop.is_set():
            try:
                with opener.open(f"http://{args.host}:{args.port}/metrics", timeout=5) as response:
                    metrics = response.read().decode()
                inflight = sum(
                    float(line.rsplit(" ", 1)[-1])
                    for line in metrics.splitlines()
                    if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
                )
                load_samples.append(inflight)
                with (root / "server-load.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(time=time.time(), inflight=inflight)) + "\n")
            except OSError as error:
                print(f"Metrics sample failed: {error}", flush=True)
            stop.wait(10)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()

    def run_stage(name, ids, limit):
        stage = root / name
        stage.mkdir()
        indices = stage / "original-ids.json"
        indices.write_text(json.dumps(ids))
        config = stage / "config.py"
        config.write_text(
            "_base_ = []\nfrom minimax_m31_gpqa import make_config\n"
            + f"datasets, models = make_config(dataset_dir={str(args.dataset_dir)!r}, "
            + f"stage_dir={str(stage)!r}, manifest_path={str(manifest_path)!r}, max_out_len={limit}, "
            + f"indices_file={str(indices)!r}, host={args.host!r}, port={args.port})\n"
        )
        command = [
            sys.executable,
            str(args.aisbench_entry),
            str(config),
            "--mode",
            "all",
            "--num-warmups",
            "0",
            "--dump-eval-details",
            "--work-dir",
            str(stage / "results"),
        ]
        environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        with (stage / "aisbench.log").open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        (stage / "exit").write_text(str(result.returncode))
        result.check_returncode()
        return audit_stage(stage, manifest, ids, limit)

    try:
        first = run_stage("first-pass", list(range(198)), FIRST_PASS_TOKENS)
        ids = truncated_ids(first)
        retries = run_stage("length-retry", ids, MAX_OUTPUT_TOKENS) if ids else []
        combined = replace_truncated_results(first, retries)
        maximum = max(load_samples, default=None)
        if maximum is None or maximum > 32:
            raise RuntimeError(f"Invalid concurrency evidence: {maximum}")
        (root / "combined-records.json").write_text(json.dumps(combined, indent=2))
        result = dict(
            first_pass=score_records(first),
            after_one_length_retry=score_records(combined),
            retried_original_ids=ids,
            maximum_inflight=maximum,
            note="First-pass and post-retry scores are separate; no correct-answer cherry-picking",
        )
        (root / "final-audit.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        stop.set()
        watcher.join(timeout=10)


if __name__ == "__main__":
    main()
