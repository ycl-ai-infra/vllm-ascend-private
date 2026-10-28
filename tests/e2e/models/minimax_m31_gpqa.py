# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AISBench adapter: literal-safe prompts, stable sample IDs and response audit.

Put this directory on PYTHONPATH in the AISBench environment. The dataset is
loaded in full before selecting retries, preserving official option rotation.
"""

import hashlib
import json
from pathlib import Path

from ais_bench.benchmark.datasets import GPQA_Simple_Eval_postprocess, GPQADataset, GPQAEvaluator
from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
from datasets import Dataset
from minimax_m31_gpqa_utils import render_prompt, validate_output_limit


class LiteralSafeGPQADataset(GPQADataset):
    @staticmethod
    def load(path, name, indices_file=None, **kwargs):
        original = GPQADataset.load(path, name, **kwargs)
        indices = json.loads(Path(indices_file).read_text()) if indices_file else list(range(len(original)))
        if len(indices) != len(set(indices)) or any(
            type(index) is not int or not 0 <= index < len(original) for index in indices
        ):
            raise ValueError("Retry selection contains duplicate or invalid original IDs")
        return Dataset.from_list(
            [
                dict(gpqa_prompt=render_prompt(original[index]), answer=original[index]["answer"], original_id=index)
                for index in indices
            ]
        )


class LiteralSafePromptTemplate(PromptTemplate):
    def generate_item(self, entry, output_field=None, output_field_replace_token="", ice_field_replace_token=""):
        # AISBench safe_format sequentially replaces fields. Only expose the one
        # pre-rendered input, so it cannot reinterpret LaTeX braces in question/options.
        return super().generate_item(
            {"gpqa_prompt": entry["gpqa_prompt"]}, ice_field_replace_token=ice_field_replace_token
        )


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


@MODELS.register_module()
class AuditedNativeVLLMChat(VLLMCustomAPIChat):
    def __init__(self, *args, audit_dir, manifest_path, **kwargs):
        validate_output_limit(kwargs.get("max_out_len", 4096))
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {row["prompt_sha256"]: row for row in json.loads(Path(manifest_path).read_text())}
        super().__init__(*args, **kwargs)

    async def get_request_body(self, input, max_out_len, output, **args):
        validate_output_limit(max_out_len)
        body = await super().get_request_body(input, max_out_len, output, **args)
        validate_output_limit(body["max_tokens"])
        messages = body["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("Expected exactly one unmodified GPQA user prompt")
        digest = hashlib.sha256(messages[0]["content"].encode()).hexdigest()
        row = self.manifest[digest]
        output.extra_details_data["original_id"] = row["original_id"]
        _append(self.audit_dir / "requests.jsonl", dict(uuid=output.uuid, original_id=row["original_id"], body=body))
        return body

    def _audit(self, output, response):
        for choice in response.get("choices", []):
            if choice.get("finish_reason") is not None:
                output.extra_details_data["finish_reason"] = choice["finish_reason"]
        if response.get("usage") and output.extra_details_data.get("finish_reason") is not None:
            _append(
                self.audit_dir / "responses.jsonl",
                dict(
                    uuid=output.uuid,
                    original_id=output.extra_details_data["original_id"],
                    finish_reason=output.extra_details_data["finish_reason"],
                    content=output.content,
                    reasoning_content=output.reasoning_content,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    usage=response["usage"],
                ),
            )

    async def parse_stream_response(self, json_content, output):
        await super().parse_stream_response(json_content, output)
        self._audit(output, json_content)

    async def parse_text_response(self, json_content, output):
        await super().parse_text_response(json_content, output)
        self._audit(output, json_content)


def make_config(
    dataset_dir, stage_dir, manifest_path, max_out_len=36000, indices_file=None, host="127.0.0.1", port=11223
):
    validate_output_limit(max_out_len)
    datasets = [
        dict(
            abbr="GPQA_diamond",
            type=LiteralSafeGPQADataset,
            path=dataset_dir,
            name="gpqa_diamond.csv",
            indices_file=indices_file,
            reader_cfg=dict(input_columns=["gpqa_prompt"], output_column="answer"),
            infer_cfg=dict(
                prompt_template=dict(
                    type=LiteralSafePromptTemplate, template=dict(round=[dict(role="HUMAN", prompt="{gpqa_prompt}")])
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            eval_cfg=dict(
                evaluator=dict(type=GPQAEvaluator), pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess)
            ),
        )
    ]
    models = [
        dict(
            attr="service",
            type=AuditedNativeVLLMChat,
            abbr="minimax-m31-native",
            path="/mnt/native/minimax_m3dot1_flash",
            model="minimax-m3",
            stream=True,
            request_rate=0,
            use_timestamp=False,
            retry=1,
            api_key="",
            host_ip=host,
            host_port=port,
            max_out_len=max_out_len,
            batch_size=32,
            trust_remote_code=True,
            generation_kwargs=dict(
                temperature=0.6, top_p=0.95, reasoning_effort="max", chat_template_kwargs=dict(reasoning_effort="max")
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content),
            audit_dir=stage_dir,
            manifest_path=manifest_path,
        )
    ]
    return datasets, models
