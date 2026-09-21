# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check real chat responses against a CPU Transformers reference."""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pybase64 as base64
import torch
from openai import OpenAI
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    CompressedTensorsConfig,
)


def cases(long_prompts=False):
    image = Image.new("RGB", (112, 112), (230, 30, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    prefix = '{\n  "short description": "'
    messages = [
        [
            {"role": "user", "content": "Describe a yellow bicycle in one sentence."},
            {"role": "assistant", "content": prefix},
        ],
        [
            {"role": "user", "content": "Describe a snowy mountain in one sentence."},
            {"role": "assistant", "content": prefix},
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            'Extract {"short description": "string"}. '
                            "Describe the image briefly, including its main color."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
            {"role": "assistant", "content": prefix},
        ],
    ]
    if long_prompts:
        context = "Background: A bicycle rests beside a stone wall. " * 160
        for conversation in messages[:2]:
            conversation[0]["content"] = context + conversation[0]["content"]
        messages[2][0]["content"][0]["text"] = (
            context + messages[2][0]["content"][0]["text"]
        )
    return messages


def reference(args):
    torch.set_num_threads(args.threads)
    processor_class = AutoTokenizer if args.text_only else AutoProcessor
    model_class = (
        AutoModelForCausalLM if args.text_only else AutoModelForImageTextToText
    )
    processor = processor_class.from_pretrained(args.model)
    model = model_class.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
        **(
            {"quantization_config": CompressedTensorsConfig(dequantize=True)}
            if args.dequantize
            else {}
        ),
    ).eval()
    results = []
    messages_to_test = cases(args.long_prompts)
    if args.text_only:
        messages_to_test = messages_to_test[:2]
    for messages in messages_to_test:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            continue_final_message=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        start = time.monotonic()
        with torch.inference_mode():
            outputs = model.model(**inputs, return_dict=True, use_cache=False)
            positions = torch.arange(inputs.input_ids.shape[1])[None, :]
            last = (
                positions.masked_fill(~inputs.attention_mask.bool(), -1).max(1).values
            )
            vector = outputs.last_hidden_state[torch.arange(len(last)), last][0].float()
            token = model.lm_head(vector.to(model.dtype)).argmax().item()
        results.append(
            {
                "vector": vector.tolist(),
                "token_ids": inputs.input_ids[0].tolist(),
                "position": last.item(),
                "first_generated_token": token,
                "seconds": time.monotonic() - start,
            }
        )
        print(json.dumps({k: v for k, v in results[-1].items() if k != "vector"}))
    Path(args.reference).write_text(
        json.dumps(
            {
                "model": args.model,
                "dtype": args.dtype,
                "text_only": args.text_only,
                "dequantize": args.dequantize,
                "long_prompts": args.long_prompts,
                "cases": results,
            }
        )
    )


def check_http(args):
    expected = json.loads(Path(args.reference).read_text())
    assert expected["model"] == args.model
    assert expected.get("text_only", False) == args.text_only
    assert expected.get("long_prompts", False) == args.long_prompts
    messages_to_test = cases(args.long_prompts)
    if args.text_only:
        messages_to_test = messages_to_test[:2]
    client = OpenAI(base_url=args.url + "/v1", api_key="unused", timeout=180)
    reports = []

    def prefix_cache_hits():
        response = httpx.get(args.url + "/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines()
            if line.startswith("vllm:prefix_cache_hits_total{")
        )

    initial_cache_hits = prefix_cache_hits() if args.require_prefix_cache_hit else None

    def request(messages, extract=True, max_tokens=1):
        start = time.monotonic()
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            n=1,
            stream=False,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "return_token_ids": True,
                **(
                    {"kv_transfer_params": {"return_last_hidden_state": True}}
                    if extract
                    else {}
                ),
            },
        ).model_dump()
        return response, time.monotonic() - start

    if args.mode == "baseline":
        baseline = [request(messages, extract=False) for messages in messages_to_test]
        baseline.append(request(messages_to_test[0], extract=False, max_tokens=3))
        Path(args.baseline).write_text(json.dumps(baseline, indent=2))
        print("Saved ordinary vLLM baseline responses")
        return

    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None

    def compare_normal(index, response, seconds):
        if baseline is not None:
            prior, prior_seconds = baseline[index]
            for key in ("choices", "usage", "prompt_token_ids"):
                assert response[key] == prior[key], (key, response[key], prior[key])
            print(
                json.dumps(
                    {
                        "ordinary_case": index,
                        "baseline_seconds": prior_seconds,
                        "plugin_seconds": seconds,
                    }
                )
            )

    def compare(index, response, seconds):
        target = expected["cases"][index]
        assert response["prompt_token_ids"] == target["token_ids"]
        assert response["usage"]["prompt_tokens"] == len(target["token_ids"])
        assert response["usage"]["completion_tokens"] == 1
        assert response["choices"][0]["token_ids"] == [target["first_generated_token"]]
        actual = torch.tensor(response["kv_transfer_params"]["last_hidden_state"])
        reference_vector = torch.tensor(target["vector"])
        assert actual.shape == reference_vector.shape and actual.isfinite().all()
        cosine = torch.nn.functional.cosine_similarity(actual, reference_vector, dim=0)
        delta = (actual - reference_vector).abs()
        relative_error = torch.linalg.vector_norm(actual - reference_vector) / (
            torch.linalg.vector_norm(reference_vector).clamp_min(1e-12)
        )
        report = {
            "case": index,
            "reference_backend": expected.get("backend", "transformers"),
            "hidden_size": actual.numel(),
            "position": target["position"],
            "last_token_id": target["token_ids"][-1],
            "max_abs_error": delta.max().item(),
            "mean_abs_error": delta.mean().item(),
            "cosine": cosine.item(),
            "relative_l2_error": relative_error.item(),
            "seconds": seconds,
            "completion": response["choices"][0],
        }
        print(json.dumps(report), flush=True)
        assert cosine >= args.min_cosine
        assert relative_error <= args.max_relative_error
        reports.append(report)

    for index, messages in enumerate(messages_to_test):
        compare(index, *request(messages))
        normal, seconds = request(messages, extract=False)
        assert normal["kv_transfer_params"] is None
        assert normal["usage"]["completion_tokens"] == 1
        compare_normal(index, normal, seconds)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(request, messages) for messages in messages_to_test]
        for index, future in enumerate(futures):
            compare(index, *future.result())
    normal, seconds = request(messages_to_test[0], extract=False, max_tokens=3)
    assert normal["usage"]["completion_tokens"] == 3
    compare_normal(len(messages_to_test), normal, seconds)
    for override in [{"max_tokens": 2}, {"n": 2}, {"stream": True}]:
        body = {
            "model": args.model,
            "messages": messages_to_test[0],
            "max_tokens": 1,
            "kv_transfer_params": {"return_last_hidden_state": True},
            **override,
        }
        response = httpx.post(args.url + "/v1/chat/completions", json=body, timeout=30)
        assert response.status_code == 400, response.text
    chunks = client.chat.completions.create(
        model=args.model,
        messages=messages_to_test[0],
        max_tokens=3,
        temperature=0,
        stream=True,
        extra_body={
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    streamed = "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
    )
    assert streamed == normal["choices"][0]["message"]["content"]
    if initial_cache_hits is not None:
        cache_hits = prefix_cache_hits() - initial_cache_hits
        assert cache_hits > 0, "No actual prefix-cache hits during validation"
        for report in reports:
            report["prefix_cache_hit_tokens_during_suite"] = cache_hits
        print(f"Prefix-cache hit tokens during validation: {cache_hits}")
    Path(args.report).write_text(json.dumps(reports, indent=2))
    print(
        "HTTP, token alignment, numerical parity, concurrency, and restrictions passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "baseline", "http"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--long-prompts", action="store_true")
    parser.add_argument("--require-prefix-cache-hit", action="store_true")
    parser.add_argument(
        "--dequantize", action="store_true", help="Decompress CT reference at load time"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--reference", default="/tmp/last-hidden-reference.json")
    parser.add_argument("--report", default="/tmp/last-hidden-report.json")
    parser.add_argument("--baseline")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-error", type=float, default=0.05)
    args = parser.parse_args()
    (reference if args.mode == "reference" else check_http)(args)


if __name__ == "__main__":
    main()
