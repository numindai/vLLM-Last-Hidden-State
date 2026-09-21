"""Compare MTP extraction with a same-checkpoint non-MTP control server.

Run once with --snapshot on the control, then with --reference on MTP.
This is an integration check, not an independent Transformers reference.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import torch
from validate_hidden_state import cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8137")
    parser.add_argument("--model", default="mtp-test")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-mtp", action="store_true")
    args = parser.parse_args()
    prompts = cases() + cases(long_prompts=True)

    def metrics(client):
        response = client.get(args.url + "/metrics")
        response.raise_for_status()
        names = (
            "spec_decode_num_drafts_total",
            "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total",
            "prefix_cache_hits_total",
        )
        return {
            name: sum(
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith("vllm:" + name + "{")
            )
            for name in names
        }

    with httpx.Client(timeout=300) as client:

        def request(messages, extract=True, tokens=1, stream=False):
            body = dict(
                model=args.model,
                messages=messages,
                max_completion_tokens=tokens,
                temperature=0,
                seed=7,
                stream=stream,
                return_token_ids=True,
                continue_final_message=True,
                add_generation_prompt=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            if extract:
                body["kv_transfer_params"] = {"return_last_hidden_state": True}
            response = client.post(args.url + "/v1/chat/completions", json=body)
            response.raise_for_status()
            if stream:
                assert "data: [DONE]" in response.text
                assert "last_hidden_state" not in response.text
                return None
            data = response.json()
            if not extract:
                assert not (data.get("kv_transfer_params") or {}).get(
                    "last_hidden_state"
                )
            return data

        before = metrics(client)
        extracted = [request(prompt) for prompt in prompts]
        for response in extracted:
            vector = torch.tensor(response["kv_transfer_params"]["last_hidden_state"])
            assert vector.ndim == 1 and vector.numel() > 0 and vector.isfinite().all()
            assert response["usage"]["completion_tokens"] == 1
        normal = [request(prompt, False, 16) for prompt in prompts[:3]]
        assert any(item["usage"]["completion_tokens"] > 3 for item in normal)
        request(prompts[0], False, 16, stream=True)
        # Mixed traffic exercises prompt capture while another request is decoding.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(request, prompts[0], False, 16)]
            futures += [pool.submit(request, prompt) for prompt in prompts]
            futures[0].result()
            repeated = [future.result() for future in futures[1:]]
        after = metrics(client)
        delta = {key: after[key] - before[key] for key in before}
        if args.require_mtp:
            assert delta["spec_decode_num_drafts_total"] > 0, delta
            assert delta["spec_decode_num_accepted_tokens_total"] > 0, delta

    def compare(actual, expected):
        assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
        assert actual["choices"][0]["token_ids"] == expected["choices"][0]["token_ids"]
        a = torch.tensor(actual["kv_transfer_params"]["last_hidden_state"])
        b = torch.tensor(expected["kv_transfer_params"]["last_hidden_state"])
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        relative_l2 = ((a - b).norm() / b.norm()).item()
        assert cosine >= 0.999 and relative_l2 <= 0.05, (cosine, relative_l2)
        return dict(
            cosine=cosine,
            relative_l2=relative_l2,
            prompt_tokens=len(actual["prompt_token_ids"]),
            hidden_size=a.numel(),
        )

    repeated_metrics = [compare(a, b) for a, b in zip(repeated, extracted, strict=True)]
    control_metrics = None
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        control_metrics = [
            compare(a, b)
            for a, b in zip(extracted, reference["extracted"], strict=True)
        ]
        for a, b in zip(normal, reference["normal"], strict=True):
            assert a["prompt_token_ids"] == b["prompt_token_ids"]
            assert a["choices"][0]["token_ids"] == b["choices"][0]["token_ids"]
    Path(args.snapshot).write_text(json.dumps(dict(extracted=extracted, normal=normal)))
    report = dict(
        model=args.model,
        metrics_delta=delta,
        repeated=repeated_metrics,
        control=control_metrics,
        streaming=True,
        mixed_traffic=True,
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
