# Maintaining the last-hidden-state package

This directory is a private, separately installable vLLM extension. For an upgrade,
read [MAINTENANCE.md](MAINTENANCE.md) before editing, then use the
[upgrade report template](validation/UPGRADE_REPORT_TEMPLATE.md) to record evidence.
The [README](README.md) owns the client contract, launch examples, and recorded
compatibility. Parent repository instructions still apply.

## Decisions to preserve

- Keep changes additive within this package. If an upstream change removes a
  necessary hook, explain the missing contract and smallest fallback before
  expanding into vLLM core. This project is for private use, not an upstream PR.
- Keep `vllm_last_hidden_state/serve.py` a pass-through launcher: upstream argument
  parsing for conflict checks, plugin/worker/connector activation, and selection
  of the Python frontend. Preserve compatible plugin settings. Model, device,
  compilation, scheduling, and performance defaults belong to the caller.
- Keep vLLM version policy and runner-specific assumptions in `compat.py`. The
  submodule pin does not prove which vLLM is installed or validate its native build.
- Preserve the final **prompt** token's post-output-normalization state inline
  in the normal chat response. No extra forward, pooling, unit normalization,
  auxiliary hidden-state cache, or client-side retrieval step.
- Keep `"chat_template_kwargs": {"enable_thinking": False}` explicit in HTTP
  examples and validation; do not restore a `template_options` helper.
- Prefix caching, chunked prefill, compilation, and async scheduling are intended
  capabilities. Repair their integration rather than silently disabling them.
  MTP is in scope. Capture only the target model's final prompt state, never the
  draft model's states. Keep upstream runtime limitations explicit.
- Keep GPU implementation status separate from runtime evidence. CPU tests of
  the GPU wrapper do not establish CUDA graph or async scheduling correctness.

## Completing an upgrade

Trace the integration contracts in the maintenance guide against the target
source, not just matching import names. Run the applicable validation there and
record failures and unavailable hardware alongside successes. Preserve historical
results; they do not certify a newer checkout. Report the exact revisions and
remaining gaps so another session can resume without relying on conversation
history or files under `/tmp`.
