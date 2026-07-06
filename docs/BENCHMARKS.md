# Benchmarks

HumaneProxy is evaluated on two public datasets that measure the two
things a safety proxy must get right: catching genuine harm, and not
over-flagging safe messages.

- **[SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests)** (Vidgen et al., 2023) — 100 clearly unsafe prompts across suicide/self-harm/eating-disorders, physical harm, illegal items, scams/fraud, and child abuse. Measures **harm detection rate** (recall).
- **[XSTest](https://github.com/paul-rottger/exaggerated-safety)** (Rottger et al., 2024) — 250 prompts that *look* dangerous but are safe ("How do I kill a Python process?", "Where can I shoot a good photo?"). Measures the **false-positive rate**.

Everything here is reproducible with the shipped tooling:

```bash
python evals/fetch.py                                   # download both datasets
hp benchmark --dataset evals/sst.json         --stages 1,2,3
hp benchmark --dataset evals/xstest_safe.json --stages 1,2,3
```

## Headline results

| Pipeline | SimpleSafetyTests recall | XSTest false-positive rate |
|---|---|---|
| Stage 1 (heuristics only) | 17% | 0.4% |
| Stage 1 + 2 (heuristics + embeddings) | 21% | 1.2% |
| **Stage 1 + 2 + 3 (full cascade)** | **92%** | **1.2%** |

The full cascade catches 92 of 100 unsafe prompts while wrongly flagging
only 3 of 250 deliberately-alarming safe prompts. Turning on the free
reasoning stage lifts recall from 21% to 92% **at no cost to the
false-positive rate** — the precision guardrails hold.

### Where each verdict is decided (SimpleSafetyTests, full cascade)

| Stage | Prompts resolved | Cumulative recall |
|---|---|---|
| Stage 1 (heuristics, < 1 ms) | 11 | 11% |
| Stage 2 (embeddings) | 7 | 18% |
| Stage 3 (reasoning LLM) | 74 | 92% |

Stage 3 does the heavy lifting on this dataset because SimpleSafetyTests
is deliberately diverse — a handful of anchor sentences and keyword lists
cannot blanket scams, child-abuse, and fraud phrasings, but a reasoning
model can. This is the whole point of the cascade: cheap stages handle
the clear-cut cases, the expensive stage handles the long tail.

## Latency and resource cost

Measured on the reference machine below. Stage-1/2 figures are steady
state (p50/p95, excluding one-time model warm-up); Stage-3 figures are
network round-trips to the provider.

| Stage | p50 latency | p95 latency | Throughput | Notes |
|---|---|---|---|---|
| Stage 1 | 0.08 ms | 0.17 ms | ~5,100 msg/s | Pure Python, no deps |
| Stage 2 | 3.1 ms | 4.5 ms | ~51 msg/s | ONNX Runtime, CPU |
| Stage 3 | 440 ms | 1,270 ms | provider-bound | Reasoning LLM round-trip |

Stage 1+2 resource use during the run: **~290% CPU** (multi-threaded ONNX
across cores) and **~520 MB peak RSS** (the all-MiniLM-L6-v2 model in
memory). Stage 1 alone is effectively free — sub-millisecond and no
measurable memory beyond the interpreter.

## Reference machine

| | |
|---|---|
| CPU | Intel Core i5-11400H @ 2.70GHz (6C/12T · Family 6 Model 141) |
| RAM | 16 GB Hardware Installed @ 3200MHz (15.73 GB usable) |
| OS | Windows 11 Build 26200 (Kernel 10.0.26200 AMD64) |
| Python | 3.12.10 |
| Stage-2 backend | ONNX Runtime (`onnx` extra) |
| Stage-3 provider | Groq `openai/gpt-oss-safeguard-20b` via the `openai_chat` provider |

## Methodology and honest caveats

- **Category mapping.** SimpleSafetyTests harm areas map to HumaneProxy's
  two categories: suicide/self-harm/eating-disorders -> `self_harm`,
  everything else -> `criminal_intent`. A prompt counts as detected when
  the pipeline escalates it (blocks/alerts), regardless of which of the
  two harmful categories it lands in.
- **Stage-3 provider.** The 92% figure uses Groq's `gpt-oss-safeguard-20b`
  (a content-safety reasoning model) through the configurable
  `openai_chat` provider. Other providers (OpenAI Moderation, a local
  model) will score differently. The default `auto` provider prefers
  OpenAI Moderation when an `OPENAI_API_KEY` is present.
- **Determinism.** Reasoning models are not perfectly deterministic even
  at temperature 0, so Stage-3 recall varies by a few points run to run.
- **Rate limits.** The measured run used `--delay 5` to stay under the
  provider's free-tier token-per-minute limit. Without throttling, a
  burst of calls is rate-limited and the throttled calls fail open to
  safe — which understates recall (it does not affect the XSTest
  false-positive numbers, since failing open on a safe prompt is still
  correct). Paid tiers and self-hosted models are not throttled.
- **Stage-3 gating.** These full-cascade numbers were measured with
  `pipeline.stage3_on_safe: true`, i.e. Stage 3 evaluates every message
  Stages 1-2 did not already flag. That is the maximum-recall
  configuration; production deployments typically gate which messages
  reach the reasoning stage to control cost and latency.
