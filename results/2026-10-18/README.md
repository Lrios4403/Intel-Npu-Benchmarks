# NPU follow-up: the Medium article's method, prompt limits, models, tool calling and load (2026-10-18)

Follows [2026-10-17](../2026-10-17/README.md), on the same laptop, drivers and software. That report covers the hardware and driver records and the four-device speed comparison.

This round tests the setup from Trevisan's ["How to use your Intel NPU to run local LLM's"](https://medium.com/@trevisan3008/how-to-use-your-intel-npu-to-run-local-llms-a8d17f835f37) (Jan 30, 2026). That article runs OpenVINO GenAI's `LLMPipeline` directly, with `{"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 1}` and `llmware/llama-3.1-8b-instruct-npu-ov`. This round also adds tool-calling and load tests.

Software: OpenVINO GenAI 2026.4.1 (system Python 3.11) for the direct runs, and OVMS 2026.4.0 for the server runs.

New downloads:
- `llmware/llama-3.1-8b-instruct-npu-ov`, 4.57 GB, Llama 3.1 license
- `OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov`, 0.64 GB, Apache-2.0

## The article's claims, checked

| Claim | Result here |
|---|---|
| 8–12 tokens/s for an 8B model on a Core Ultra NPU | **Confirmed.** The article's own model ran at 10.35 tokens/s, and Qwen3 8B at ~10. |
| The NPU is 3–5x more energy-efficient than a GPU (5–15 W vs 120–190 W) | **Not verified.** The article gives no measurement method. Here, NPU generation showed no measurable rise in CPU package power, and the RTX 5080 drew ~146 W. A battery test is still needed for a real efficiency number. |
| "Intel's NPU architecture is physically tuned for 4-bit math" | **Wrong for this NPU.** OpenVINO reports the NPU's optimization capabilities as `FP16` and `INT8` only. INT4 is weight compression, unpacked at runtime. |
| The `LLMPipeline(path, "NPU", config)` script | Works, but passing the config dict positionally is deprecated in GenAI 2026.4. Use `LLMPipeline(path, "NPU", **config)`. |
| `MAX_PROMPT_LEN: 1024` | Works, but any longer prompt is rejected: `Stateful LLM pipeline on NPU may only process prompts or hold chat history up to 1024 tokens. 3619 is passed.` That's fine for a chat script and too small for agent tools. `MIN_RESPONSE_LEN: 1` didn't cut responses short. |
| Efficiency table row "NPU (ARC B580)" | Typo; that row is the B580 GPU. |

## Prompt limit (`MAX_PROMPT_LEN`) on Qwen3 8B, NPU, GenAI direct

Each config was compiled from an empty cache in a fresh process, then reloaded from that cache in a second process. Prompts: ~60, ~770, ~3.6k and ~11.5k tokens, with 256 output tokens.

| Config | First compile | Cached reload | First token (60 / 770 / 3.6k / 11.5k tokens) | Generation (tokens/s) |
|---|---|---|---|---|
| 1024 + `MIN_RESPONSE_LEN` 1 (the article's) | 110 s* | 21 s | 4.4 / 4.3 s / rejected / — | 9.8 |
| 1024 | 39 s | 18 s | 4.3 / 4.3 s | 9.9 |
| 2048 | 59 s | 21 s | 4.5 / 4.5 s | 9.6–9.9 |
| 4096 | 56 s | failed | 4.5 / 4.5 / 21.5 s | 10.2 → 8.7 at 3.6k |
| 16384 | **514 s** | failed | 4.7 / 4.7 / 21.9 / 171 s | 10.1 → 8.2 → 5.9 |
| 16384 + PYRAMID (the post's setup) | **69 s** | 42 s (failed once on a rerun) | 5.2 / 4.7 / 21.9 / 148 s | 10.1 → 8.3 → 6.1 |

\*The first config to run, so it probably includes reading the 4.4 GB model from disk for the first time.

- **The ~4.3 s wait for the first token is the NPU's prompt-block size, not the limit setting.** Prompts are processed in blocks of about 1,000 tokens, so a 60-token prompt waits as long as a 770-token one. Each further block adds ~5 s, rising with context: ~21.5 s at 3.6k tokens and 2.5–3 minutes at 11.5k.
- **PYRAMID cuts the 16k compile from 8.5 minutes to about 1 minute.**
- **Generation speed depends on the actual prompt length, not the limit.**
- **Cached reloads are unreliable in GenAI 2026.4.1:** the compile and the first run worked, but loading the cached compile in a new process failed with `Cannot find tensor for port opset1::Parameter` for Qwen3 8B at 4096, at 16384, and once at 16384 + PYRAMID. Llama 3.1 8B reloaded fine. OVMS 2026.4.0's own cache reloaded on every launch across both days.

### RAM vs prompt limit (Qwen3 8B, GenAI direct)

| `MAX_PROMPT_LEN` | Peak working set | Peak committed (private) |
|---|---|---|
| 1024 | 7.8 GB | 3.6 GB |
| 16384 + PYRAMID | 7.0 GB | 11.3 GB |

The memory reserved for a 16k context costs ~7.7 GB of extra committed memory, which matches OVMS's 11.2 GB private (16.8 GB peak working set) the day before. A small limit is how an 8B model fits on a 16 GB machine like the article's.

## Which models run on this NPU (NPU 3720, driver 32.0.100.5540)

| Model | Format | Result | First compile | Cached load | First token | Generation | Peak RAM |
|---|---|---|---|---|---|---|---|
| `OpenVINO/Qwen3-8B-int4-cw-ov` | int4, channel-wise | Works | 33–69 s | 18–46 s | 4.3–5.2 s | ~10 tokens/s | 7–8 GB working set |
| `llmware/llama-3.1-8b-instruct-npu-ov` | llmware NPU export | Works | 161–179 s | 19–25 s | 4.6–4.8 s | 10.2–10.35 tokens/s (9.4 at 3.6k) | 8.0–8.7 GB |
| `OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov` | int4 | Works | 86 s | 1.8 s | 1.3 s | 33 tokens/s | 1.7 GB |
| `OpenVINO/Qwen3-8B-int4-ov` | int4, group-wise | **Compile fails** | — | — | — | — | — |
| `OpenVINO/Qwen3-0.6B-int4-ov` | int4, group-wise | **Compile fails** | — | — | — | — | — |
| `OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov` | int4, group-wise | **Compile fails** | — | — | — | — | — |

The failing models all stop in the NPU compiler with `Unsupported DequantizeOp configuration … quant.uniform<u4:f16 …> on arch NPU37XX`, which surfaces as `Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER`. On this NPU generation, use the channel-wise (`cw`) exports, or exports made for the NPU like llmware's.

## Tool calling (20 cases, all six tools offered every time, temperature 0)

| Model / backend | Passed | Typical time per case | Failures |
|---|---|---|---|
| Qwen3 8B cw, NPU (OVMS, `--tool_parser hermes3`) | **20/20** | ~8 s | none |
| Qwen3 8B Q4_K_M, RTX 5080 (Ollama native API, `think: false`) | **20/20** | ~1.5 s | none |
| Llama 3.1 8B (llmware), NPU (OVMS, `--tool_parser llama3`) | **11/20** | ~8 s | Empty arguments (`get_time {}`), tools invented when none were needed (`calculate`, `explain`), only one call when two were asked for, and the tool result ignored |

The cases cover single calls (numbers, enums, lists, an optional argument, Unicode text, a comma in a number, details pulled from a sentence), three prompts that need no tool, two that need two calls, and a follow-up that must use a tool result.

Through Ollama's **OpenAI-compatible** endpoint, Qwen3 ignores the `/no_think` switch: it reasons until it hits the token limit and returns no answer. That affects any client pointed at `http://localhost:11434/v1`, including Claude Code Router and OpenWork. Use the native API with `think: false`, or allow for the thinking time.

## Does the LLM slow down other work? (Qwen3 8B, OVMS)

### All-core CPU stress (24 processes of SHA-256 for 40 s; 7.18 M hashes/s on its own)

| LLM on | LLM alone | LLM under the stress | Stress throughput |
|---|---|---|---|
| NPU | 10.55 tokens/s | 10.28 (−3%) | 7.73 M/s (no loss) |
| CPU | 7.92 tokens/s | 5.69 (**−28%**) | 6.42 M/s (**−11%**) |
| Intel iGPU | 12.78 tokens/s | 12.80 (0%) | 7.40 M/s (no loss) |

### NPU and RTX 5080 generating at the same time

| | Alone | Together |
|---|---|---|
| NPU (OVMS) | 10.56 tokens/s | 10.52 tokens/s |
| RTX 5080 (Ollama) | 89.3 tokens/s | 76.3 tokens/s (−15%) |

### `next build` of this site, timed twice per scenario

| Scenario | Build times | LLM speed, alone → during the builds |
|---|---|---|
| Build alone | 15.4 s, 11.7 s | — |
| While generating on the NPU | 15.0 s, 10.7 s | 10.54 → 10.29 |
| While generating on the CPU | 12.9 s, 16.2 s | 7.99 → 7.71 |
| While generating on the iGPU | 14.0 s, 9.1 s | 12.76 → 12.35 |

The build is too short (~12 s) and varies too much from run to run to show a difference, which is why the stress test was added.

## What this means for the post

- **The NPU and the iGPU both leave the CPU alone.** Under full CPU load, NPU and iGPU generation kept their speed and didn't slow the other work, while CPU inference lost 28% and took 11% from everything else.
- **On this laptop the Intel iGPU is the stronger Intel option** for interactive use: a faster first token (0.7 s vs 4.8 s for a short prompt), better speed on long prompts, and similar or faster generation. The NPU's remaining case is power (no measurable rise in package power, vs about +10 W for the iGPU, from noisy measurements) and leaving the graphics engine free.
- **Running the NPU alongside the RTX:** the NPU isn't affected, but the RTX lost about 15%, plausibly because the CPU and GPU share the laptop's power and cooling budget.
- **For agents:**
  - Qwen3 8B with `hermes3` tool calls was perfect on every backend.
  - Each NPU tool call takes ~8 s, against ~1.5 s on the RTX.
  - Long agent prompts cost minutes before the first token on the NPU (2.5–3 minutes at 11.5k tokens).
  - The article's Llama 3.1 8B export isn't reliable for tool calling through OVMS.
- **Setup advice for the post:** use `cw` model exports, keep PYRAMID, and choose `MAX_PROMPT_LEN` as a tradeoff between context length and ~7.7 GB of RAM. Prefer OVMS's cache over reloading cached GenAI compiles.

## Caveats

- Same session caveats as 2026-10-17: Brave and Roblox were open, the laptop was plugged in, and the OMEN performance mode wasn't recorded.
- The stress test is synthetic. Real heavy workloads (games, big builds) also load memory bandwidth and the GPU differently.
- The Llama tool-calling result may partly reflect llmware's export and chat template with OVMS's `llama3` parser, not only the model.
- One run per cell for the load tests; three per prompt size in the sweep (medians are shown).

## Prompt (prefix) caching

The tests above put a unique line at the start of every prompt, so caching couldn't help. This test does the opposite, the way agent tools work. Each scenario starts a fresh ~4k or ~11k token system prompt (unique first line, so nothing carries over between runs) and times the first token of each request:

- **agent:** the same system prompt plus a different short question, 4 times
- **chat:** a 4-turn conversation that resends the growing history each turn
- **identical:** the same request twice

Both runs used `cache_bench.py`. Run 1 overlapped a Rust build on the same machine during the iGPU and RTX phases and the last two minutes of the NPU caching-off phase. Run 2 was clean.

| Time to first token | Run 1: turn 1 → turns 2–4 | Run 2: turn 1 → turns 2–4 |
|---|---|---|
| NPU, caching on (the post's config), agent 4k | 27.7 s → 8.2–8.7 s | 26.7 s → 8.2–8.4 s |
| NPU, caching on, agent 11k | 124 s → 23, 17, 17 s | 121 s → 18, 15, 15 s |
| NPU, caching on, chat 4k | 28.3 s → 8.5–8.9 s | 27.5 s → 8.0–8.5 s |
| NPU, caching on, identical 4k | 24.6 s → 9.2 s | 24.0 s → 8.2 s |
| NPU, caching **off**, agent 4k | 27.2 s → 24–25 s | 26.7 s → 23.3–23.5 s |
| Intel iGPU, caching on, agent 4k | 43 s → 1.3–1.6 s | 47 s → 1.6–1.7 s |
| Intel iGPU, caching on, agent 11k | 246 s → 1.5–1.8 s | 262 s → 1.9–2.1 s |
| Intel iGPU, caching on, chat 4k | 44 s → 2.0–2.5 s | 47 s → 2.1–2.9 s |
| RTX 5080 (Ollama), agent 4k | 2.4 s → 0.5 s | 2.4 s → 0.5–0.6 s |
| RTX 5080, agent 11k | 7.5 s → 2.0–2.3 s | 7.2 s → 1.9–2.1 s |
| RTX 5080, chat 4k | 3.5 s → 0.22 s | 3.1 s → 0.2 s |

- **Caching works on the NPU** (`--enable_prefix_caching true`, which OVMS turns into `NPUW_LLM_ENABLE_PREFIX_CACHING`): repeat turns are 3x faster at 4k and 7–8x faster at 11k. With caching off there's no gain.
- **But the NPU still pays ~8 s at 4k and ~15–17 s at 11k on every cached turn.** The iGPU's cache gets repeat turns to ~1.5–2 s regardless of prompt length. For agents that resend the same long prompt, the iGPU with prefix caching is clearly the better Intel option.

### Resources during caching run 2 (from `raw/monitor.jsonl`)

| Phase | Minutes | CPU avg | Package power avg | Accelerator | Accelerator memory (max) | Server RAM (max) | Free RAM (min) |
|---|---|---|---|---|---|---|---|
| NPU, caching on | 6.8 | 27% | 36 W | NPU 96% (median) | 10.6 GB NPU shared | 13.1 GB | 0.1 GB |
| NPU, caching off | 2.6 | 25% | 39 W | NPU 95% | 8.8 GB NPU shared | 14.6 GB | 0.0 GB |
| Intel iGPU, caching on | 7.5 | 20% | 34 W (iGPU rail 9.6 W) | iGPU busy* | 9.6 GB iGPU shared | 9.5 GB | 4.1 GB |
| RTX 5080 (Ollama) | ~2 | 51% | 54 W | RTX up to 136 W | 14.8 GB VRAM | — | 0.6 GB |

\*The iGPU's compute-engine counters are unreliable (spikes in the thousands of percent), so the iGPU power rail is the better signal: 9.6 W during the run vs 0.4 W at idle.

**RAM is the constraint on this 32 GB laptop.** With the post's 16k-limit NPU config:
- OVMS holds 13–15 GB of RAM, peaking at 19 GB while loading.
- The NPU maps another 6–11 GB of shared memory.
- With everyday apps open, free RAM hit zero and Windows paged.

Even loaded and idle, the NPU server keeps ~15 GB of RAM plus 6.4 GB of NPU shared memory, with no measurable power draw.

## Files and rerun

| Script (in `bench/`) | What it does |
|---|---|
| `download_models.py` | Downloads TinyLlama and the article's Llama 3.1 8B |
| `genai_bench.py` | Plans `qwen8`, `mem` and `others`: the article's method, one config per process, cold then warm (`python bench/genai_bench.py sweep qwen8`) |
| `tools_bench.py` | `qwen-npu`, `llama-npu`, `qwen-rtx` |
| `load_bench.py` | The build test (`NPUBENCH_BUILD_DIR`) and NPU + RTX at the same time |
| `stress_bench.py` | The all-core CPU stress test |
| `cache_bench.py` | The prompt (prefix) caching test |
| `more_bench.py` | Idle cost, thinking mode, parallel requests, sustained load, the context limit |
| `ovms_launch.py` | Starts and stops OVMS for any model; backs up and restores the model's `graph.pbtxt` |
| `monitor.ps1`, `monitor_summary.py` | The system resource monitor and its per-phase summary |

The results are the `results-*.json` files here. `raw/` has the OVMS logs, the resource monitor log (`monitor.jsonl`, with `markers.jsonl` labelling the `more_bench.py` steps) and `events.txt` (things that changed on the machine during the runs). Setup is in the [top-level README](../../README.md).
