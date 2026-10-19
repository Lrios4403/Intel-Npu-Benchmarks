# Qwen3 8B on the Intel AI Boost NPU vs CPU, iGPU and RTX 5080 (2026-10-17)

Source material for expanding `/blogs/llms-on-an-intel-npu`. Everything here was
recorded on the laptop itself; the raw results are the `results-*.json` files
next to this one, and `bench.py` reruns the whole thing.

## Machine

| Item | Recorded |
|---|---|
| Laptop | HP OMEN MAX Gaming Laptop 16-ah0xxx, on AC power |
| CPU | Intel Core Ultra 9 275HX, 24 cores / 24 threads |
| RAM | 31.4 GB |
| Windows | Windows 11 Pro 25H2, build 26200.9711 |
| NPU | Intel AI Boost, OpenVINO architecture 3720, 2 tiles, 16 GiB shared memory |
| NPU peak (OpenVINO `DEVICE_GOPS`) | 13.1 TOPS INT8, 6.6 TFLOPS FP16 |
| NPU compiler | In-driver compiler (`NPU_COMPILER_TYPE` DRIVER, version 524291) |
| Intel iGPU | Intel Graphics (Xe arch 12.70.4, 64 EUs); OpenVINO reports 3.9 TFLOPS FP16 |
| dGPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, CUDA 13.2 |

## Drivers: installed vs latest (checked 2026-10-17)

| Driver | Installed | Latest | Status |
|---|---|---|---|
| Intel NPU (Intel AI Boost) | 32.0.100.5540, dated 2026-08-19, `oem78.inf`, WHQL | [32.0.100.5540](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) (released 2026-08-28, `npu_win_32.0.100.5540.exe`) | Current |
| Intel Graphics (iGPU) | 32.0.101.8860, dated 2026-06-24, `oem316.inf`, WHQL | [32.0.101.9150](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html) (2026-10-02, `gfx_win_101.9150.exe`) | Update available |
| NVIDIA (RTX 5080 Laptop) | 596.49 (32.0.15.9649), dated 2026-05-04, `oem165.inf`, WHQL | [617.42 Game Ready WHQL](https://www.nvidia.com/en-us/geforce/news/call-of-duty-modern-warfare-4-star-wars-galactic-racer-game-ready-driver/) (2026-10-06) | Update available (HP may also ship its own certified build) |

Hardware IDs: NPU `PCI\VEN_8086&DEV_AD1D`, iGPU `PCI\VEN_8086&DEV_7D67`, RTX `PCI\VEN_10DE&DEV_2C59` (subsystem `8D41103C`, HP).

## Software

| Item | Installed | Latest |
|---|---|---|
| OpenVINO Model Server | 2026.4.0 (`869b2186`), OpenVINO 2026.4.0, GenAI 2026.4.0; package `ovms_windows_2026.4.0_python_off` | [2026.4.1](https://github.com/openvinotoolkit/model_server/releases/tag/v2026.4.1) (2026-10-06, hotfix: OpenVINO 2026.4.1 plus two non-NPU fixes) |
| OpenVINO (Python) | 2026.4.1 | 2026.4.1 |
| Ollama | 0.40.5 (llama.cpp runner) | 0.40.6 (2026-10-16) |
| Claude Code | 2.1.198 | 2.1.302 |
| Claude Code Router | not installed | 3.1.3 |
| NoLlama | not installed | 1.0.0 |

## Model

- `OpenVINO/Qwen3-8B-int4-cw-ov` (channel-wise int4, the NPU-recommended export), 4.4 GB, downloaded 2026-09-30. Exported with optimum 2.1.0.dev0 and transformers 4.57.6. The Hugging Face revision wasn't saved, so it can't be pinned.
- The plain `OpenVINO/Qwen3-8B-int4-ov` was downloaded about three hours earlier. The working setup moved to the `cw` export, and the 760 MB NPU compile blob in `.ovcache` appeared five minutes after that download (2026-09-30 03:57).
- Served config, from the model folder's `graph.pbtxt`: device NPU, `NPUW_LLM_PREFILL_ATTENTION_HINT` PYRAMID, `MAX_PROMPT_LEN` 16384, prefix caching on, `tool_parser` hermes3.
- RTX comparison: Ollama `qwen3:8b` (8.2B, Q4_K_M, the same base model in a different quantization and runtime), `num_ctx` 8192, 100% on GPU.

## Method

- Same prompts on every config: a short question (52 tokens), a ~1.2k token document and a ~4.3k token document, each answered in about 200 words, with up to 256 output tokens and temperature 0.
- Every request starts with a unique line, so prompt and prefix caches can't shortcut prompt processing. Qwen3's thinking is off (`/no_think`; plus `think: false` on Ollama).
- 3 runs per prompt size; the tables show medians. "First token" is measured by the client from sending the request to the first streamed token. "Generation" is output tokens per second after that.
- The NPU run used the post's exact launch command. The CPU and iGPU runs used the same command with `--target_device CPU` / `GPU.0`, without the NPU-only flags or a compile cache.
- Power is CPU package power from Windows' RAPL energy counters (the NPU sits inside the package), plus `nvidia-smi` for the RTX. Utilization comes from Windows' GPU Engine counters; the NPU shows up as the `neural` engine.
- Each config also checked which OpenAI routes OVMS serves and ran one tool-call test (a `get_weather` tool; it passes if the model returns that call with `city: Phoenix`).

## Results

### Generation speed (output tokens/s, median of 3)

| Prompt | NPU (OVMS) | CPU (OVMS) | Intel iGPU (OVMS) | RTX 5080 (Ollama) |
|---|---|---|---|---|
| Short (52 tok) | 10.0 | 9.9 | 10.6 | 103.3 |
| Medium (1,167 tok) | 9.8 | 8.8 | 9.9 | 98.0 |
| Long (4,329 tok) | 6.5 | 6.7 | 9.8 | 86.8 |

### Time to first token (s, median of 3)

| Prompt | NPU | CPU | Intel iGPU | RTX 5080 |
|---|---|---|---|---|
| Short | 4.80 | 1.00 | 0.74 | 0.08 |
| Medium | 10.73 | 21.14 | 8.74 | 0.40 |
| Long | 30.49 | 87.81 | 53.81 | 1.47 |

### Prompt processing (prompt tokens/s, median of 3)

| Prompt | NPU | CPU | Intel iGPU | RTX 5080 |
|---|---|---|---|---|
| Short | 11 | 52 | 70 | 720 |
| Medium | 109 | 55 | 134 | 2,942 |
| Long | 142 | 49 | 80 | 2,947 |

### Startup, load, memory and power

| | NPU | CPU | Intel iGPU | RTX 5080 |
|---|---|---|---|---|
| Server ready | 30.7 s (from compile cache) | 3.2 s | 19.4 s (no cache) | 39.3 s (first load from disk) |
| Busy accelerator | `neural` engine ~95% | — | `compute` engine ~100% | GPU 86%, 146 W |
| Total CPU load during runs | 20% (idle was 20%) | 45% | 24% | 24% |
| CPU package power, runs vs idle | 26 W vs 34 W idle (no measurable increase) | 63 W vs 39 W | 42 W vs 32 W | 46 W vs 28 W, plus 146 W on the GPU |
| Server peak RAM | 16.8 GB (11.2 GB private) | 8.5 GB | 10.1 GB | ~14 GB reserved by Ollama, 15.1 GB VRAM in use |
| Tool call (`hermes3` / Ollama tools) | pass | pass | pass | pass |

OVMS 2026.4.0 answered on `/v1/models`, `/v3/models`, `/v3/v1/models` and the matching `chat/completions` routes on every device, so both `/v1` and `/v3` work as API bases.

## What the numbers say

- **Generation speed is about the same on the NPU, CPU and iGPU** (~10 tokens/s with short prompts). For an 8B model, memory bandwidth limits all three, and they share the same RAM. The RTX 5080, with its own fast VRAM, is about 10x faster.
- **The NPU's advantage is that it leaves the rest of the laptop alone.** During NPU generation, CPU load stayed at its idle level and package power didn't rise measurably. The CPU run added ~24 W and more than doubled CPU load, and the RTX drew ~146 W. That matches the post's goal: background tasks that don't tie up the GPU or slow down other work.
- **The NPU has a fixed delay before the first token.** It processes prompts in fixed-size chunks, so even a 52-token prompt waits ~4.8 s. On long prompts it's the fastest of the three Intel options (142 vs 80 vs 49 prompt tokens/s), but its generation speed drops more as context grows (6.5 tokens/s at 4.3k, while the iGPU holds 9.8).
- **The NPU used the most host RAM** (16.8 GB peak) of the OVMS runs.
- **Tool calling works** with `--tool_parser hermes3` on every device.

## Caveats

- One session, 3 runs per cell, with background apps open (Brave, Roblox). Idle package power varied between 28 and 39 W from run to run, so read the power figures as rough deltas.
- The laptop was plugged in, and the OMEN performance mode wasn't recorded. Battery drain wasn't measured.
- The RTX numbers use a different quantization (Q4_K_M) and runtime (llama.cpp via Ollama), so they're indicative, not a like-for-like kernel comparison.
- Utilization counters can briefly read over 100%. The tables use averages.

## Files and rerun

- `results-*.json`: one file per device.
- `report-tables.md`: the generated tables.
- `raw/`: the per-second power and utilization samples (`samples-*.jsonl`, `nvsmi-rtx.csv`) and the OVMS logs.

To rerun, use `bench/bench.py` and `bench/report.py` from the repo root. Setup and paths are in the [top-level README](../../README.md).

```bash
python bench/bench.py ovms NPU
python bench/bench.py ovms CPU
python bench/bench.py ovms GPU.0
python bench/bench.py ollama
python bench/report.py
```

`bench.py` backs up and restores the model's `graph.pbtxt`, which the CPU and iGPU runs rewrite. `bench/ov_devices.py` prints the device properties above.
