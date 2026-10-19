# Intel NPU LLM benchmarks

Benchmarks of running an 8B LLM (Qwen3 8B) on the Intel AI Boost NPU of a Core Ultra 9 275HX laptop, compared with the CPU, the Intel iGPU and an RTX 5080 Laptop GPU. It also checks the claims and setup from a Medium article on NPU LLMs. Every script, raw result, resource log and server log is here, so the numbers can be checked or rerun.

Written for the post [LLMs on an Intel NPU](https://m4cgyver.net/blogs/llms-on-an-intel-npu) on m4cgyver.net.

## Machine

| | |
|---|---|
| Laptop | HP OMEN MAX Gaming Laptop 16 (on AC power) |
| CPU | Intel Core Ultra 9 275HX, 24 cores; 31.4 GB RAM; Windows 11 Pro 25H2 (26200.9711) |
| NPU | Intel AI Boost (OpenVINO architecture 3720, 2 tiles), driver 32.0.100.5540, the latest at test time |
| iGPU | Intel Graphics (Xe 12.70.4, 64 EUs), driver 32.0.101.8860 |
| dGPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, driver 596.49 |
| Software | OpenVINO Model Server 2026.4.0, OpenVINO GenAI 2026.4.1, Ollama 0.40.5 |

The full hardware, driver and software records (including the latest versions available at test time) are in [results/2026-10-17](results/2026-10-17/README.md).

## Headline results (Qwen3 8B)

| | NPU | CPU | Intel iGPU | RTX 5080 |
|---|---|---|---|---|
| Generation speed, short / 4k-token prompt | 10.0 / 6.5 tokens/s | 9.9 / 6.7 | 10.6 / 9.8 | 103 / 87 |
| First token, short prompt | 4.8 s | 1.0 s | 0.7 s | 0.08 s |
| First token, 4k prompt, first turn | ~27 s | ~88 s | ~47 s | ~2.4 s |
| First token, 4k prompt, cached repeat turn | ~8.3 s | — | ~1.6 s | ~0.5 s |
| Its own speed under an all-core CPU load | −3% | −28% | 0% | — |
| Server RAM + accelerator memory | ~13–15 GB + 6–11 GB NPU shared | 8.5 GB | 9.5 GB + 9.6 GB iGPU shared | 14.8 GB VRAM |
| Tool calling (20 cases) | 20/20 | — | — | 20/20 |

- **Generation speed is the same on the NPU, CPU and iGPU** (~10 tokens/s), because memory bandwidth limits all three. The RTX is ~10x faster.
- **The NPU's strength is staying out of the way:** CPU load and package power stay at idle levels while it generates, and it isn't slowed by heavy CPU work.
- **Its weaknesses:**
  - A fixed ~4.3 s before the first token: prompts are processed in ~1k-token blocks.
  - Slow long prompts: an 11.5k-token prompt takes 2–3 minutes cold and ~15 s cached.
  - Heavy RAM use.

  For agent-style use (long, repeated prompts), the Intel iGPU with prefix caching is the better Intel option.
- **Use channel-wise (`cw`) exports on this NPU generation.** The plain `int4-ov` exports fail to compile, and only the `cw` and NPU-specific exports work.

## Layout

```
bench/       benchmark scripts (see below)
results/
  2026-10-17/  four-device speed, power and utilization comparison; hardware/driver/software records
  2026-10-18/  the Medium article's method, prompt-limit sweep, model compatibility, RAM,
               tool calling, load/stress tests, prompt caching, system resource logs
runs/        default output folder for new runs (gitignored)
```

## Setup

1. **Intel NPU driver** for Windows: <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>
2. **OpenVINO Model Server** for Windows (`ovms_windows_*_python_off.zip`): <https://github.com/openvinotoolkit/model_server/releases>. Unzip it.
3. **Python 3.11+** with the OpenVINO packages:
   ```bash
   pip install openvino openvino-genai huggingface_hub
   ```
4. **Models**, in an OVMS-style repository (`<MODELS_DIR>/<org>/<name>`):
   ```bash
   python -c "from huggingface_hub import snapshot_download as d; d('OpenVINO/Qwen3-8B-int4-cw-ov', local_dir=r'%USERPROFILE%\Downloads\models\OpenVINO\Qwen3-8B-int4-cw-ov')"
   python bench/download_models.py   # TinyLlama + the article's Llama 3.1 8B (optional)
   ```
5. **Optional, for the RTX comparison:** [Ollama](https://ollama.com) with `ollama pull qwen3:8b`.

### Paths

| Variable | Default | Meaning |
|---|---|---|
| `OVMS_DIR` | `%USERPROFILE%\Downloads\ovms_windows_2026.4.0_python_off\ovms` | Folder containing `ovms.exe` |
| `MODELS_DIR` | `%USERPROFILE%\Downloads\models` | OVMS model repository |
| `NPUBENCH_OUT` | `runs/latest` | Where results, logs and caches are written |
| `NPUBENCH_BUILD_DIR` | `%USERPROFILE%\Downloads\website\current` | A project whose `bun run build` serves as real CPU work in `load_bench.py` |

## Running

Run the scripts from the repo root. Each one starts and stops its own OVMS on `127.0.0.1:8000`. OVMS rewrites a model's `graph.pbtxt` from its command-line flags, so the scripts back it up and restore it afterwards. To log system resources while tests run, start the monitor in a second terminal first:

```bash
powershell -ExecutionPolicy Bypass -File bench/monitor.ps1 -Out runs/latest/monitor.jsonl
```

| Command | What it measures |
|---|---|
| `python bench/bench.py ovms NPU` (or `CPU`, `GPU.0`), `python bench/bench.py ollama` | Speed, first token, prompt processing, power and utilization per device; `python bench/report.py` makes the tables |
| `python bench/genai_bench.py sweep qwen8` | OpenVINO GenAI directly (the article's method): `MAX_PROMPT_LEN` 1024–16384, with and without PYRAMID, cold compile vs cached load |
| `python bench/genai_bench.py sweep mem` / `others` | Peak RAM per prompt limit; other models (Llama 3.1 8B, TinyLlama, small Qwens) on the NPU |
| `python bench/tools_bench.py qwen-npu` (`llama-npu`, `qwen-rtx`) | Tool-call reliability, 20 cases |
| `python bench/stress_bench.py` | LLM speed under an all-core CPU stress, and the stress job's slowdown |
| `python bench/load_bench.py` | A real build alongside the LLM; NPU and RTX generating at the same time |
| `python bench/cache_bench.py` | Prompt (prefix) caching: agent turns, chat history, identical repeats |
| `python bench/more_bench.py` | Idle cost, thinking mode, parallel requests, sustained load, the 16k context limit |
| `python bench/monitor_summary.py runs/latest/monitor.jsonl` | Per-phase CPU, RAM, power, NPU/iGPU/RTX utilization, VRAM and temperature from the monitor log |
| `python bench/ov_devices.py` | Every device property OpenVINO reports |

## Caveats

- One laptop, plugged in, with everyday apps open during most runs; `results/2026-10-18/raw/events.txt` notes the changes. Power figures from the CPU package (RAPL) are rough deltas.
- The RTX numbers use Ollama's Q4_K_M build of the same model (llama.cpp), not OpenVINO, so they're indicative.
- The NPU's int4 `cw` export and the RTX's Q4_K_M build are different quantizations.
- GPU and NPU engine counters in Windows occasionally spike past 100%, so the summaries use medians.

## License

[MIT](LICENSE). The result data and reports are covered by the same license.
