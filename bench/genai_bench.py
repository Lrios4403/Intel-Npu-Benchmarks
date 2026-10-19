"""OpenVINO GenAI LLMPipeline benchmarks (the Medium article's method), one config per process.

    python genai_bench.py one <label> <model_dir> <device> <json config> <sizes> [warm]
    python genai_bench.py sweep <plan name>

A config is the dict passed as LLMPipeline's properties, e.g.
{"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 1}. Each config runs twice: cold
(empty CACHE_DIR, so construction includes the NPU compile, then the prompt
suite) and warm (new process, same cache: load time plus one short prompt).
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
from paths import MODELS
CACHE_ROOT = os.path.join(HERE, "cache")
MAX_NEW = 256

PARAGRAPH = (
    "In the early days of personal computing, hobbyists assembled machines from kits, "
    "wrote programs in BASIC, and traded software on cassette tapes and floppy disks. "
    "Memory was measured in kilobytes, storage was slow, and every byte mattered, so "
    "programmers learned to squeeze routines into tight loops and lookup tables. "
    "Over the following decades processors gained caches, pipelines, multiple cores and "
    "dedicated accelerators, while operating systems grew from simple monitors into "
    "complex platforms that schedule work across many kinds of hardware. "
)
# ~106 tokens per numbered paragraph with the Qwen tokenizer (measured in the OVMS runs)
SIZES = {"short": 0, "medium": 7, "long": 35, "xl": 112}
REPEATS = {"short": 3, "medium": 3, "long": 3, "xl": 2}


def prompt(size, nonce):
    head = f"Benchmark request {nonce}."
    if size == "short":
        return f"{head} Explain how a CPU cache works in about 200 words. /no_think"
    body = "\n\n".join(f"Section {i + 1}. {PARAGRAPH}" for i in range(SIZES[size]))
    return f"{head}\n\n{body}\n\nSummarize the text above in detail in about 200 words. /no_think"


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_memory():
    kernel32 = ctypes.windll.kernel32
    # 64-bit handles: without these, ctypes truncates the pseudo-handle to a C int and the call fails
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    gb = 1024 ** 3
    return {"peak_working_set_gb": round(counters.PeakWorkingSetSize / gb, 2),
            "peak_private_gb": round(counters.PeakPagefileUsage / gb, 2)}


def run_one(label, model_dir, device, config, sizes, warm):
    import openvino_genai as ov_genai

    result = {"label": label, "model": os.path.relpath(model_dir, MODELS), "device": device,
              "config": config, "warm": warm, "openvino_genai": ov_genai.__version__}
    t0 = time.perf_counter()
    try:
        # The article passes config positionally; GenAI 2026.4 deprecates that in favour of kwargs
        pipe = ov_genai.LLMPipeline(model_dir, device, **config)
    except Exception as error:
        result["load_error"] = str(error)[:2000]
        result["load_s"] = round(time.perf_counter() - t0, 1)
        result["memory"] = peak_memory()
        return result
    result["load_s"] = round(time.perf_counter() - t0, 1)
    print(f"  [{label}{' warm' if warm else ''}] loaded in {result['load_s']}s", flush=True)

    gen = ov_genai.GenerationConfig()
    gen.max_new_tokens = MAX_NEW
    gen.do_sample = False
    runs = []
    for size in sizes:
        for i in range(1 if warm else REPEATS[size]):
            text = prompt(size, f"{label}-{size}-{i}-{time.time_ns()}")
            t = time.perf_counter()
            try:
                res = pipe.generate([text], gen)
            except Exception as error:
                runs.append({"size": size, "error": str(error)[:1500], "wall_s": round(time.perf_counter() - t, 1)})
                print(f"    {size} #{i + 1}: ERROR {str(error)[:200]}", flush=True)
                break
            m = res.perf_metrics
            run = {"size": size,
                   "input_tokens": m.get_num_input_tokens(),
                   "output_tokens": m.get_num_generated_tokens(),
                   "ttft_s": round(m.get_ttft().mean / 1000, 3),
                   "tpot_ms": round(m.get_tpot().mean, 1),
                   "decode_tps": round(1000 / m.get_tpot().mean, 2) if m.get_tpot().mean else None,
                   "throughput_tps": round(m.get_throughput().mean, 2),
                   "wall_s": round(time.perf_counter() - t, 2)}
            run["prefill_tps"] = round(run["input_tokens"] / run["ttft_s"], 1) if run["ttft_s"] else None
            if not runs or runs[-1]["size"] != size:
                run["text"] = res.texts[0][:600]
            runs.append(run)
            print(f"    {size} #{i + 1}: in {run['input_tokens']} tok, ttft {run['ttft_s']}s, "
                  f"decode {run['decode_tps']} tok/s, out {run['output_tokens']} tok", flush=True)
    result["runs"] = runs
    result["memory"] = peak_memory()
    return result


def summarize(runs):
    out = {}
    for size in SIZES:
        group = [r for r in runs if r["size"] == size and "error" not in r]
        if group:
            mid = lambda key: sorted(r[key] for r in group)[len(group) // 2]
            out[size] = {k: mid(k) for k in ("input_tokens", "output_tokens", "ttft_s", "decode_tps", "prefill_tps")}
        errors = [r["error"] for r in runs if r["size"] == size and "error" in r]
        if errors:
            out.setdefault(size, {})["error"] = errors[0][:400]
    return out


def m(path):
    return os.path.join(MODELS, *path.split("/"))


QWEN8 = m("OpenVINO/Qwen3-8B-int4-cw-ov")
PYRAMID = {"NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID"}
PLANS = {
    # Prompt-limit sweep on the production model, plus the article's exact config
    "qwen8": [
        ("qwen8-article", QWEN8, "NPU", {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 1}, ["short", "medium", "long"]),
        ("qwen8-p1024", QWEN8, "NPU", {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 256}, ["short", "medium"]),
        ("qwen8-p2048", QWEN8, "NPU", {"MAX_PROMPT_LEN": 2048, "MIN_RESPONSE_LEN": 256}, ["short", "medium"]),
        ("qwen8-p4096", QWEN8, "NPU", {"MAX_PROMPT_LEN": 4096, "MIN_RESPONSE_LEN": 256}, ["short", "medium", "long"]),
        ("qwen8-p16384", QWEN8, "NPU", {"MAX_PROMPT_LEN": 16384, "MIN_RESPONSE_LEN": 256}, ["short", "medium", "long", "xl"]),
        ("qwen8-p16384-pyramid", QWEN8, "NPU", {"MAX_PROMPT_LEN": 16384, "MIN_RESPONSE_LEN": 256, **PYRAMID},
         ["short", "medium", "long", "xl"]),
    ],
    # Peak RAM of the 8B pipeline at the smallest and the production prompt limit
    "mem": [
        ("mem-qwen8-p1024", QWEN8, "NPU", {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 256}, ["short"]),
        ("mem-qwen8-p16384-pyramid", QWEN8, "NPU", {"MAX_PROMPT_LEN": 16384, "MIN_RESPONSE_LEN": 256, **PYRAMID}, ["short"]),
    ],
    # The article's model and config, other small models, and the plain (non-cw) Qwen3 8B
    "others": [
        ("llama8-article", m("llmware/llama-3.1-8b-instruct-npu-ov"), "NPU",
         {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 1}, ["short", "medium"]),
        ("llama8-p4096", m("llmware/llama-3.1-8b-instruct-npu-ov"), "NPU",
         {"MAX_PROMPT_LEN": 4096, "MIN_RESPONSE_LEN": 256}, ["short", "medium", "long"]),
        ("tinyllama-p4096", m("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"), "NPU",
         {"MAX_PROMPT_LEN": 4096, "MIN_RESPONSE_LEN": 256}, ["short", "medium"]),
        ("qwen06-p4096", m("OpenVINO/Qwen3-0.6B-int4-ov"), "NPU",
         {"MAX_PROMPT_LEN": 4096, "MIN_RESPONSE_LEN": 256}, ["short", "medium", "long"]),
        ("qwencoder05-p4096", m("OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov"), "NPU",
         {"MAX_PROMPT_LEN": 4096, "MIN_RESPONSE_LEN": 256}, ["short", "medium", "long"]),
        ("qwen8-plain-p4096", m("OpenVINO/Qwen3-8B-int4-ov"), "NPU",
         {"MAX_PROMPT_LEN": 4096, "MIN_RESPONSE_LEN": 256}, ["short", "medium"]),
    ],
}


def sweep(plan):
    for label, model_dir, device, config, sizes in PLANS[plan]:
        out_path = os.path.join(HERE, f"results-genai-{label}.json")
        if os.path.exists(out_path):
            print(f"skip {label} (done)", flush=True)
            continue
        if not os.path.isdir(model_dir):
            print(f"skip {label}: {model_dir} missing", flush=True)
            continue
        cache = os.path.join(CACHE_ROOT, label)
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache)
        full = {**config, "CACHE_DIR": cache}
        record = {"label": label}
        for warm in (False, True):
            proc = subprocess.run([sys.executable, "-u", __file__, "one", label, model_dir, device,
                                   json.dumps(full), ",".join(sizes), "warm" if warm else "cold"],
                                  capture_output=True, text=True, timeout=7200)
            sys.stdout.write(proc.stdout.split("@@RESULT@@")[0])
            if "@@RESULT@@" not in proc.stdout:
                record["cold" if not warm else "warm"] = {"crash": (proc.stderr or proc.stdout)[-2000:],
                                                          "exit": proc.returncode}
                print(f"  [{label}] process failed (exit {proc.returncode}): {(proc.stderr or '')[-400:]}", flush=True)
                break
            res = json.loads(proc.stdout.split("@@RESULT@@")[1])
            record["warm" if warm else "cold"] = res
            if "load_error" in res:
                print(f"  [{label}] load error: {res['load_error'][:400]}", flush=True)
                break
        cold = record.get("cold", {})
        record["summary"] = {
            "model": os.path.relpath(model_dir, MODELS), "config": config,
            "cold_load_s": cold.get("load_s"), "warm_load_s": record.get("warm", {}).get("load_s"),
            "cache_mb": round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(cache) for f in fs) / 2**20),
            "memory": cold.get("memory"), "load_error": cold.get("load_error") or cold.get("crash"),
            "per_size": summarize(cold.get("runs", [])),
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  [{label}] summary: {json.dumps(record['summary'])[:900]}", flush=True)
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "one":
        _, _, label, model_dir, device, config, sizes, mode = sys.argv
        res = run_one(label, model_dir, device, json.loads(config), sizes.split(","), mode == "warm")
        print("@@RESULT@@" + json.dumps(res), flush=True)
    else:
        sweep(sys.argv[2])
