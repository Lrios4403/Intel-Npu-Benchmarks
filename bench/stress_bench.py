"""All-core CPU stress vs LLM generation on each device.

    python stress_bench.py

A 40 s, 24-process SHA-256 loop stands in for heavy CPU work (builds, games).
Measured alone, then while Qwen3 8B generates continuously on the NPU, the CPU
and the iGPU: the stress workload's throughput, and the LLM's generation speed
alone vs under the stress.
"""
import hashlib
import json
import multiprocessing
import os
import time

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
SECONDS = 40
WORKERS = 24


def burn(seconds):
    end = time.perf_counter() + seconds
    block = b"x" * 4096
    count = 0
    while time.perf_counter() < end:
        hashlib.sha256(block).digest()
        count += 1
    return count


def stress():
    with multiprocessing.Pool(WORKERS) as pool:
        t0 = time.time()
        counts = pool.map(burn, [SECONDS] * WORKERS)
        t1 = time.time()
    return round(sum(counts) / SECONDS / 1e6, 3), t0, t1  # million hashes per second


def main():
    from load_bench import Loop, alone_speed
    from ovms_launch import Ovms

    results = {}
    stress()  # warm up
    results["stress_alone_mhs"] = [stress()[0], stress()[0]]
    print(f"stress alone: {results['stress_alone_mhs']} M hashes/s", flush=True)
    for label, device, extra in [
        ("npu", "NPU", ["--max_prompt_len", "16384", "--plugin_config",
                        '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}', "--cache_dir", ".ovcache"]),
        ("cpu", "CPU", []), ("igpu", "GPU.0", [])]:
        with Ovms("OpenVINO/Qwen3-8B-int4-cw-ov", "Qwen3-8B", device, "hermes3", extra, log_label=f"stress-{label}"):
            alone = alone_speed(8000, "/v3/chat/completions", "Qwen3-8B")
            loop = Loop(8000, "/v3/chat/completions", "Qwen3-8B")
            loop.start()
            time.sleep(10)
            mhs, t0, t1 = stress()
            loop.running = False
            loop.join(timeout=300)
            # Requests that finished during the stress window, or within 20 s after it
            during = loop.between(t0 + 5, t1 + 20)
        results[label] = {"llm_alone_tps": alone, "llm_under_stress_tps": during, "stress_mhs": mhs}
        print(f"{label}: {results[label]}", flush=True)
    with open(os.path.join(HERE, "results-stress.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
