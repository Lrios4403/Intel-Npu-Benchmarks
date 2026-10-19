"""Does running the LLM slow down real work, and does real work slow down the LLM?

    python load_bench.py

1. `bun run build` of the site, alone (x2).
2. The same build while Qwen3 8B generates continuously on the NPU, the CPU and the iGPU (OVMS), x2 each.
   For each, the LLM's own generation speed alone vs during the builds.
3. NPU generation and RTX 5080 generation (Ollama) at the same time vs each alone.
"""
import http.client
import json
import os
import statistics
import subprocess
import threading
import time

from ovms_launch import Ovms

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
# Any project whose `bun run build` makes a realistic CPU-heavy job (default: the m4cgyver.net site)
SITE = os.environ.get("NPUBENCH_BUILD_DIR", os.path.join(os.path.expanduser("~"), "Downloads", "website", "current"))
PARAGRAPH = ("In the early days of personal computing, hobbyists assembled machines from kits, wrote programs in "
             "BASIC, and traded software on cassette tapes and floppy disks. Memory was measured in kilobytes, "
             "storage was slow, and every byte mattered. ")


def medium_prompt(nonce):
    body = "\n\n".join(f"Section {i + 1}. {PARAGRAPH * 2}" for i in range(7))
    return f"Benchmark request {nonce}.\n\n{body}\n\nSummarize the text above in about 200 words. /no_think"


def stream(port, path, model, text, ollama=False):
    if ollama:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "stream": True, "think": False,
                "options": {"temperature": 0, "num_predict": 256, "num_ctx": 8192}}
        path = "/api/chat"
    else:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 256,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    res = conn.getresponse()
    first = last = None
    tokens = 0
    while True:
        line = res.readline()
        if not line:
            break
        line = line.strip()
        if ollama:
            if not line:
                continue
            event = json.loads(line)
            piece = (event.get("message") or {}).get("content")
            if event.get("done"):
                tokens = event.get("eval_count") or tokens
        else:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                tokens = event["usage"].get("completion_tokens") or tokens
            piece = "".join((c.get("delta") or {}).get("content") or "" for c in event.get("choices") or [])
        if piece:
            now = time.perf_counter()
            first = first or now
            last = now
    conn.close()
    if first and last and last > first and tokens > 1:
        return (tokens - 1) / (last - first)
    return None


class Loop(threading.Thread):
    """Generates continuously and records each request's decode speed with its end time."""

    def __init__(self, port, path, model, ollama=False):
        super().__init__(daemon=True)
        self.args = (port, path, model)
        self.ollama = ollama
        self.samples = []
        self.running = True

    def run(self):
        i = 0
        while self.running:
            tps = stream(*self.args, medium_prompt(f"load-{i}-{time.time_ns()}"), ollama=self.ollama)
            if tps:
                self.samples.append((time.time(), tps))
            i += 1

    def between(self, t0, t1):
        values = [v for t, v in self.samples if t0 <= t <= t1]
        return round(statistics.median(values), 2) if values else None


def build():
    t = time.perf_counter()
    proc = subprocess.run("bun run build", cwd=SITE, shell=True, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - t, 1)
    if proc.returncode != 0:
        raise RuntimeError(f"build failed: {proc.stdout[-500:]} {proc.stderr[-500:]}")
    return elapsed


def alone_speed(port, path, model, n=2, ollama=False):
    values = [stream(port, path, model, medium_prompt(f"alone-{i}-{time.time_ns()}"), ollama=ollama) for i in range(n)]
    values = [v for v in values if v]
    return round(statistics.median(values), 2) if values else None


def main():
    results = {}
    print("== build alone", flush=True)
    build()  # warm the build caches first
    results["build_alone_s"] = [build(), build()]
    print(f"  {results['build_alone_s']}", flush=True)

    devices = [("npu", "NPU", ["--max_prompt_len", "16384", "--plugin_config",
                               '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}', "--cache_dir", ".ovcache"]),
               ("cpu", "CPU", []), ("igpu", "GPU.0", [])]
    for label, device, extra in devices:
        print(f"== build while generating on {label}", flush=True)
        with Ovms("OpenVINO/Qwen3-8B-int4-cw-ov", "Qwen3-8B", device, "hermes3", extra, log_label=f"load-{label}"):
            alone = alone_speed(8000, "/v3/chat/completions", "Qwen3-8B")
            loop = Loop(8000, "/v3/chat/completions", "Qwen3-8B")
            loop.start()
            time.sleep(8)
            t0 = time.time()
            builds = [build(), build()]
            t1 = time.time()
            loop.running = False
            loop.join(timeout=300)
            during = loop.between(t0, t1 + 30)
        results[label] = {"llm_alone_tps": alone, "llm_during_build_tps": during, "build_s": builds}
        print(f"  {results[label]}", flush=True)

    print("== NPU and RTX generating at the same time", flush=True)
    rtx_alone = alone_speed(11434, None, "qwen3:8b", ollama=True)
    with Ovms("OpenVINO/Qwen3-8B-int4-cw-ov", "Qwen3-8B", "NPU", "hermes3", devices[0][2], log_label="load-npu-rtx"):
        npu_alone = alone_speed(8000, "/v3/chat/completions", "Qwen3-8B")
        npu = Loop(8000, "/v3/chat/completions", "Qwen3-8B")
        rtx = Loop(11434, None, "qwen3:8b", ollama=True)
        npu.start(); rtx.start()
        t0 = time.time()
        time.sleep(150)
        t1 = time.time()
        npu.running = rtx.running = False
        npu.join(timeout=300); rtx.join(timeout=300)
    results["npu_plus_rtx"] = {"npu_alone_tps": npu_alone, "npu_with_rtx_tps": npu.between(t0, t1 + 30),
                               "rtx_alone_tps": rtx_alone, "rtx_with_npu_tps": rtx.between(t0, t1)}
    print(f"  {results['npu_plus_rtx']}", flush=True)
    subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True)

    with open(os.path.join(HERE, "results-load.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
