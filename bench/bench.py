"""Qwen3 8B benchmark: OVMS on NPU / CPU / Intel iGPU, and Ollama on the RTX 5080.

    python bench.py ovms NPU        python bench.py ovms CPU
    python bench.py ovms GPU.0      python bench.py ollama

Each config: wait for the server, probe endpoints, a cold first request, 3 timed
runs each of a short, ~1k and ~4k token prompt, then a tool-call test. Every run
starts with a unique prefix so prompt/prefix caching can't shortcut prefill, and
Qwen3's thinking is switched off (/no_think) so output lengths are comparable.
Results go to results-<label>.json next to this script.
"""
import ctypes
import http.client
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
from paths import MODELS, OVMS_DIR
MODEL_DIR = os.path.join(MODELS, "OpenVINO", "Qwen3-8B-int4-cw-ov")
GRAPH = os.path.join(MODEL_DIR, "graph.pbtxt")
GRAPH_BACKUP = os.path.join(HERE, "graph.pbtxt.original")
MAX_TOKENS = 256
RUNS = 3

PARAGRAPH = (
    "In the early days of personal computing, hobbyists assembled machines from kits, "
    "wrote programs in BASIC, and traded software on cassette tapes and floppy disks. "
    "Memory was measured in kilobytes, storage was slow, and every byte mattered, so "
    "programmers learned to squeeze routines into tight loops and lookup tables. "
    "Over the following decades processors gained caches, pipelines, multiple cores and "
    "dedicated accelerators, while operating systems grew from simple monitors into "
    "complex platforms that schedule work across many kinds of hardware. "
)


def context(target_tokens):
    # ~95 tokens per numbered paragraph with Qwen's tokenizer
    count = max(1, round(target_tokens / 95))
    return "\n\n".join(f"Section {i + 1}. {PARAGRAPH}" for i in range(count))


def prompt(size, nonce):
    head = f"Benchmark request {nonce}."
    if size == "short":
        return f"{head} Explain how a CPU cache works in about 200 words. /no_think"
    target = 1000 if size == "medium" else 4000
    return (f"{head}\n\n{context(target)}\n\n"
            "Summarize the text above in detail in about 200 words. /no_think")


# --- HTTP ------------------------------------------------------------------

def request(port, method, path, body=None, timeout=900):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Content-Type": "application/json"})
    return conn, conn.getresponse()


def get_json(port, path):
    try:
        conn, res = request(port, "GET", path, timeout=10)
        data = res.read()
        conn.close()
        return res.status, (json.loads(data) if data else None)
    except Exception as error:
        return None, str(error)


def stream_openai(port, path, model, text, max_tokens=MAX_TOKENS):
    """Streams a chat completion; returns client-side timings and token counts."""
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    conn, res = request(port, "POST", path, body)
    if res.status == 400:
        # Server without stream_options support: count streamed chunks instead
        res.read(); conn.close()
        body.pop("stream_options")
        start = time.perf_counter()
        conn, res = request(port, "POST", path, body)
    if res.status != 200:
        raise RuntimeError(f"HTTP {res.status}: {res.read()[:300]!r}")
    first = last = None
    chunks, usage, out = 0, None, []
    while True:
        line = res.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
                out.append(piece)
    conn.close()
    end = time.perf_counter()
    return finish(start, first, last, end, chunks, usage, "".join(out))


def stream_ollama(port, model, text, max_tokens=MAX_TOKENS):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "stream": True,
            "think": False, "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192}}
    start = time.perf_counter()
    conn, res = request(port, "POST", "/api/chat", body)
    if res.status != 200:
        raise RuntimeError(f"HTTP {res.status}: {res.read()[:300]!r}")
    first = last = None
    chunks, final, out = 0, {}, []
    while True:
        line = res.readline()
        if not line:
            break
        event = json.loads(line)
        piece = (event.get("message") or {}).get("content") or ""
        if piece:
            now = time.perf_counter()
            first = first or now
            last = now
            chunks += 1
            out.append(piece)
        if event.get("done"):
            final = event
            break
    conn.close()
    end = time.perf_counter()
    usage = {"prompt_tokens": final.get("prompt_eval_count"), "completion_tokens": final.get("eval_count")}
    result = finish(start, first, last, end, chunks, usage, "".join(out))
    # Ollama's own server-side timings (nanoseconds)
    ns = 1e9
    if final.get("eval_duration"):
        result["server_decode_tps"] = round(final["eval_count"] / (final["eval_duration"] / ns), 2)
    if final.get("prompt_eval_duration"):
        result["server_prefill_tps"] = round(final["prompt_eval_count"] / (final["prompt_eval_duration"] / ns), 1)
    result["server_load_s"] = round(final.get("load_duration", 0) / ns, 3)
    return result


def finish(start, first, last, end, chunks, usage, text):
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion = (usage or {}).get("completion_tokens") or chunks
    ttft = (first - start) if first else None
    decode_window = (last - first) if first and last and last > first else None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "chunks": chunks,
        "ttft_s": round(ttft, 3) if ttft else None,
        "decode_tps": round((completion - 1) / decode_window, 2) if decode_window else None,
        "prefill_tps": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None,
        "total_s": round(end - start, 3),
        "text": text,
    }


# --- tool calling ----------------------------------------------------------

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}
TOOL_PROMPT = "What's the weather in Phoenix right now? Use the get_weather tool. /no_think"


def tool_test_openai(port, path, model):
    body = {"model": model, "messages": [{"role": "user", "content": TOOL_PROMPT}],
            "tools": [WEATHER_TOOL], "tool_choice": "auto", "max_tokens": 200, "temperature": 0}
    conn, res = request(port, "POST", path, body)
    raw = res.read()
    conn.close()
    if res.status != 200:
        return {"pass": False, "status": res.status, "raw": raw[:400].decode(errors="replace")}
    message = json.loads(raw)["choices"][0]["message"]
    return judge_tool(message.get("tool_calls") or [], message.get("content"))


def tool_test_ollama(port, model):
    body = {"model": model, "messages": [{"role": "user", "content": TOOL_PROMPT}], "tools": [WEATHER_TOOL],
            "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 200}}
    conn, res = request(port, "POST", "/api/chat", body)
    raw = res.read()
    conn.close()
    message = json.loads(raw).get("message") or {}
    return judge_tool(message.get("tool_calls") or [], message.get("content"))


def judge_tool(calls, content):
    if not calls:
        return {"pass": False, "reason": "no tool_calls", "content": (content or "")[:300]}
    fn = calls[0].get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {"pass": False, "reason": "arguments not JSON", "call": fn}
    ok = fn.get("name") == "get_weather" and isinstance(args, dict) and "phoenix" in str(args.get("city", "")).lower()
    return {"pass": ok, "name": fn.get("name"), "arguments": args}


# --- processes, memory, sampling ------------------------------------------

class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def memory(pid):
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return None
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    kernel32.CloseHandle(handle)
    if not ok:
        return None
    gb = 1024 ** 3
    return {"peak_working_set_gb": round(counters.PeakWorkingSetSize / gb, 2),
            "peak_private_gb": round(counters.PeakPagefileUsage / gb, 2)}


def pids(image):
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return [int(line.split('","')[1]) for line in out.splitlines() if line.startswith('"')]


class Sampler:
    def __init__(self, label):
        self.out = os.path.join(HERE, f"samples-{label}.jsonl")
        self.pidfile = os.path.join(HERE, f"pids-{label}.txt")
        for path in (self.out, self.pidfile):
            if os.path.exists(path):
                os.remove(path)
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             os.path.join(SCRIPT_DIR, "sampler.ps1"), "-Out", self.out, "-PidFile", self.pidfile],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.nvsmi = None

    def watch(self, pid_list):
        with open(self.pidfile, "w") as f:
            f.write(",".join(str(p) for p in pid_list))

    def start_nvidia(self, label):
        self.nvout = os.path.join(HERE, f"nvsmi-{label}.csv")
        self.nvsmi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", "1000"],
            stdout=open(self.nvout, "w"), stderr=subprocess.DEVNULL)

    def stop(self):
        for proc in (self.proc, self.nvsmi):
            if proc:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)

    def summarize(self, windows):
        rows = []
        if os.path.exists(self.out):
            with open(self.out, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            pass
        result = {}
        for name, (t0, t1) in windows.items():
            inside = [r for r in rows if t0 <= r["t"] <= t1]
            if not inside:
                continue
            stats = {}
            for key, label in (("rapl_package0_pkg", "package_w"), ("rapl_package0_pp0", "cores_w"),
                               ("rapl_package0_pp1", "igpu_w")):
                values = [v / 1000 for r in inside for p, v in r["v"].items() if key in p.lower()]
                if values:
                    stats[label] = round(statistics.mean(values), 1)
            cpu = [v for r in inside for p, v in r["v"].items() if "processor(_total)" in p.lower()]
            if cpu:
                stats["cpu_util_pct"] = round(statistics.mean(cpu), 1)
            engines = {}
            for r in inside:
                per_sample = {}
                for p, v in r["v"].items():
                    p = p.lower()
                    if "gpu engine(" in p and "engtype_" in p:
                        luid = p.split("luid_")[1].split("_phys")[0]
                        engtype = p.split("engtype_")[1].split(")")[0]
                        key = f"{engtype}@{luid}"
                        per_sample[key] = per_sample.get(key, 0) + v
                for key, v in per_sample.items():
                    engines.setdefault(key, []).append(v)
            busy = {k: {"mean_pct": round(statistics.mean(v), 1), "max_pct": round(max(v), 1)}
                    for k, v in engines.items() if max(v) > 0}
            if busy:
                stats["engines"] = busy
            stats["samples"] = len(inside)
            result[name] = stats
        if self.nvsmi and os.path.exists(self.nvout):
            for name, (t0, t1) in windows.items():
                power, util, mem = [], [], []
                with open(self.nvout) as f:
                    for line in f:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 4:
                            continue
                        try:
                            ts = time.mktime(time.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
                        except ValueError:
                            continue
                        if t0 - 1 <= ts <= t1 + 1:
                            try:
                                power.append(float(parts[1])); util.append(float(parts[2])); mem.append(float(parts[3]))
                            except ValueError:
                                pass
                if power:
                    result.setdefault(name, {})["rtx"] = {
                        "power_w": round(statistics.mean(power), 1), "util_pct": round(statistics.mean(util), 1),
                        "vram_used_gb": round(max(mem) / 1024, 2)}
        return result


# --- suites ----------------------------------------------------------------

def timed_suite(stream):
    """Cold first request, then RUNS x (short, medium, long). Returns runs + time windows."""
    runs = []
    windows = {}
    t0 = time.time()
    cold = stream(prompt("short", "cold-0"))
    windows["cold_first"] = (t0, time.time())
    print(f"  cold first request: ttft {cold['ttft_s']}s, {cold['decode_tps']} tok/s", flush=True)
    for size in ("short", "medium", "long"):
        w0 = time.time()
        for i in range(RUNS):
            r = stream(prompt(size, f"{size}-{i}-{time.time_ns()}"))
            r["size"] = size
            runs.append(r)
            print(f"  {size:6} #{i + 1}: prompt {r['prompt_tokens']} tok, ttft {r['ttft_s']}s, "
                  f"prefill {r['prefill_tps']} tok/s, decode {r['decode_tps']} tok/s, "
                  f"{r['completion_tokens']} tokens", flush=True)
        windows[size] = (w0, time.time())
    windows["all_runs"] = (windows["short"][0], windows["long"][1])
    return cold, runs, windows


def medians(runs):
    out = {}
    for size in ("short", "medium", "long"):
        group = [r for r in runs if r["size"] == size]
        out[size] = {key: (statistics.median([r[key] for r in group if r.get(key) is not None])
                           if any(r.get(key) is not None for r in group) else None)
                     for key in ("prompt_tokens", "completion_tokens", "ttft_s", "prefill_tps", "decode_tps",
                                 "server_prefill_tps", "server_decode_tps")}
        out[size] = {k: v for k, v in out[size].items() if v is not None}
    return out


def idle_baseline(sampler, seconds=12):
    t0 = time.time()
    time.sleep(seconds)
    return {"idle": (t0, time.time())}


def run_ovms(device):
    label = {"NPU": "npu", "CPU": "cpu", "GPU.0": "igpu"}[device]
    if pids("ovms.exe"):
        sys.exit("ovms.exe is already running")
    if not os.path.exists(GRAPH_BACKUP):
        shutil.copy2(GRAPH, GRAPH_BACKUP)
    args = [os.path.join(OVMS_DIR, "ovms.exe"), "--rest_port", "8000", "--rest_bind_address", "127.0.0.1",
            "--model_repository_path", MODELS, "--source_model", "OpenVINO/Qwen3-8B-int4-cw-ov",
            "--model_name", "Qwen3-8B", "--task", "text_generation", "--target_device", device,
            "--tool_parser", "hermes3"]
    if device == "NPU":
        # The post's exact NPU command (its plugin config and existing compile cache)
        args += ["--max_prompt_len", "16384",
                 "--plugin_config", '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}',
                 "--cache_dir", ".ovcache"]
    log_path = os.path.join(HERE, f"ovms-{label}.log")
    sampler = Sampler(label)
    windows = idle_baseline(sampler)
    # What setupvars.bat sets, without going through cmd's quoting rules
    env = dict(os.environ)
    env["OVMS_DIR"] = OVMS_DIR
    env["PYTHONHOME"] = os.path.join(OVMS_DIR, "python")
    env["PATH"] = os.pathsep.join([OVMS_DIR, env["PYTHONHOME"], os.path.join(env["PYTHONHOME"], "Scripts"), env["PATH"]])
    launch = time.time()
    shell = subprocess.Popen(args, cwd=OVMS_DIR, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    result = {"label": label, "backend": "OVMS 2026.4.0", "device": device, "model": "OpenVINO/Qwen3-8B-int4-cw-ov",
              "command": subprocess.list2cmdline(args)}
    try:
        server = None
        ready = None
        while time.time() - launch < 1800:
            if not server:
                found = pids("ovms.exe")
                if found:
                    server = found[0]
                    sampler.watch([server])
            status, _ = get_json(8000, "/v2/models/Qwen3-8B/ready")
            _, config = get_json(8000, "/v1/config")
            if status == 200 or (isinstance(config, dict) and "AVAILABLE" in json.dumps(config.get("Qwen3-8B", {}))):
                ready = time.time()
                break
            if shell.poll() is not None:
                raise RuntimeError(f"OVMS exited early, see {log_path}")
            time.sleep(0.5)
        if not ready:
            raise RuntimeError("OVMS never became ready")
        windows["startup"] = (launch, ready)
        result["startup_s"] = round(ready - launch, 1)
        print(f"[{label}] ready after {result['startup_s']}s", flush=True)

        # Which routes this release actually serves
        endpoints = {}
        for path in ("/v1/models", "/v3/models", "/v3/v1/models"):
            endpoints[f"GET {path}"] = get_json(8000, path)[0]
        for path in ("/v1/chat/completions", "/v3/chat/completions", "/v3/v1/chat/completions"):
            try:
                conn, res = request(8000, "POST", path, {"model": "Qwen3-8B", "max_tokens": 1,
                                                          "messages": [{"role": "user", "content": "hi /no_think"}]})
                res.read(); conn.close()
                endpoints[f"POST {path}"] = res.status
            except Exception as error:
                endpoints[f"POST {path}"] = str(error)
        result["endpoints"] = endpoints
        print(f"  endpoints: {endpoints}", flush=True)
        chat = "/v3/chat/completions" if endpoints.get("POST /v3/chat/completions") == 200 else "/v1/chat/completions"
        result["chat_path"] = chat

        cold, runs, run_windows = timed_suite(lambda text: stream_openai(8000, chat, "Qwen3-8B", text))
        windows.update(run_windows)
        t = time.time()
        result["tool_call"] = tool_test_openai(8000, chat, "Qwen3-8B")
        windows["tool_call"] = (t, time.time())
        print(f"  tool call: {result['tool_call']}", flush=True)
        result["memory"] = memory(server)
        result["cold_first"] = {k: v for k, v in cold.items() if k != "text"}
        result["sample_response"] = runs[0]["text"][:1200]
        result["runs"] = [{k: v for k, v in r.items() if k != "text"} for r in runs]
        result["median"] = medians(runs)
    finally:
        subprocess.run(["taskkill", "/PID", str(shell.pid), "/T", "/F"], capture_output=True)
        for pid in pids("ovms.exe"):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        time.sleep(2)
        sampler.stop()
        result["power_and_util"] = sampler.summarize(windows)
        shutil.copy2(GRAPH_BACKUP, GRAPH)
        save(label, result)


def run_ollama():
    label = "rtx"
    model = "qwen3:8b"
    subprocess.run(["ollama", "stop", model], capture_output=True)
    time.sleep(3)
    sampler = Sampler(label)
    sampler.start_nvidia(label)
    windows = idle_baseline(sampler)
    result = {"label": label, "backend": "Ollama 0.40.5", "device": "NVIDIA GeForce RTX 5080 Laptop GPU",
              "model": "qwen3:8b (Q4_K_M, 8.2B)", "num_ctx": 8192}
    try:
        t = time.time()
        load = stream_ollama(11434, model, "Say hi. /no_think", max_tokens=1)
        windows["startup"] = (t, time.time())
        result["startup_s"] = round(time.time() - t, 1)
        result["model_load_s"] = load.get("server_load_s")
        sampler.watch(pids("ollama.exe"))
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
        result["ollama_ps"] = ps.strip()
        print(f"[{label}] loaded in {result['startup_s']}s\n{ps}", flush=True)
        cold, runs, run_windows = timed_suite(lambda text: stream_ollama(11434, model, text))
        windows.update(run_windows)
        t = time.time()
        result["tool_call"] = tool_test_ollama(11434, model)
        windows["tool_call"] = (t, time.time())
        print(f"  tool call: {result['tool_call']}", flush=True)
        runner = [p for p in pids("ollama.exe")]
        result["memory"] = {str(p): memory(p) for p in runner}
        result["cold_first"] = {k: v for k, v in cold.items() if k != "text"}
        result["sample_response"] = runs[0]["text"][:1200]
        result["runs"] = [{k: v for k, v in r.items() if k != "text"} for r in runs]
        result["median"] = medians(runs)
    finally:
        sampler.stop()
        result["power_and_util"] = sampler.summarize(windows)
        save(label, result)


def save(label, result):
    path = os.path.join(HERE, f"results-{label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {path}", flush=True)
    print(json.dumps({k: result.get(k) for k in ("startup_s", "median", "tool_call", "memory", "power_and_util")},
                     indent=1), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "ovms":
        run_ovms(sys.argv[2])
    elif sys.argv[1] == "ollama":
        run_ollama()
