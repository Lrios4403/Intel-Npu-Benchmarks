"""Follow-up tests, run while monitor.ps1 records system resources.

    python more_bench.py

Per device (NPU and iGPU via OVMS, RTX 5080 via Ollama):
  idle      the server loaded but doing nothing (after a no-server baseline)
  thinking  Qwen3 with thinking on (its default in agent tools): time to the answer
  parallel  1, 2 and 4 simultaneous requests
  sustained continuous generation for 8 minutes (NPU and iGPU)
  context   a ~15.5k token prompt (under the NPU's 16,384 limit) and a ~17.5k one (over)

Each step appends a marker to markers.jsonl so monitor_summary.py can split the
resource log per step. Results go to results-more.json.
"""
import http.client
import json
import os
import subprocess
import sys
import threading
import time

from ovms_launch import Ovms

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
MARKERS = os.path.join(HERE, "markers.jsonl")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARAGRAPH = (
    "In the early days of personal computing, hobbyists assembled machines from kits, "
    "wrote programs in BASIC, and traded software on cassette tapes and floppy disks. "
    "Memory was measured in kilobytes, storage was slow, and every byte mattered, so "
    "programmers learned to squeeze routines into tight loops and lookup tables. "
    "Over the following decades processors gained caches, pipelines, multiple cores and "
    "dedicated accelerators, while operating systems grew from simple monitors into "
    "complex platforms that schedule work across many kinds of hardware. "
)
WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def mark(label):
    with open(MARKERS, "a") as f:
        f.write(json.dumps({"t": time.time(), "label": label}) + "\n")
    print(f"--- {label} ({time.strftime('%H:%M:%S')})", flush=True)


def doc(sections, nonce):
    body = "\n\n".join(f"Section {i + 1}. {PARAGRAPH}" for i in range(sections))
    return f"Request {nonce}.\n\n{body}\n\nSummarize the text above in about 100 words. /no_think"


def request(backend, messages, max_tokens, think=False, tools=None, num_ctx=8192):
    """One streamed chat request. Returns timing, token counts and the full reply."""
    if backend["kind"] == "ollama":
        body = {"model": backend["model"], "messages": messages, "stream": True, "think": think,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": num_ctx}}
        if tools:
            body["tools"] = tools
        path = "/api/chat"
    else:
        body = {"model": backend["model"], "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        path = "/v3/chat/completions"
    conn = http.client.HTTPConnection("127.0.0.1", backend["port"], timeout=3600)
    start = time.perf_counter()
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    res = conn.getresponse()
    if res.status != 200:
        detail = res.read()[:600].decode(errors="replace")
        conn.close()
        return {"error": f"HTTP {res.status}: {detail}", "total_s": round(time.perf_counter() - start, 2)}
    first, text, thinking, tool_calls, completion, final = None, [], [], [], None, {}
    while True:
        line = res.readline()
        if not line:
            break
        line = line.strip()
        if backend["kind"] == "ollama":
            if not line:
                continue
            event = json.loads(line)
            msg = event.get("message") or {}
            piece, think_piece = msg.get("content") or "", msg.get("thinking") or ""
            tool_calls += msg.get("tool_calls") or []
            if event.get("done"):
                final, completion = event, event.get("eval_count")
        else:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                completion = event["usage"].get("completion_tokens")
            piece, think_piece = "", ""
            for c in event.get("choices") or []:
                d = c.get("delta") or {}
                piece += d.get("content") or ""
                think_piece += d.get("reasoning_content") or ""
                tool_calls += d.get("tool_calls") or []
        if piece or think_piece or tool_calls:
            first = first or time.perf_counter()
        text.append(piece)
        thinking.append(think_piece)
        if final:
            break
    conn.close()
    end = time.perf_counter()
    reply, thought = "".join(text), "".join(thinking)
    if "</think>" in reply:  # OVMS without a reasoning parser streams the thinking inline
        thought, reply = reply.split("</think>", 1)
        thought = thought.replace("<think>", "")
    return {"ttft_s": round(first - start, 2) if first else None, "total_s": round(end - start, 2),
            "completion_tokens": completion, "thinking_chars": len(thought.strip()), "reply": reply.strip()[:500],
            "tool_calls": bool(tool_calls)}


THINK_TASKS = [
    ("bat and ball", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
     lambda r: any(s in r["reply"].replace(" ", "").lower() for s in ("$0.05", "5cents", "0.05", "fivecents"))),
    ("fibonacci code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
     lambda r: "def " in r["reply"]),
    ("tool call", "What's the weather in Phoenix right now?", lambda r: r["tool_calls"]),
]


def thinking(backend, out):
    rows = []
    for name, text, check in THINK_TASKS:
        tools = [WEATHER_TOOL] if name == "tool call" else None
        r = request(backend, [{"role": "user", "content": text}], 3000, think=True, tools=tools)
        r["task"], r["ok"] = name, (check(r) if "error" not in r else False)
        rows.append(r)
        print(f"    thinking {name}: total {r.get('total_s')}s, first token {r.get('ttft_s')}s, "
              f"{r.get('completion_tokens')} tokens, thinking {r.get('thinking_chars')} chars, ok={r['ok']}", flush=True)
    out["thinking"] = rows


def parallel(backend, out):
    rows = {}
    for n in (1, 2, 4):
        results = [None] * n
        def worker(i):
            results[i] = request(backend, [{"role": "user", "content": doc(9, f"par{n}-{i}-{time.time_ns()}")}], 128)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        tokens = sum((r or {}).get("completion_tokens") or 0 for r in results)
        rows[str(n)] = {"wall_s": round(wall, 1), "aggregate_tps": round(tokens / wall, 2),
                        "per_request_total_s": [r.get("total_s") for r in results],
                        "per_request_ttft_s": [r.get("ttft_s") for r in results],
                        "errors": [r["error"] for r in results if r and "error" in r]}
        print(f"    parallel x{n}: wall {rows[str(n)]['wall_s']}s, aggregate {rows[str(n)]['aggregate_tps']} tok/s, "
              f"per request {rows[str(n)]['per_request_total_s']}", flush=True)
    out["parallel"] = rows


def sustained(backend, out, minutes=8):
    samples = []
    end = time.time() + minutes * 60
    i = 0
    while time.time() < end:
        r = request(backend, [{"role": "user", "content": f"Run {i} {time.time_ns()}. Explain how a CPU cache works in about 200 words. /no_think"}], 256)
        if r.get("completion_tokens") and r.get("ttft_s"):
            gen_s = r["total_s"] - r["ttft_s"]
            samples.append({"t": round(time.time(), 1), "decode_tps": round((r["completion_tokens"] - 1) / gen_s, 2) if gen_s > 0 else None})
        i += 1
    first = [s["decode_tps"] for s in samples[:3] if s["decode_tps"]]
    last = [s["decode_tps"] for s in samples[-3:] if s["decode_tps"]]
    out["sustained"] = {"minutes": minutes, "requests": len(samples), "samples": samples,
                        "first_3_avg_tps": round(sum(first) / len(first), 2) if first else None,
                        "last_3_avg_tps": round(sum(last) / len(last), 2) if last else None}
    print(f"    sustained {minutes} min: {len(samples)} requests, first {out['sustained']['first_3_avg_tps']} -> "
          f"last {out['sustained']['last_3_avg_tps']} tok/s", flush=True)


def context(backend, out, num_ctx=8192):
    rows = {}
    for label, sections in (("15.5k", 146), ("17.5k", 165)):
        r = request(backend, [{"role": "user", "content": doc(sections, f"ctx-{label}-{time.time_ns()}")}], 64, num_ctx=num_ctx)
        rows[label] = {k: v for k, v in r.items() if k != "reply"}
        rows[label]["reply_start"] = (r.get("reply") or "")[:120]
        print(f"    context {label}: {json.dumps(rows[label])[:300]}", flush=True)
    out["context"] = rows


def idle(seconds, label):
    mark(label)
    time.sleep(seconds)


def main():
    results = {}
    mark("baseline-idle (no server)")
    time.sleep(120)

    npu_flags = ["--max_prompt_len", "16384", "--plugin_config", '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}',
                 "--cache_dir", ".ovcache"]
    for label, device, flags in (("npu", "NPU", npu_flags), ("igpu", "GPU.0", [])):
        mark(f"{label}:startup")
        with Ovms("OpenVINO/Qwen3-8B-int4-cw-ov", "Qwen3-8B", device, "hermes3", flags, log_label=f"more-{label}") as server:
            out = {"startup_s": server.startup_s}
            backend = {"kind": "openai", "port": 8000, "model": "Qwen3-8B"}
            idle(180, f"{label}:idle (server loaded)")
            mark(f"{label}:thinking"); thinking(backend, out)
            mark(f"{label}:parallel"); parallel(backend, out)
            mark(f"{label}:sustained"); sustained(backend, out)
            if label == "npu":
                mark(f"{label}:context"); context(backend, out)
            results[label] = out
        mark(f"{label}:stopped")
        time.sleep(30)

    rtx = {"kind": "ollama", "port": 11434, "model": "qwen3:8b"}
    mark("rtx:startup")
    request(rtx, [{"role": "user", "content": "hi"}], 1)
    out = {}
    idle(120, "rtx:idle (model loaded)")
    mark("rtx:thinking"); thinking(rtx, out)
    mark("rtx:parallel"); parallel(rtx, out)
    mark("rtx:sustained"); sustained(rtx, out, minutes=4)
    mark("rtx:context"); context(rtx, out, num_ctx=20480)
    results["rtx"] = out
    subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True)
    mark("done")

    with open(os.path.join(HERE, "results-more.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved results-more.json", flush=True)


if __name__ == "__main__":
    main()
