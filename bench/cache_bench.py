"""Prompt (prefix) caching: does a repeated long prefix skip prompt processing?

    python cache_bench.py

Agent tools resend the same long system prompt every turn. Each scenario uses a
fresh ~4k or ~11k token system prompt (unique first line, so nothing is cached
from earlier runs) and measures time to first token per request:

  agent turns   the same system prompt + a different short question, 4 times
  chat history  a 4-turn conversation, each turn resending the growing history
  identical     the exact same request twice

Backends: NPU (OVMS, prefix caching on, the post's config), NPU with prefix
caching off, Intel iGPU (OVMS, prefix caching on), RTX 5080 (Ollama).
"""
import http.client
import json
import os
import subprocess
import sys
import time

from ovms_launch import Ovms

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
SCRATCH_CACHE = os.path.join(HERE, "cache", "prefix-off")
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
SECTIONS = {"4k": 37, "11k": 106}
QUESTIONS = ["In one sentence, what is section 3 about?", "Name two things that grew over the decades.",
             "What was memory measured in?", "Summarize section 10 in five words."]


def system_prompt(size):
    nonce = f"Session {time.time_ns()}. You are a helpful assistant. Reference material follows."
    body = "\n\n".join(f"Section {i + 1}. {PARAGRAPH}" for i in range(SECTIONS[size]))
    return f"{nonce}\n\n{body}"


def request(backend, messages, max_tokens=64):
    """Streams one chat request; returns time to first token, prompt tokens and server-side detail."""
    if backend["kind"] == "ollama":
        body = {"model": backend["model"], "messages": messages, "stream": True, "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384}}
        path = "/api/chat"
    else:
        body = {"model": backend["model"], "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}}
        path = "/v3/chat/completions"
    conn = http.client.HTTPConnection("127.0.0.1", backend["port"], timeout=1800)
    start = time.perf_counter()
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    res = conn.getresponse()
    first, text, usage, final = None, [], {}, {}
    while True:
        line = res.readline()
        if not line:
            break
        line = line.strip()
        if backend["kind"] == "ollama":
            if not line:
                continue
            event = json.loads(line)
            piece = (event.get("message") or {}).get("content") or ""
            if event.get("done"):
                final = event
        else:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            usage = event.get("usage") or usage
            piece = "".join((c.get("delta") or {}).get("content") or "" for c in event.get("choices") or [])
        if piece:
            first = first or time.perf_counter()
            text.append(piece)
        if final:
            break
    conn.close()
    out = {"ttft_s": round(first - start, 2) if first else None, "total_s": round(time.perf_counter() - start, 2),
           "reply": "".join(text).strip()}
    if backend["kind"] == "ollama":
        out["prompt_tokens"] = None
        out["evaluated_prompt_tokens"] = final.get("prompt_eval_count")  # only tokens not reused from cache
        if final.get("prompt_eval_duration"):
            out["server_prefill_s"] = round(final["prompt_eval_duration"] / 1e9, 2)
    else:
        out["prompt_tokens"] = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details") or {}
        if details:
            out["cached_tokens"] = details.get("cached_tokens")
    return out


def agent_turns(backend, size):
    system = system_prompt(size)
    rows = []
    for i, question in enumerate(QUESTIONS):
        r = request(backend, [{"role": "system", "content": system}, {"role": "user", "content": question + " /no_think"}])
        rows.append({"turn": i + 1, **{k: v for k, v in r.items() if k != "reply"}})
        print(f"    agent {size} turn {i + 1}: ttft {r['ttft_s']}s {json.dumps({k: v for k, v in r.items() if k not in ('reply', 'ttft_s')})}", flush=True)
    return rows


def chat_history(backend, size):
    messages = [{"role": "system", "content": system_prompt(size)}]
    rows = []
    for i, question in enumerate(QUESTIONS):
        messages.append({"role": "user", "content": question + " /no_think"})
        r = request(backend, messages)
        messages.append({"role": "assistant", "content": r["reply"] or "OK."})
        rows.append({"turn": i + 1, **{k: v for k, v in r.items() if k != "reply"}})
        print(f"    chat {size} turn {i + 1}: ttft {r['ttft_s']}s {json.dumps({k: v for k, v in r.items() if k not in ('reply', 'ttft_s')})}", flush=True)
    return rows


def identical(backend, size):
    messages = [{"role": "system", "content": system_prompt(size)}, {"role": "user", "content": QUESTIONS[0] + " /no_think"}]
    rows = []
    for i in range(2):
        r = request(backend, messages)
        rows.append({"run": i + 1, **{k: v for k, v in r.items() if k != "reply"}})
        print(f"    identical {size} run {i + 1}: ttft {r['ttft_s']}s", flush=True)
    return rows


def suite(backend, plan):
    out = {}
    for name, size in plan:
        fn = {"agent": agent_turns, "chat": chat_history, "identical": identical}[name]
        out[f"{name}-{size}"] = fn(backend, size)
    return out


NPU_FLAGS = ["--max_prompt_len", "16384", "--plugin_config", '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}']


def main():
    results = {}
    model = "OpenVINO/Qwen3-8B-int4-cw-ov"
    backend = {"kind": "openai", "port": 8000, "model": "Qwen3-8B"}

    print("== NPU, prefix caching on (the post's config)", flush=True)
    with Ovms(model, "Qwen3-8B", "NPU", "hermes3", NPU_FLAGS + ["--cache_dir", ".ovcache", "--enable_prefix_caching", "true"],
              log_label="cache-npu-on") as server:
        graph = open(server.graph).read()
        results["npu_prefix_on"] = {"startup_s": server.startup_s, "graph_prefix_lines": [
            l.strip() for l in graph.splitlines() if "prefix" in l.lower()],
            **suite(backend, [("agent", "4k"), ("agent", "11k"), ("chat", "4k"), ("identical", "4k")])}

    print("== NPU, prefix caching off", flush=True)
    os.makedirs(SCRATCH_CACHE, exist_ok=True)
    with Ovms(model, "Qwen3-8B", "NPU", "hermes3", NPU_FLAGS + ["--cache_dir", SCRATCH_CACHE, "--enable_prefix_caching", "false"],
              log_label="cache-npu-off") as server:
        graph = open(server.graph).read()
        results["npu_prefix_off"] = {"startup_s": server.startup_s, "graph_prefix_lines": [
            l.strip() for l in graph.splitlines() if "prefix" in l.lower()],
            **suite(backend, [("agent", "4k")])}

    print("== Intel iGPU, prefix caching on", flush=True)
    with Ovms(model, "Qwen3-8B", "GPU.0", "hermes3", ["--enable_prefix_caching", "true"], log_label="cache-igpu") as server:
        results["igpu_prefix_on"] = {"startup_s": server.startup_s,
                                     **suite(backend, [("agent", "4k"), ("agent", "11k"), ("chat", "4k")])}

    print("== RTX 5080 (Ollama)", flush=True)
    rtx = {"kind": "ollama", "port": 11434, "model": "qwen3:8b"}
    request(rtx, [{"role": "user", "content": "hi /no_think"}], max_tokens=1)  # load the model
    results["rtx_ollama"] = suite(rtx, [("agent", "4k"), ("agent", "11k"), ("chat", "4k")])
    subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True)

    with open(os.path.join(HERE, "results-cache.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved results-cache.json", flush=True)


if __name__ == "__main__":
    main()
