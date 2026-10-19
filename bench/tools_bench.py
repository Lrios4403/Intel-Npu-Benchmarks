"""Tool-call reliability: 20 cases against an OpenAI-compatible chat endpoint.

    python tools_bench.py qwen-npu      Qwen3 8B cw on the NPU (OVMS, hermes3)
    python tools_bench.py qwen-rtx      qwen3:8b on the RTX 5080 (Ollama)
    python tools_bench.py llama-npu     Llama 3.1 8B (llmware) on the NPU (OVMS, llama3 parser)

Every request carries all six tools; temperature 0. A case passes only if the
returned tool calls (or the absence of one) match what was asked.
"""
import http.client
import json
import os
import subprocess
import sys
import time

from ovms_launch import Ovms

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
# Model output can include emoji; the Windows console's default code page can't print them
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


TOOLS = [
    fn("get_weather", "Get the current weather for a city.",
       {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("get_time", "Get the current local time in a timezone or city.", {"timezone": {"type": "string"}}, ["timezone"]),
    fn("convert_currency", "Convert an amount between currencies (ISO 4217 codes).",
       {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
       ["amount", "from_currency", "to_currency"]),
    fn("create_event", "Create a calendar event.",
       {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
        "attendees": {"type": "array", "items": {"type": "string"}}}, ["title", "date"]),
    fn("set_thermostat", "Set the home thermostat.",
       {"mode": {"type": "string", "enum": ["heat", "cool", "off"]}, "temperature": {"type": "number"}}, ["mode"]),
    fn("search_files", "Search the user's files.",
       {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
]


def low(v):
    return str(v).strip().lower()


def has(call, name, **checks):
    if call["name"] != name:
        return False
    a = call["args"]
    for key, expect in checks.items():
        if callable(expect):
            if not expect(a.get(key)):
                return False
        elif isinstance(expect, (int, float)):
            try:
                if float(a.get(key)) != float(expect):
                    return False
            except (TypeError, ValueError):
                return False
        elif low(a.get(key)) != low(expect):
            return False
    return True


contains = lambda word: (lambda v: word in low(v))
CASES = [
    ("weather simple", "What's the weather in Phoenix right now?", lambda c, t: len(c) == 1 and has(c[0], "get_weather", city=contains("phoenix"))),
    ("weather with unit", "Weather in Paris, in Celsius please.", lambda c, t: len(c) == 1 and has(c[0], "get_weather", city=contains("paris"), unit="celsius")),
    ("time", "What time is it in Tokyo?", lambda c, t: len(c) == 1 and has(c[0], "get_time", timezone=contains("tokyo"))),
    ("currency", "Convert 250 US dollars to euros.", lambda c, t: len(c) == 1 and has(c[0], "convert_currency", amount=250, from_currency="USD", to_currency="EUR")),
    ("currency paraphrase", "How many yen would I get for 40 British pounds?", lambda c, t: len(c) == 1 and has(c[0], "convert_currency", amount=40, from_currency="GBP", to_currency="JPY")),
    ("thermostat heat", "Set the heat to 70 degrees.", lambda c, t: len(c) == 1 and has(c[0], "set_thermostat", mode="heat", temperature=70)),
    ("thermostat cool", "Turn the AC on and set it to 68.", lambda c, t: len(c) == 1 and has(c[0], "set_thermostat", mode="cool", temperature=68)),
    ("event with list", "Schedule a meeting called Sprint Review on 2026-10-20 with Ana and Ben.",
     lambda c, t: len(c) == 1 and has(c[0], "create_event", title=contains("sprint review"), date="2026-10-20",
                                      attendees=lambda v: isinstance(v, list) and {low(x) for x in v} >= {"ana", "ben"})),
    ("search with int", "Find files that mention 'invoice', at most 5 results.", lambda c, t: len(c) == 1 and has(c[0], "search_files", query=contains("invoice"), max_results=5)),
    ("no tool: math", "What is 17 times 3?", lambda c, t: not c and "51" in t),
    ("no tool: haiku", "Write a haiku about autumn.", lambda c, t: not c and len(t.strip()) > 10),
    ("no tool: definition", "Explain what an NPU is in one sentence.", lambda c, t: not c and len(t.strip()) > 10),
    ("two calls, same tool", "What's the weather in Phoenix and in Seattle?",
     lambda c, t: len(c) == 2 and {low(x["args"].get("city")) for x in c if x["name"] == "get_weather"} >= {"phoenix", "seattle"}),
    ("two calls, two tools", "What's the time in London, and what's the weather there?",
     lambda c, t: len(c) == 2 and {x["name"] for x in c} == {"get_time", "get_weather"}),
    ("extract from sentence", "My flight lands in Denver tomorrow and I want to know if it's cold there.", lambda c, t: len(c) == 1 and has(c[0], "get_weather", city=contains("denver"))),
    ("unicode", "Look up the weather. City: São Paulo", lambda c, t: len(c) == 1 and has(c[0], "get_weather", city=lambda v: "paulo" in low(v))),
    ("optional arg omitted", "Turn the thermostat off.", lambda c, t: len(c) == 1 and has(c[0], "set_thermostat", mode="off")),
    ("number with comma", "Convert 1,000 Canadian dollars to US dollars.", lambda c, t: len(c) == 1 and has(c[0], "convert_currency", amount=1000, from_currency="CAD", to_currency="USD")),
    ("search phrase", "Search my files for 'tax 2025' and show 3 results.", lambda c, t: len(c) == 1 and has(c[0], "search_files", query=contains("tax 2025"), max_results=3)),
    ("use tool result", None, lambda c, t: not c and "98" in t and "sunny" in low(t)),
]
FOLLOW_UP = [
    {"role": "user", "content": "What's the weather in Phoenix?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "get_weather", "arguments": json.dumps({"city": "Phoenix"})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"temp_f": 98, "conditions": "sunny"})},
]


def chat(port, path, model, messages, suffix):
    msgs = [dict(m) for m in messages]
    if suffix:
        for m in msgs:
            if m["role"] == "user":
                m["content"] += suffix
    body = {"model": model, "messages": msgs, "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 300}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t = time.perf_counter()
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    res = conn.getresponse()
    raw = res.read()
    conn.close()
    elapsed = time.perf_counter() - t
    if res.status != 200:
        return None, f"HTTP {res.status}: {raw[:300]!r}", elapsed
    message = json.loads(raw)["choices"][0]["message"]
    calls = []
    for call in message.get("tool_calls") or []:
        f = call.get("function") or {}
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {"__unparsed__": args}
        calls.append({"name": f.get("name"), "args": args if isinstance(args, dict) else {}})
    return calls, message.get("content") or "", elapsed


def chat_ollama(port, path, model, messages, suffix):
    """Ollama's native API with think: false. Its OpenAI-compatible endpoint ignores
    Qwen3's /no_think switch, so the model reasons until it hits max_tokens."""
    msgs = []
    for m in messages:
        m = dict(m)
        if m["role"] == "user" and suffix:
            m["content"] += suffix
        if m.get("tool_calls"):
            m["tool_calls"] = [{"function": {"name": c["function"]["name"],
                                             "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        if m["role"] == "tool":
            m = {"role": "tool", "content": m["content"], "tool_name": "get_weather"}
        msgs.append(m)
    body = {"model": model, "messages": msgs, "tools": TOOLS, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 300}}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t = time.perf_counter()
    conn.request("POST", "/api/chat", json.dumps(body), {"Content-Type": "application/json"})
    res = conn.getresponse()
    raw = res.read()
    conn.close()
    elapsed = time.perf_counter() - t
    if res.status != 200:
        return None, f"HTTP {res.status}: {raw[:300]!r}", elapsed
    message = json.loads(raw).get("message") or {}
    calls = [{"name": (c.get("function") or {}).get("name"),
              "args": (c.get("function") or {}).get("arguments") or {}} for c in message.get("tool_calls") or []]
    return calls, message.get("content") or "", elapsed


def run_suite(label, port, path, model, suffix, send=chat):
    results = []
    for name, text, check in CASES:
        messages = FOLLOW_UP if text is None else [{"role": "user", "content": text}]
        try:
            calls, content, elapsed = send(port, path, model, messages, suffix)
            ok = calls is not None and bool(check(calls, content))
        except Exception as error:
            calls, content, elapsed, ok = None, f"error: {error}", 0, False
        results.append({"case": name, "pass": ok, "seconds": round(elapsed, 1), "calls": calls,
                        "content": (content or "")[:240]})
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24} {elapsed:5.1f}s  {json.dumps(calls)[:160] if calls else (content or '')[:120]!r}", flush=True)
    passed = sum(r["pass"] for r in results)
    summary = {"label": label, "passed": passed, "total": len(results),
               "median_seconds": sorted(r["seconds"] for r in results)[len(results) // 2], "cases": results}
    with open(os.path.join(HERE, f"results-tools-{label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{label}] {passed}/{len(results)} passed", flush=True)
    return summary


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "qwen-npu":
        with Ovms("OpenVINO/Qwen3-8B-int4-cw-ov", "Qwen3-8B", "NPU", "hermes3",
                  ["--max_prompt_len", "16384", "--plugin_config", '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}',
                   "--cache_dir", ".ovcache"]) as server:
            print(f"[{which}] ready in {server.startup_s}s", flush=True)
            run_suite(which, 8000, "/v3/chat/completions", "Qwen3-8B", " /no_think")
    elif which == "llama-npu":
        with Ovms("llmware/llama-3.1-8b-instruct-npu-ov", "Llama-3.1-8B", "NPU", "llama3",
                  ["--max_prompt_len", "4096"], log_label="llama-tools") as server:
            print(f"[{which}] ready in {server.startup_s}s", flush=True)
            run_suite(which, 8000, "/v3/chat/completions", "Llama-3.1-8B", "")
    elif which == "qwen-rtx":
        # Load the model first so the first case doesn't include Ollama's model load
        chat_ollama(11434, None, "qwen3:8b", [{"role": "user", "content": "hi"}], " /no_think")
        run_suite(which, 11434, None, "qwen3:8b", " /no_think", send=chat_ollama)
        subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True)
