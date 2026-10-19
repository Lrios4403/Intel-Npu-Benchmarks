"""Combines results-*.json into comparison tables (markdown)."""
import glob
import json
import os

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)
ORDER = ["npu", "cpu", "igpu", "rtx"]
NAMES = {"npu": "NPU (OVMS)", "cpu": "CPU (OVMS)", "igpu": "Intel iGPU (OVMS)", "rtx": "RTX 5080 (Ollama)"}

results = {}
for path in glob.glob(os.path.join(HERE, "results-*.json")):
    with open(path) as f:
        r = json.load(f)
    if r.get("median"):
        results[r["label"]] = r
labels = [l for l in ORDER if l in results]


def cell(value, fmt="{:.1f}"):
    return "-" if value is None else fmt.format(value)


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


print("## Generation speed (tokens/s, median of 3)\n")
table(["Prompt"] + [NAMES[l] for l in labels],
      [[f"{size} (~{results[labels[0]]['median'][size].get('prompt_tokens', '?')} tok)"]
       + [cell(results[l]["median"][size].get("decode_tps")) for l in labels]
       for size in ("short", "medium", "long")])

print("## Time to first token (s, median of 3)\n")
table(["Prompt"] + [NAMES[l] for l in labels],
      [[size] + [cell(results[l]["median"][size].get("ttft_s"), "{:.2f}") for l in labels]
       for size in ("short", "medium", "long")])

print("## Prompt processing (tokens/s, median of 3)\n")
table(["Prompt"] + [NAMES[l] for l in labels],
      [[size] + [cell(results[l]["median"][size].get("prefill_tps"), "{:.0f}") for l in labels]
       for size in ("short", "medium", "long")])

print("## Startup, memory, power, tool calling\n")
rows = []
for l in labels:
    r = results[l]
    pu = r.get("power_and_util", {})
    runs, idle = pu.get("all_runs", {}), pu.get("idle", {})
    engines = runs.get("engines", {})
    accel = ", ".join(f"{k.split('@')[0]} {v['mean_pct']}% avg / {v['max_pct']}% max" for k, v in engines.items()) or "-"
    mem = r.get("memory") or {}
    if l == "rtx":
        mem_text = f"VRAM {runs.get('rtx', {}).get('vram_used_gb', '-')} GB"
        rtx = runs.get("rtx", {})
        accel = f"GPU {rtx.get('util_pct', '-')}% avg, {rtx.get('power_w', '-')} W"
    else:
        mem_text = f"{mem.get('peak_working_set_gb', '-')} GB peak RAM"
    rows.append([NAMES[l], cell(r.get("startup_s")) + " s",
                 cell(r.get("cold_first", {}).get("ttft_s"), "{:.2f}") + " s",
                 mem_text,
                 f"{cell(runs.get('package_w'))} W (idle {cell(idle.get('package_w'))} W)",
                 cell(runs.get("cpu_util_pct")) + "%", accel,
                 "pass" if (r.get("tool_call") or {}).get("pass") else "fail"])
table(["Config", "Startup", "Cold 1st token", "Memory", "CPU package power", "CPU load", "Accelerator load",
       "Tool call"], rows)

for l in labels:
    r = results[l]
    print(f"### {NAMES[l]}\n- command/backend: {r.get('backend')} {r.get('command', r.get('model'))}")
    if r.get("endpoints"):
        print(f"- endpoints: {r['endpoints']}")
    print(f"- tool call: {r.get('tool_call')}")
    print(f"- sample response (short prompt): {r.get('sample_response', '')[:300]!r}\n")
