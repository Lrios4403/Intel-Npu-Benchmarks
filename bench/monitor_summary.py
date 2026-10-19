"""Per-phase resource summary of monitor.jsonl (written by monitor.ps1).

    python monitor_summary.py [monitor.jsonl] [--since <epoch seconds>]

Phases are contiguous runs of the same `phase` value (idle, ovms-NPU-prefix-true,
ovms-GPU.0-..., ollama). Prints a markdown table and writes monitor-summary.json.
"""
import json
import os
import statistics
import sys

from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)


def load(path, since=0):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row["t"] >= since:
                rows.append(row)
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]


def stat(values, fn=statistics.mean, digits=1):
    values = [v for v in values if v is not None]
    return round(fn(values), digits) if values else None


def apply_markers(rows, path):
    """Label each sample with the latest step marker (markers.jsonl from more_bench.py).
    Samples from before the first marker keep the process-based phase."""
    if not os.path.exists(path):
        return rows
    marks = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda m: m["t"])
    i = -1
    for row in rows:
        while i + 1 < len(marks) and marks[i + 1]["t"] <= row["t"]:
            i += 1
        if i >= 0:
            row["phase"] = marks[i]["label"]
    return rows


def segments(rows):
    out = []
    for row in rows:
        if out and out[-1]["phase"] == row["phase"]:
            out[-1]["rows"].append(row)
        else:
            out.append({"phase": row["phase"], "rows": [row]})
    return [s for s in out if len(s["rows"]) >= 3]


def value(row, key):
    return next((v for k, v in row["v"].items() if key in k), None)


def summarize(seg):
    rows = seg["rows"]
    eng = lambda name: [sum(v for k, v in r["eng"].items() if k.startswith(name + "|")) for r in rows]
    mem = lambda name: [r["adapter_mem_gb"].get(name) for r in rows]
    proc = lambda name: [r["procs_ws_gb"].get(name) for r in rows]
    temps = [max(r["temps_c"].values()) for r in rows if r.get("temps_c")]
    return {
        "phase": seg["phase"],
        "minutes": round((rows[-1]["t"] - rows[0]["t"]) / 60, 1),
        "samples": len(rows),
        "cpu_util_pct": {"mean": stat([value(r, "processor utility") for r in rows]),
                         "max": stat([value(r, "processor utility") for r in rows], max)},
        "package_w": {"mean": stat([num(value(r, "rapl_package0_pkg")) and value(r, "rapl_package0_pkg") / 1000 for r in rows]),
                      "max": stat([num(value(r, "rapl_package0_pkg")) and value(r, "rapl_package0_pkg") / 1000 for r in rows], max)},
        "cores_w": stat([num(value(r, "rapl_package0_pp0")) and value(r, "rapl_package0_pp0") / 1000 for r in rows]),
        "igpu_w": stat([num(value(r, "rapl_package0_pp1")) and value(r, "rapl_package0_pp1") / 1000 for r in rows]),
        "ram_available_gb_min": stat([num(value(r, "available mbytes")) and value(r, "available mbytes") / 1024 for r in rows], min),
        "ram_committed_gb_max": stat([num(value(r, "committed bytes")) and value(r, "committed bytes") / 2**30 for r in rows], max),
        # Engine counters spike far past 100% now and then, so median and p95 rather than mean and max
        "npu_util_pct": {"median": stat(eng("npu"), statistics.median), "p95": stat(eng("npu"), p95)},
        "igpu_util_pct": {"median": stat(eng("igpu"), statistics.median), "p95": stat(eng("igpu"), p95)},
        "npu_shared_mem_gb_max": stat(mem("npu|shared"), max, 2),
        "igpu_shared_mem_gb_max": stat(mem("igpu|shared"), max, 2),
        "rtx": {"power_w_mean": stat([num(r["rtx"].get("power_w")) for r in rows]),
                "power_w_max": stat([num(r["rtx"].get("power_w")) for r in rows], max),
                "util_pct_mean": stat([num(r["rtx"].get("util_pct")) for r in rows]),
                "vram_gb_max": stat([num(r["rtx"].get("vram_mb")) and num(r["rtx"]["vram_mb"]) / 1024 for r in rows], max, 2),
                "temp_c_max": stat([num(r["rtx"].get("temp_c")) for r in rows], max)},
        "thermal_zone_c_max": stat(temps, max),
        "ovms_ws_gb_max": stat(proc("ovms"), max, 2),
        "ollama_ws_gb_max": stat(proc("ollama"), max, 2),
        "disk_mb_s_mean": stat([num(value(r, "disk bytes/sec")) and value(r, "disk bytes/sec") / 2**20 for r in rows]),
    }


def main():
    args = sys.argv[1:]
    since = 0
    if "--since" in args:
        since = float(args[args.index("--since") + 1])
        del args[args.index("--since"):args.index("--since") + 2]
    path = args[0] if args else os.path.join(HERE, "monitor.jsonl")
    rows = apply_markers(load(path, since), os.path.join(HERE, "markers.jsonl"))
    summary = [summarize(s) for s in segments(rows)]
    with open(os.path.join(HERE, "monitor-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    cols = ["phase", "min", "CPU % avg/max", "package W avg/max", "iGPU W", "RAM free min", "commit max",
            "NPU % median/p95", "iGPU % median/p95", "NPU/iGPU shared GB", "RTX W avg/max", "RTX % avg", "VRAM GB", "temp °C max",
            "OVMS RAM", "Ollama RAM"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in summary:
        cells = [s["phase"], s["minutes"], f"{s['cpu_util_pct']['mean']}/{s['cpu_util_pct']['max']}",
                 f"{s['package_w']['mean']}/{s['package_w']['max']}", s["igpu_w"], s["ram_available_gb_min"],
                 s["ram_committed_gb_max"], f"{s['npu_util_pct']['median']}/{s['npu_util_pct']['p95']}",
                 f"{s['igpu_util_pct']['median']}/{s['igpu_util_pct']['p95']}",
                 f"{s['npu_shared_mem_gb_max']}/{s['igpu_shared_mem_gb_max']}",
                 f"{s['rtx']['power_w_mean']}/{s['rtx']['power_w_max']}", s["rtx"]["util_pct_mean"],
                 s["rtx"]["vram_gb_max"], max(filter(None, [s["thermal_zone_c_max"], s["rtx"]["temp_c_max"]]), default=None),
                 s["ovms_ws_gb_max"], s["ollama_ws_gb_max"]]
        print("| " + " | ".join("-" if c is None else str(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
