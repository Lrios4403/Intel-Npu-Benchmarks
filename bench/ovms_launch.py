"""Start and stop OVMS 2026.4.0 for one model, the way the blog post's commands do.

Backs up the model folder's graph.pbtxt before launching (OVMS rewrites it from
the CLI flags) and restores it on stop, so the user's own config is untouched.
"""
import http.client
import json
import os
import shutil
import subprocess
import time

from paths import MODELS, OVMS_DIR
from paths import OUT as HERE  # outputs go to $NPUBENCH_OUT (default runs/latest)


def get_status(path, port=8000):
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path)
        res = conn.getresponse()
        body = res.read()
        conn.close()
        return res.status, body
    except Exception:
        return None, b""


class Ovms:
    def __init__(self, source, name, device, tool_parser, extra=(), log_label=None):
        self.source, self.name, self.device = source, name, device
        self.model_dir = os.path.join(MODELS, *source.split("/"))
        self.graph = os.path.join(self.model_dir, "graph.pbtxt")
        self.backup = os.path.join(HERE, "graph-backups", source.replace("/", "__") + ".pbtxt")
        self.args = [os.path.join(OVMS_DIR, "ovms.exe"), "--rest_port", "8000", "--rest_bind_address", "127.0.0.1",
                     "--model_repository_path", MODELS, "--source_model", source, "--model_name", name,
                     "--task", "text_generation", "--target_device", device, *(
                         ["--tool_parser", tool_parser] if tool_parser else []), *extra]
        self.log = os.path.join(HERE, f"ovms-{log_label or name}-{device}.log")
        self.proc = None

    def start(self, timeout=1800):
        if os.path.exists(self.graph) and not os.path.exists(self.backup):
            os.makedirs(os.path.dirname(self.backup), exist_ok=True)
            shutil.copy2(self.graph, self.backup)
        env = dict(os.environ)
        env["OVMS_DIR"] = OVMS_DIR
        env["PYTHONHOME"] = os.path.join(OVMS_DIR, "python")
        env["PATH"] = os.pathsep.join([OVMS_DIR, env["PYTHONHOME"], os.path.join(env["PYTHONHOME"], "Scripts"), env["PATH"]])
        started = time.time()
        self.proc = subprocess.Popen(self.args, cwd=OVMS_DIR, env=env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        while time.time() - started < timeout:
            status, _ = get_status(f"/v2/models/{self.name}/ready")
            _, config = get_status("/v1/config")
            if status == 200 or (b"AVAILABLE" in config and self.name.encode() in config):
                self.startup_s = round(time.time() - started, 1)
                return self.startup_s
            if self.proc.poll() is not None:
                raise RuntimeError(f"OVMS exited early (code {self.proc.returncode}); log: {open(self.log).read()[-1500:]}")
            time.sleep(0.5)
        raise RuntimeError("OVMS never became ready")

    def stop(self):
        if self.proc:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            self.proc = None
            time.sleep(2)
        if os.path.exists(self.backup):
            shutil.copy2(self.backup, self.graph)
        elif os.path.exists(self.graph):
            # No graph existed before this launch (a freshly downloaded model): remove ours
            os.remove(self.graph)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
