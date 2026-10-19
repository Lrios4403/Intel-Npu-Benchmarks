"""Where the benchmarks find OVMS and the models, and where they write results.

Override with environment variables:
  OVMS_DIR      unpacked OVMS package (the folder containing ovms.exe)
  MODELS_DIR    OVMS model repository (models laid out as <org>/<name>)
  NPUBENCH_OUT  output folder for results, logs and caches (default: runs/latest)
"""
import os

HOME = os.environ.get("USERPROFILE", os.path.expanduser("~"))
OVMS_DIR = os.environ.get("OVMS_DIR", os.path.join(HOME, "Downloads", "ovms_windows_2026.4.0_python_off", "ovms"))
MODELS = os.environ.get("MODELS_DIR", os.path.join(HOME, "Downloads", "models"))
OUT = os.path.abspath(os.environ.get("NPUBENCH_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "latest")))
os.makedirs(OUT, exist_ok=True)
