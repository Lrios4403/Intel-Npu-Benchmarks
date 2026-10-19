# Downloads the two models from the Medium article's setup into the OVMS model repository.
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import snapshot_download

from paths import MODELS as models
for repo in ("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", "llmware/llama-3.1-8b-instruct-npu-ov"):
    target = os.path.join(models, *repo.split("/"))
    start = time.time()
    path = snapshot_download(repo_id=repo, local_dir=target)
    size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)
    print(f"{repo}: {size / 1e9:.2f} GB in {time.time() - start:.0f}s -> {path}", flush=True)
