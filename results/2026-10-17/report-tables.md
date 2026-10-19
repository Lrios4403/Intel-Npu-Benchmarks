## Generation speed (tokens/s, median of 3)

| Prompt | NPU (OVMS) | CPU (OVMS) | Intel iGPU (OVMS) | RTX 5080 (Ollama) |
|---|---|---|---|---|
| short (~52 tok) | 10.0 | 9.9 | 10.6 | 103.3 |
| medium (~1167 tok) | 9.8 | 8.8 | 9.9 | 98.0 |
| long (~4329 tok) | 6.5 | 6.7 | 9.8 | 86.8 |

## Time to first token (s, median of 3)

| Prompt | NPU (OVMS) | CPU (OVMS) | Intel iGPU (OVMS) | RTX 5080 (Ollama) |
|---|---|---|---|---|
| short | 4.80 | 1.00 | 0.74 | 0.08 |
| medium | 10.73 | 21.14 | 8.74 | 0.40 |
| long | 30.49 | 87.81 | 53.81 | 1.47 |

## Prompt processing (tokens/s, median of 3)

| Prompt | NPU (OVMS) | CPU (OVMS) | Intel iGPU (OVMS) | RTX 5080 (Ollama) |
|---|---|---|---|---|
| short | 11 | 52 | 70 | 720 |
| medium | 109 | 55 | 134 | 2942 |
| long | 142 | 49 | 80 | 2947 |

## Startup, memory, power, tool calling

| Config | Startup | Cold 1st token | Memory | CPU package power | CPU load | Accelerator load | Tool call |
|---|---|---|---|---|---|---|---|
| NPU (OVMS) | 30.7 s | 4.97 s | 16.79 GB peak RAM | 26.4 W (idle 34.4 W) | 20.2% | neural 94.6% avg / 108.2% max | pass |
| CPU (OVMS) | 3.2 s | 0.71 s | 8.45 GB peak RAM | 62.5 W (idle 38.9 W) | 44.9% | - | pass |
| Intel iGPU (OVMS) | 19.4 s | 0.47 s | 10.05 GB peak RAM | 41.8 W (idle 31.5 W) | 24.2% | compute 100.7% avg / 1962.5% max | pass |
| RTX 5080 (Ollama) | 39.3 s | 0.10 s | VRAM 15.14 GB | 46.3 W (idle 28.4 W) | 23.9% | GPU 86.2% avg, 145.8 W | pass |

### NPU (OVMS)
- command/backend: OVMS 2026.4.0 %USERPROFILE%\Downloads\ovms_windows_2026.4.0_python_off\ovms\ovms.exe --rest_port 8000 --rest_bind_address 127.0.0.1 --model_repository_path %USERPROFILE%\Downloads\models --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_name Qwen3-8B --task text_generation --target_device NPU --tool_parser hermes3 --max_prompt_len 16384 --plugin_config {\"NPUW_LLM_PREFILL_ATTENTION_HINT\":\"PYRAMID\"} --cache_dir .ovcache
- endpoints: {'GET /v1/models': 200, 'GET /v3/models': 200, 'GET /v3/v1/models': 200, 'POST /v1/chat/completions': 200, 'POST /v3/chat/completions': 200, 'POST /v3/v1/chat/completions': 200}
- tool call: {'pass': True, 'name': 'get_weather', 'arguments': {'city': 'Phoenix'}}
- sample response (short prompt): '\n\n\n\nA CPU cache is a small, fast memory located close to the CPU that stores frequently accessed data to reduce the time required to fetch it from main memory. It operates on the principle of locality, which includes temporal and spatial locality—meaning data that is recently used or located near re'

### CPU (OVMS)
- command/backend: OVMS 2026.4.0 %USERPROFILE%\Downloads\ovms_windows_2026.4.0_python_off\ovms\ovms.exe --rest_port 8000 --rest_bind_address 127.0.0.1 --model_repository_path %USERPROFILE%\Downloads\models --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_name Qwen3-8B --task text_generation --target_device CPU --tool_parser hermes3
- endpoints: {'GET /v1/models': 200, 'GET /v3/models': 200, 'GET /v3/v1/models': 200, 'POST /v1/chat/completions': 200, 'POST /v3/chat/completions': 200, 'POST /v3/v1/chat/completions': 200}
- tool call: {'pass': True, 'name': 'get_weather', 'arguments': {'city': 'Phoenix'}}
- sample response (short prompt): '\n\n\n\nA CPU cache is a small, fast memory located close to the CPU that stores frequently accessed data to reduce the time required to access it from main memory. It operates on the principle that "locality of reference," meaning programs often reuse data or instructions from the same location. Caches'

### Intel iGPU (OVMS)
- command/backend: OVMS 2026.4.0 %USERPROFILE%\Downloads\ovms_windows_2026.4.0_python_off\ovms\ovms.exe --rest_port 8000 --rest_bind_address 127.0.0.1 --model_repository_path %USERPROFILE%\Downloads\models --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_name Qwen3-8B --task text_generation --target_device GPU.0 --tool_parser hermes3
- endpoints: {'GET /v1/models': 200, 'GET /v3/models': 200, 'GET /v3/v1/models': 200, 'POST /v1/chat/completions': 200, 'POST /v3/chat/completions': 200, 'POST /v3/v1/chat/completions': 200}
- tool call: {'pass': True, 'name': 'get_weather', 'arguments': {'city': 'Phoenix'}}
- sample response (short prompt): '\n\n\n\nA CPU cache is a small, fast memory located close to the CPU that stores frequently accessed data to reduce the time required to access it from main memory. It operates on the principle that "locality of reference," meaning programs often access the same or nearby memory locations. Caches are or'

### RTX 5080 (Ollama)
- command/backend: Ollama 0.40.5 qwen3:8b (Q4_K_M, 8.2B)
- tool call: {'pass': True, 'name': 'get_weather', 'arguments': {'city': 'Phoenix'}}
- sample response (short prompt): 'A CPU cache is a small, high-speed memory located close to the CPU that stores frequently accessed data to reduce the time it takes to fetch information from slower main memory (RAM). The cache operates on the principle of locality—data that is accessed repeatedly or in a predictable pattern is stor'

