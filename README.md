# IELTS Qwen3-TTS Vast worker

This worker follows Vast's PyWorker layout: the PyWorker on port `3000` is a thin JSON adapter, and it starts one online `vllm-omni:v0.30.0` server on `127.0.0.1:18000` per GPU. vLLM-Omni owns continuous batching; the controller owns durable state and bounded dispatch.

## Deployment

Use the private template attached to endpoint `ielts-qwen3-tts` (endpoint `42781`). It uses `vllm/vllm-omni:v0.30.0`, a 50 GB disk, direct SSH runtime, and the pinned worker commit. Keep endpoint `cold_workers=0`; workers are created on demand and removed after the inactivity timeout.

The startup script clones this repository, installs the small adapter requirements and Vast SDK, then invokes Vast's official `start_server.sh`. The model executable and deploy YAML are resolved from the image's system Python so the bootstrap virtualenv cannot hide vLLM-Omni.

## API

Call the native Vast SDK route (the public OpenAI proxy does not expose audio routes):

```python
from vastai import CoroutineServerless
client = CoroutineServerless(api_key=VAST_API_KEY)
endpoint = await client.get_endpoint(name="ielts-qwen3-tts")
result = await endpoint.request("/v1/audio/speech", {
    "input": "Ready for today's challenge?",
    "voice": "alex",                 # alex, ben, jane, june, and *_xvec
    "mode": "icl",                   # inferred from voice when omitted
    "response_format": "wav",
    "language": "English",
    "request_id": "stable-job-id",
    "seed": 0,
}, cost=100, timeout=900)
```

The adapter registers all eight clone voices once at worker startup. ICL voices use the stored reference text; `_xvec` voices use the same WAV with x-vector-only mode. It validates mono 16-bit 24 kHz non-silent WAV output and returns JSON metadata plus base64 audio.

## Durable controller

```bash
export VAST_API_KEY=...
python tts_controller.py --db jobs.sqlite3 enqueue --voice alex --text 'Hello'
python tts_controller.py --db jobs.sqlite3 run --endpoint ielts-qwen3-tts --concurrency 16 --max-concurrency 64
python tts_controller.py --db jobs.sqlite3 run --continuous --concurrency 16 --max-concurrency 64
```

SQLite WAL stores the content hash, request ID, attempts, status, output path and error. The hash includes text, voice, mode and generation parameters. Running jobs are recovered as retries after a controller restart. Requests are bounded, costed from text and output budget, and retried for timeouts, 408, 429 and 5xx/cold-start errors. Malformed requests, unsupported voices, metadata mismatches and invalid audio are permanent failures. WAV files are fsynced before the atomic rename.

## Voices

`voices.json` maps `alex`, `ben`, `jane`, and `june` to the reference WAV and ICL text. The four `_xvec` names share those WAVs and omit reference text. The source assets are the character files under `public/content/characters`.

## Benchmarking

```bash
python benchmark.py --endpoint ielts-qwen3-tts --sizes 8,16,32,64,128 --jobs 128 --output benchmark-results.json
```

Run one request first to measure cold-start latency, repeat it for warm latency, then run each concurrency level and inspect endpoint workers/logs for scaling and GPU memory. Do not claim a throughput comparison until both backends have been measured on the same GPU and voice mix.

The deployed concurrency sweep passed 640/640 requests. On one RTX 5090, throughput leveled off at concurrency 16 (~238 jobs/min); 128 in flight also succeeded but raised p95 latency to ~31s. See [the report](benchmarks/concurrency-2026-10-27.md) for the workload, per-request evidence, GPU readings, and limits. Fixed seeds did not give byte-identical audio under continuous batching.
