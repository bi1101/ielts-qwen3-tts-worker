# IELTS Qwen3-TTS Vast Serverless worker

This repository serves `Qwen/Qwen3-TTS-12Hz-1.7B-Base` through Vast Serverless. The worker loads one model per GPU, builds all eight clone prompts once, and runs a single generation loop with a bounded microbatch accumulator.

## API

Use the Vast native SDK; Vast's public OpenAI proxy does not support audio routes.

```python
from vastai import CoroutineServerless
client = CoroutineServerless(api_key=VAST_API_KEY)
endpoint = await client.get_endpoint(name="ielts-qwen3-tts")
result = await endpoint.request("/v1/audio/speech", {
    "input": "Ready for today's challenge?",
    "voice": "alex",                 # alex, ben, jane, june, and *_xvec
    "mode": "icl",
    "response_format": "wav",
    "language": "English",
    "request_id": "your-idempotency-id",
}, cost=100)
```

The SDK response contains a JSON `response` with `request_id`, `voice`, `mode`, `format`, `sample_rate`, `duration`, and `audio_base64`. WAV is validated before it is returned.

## Durable controller

The controller uses SQLite WAL mode, a unique content hash for idempotency, atomic WAV writes, retry/permanent-failure classification, endpoint request IDs, adaptive bounded concurrency, and a text/audio-token workload estimate.

```bash
export VAST_API_KEY=...
python tts_controller.py --db jobs.sqlite3 enqueue --voice alex --text 'Hello'
python tts_controller.py --db jobs.sqlite3 run --endpoint ielts-qwen3-tts --concurrency 8 --max-concurrency 8
python tts_controller.py --db jobs.sqlite3 run --continuous
```

Malformed input, unsupported voices, mode mismatches, HTTP 400 responses, metadata mismatches, and invalid audio are permanent failures. 408, 429, 5xx, timeouts, and cold-start/transport failures are retried up to `--max-attempts`.

## Microbatching

Set `TTS_MAX_BATCH_SIZE` (default 4) and `TTS_MAX_WAIT_MS` (default 35) in the template. Requests are grouped by voice family/mode and text-length bucket. Successful members are preserved; a failed batch is retried per item. Generation remains single-loop per GPU.

## Backend decision

`inference_backend.py` includes an explicit `probe_vllm_omni()` capability probe. Direct `Qwen3TTSModel` is production default until a real vLLM-Omni feature probe and throughput benchmark prove support for Base ICL, x-vector-only, reusable prompts, and batched generation. No speed claim is made without benchmark data.

## Benchmarking and operations

```bash
python benchmark.py --endpoint ielts-qwen3-tts --sizes 1,2,4,8 --output benchmark-results.json
```

The endpoint is dedicated to this worker and remains configured with `cold_workers=0`, high `max_workers`, queue-time autoscaling, and a five-minute inactivity timeout. Worker state is stateless; SQLite and output files belong on the controller or durable storage.

Reference mapping is recorded in `voices.json`; ICL and x-vector entries share their character WAV file, while only ICL entries use `ref_text`.
