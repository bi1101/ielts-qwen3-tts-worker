# IELTS Qwen3-TTS Vast worker

Durable SQLite controller → native Vast Serverless routing → official PyWorker on port 3000 → online vLLM-Omni on localhost:18000 → one Qwen3-TTS model per GPU. The small localhost:18001 adapter validates requests/audio and returns JSON; vLLM-Omni owns continuous batching. `allow_parallel_requests=True` keeps PyWorker from serializing generation.

## Deployment and operation

TTS endpoint **42781** (`ielts-qwen3-tts`), workergroup **53004**. The original group 53002 was replaced. See [production.json](deployment/production.json) for the pinned image, worker commit, template and autoscaler settings. LLM endpoint 42668 is separate.

The image pins vLLM-Omni 0.30.0/CUDA/PyTorch and Vast SDK 1.9.1. The template uses direct SSH runtime with a shell-compatible onstart script; it fetches the pinned worker and runs Vast's official bootstrap. Each GPU starts one online server with `max_num_seqs=16` per pipeline stage with CUDA Model Runner V2 and decoder graph buckets `[1, 2, 3, 4]`. All eight precomputed voice profiles load at startup, followed by eight warmups before readiness. No reference audio or embeddings are processed per request.

Keep `cold_workers=0`, `max_workers=8`, target queue time 5 seconds, maximum queue time 600 seconds and inactivity timeout 300 seconds. Cold starts include image/model download and engine startup; send enough concurrent work for the autoscaler to add replicas. Workers hold no durable job state. A stopped instance may retain a model cache and incur storage charges after GPU scale-down. Monitor stuck image pulls separately: one never-ready replica outlived the inactivity window and needed explicit removal during acceptance.

To deploy a code update, push it, create a replacement template with the pinned image and commit in `deployment/onstart.sh`, then attach it to workergroup 53004 and roll workers. This account returned `410 Gone` for template updates, so replacement templates are used. Do not put credentials in templates, source files or logs. Inject the API key through the controller environment.

## Native SDK request

The public OpenAI proxy does not expose audio routes. Use the native SDK:

```python
import asyncio
import os
from vastai import CoroutineServerless
from speech_contract import estimate_cost

async def main():
    client = CoroutineServerless(api_key=os.environ["VAST_API_KEY"])
    try:
        endpoint = await client.get_endpoint(name="ielts-qwen3-tts")
        payload = {
            "input": "Ready for today's challenge?",
            "voice": "alex",
            "response_format": "wav",
            "language": "English",
            "request_id": "stable-job-id",
            "seed": 0,
            "max_new_tokens": 256,
        }
        result = await endpoint.request(
            "/v1/audio/speech", payload,
            cost=estimate_cost(payload["input"], payload), timeout=900,
        )
        if not result.get("ok"):
            raise RuntimeError(f"worker HTTP {result.get('status')}")
        response = result["response"]
        # JSON: request_id, voice, mode, format, sample_rate, duration, audio_base64.
        print({key: response[key] for key in ("request_id", "voice", "mode", "duration")})
    finally:
        await client.close()

asyncio.run(main())
```

Only WAV output is supported. Requests accept `input`, `voice`, `response_format`, `language`, `request_id`, optional matching `mode`, and the generation fields validated in `speech_contract.py`. Invalid requests return HTTP 400. Every success contains a validated, non-silent mono PCM16 WAV at 24 kHz, with matching request/voice/mode metadata and duration.

## Durable controller

```bash
pip install -r requirements.txt
# Set VAST_API_KEY in your shell without committing it.
python tts_controller.py --db jobs.sqlite3 enqueue --voice alex --text 'Hello'
python tts_controller.py --db jobs.sqlite3 run --concurrency 16 --max-concurrency 128
# Continue polling for new jobs after the queue drains:
python tts_controller.py --db jobs.sqlite3 run --continuous --concurrency 16 --max-concurrency 128
```

The controller starts with 16 in-flight jobs when no worker is ready, then targets 16 jobs per ready worker up to 128 by default. It estimates queue time from each ready worker's reported `cur_load / throughput`, and expands only after that estimate exceeds 10 seconds for two consecutive five-second capacity polls; it shrinks immediately when ready capacity falls. Vast's endpoint autoscaler independently uses `target_queue_time=10`, `cold_workers=0` and its 300-second inactivity timeout. AIMD still reduces concurrency on retryable failures. A larger in-flight count is queue depth, not a guarantee of more simultaneous GPU generation.

`JobStore.enqueue(text, voice, mode, generation, output_path=None)` also accepts jobs from another local producer process. SQLite WAL with FULL synchronization stores canonical job hashes, attempts, timestamps, route request IDs, retry deadlines, output locations, metadata and errors. One dispatcher per database holds an OS file lock. Keep this database on persistent local storage; it is not a distributed queue for multiple hosts.

Jobs deduplicate by text, voice, mode and normalized generation parameters. Route IDs are persisted before waiting for GPU completion. Results are validated and atomically written with file/directory fsync before marking success. Restart recovery reuses completed WAV artifacts. Bounded retries with exponential backoff/jitter cover 408, 429, all 5xx, transport/timeouts and cold starts. Other HTTP 4xx are permanent failures. Invalid response audio/metadata is retried within the attempt limit. `run` drains scheduled retries; `--continuous` waits for new arrivals.

Continuous batching does **not** yield byte-identical audio for repeated fixed-seed requests. Deterministic job identity and persisted artifact reuse prevent retry results from replacing a completed output. Use a separate database/output directory when intentionally regenerating a voice/model version.

## Voices and reference assets

| ICL | x-vector-only | Source |
|---|---|---|
| alex | alex_xvec | alex reference WAV/text |
| ben | ben_xvec | ben reference WAV/text |
| jane | jane_xvec | jane reference WAV/text |
| june | june_xvec | june reference WAV/text |

`voices.json` records source mappings. `voices/*.wav` match the character audio under `public/content/characters`. `voice_profiles/` contains eight safetensors profiles, the upstream custom-voice manifest and source hashes. Both modes share the same speaker embedding; only ICL retains reference speech codes/text. Actual embeddings from this 1.7B checkpoint are **2048-dimensional**, rather than the 1024-dimensional example supplied initially.

Rebuild profiles inside the pinned GPU image with `python build_voice_profiles.py --characters /path/to/public/content/characters`. It uses the installed vLLM-Omni precompute utility once for each source voice, then derives its x-vector variant. Commit regenerated profiles with their provenance and roll workers to use them.

## Validation and benchmarks

```bash
python -m unittest discover -v
python integration_check.py --output benchmarks/acceptance.json
python benchmark.py --sizes 1,2,4,8 --output benchmarks/bursts.json
python benchmark.py --sizes 8,16,32,64,128 --jobs 128 --output benchmarks/concurrency.json
python benchmark.py --sizes 256 --jobs 2048 --monitor-endpoint-id 42781 --output benchmarks/scaling.json
```

Acceptance checks all eight voices, malformed requests, and injected 503/timeout retries followed by a real endpoint response. Benchmarks use bounded dispatch loops and validate every result. Reports retain metadata/hashes, never audio or credentials. Worker snapshots distinguish ready capacity from replicas still booting. For GPU readings, inspect `nvidia-smi` over the instance's registered SSH connection; do not assume every replica has the same GPU.

[The concurrency report](benchmarks/concurrency-2026-10-27.md) records the earlier single-GPU sweep. Its throughput plateau was approximately 238–242 jobs/min at concurrency 16–128. A longer pre-rollout test passed 1,024/1,024 jobs at concurrency 256, averaging 257 jobs/min, p95 61.843 seconds. These measurements use short challenge text, not long-form narration. Direct-Qwen comparison, compiled-mode tuning and subjective voice-quality assessment remain unmeasured; no relative-speed claim is made.

[Final deployment results](benchmarks/final-2026-10-27.md) include all eight preloaded voices, 4,096 successful sustained jobs at concurrency 256/512, and observed processing across six workers. The 512 run averaged 556 jobs/min while replicas warmed; eight fully warm GPUs were not benchmarked.
