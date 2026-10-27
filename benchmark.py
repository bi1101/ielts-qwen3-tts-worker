"""Bounded native Vast TTS load test; persist metadata, never audio or credentials."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import time
from pathlib import Path

from audio_validation import validate_wav

VOICES = ["alex", "alex_xvec", "ben", "ben_xvec", "jane", "jane_xvec", "june", "june_xvec"]
TEXT = "Ready for today's science challenge? Think carefully, and explain how you reached your answer."


def percentile(values: list[float], fraction: float) -> float | None:
    return round(sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)], 3) if values else None


async def run_batch(endpoint, size: int, jobs: int | None = None) -> dict:
    count = jobs or size
    if not 1 <= size <= 512 or count < 1:
        raise ValueError("concurrency must be 1-512 and jobs must be positive")
    started = time.perf_counter()
    indices = iter(range(count))
    results = []
    run_id = time.time_ns()

    async def worker():
        for index in indices:
            voice = VOICES[index % len(VOICES)]
            request_id = f"bench-{run_id}-{size}-{index}"
            payload = {"input": TEXT, "voice": voice, "mode": "xvec" if voice.endswith("_xvec") else "icl", "response_format": "wav", "language": "English", "request_id": request_id, "seed": 0, "max_new_tokens": 256}
            record = {"voice": voice, "request_id": request_id}
            request_start = time.perf_counter()
            try:
                result = await endpoint.request("/v1/audio/speech", payload, cost=len(TEXT) * 12 + 256 // 8, timeout=900)
                record["status"] = result.get("status")
                if not result.get("ok"):
                    raise RuntimeError(f"worker HTTP {result.get('status')}")
                response = result["response"]
                audio = base64.b64decode(response["audio_base64"], validate=True)
                metadata = validate_wav(audio)
                if any(response.get(key) != payload[key] for key in ("request_id", "voice", "mode")):
                    raise ValueError("response metadata does not match request")
                if response.get("format") != "wav" or response.get("sample_rate") != metadata["sample_rate"] or abs(response.get("duration", 0) - metadata["duration"]) > 0.001:
                    raise ValueError("audio metadata does not match WAV")
                record.update(ok=True, sample_rate=metadata["sample_rate"], duration=metadata["duration"], audio_sha256=hashlib.sha256(audio).hexdigest(), sdk_latency=result.get("latency"), endpoint_request_id=result.get("request_idx"))
            except Exception as exc:
                # Never log SDK result/auth mappings or binary audio.
                record.update(ok=False, error_type=type(exc).__name__)
            record["latency_seconds"] = round(time.perf_counter() - request_start, 3)
            results.append(record)

    await asyncio.gather(*(worker() for _ in range(min(size, count))))
    elapsed = time.perf_counter() - started
    successes = [item for item in results if item["ok"]]
    latencies = [item["latency_seconds"] for item in successes]
    return {"concurrency": size, "jobs": count, "succeeded": len(successes), "failed": count - len(successes), "elapsed_seconds": round(elapsed, 3), "jobs_per_minute": round(len(successes) / elapsed * 60, 2), "latency_p50_seconds": percentile(latencies, 0.5), "latency_p95_seconds": percentile(latencies, 0.95), "requests": results}


async def main() -> None:
    from vastai import CoroutineServerless
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="ielts-qwen3-tts")
    parser.add_argument("--sizes", default="1,2,4,8")
    parser.add_argument("--jobs", type=int, help="Total requests per concurrency level; default one burst")
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(not 1 <= size <= 512 for size in sizes) or (args.jobs is not None and args.jobs < 1):
        parser.error("sizes must be 1-512 and jobs positive")
    client = CoroutineServerless(api_key=os.environ["VAST_API_KEY"], default_request_timeout=900)
    results = {"backend": "vllm-omni-online", "endpoint": args.endpoint, "started_at": time.time(), "text": TEXT, "seed": 0, "max_new_tokens": 256, "runs": []}
    try:
        endpoint = await client.get_endpoint(name=args.endpoint)
        for size in sizes:
            row = await run_batch(endpoint, size, args.jobs)
            results["runs"].append(row)
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_text(json.dumps(results, indent=2) + "\n")
            temp.replace(target)
            print(json.dumps({key: value for key, value in row.items() if key != "requests"}), flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
