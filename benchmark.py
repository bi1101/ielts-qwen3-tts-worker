"""Real endpoint benchmark for concurrency/batch sizes and vLLM-Omni capability probe."""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
import wave

from audio_validation import validate_wav
from inference_backend import probe_vllm_omni

VOICES = ["alex", "alex_xvec", "ben", "ben_xvec", "jane", "jane_xvec", "june", "june_xvec"]


async def run_batch(endpoint, size: int) -> dict:
    started = time.perf_counter()
    async def one(index: int):
        voice = VOICES[index % len(VOICES)]
        result = await endpoint.request("/v1/audio/speech", {"input": f"Benchmark sample {index}, ready for the science challenge?", "voice": voice, "mode": "xvec" if voice.endswith("_xvec") else "icl", "response_format": "wav", "language": "English", "request_id": f"bench-{size}-{index}"}, cost=100, timeout=900)
        if not result.get("ok"):
            raise RuntimeError(result)
        response = result["response"]
        audio = base64.b64decode(response["audio_base64"], validate=True)
        validate_wav(audio)
        return result
    results = await asyncio.gather(*(one(i) for i in range(size)))
    elapsed = time.perf_counter() - started
    return {"batch_size": size, "elapsed_seconds": elapsed, "jobs_per_minute": size / elapsed * 60, "latencies": [round(r["latency"], 3) for r in results]}


async def main() -> None:
    from vastai import CoroutineServerless
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="ielts-qwen3-tts")
    parser.add_argument("--sizes", default="1,2,4,8")
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()
    client = CoroutineServerless(api_key=os.environ["VAST_API_KEY"], default_request_timeout=900)
    try:
        endpoint = await client.get_endpoint(name=args.endpoint)
        results = {"vllm_omni_probe": probe_vllm_omni(), "endpoint": args.endpoint, "runs": []}
        for size in [int(value) for value in args.sizes.split(",")]:
            results["runs"].append(await run_batch(endpoint, size))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
