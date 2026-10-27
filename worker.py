from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp.web
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from audio_validation import validate_wav
from batch_engine import BatchEngine
from inference_backend import DirectQwenBackend

ROOT = Path(__file__).resolve().parent
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE = "/workspace/qwen-tts.log"
MODEL_HEALTH_ENDPOINT = "/health"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_BATCH_SIZE = int(os.getenv("TTS_MAX_BATCH_SIZE", "4"))
MAX_WAIT_MS = int(os.getenv("TTS_MAX_WAIT_MS", "35"))

log = logging.getLogger("qwen3-tts-worker")
_model_ready = False
_http_runner: aiohttp.web.AppRunner | None = None
_engine: BatchEngine | None = None

with (ROOT / "voices.json").open(encoding="utf-8") as f:
    VOICES: dict[str, dict[str, Any]] = json.load(f)


def write_log(message: str) -> None:
    Path(MODEL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def validate_payload(body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    text = body.get("input")
    voice = body.get("voice", "alex")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input must be a non-empty string")
    if len(text) > 4_000:
        raise ValueError("input exceeds the 4000-character limit")
    if voice not in VOICES:
        raise ValueError(f"unsupported voice: {voice}")
    requested_mode = body.get("mode")
    actual_mode = "xvec" if voice.endswith("_xvec") else "icl"
    if requested_mode is not None and requested_mode != actual_mode:
        raise ValueError(f"voice {voice} requires mode={actual_mode}")
    if body.get("response_format", "wav") not in ("wav", "wave"):
        raise ValueError("response_format must be wav")


async def health(_request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"status": "ok" if _model_ready else "loading"}, status=200 if _model_ready else 503)


async def speech(request: aiohttp.web.Request) -> aiohttp.web.Response:
    try:
        body = await request.json()
        validate_payload(body)
        body.setdefault("request_id", f"worker-{id(request)}")
        body.setdefault("language", "Auto")
        if _engine is None:
            raise RuntimeError("batch engine is not ready")
        result = await _engine.submit(body)
        metadata = validate_wav(result["audio"], expected_sample_rate=result["sample_rate"])
        return aiohttp.web.json_response({
            "request_id": body["request_id"],
            "voice": body["voice"],
            "mode": "xvec" if body["voice"].endswith("_xvec") else "icl",
            "format": "wav",
            "sample_rate": result["sample_rate"],
            "duration": metadata["duration"],
            "audio_base64": base64.b64encode(result["audio"]).decode("ascii"),
        })
    except ValueError as exc:
        return aiohttp.web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("TTS request failed")
        return aiohttp.web.json_response({"error": "TTS generation failed"}, status=500)


@asynccontextmanager
async def lifecycle():
    global _model_ready, _http_runner, _engine
    write_log("QWEN_TTS_LOADING")
    backend = DirectQwenBackend(MODEL_ID)
    backend.load(VOICES, ROOT / "voices")
    _engine = BatchEngine(backend, MAX_BATCH_SIZE, MAX_WAIT_MS)
    await _engine.start()
    app = aiohttp.web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/audio/speech", speech)
    _http_runner = aiohttp.web.AppRunner(app)
    await _http_runner.setup()
    await aiohttp.web.TCPSite(_http_runner, "127.0.0.1", MODEL_SERVER_PORT).start()
    _model_ready = True
    write_log("QWEN_TTS_READY")
    try:
        yield
    finally:
        _model_ready = False
        if _engine:
            await _engine.close()
        if _http_runner:
            await _http_runner.cleanup()
        write_log("QWEN_TTS_STOPPED")


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTH_ENDPOINT,
    lifecycle=lifecycle(),
    handlers=[
        HandlerConfig(
            route="/v1/audio/speech",
            allow_parallel_requests=True,
            max_queue_time=600.0,
            workload_calculator=lambda body: max(1.0, len(body.get("input", "")) * 3.0),
            benchmark_config=BenchmarkConfig(
                dataset=[{"input": "Ready for today's challenge?", "voice": "alex", "response_format": "wav", "language": "English"}],
                runs=1,
                concurrency=1,
                do_warmup=False,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["QWEN_TTS_READY"],
        on_error=["QWEN_TTS_FATAL_ERROR"],
        on_info=["QWEN_TTS_LOADING"],
    ),
)

Worker(worker_config).run()
