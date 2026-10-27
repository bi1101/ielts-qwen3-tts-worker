from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp.web
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE = "/workspace/qwen-tts.log"
MODEL_HEALTH_ENDPOINT = "http://127.0.0.1:18000/health"

log = logging.getLogger("qwen3-tts-worker")
_model: Qwen3TTSModel | None = None
_prompts: dict[str, Any] = {}
_model_lock = threading.Lock()
_http_runner: aiohttp.web.AppRunner | None = None

with (ROOT / "voices.json").open() as f:
    VOICES = json.load(f)


def _write_log(line: str) -> None:
    Path(MODEL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_model() -> None:
    global _model, _prompts
    _write_log("QWEN_TTS_LOADING")
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    _prompts = {}
    for voice, config in VOICES.items():
        _prompts[voice] = _model.create_voice_clone_prompt(
            ref_audio=str(ROOT / "voices" / config["file"]),
            ref_text=config["ref_text"],
            x_vector_only_mode=config["x_vector_only_mode"],
        )
    _write_log("QWEN_TTS_READY")


def _generate(body: dict[str, Any]) -> bytes:
    text = body.get("input")
    voice = body.get("voice", "alex")
    response_format = body.get("response_format", "wav")
    language = body.get("language", "Auto")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input must be a non-empty string")
    if voice not in _prompts:
        raise ValueError(f"unknown voice: {voice}")
    if response_format not in ("wav", "wave"):
        raise ValueError("Qwen3-TTS worker currently returns response_format=wav")
    with _model_lock:
        if _model is None:
            raise RuntimeError("model is not loaded")
        with torch.inference_mode():
            wavs, sample_rate = _model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=_prompts[voice],
                max_new_tokens=int(body.get("max_new_tokens", 2048)),
            )
        output = io.BytesIO()
        sf.write(output, wavs[0], sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()


async def health(_request: aiohttp.web.Request) -> aiohttp.web.Response:
    ready = _model is not None
    return aiohttp.web.json_response({"status": "ok" if ready else "loading"}, status=200 if ready else 503)


async def speech(request: aiohttp.web.Request) -> aiohttp.web.Response:
    try:
        body = await request.json()
        audio = await asyncio.to_thread(_generate, body)
    except ValueError as exc:
        return aiohttp.web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("TTS request failed")
        return aiohttp.web.json_response({"error": "TTS generation failed"}, status=500)
    return aiohttp.web.Response(body=audio, content_type="audio/wav", headers={"Content-Disposition": "attachment; filename=speech.wav"})


@asynccontextmanager
async def lifecycle():
    global _http_runner
    await asyncio.to_thread(_load_model)
    app = aiohttp.web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/audio/speech", speech)
    _http_runner = aiohttp.web.AppRunner(app)
    await _http_runner.setup()
    site = aiohttp.web.TCPSite(_http_runner, "127.0.0.1", MODEL_SERVER_PORT)
    await site.start()
    try:
        yield
    finally:
        if _http_runner is not None:
            await _http_runner.cleanup()
        _write_log("QWEN_TTS_STOPPED")


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTH_ENDPOINT,
    lifecycle=lifecycle(),
    handlers=[
        HandlerConfig(
            route="/v1/audio/speech",
            allow_parallel_requests=False,
            max_queue_time=600.0,
            workload_calculator=lambda body: max(1.0, len(body.get("input", ""))),
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
