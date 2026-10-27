"""Thin Vast PyWorker adapter for the local vLLM-Omni speech server."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import importlib.util
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import aiohttp.web
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from audio_validation import validate_wav

ROOT = Path(__file__).resolve().parent
VLLM_URL = "http://127.0.0.1:18000"
MODEL_LOG_FILE = os.getenv("MODEL_LOG", "/workspace/vllm-omni.log")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_TEXT = 4_000
READY_TIMEOUT = float(os.getenv("VLLM_READY_TIMEOUT", "1200"))
WARMUP_TEXT = os.getenv("TTS_WARMUP_TEXT", "Ready.")
log = logging.getLogger("qwen3-tts-pyworker")
_ready = False
_server_process: subprocess.Popen | None = None
_http_runner: aiohttp.web.AppRunner | None = None
_http_session: aiohttp.ClientSession | None = None

with (ROOT / "voices.json").open(encoding="utf-8") as f:
    VOICES: dict[str, dict[str, Any]] = json.load(f)


def _mode(voice: str) -> str:
    return "xvec" if voice.endswith("_xvec") else "icl"


def _write_log(message: str) -> None:
    Path(MODEL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _vllm_command() -> list[str]:
    # v0.30.0 is pinned in the Vast template. vLLM-Omni owns batching/scheduling.
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID, "--omni", "--host", "127.0.0.1", "--port", "18000",
        "--deploy-config", str(Path(importlib.util.find_spec("vllm_omni").origin).parent / "deploy/qwen3_tts.yaml"), "--trust-remote-code", "--enforce-eager",
        "--stage-overrides", '{"0":{"max_num_seqs":8},"1":{"max_num_seqs":8}}',
    ]


def _validate_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    text, voice = body.get("input"), body.get("voice", "alex")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input must be a non-empty string")
    if len(text) > MAX_TEXT:
        raise ValueError(f"input exceeds the {MAX_TEXT}-character limit")
    if not isinstance(voice, str) or voice not in VOICES:
        raise ValueError(f"unsupported voice: {voice}")
    expected_mode = _mode(voice)
    if body.get("mode") not in (None, expected_mode):
        raise ValueError(f"voice {voice} requires mode={expected_mode}")
    if body.get("response_format", "wav") not in ("wav", "wave"):
        raise ValueError("response_format must be wav")
    request_id = body.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
        raise ValueError("request_id must be a non-empty string")
    body = dict(body)
    body.setdefault("voice", voice)
    body.setdefault("seed", 0)
    if not isinstance(body["seed"], int) or isinstance(body["seed"], bool) or not 0 <= body["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if "max_new_tokens" in body and (type(body["max_new_tokens"]) is not int or not 1 <= body["max_new_tokens"] <= 4096):
        raise ValueError("max_new_tokens must be an integer in [1,4096]")
    if not isinstance(body.get("language", "Auto"), str):
        raise ValueError("language must be a string")
    return body


def _speech_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": body["input"], "voice": body["voice"], "task_type": "Base",
        "response_format": "wav", "language": body.get("language", "Auto"),
    }
    # Forward only documented vLLM-Omni speech parameters.
    for key in ("max_new_tokens", "speed", "seed", "instructions", "non_streaming_mode", "extra_params"):
        if key in body:
            payload[key] = body[key]
    return payload


async def _wait_for_vllm() -> None:
    assert _http_session is not None
    deadline = time.monotonic() + READY_TIMEOUT
    last_error = "not started"
    while time.monotonic() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            raise RuntimeError(f"vLLM-Omni exited with code {_server_process.returncode}; see {MODEL_LOG_FILE}")
        try:
            async with _http_session.get(f"{VLLM_URL}/health", timeout=10) as response:
                if response.status == 200:
                    return
                last_error = f"health HTTP {response.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        await asyncio.sleep(2)
    raise RuntimeError(f"vLLM-Omni did not become ready: {last_error}")


async def _register_voice(name: str, config: dict[str, Any]) -> None:
    assert _http_session is not None
    ref_audio = ROOT / "voices" / config["file"]
    if not ref_audio.is_file():
        raise FileNotFoundError(ref_audio)
    form = aiohttp.FormData()
    form.add_field("audio_sample", ref_audio.read_bytes(), filename=ref_audio.name, content_type="audio/wav")
    form.add_field("consent", "ielts-science-character-assets")
    form.add_field("name", name)
    if config.get("ref_text"):
        form.add_field("ref_text", config["ref_text"])
    async with _http_session.post(f"{VLLM_URL}/v1/audio/voices", data=form, timeout=120) as response:
        if response.status not in (200, 201):
            detail = (await response.text())[:1000]
            raise RuntimeError(f"voice registration failed for {name}: HTTP {response.status}: {detail}")


async def _warm_voice(name: str) -> None:
    assert _http_session is not None
    payload = {"input": WARMUP_TEXT, "voice": name, "task_type": "Base", "response_format": "wav", "language": "English", "max_new_tokens": 64}
    async with _http_session.post(f"{VLLM_URL}/v1/audio/speech", json=payload, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"voice warmup failed for {name}: HTTP {response.status}: {(await response.text())[:1000]}")
        validate_wav(await response.read(), expected_sample_rate=24_000)


async def _prepare_voices() -> None:
    # Registry writes are sequential; warmups run concurrently and are batched by vLLM.
    for name, config in VOICES.items():
        await _register_voice(name, config)
    await asyncio.gather(*(_warm_voice(name) for name in VOICES))


async def _health(_request: aiohttp.web.Request) -> aiohttp.web.Response:
    healthy = _ready and _server_process is not None and _server_process.poll() is None
    return aiohttp.web.json_response({"status": "ok" if healthy else "loading"}, status=200 if healthy else 503)


async def _speech(request: aiohttp.web.Request) -> aiohttp.web.Response:
    try:
        try:
            body = _validate_request(await request.json())
        except (ValueError, TypeError) as exc:
            return aiohttp.web.json_response({"error": str(exc)}, status=400)
        request_id = body.get("request_id") or f"worker-{time.time_ns()}"
        if not _ready or _http_session is None:
            raise RuntimeError("vLLM-Omni is not ready")
        async with _http_session.post(
            f"{VLLM_URL}/v1/audio/speech", json=_speech_payload(body),
            timeout=aiohttp.ClientTimeout(total=900),
        ) as response:
            audio = await response.read()
            if response.status != 200:
                return aiohttp.web.json_response({"error": f"vLLM HTTP {response.status}: {audio[:1000].decode(errors='replace')}"}, status=response.status)
        metadata = validate_wav(audio, expected_sample_rate=24_000)
        return aiohttp.web.json_response({
            "request_id": request_id, "voice": body["voice"], "mode": _mode(body["voice"]),
            "format": "wav", "sample_rate": metadata["sample_rate"], "duration": metadata["duration"],
            "audio_base64": base64.b64encode(audio).decode("ascii"),
        })
    except Exception:
        log.exception("TTS request failed")
        return aiohttp.web.json_response({"error": "TTS generation failed"}, status=500)


@asynccontextmanager
async def lifecycle():
    global _ready, _http_runner, _http_session, _server_process
    _write_log("VLLM_OMNI_LOADING")
    log_path = Path(MODEL_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    _server_process = subprocess.Popen(_vllm_command(), stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    _http_session = aiohttp.ClientSession()
    try:
        await _wait_for_vllm()
        await _prepare_voices()
        app = aiohttp.web.Application()
        app.router.add_get("/health", _health)
        app.router.add_post("/v1/audio/speech", _speech)
        _http_runner = aiohttp.web.AppRunner(app)
        await _http_runner.setup()
        await aiohttp.web.TCPSite(_http_runner, "127.0.0.1", 18001).start()
        _ready = True
        _write_log("VLLM_OMNI_READY")
        yield
    finally:
        _ready = False
        if _http_runner:
            await _http_runner.cleanup()
        if _http_session:
            await _http_session.close()
        if _server_process and _server_process.poll() is None:
            os.killpg(_server_process.pid, signal.SIGTERM)
            try:
                _server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(_server_process.pid, signal.SIGKILL)
        log_file.close()
        _write_log("VLLM_OMNI_STOPPED")


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1", model_server_port=18001,
    model_log_file=MODEL_LOG_FILE, model_healthcheck_url="/health", lifecycle=lifecycle(),
    handlers=[HandlerConfig(
        route="/v1/audio/speech", allow_parallel_requests=True, max_queue_time=600.0,
        workload_calculator=lambda body: max(1.0, len(body.get("input", "")) * 12 + int(body.get("max_new_tokens", 2048)) / 8),
        benchmark_config=BenchmarkConfig(
            dataset=[{"input": "Ready for today's challenge?", "voice": "alex", "response_format": "wav", "language": "English"}],
            runs=1, concurrency=1, do_warmup=False,
        ),
    )],
    log_action_config=LogActionConfig(
        on_load=["VLLM_OMNI_READY"], on_error=["VLLM_OMNI_FATAL_ERROR"], on_info=["VLLM_OMNI_LOADING"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
