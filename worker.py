"""Thin Vast PyWorker adapter for the local vLLM-Omni speech server."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import signal
import sys
import importlib.util
import json
import re
import subprocess
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import aiohttp.web
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from audio_validation import validate_wav
from speech_contract import VOICES, GENERATION_FIELDS, estimate_cost, validate_request, voice_mode

ROOT = Path(__file__).resolve().parent
MODEL_SERVER_HOST = os.getenv("MODEL_SERVER_HOST", "127.0.0.1")
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "18000"))
WRAPPER_PORT = int(os.getenv("WORKER_MODEL_PORT", "18001"))
VLLM_URL = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"
MODEL_LOG_FILE = os.getenv("MODEL_LOG", "/workspace/vllm-omni.log")
MODEL_HEALTH_ENDPOINT = os.getenv("MODEL_HEALTH_ENDPOINT", "/health")
MODEL_LOG_MAX_BYTES = int(os.getenv("MODEL_LOG_MAX_BYTES", str(64 * 1024 * 1024)))
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_TEXT = 4_000
READY_TIMEOUT = float(os.getenv("VLLM_READY_TIMEOUT", "1200"))
WARMUP_TEXT = os.getenv("TTS_WARMUP_TEXT", "Ready.")
log = logging.getLogger("qwen3-tts-pyworker")
_ready = False
_server_process: subprocess.Popen | None = None
_http_runner: aiohttp.web.AppRunner | None = None
_http_session: aiohttp.ClientSession | None = None
_log_forwarder: threading.Thread | None = None
_fatal_logged = False


def _env_lines(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    return [line.strip() for line in raw.splitlines() if line.strip()] if raw else default


class _AccessLogRedactor(logging.Filter):
    """Prevent Vast's query-string API key from entering aiohttp access logs."""
    _secret = re.compile(r"([?&](?:api_key|token)=)[^&\s\"]+", re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        original = record.getMessage()
        message = self._secret.sub(r"\1[REDACTED]", original)
        if message != original:
            record.msg, record.args = message, ()
        return True


def _configure_log_redaction() -> None:
    logging.getLogger("aiohttp.access").addFilter(_AccessLogRedactor())


def _rotate_model_log() -> None:
    path = Path(MODEL_LOG_FILE)
    if path.exists() and path.stat().st_size >= MODEL_LOG_MAX_BYTES:
        backup = path.with_suffix(path.suffix + ".1")
        backup.unlink(missing_ok=True)
        path.replace(backup)

def _write_log(message: str) -> None:
    Path(MODEL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _forward_model_output(stream, log_file) -> None:
    """Keep Vast stdout useful while retaining a bounded local model log."""
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            log_file.write(line)
            log_file.flush()
            print(f"[vllm] {line.rstrip()}", flush=True)
    finally:
        stream.close()


def _startup_diagnostics() -> None:
    gpu = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    _write_log(
        f"VLLM_OMNI_CONFIG model={MODEL_ID} model_server={VLLM_URL} "
        f"wrapper_port={WRAPPER_PORT} gpu={gpu} pid={os.getpid()}"
    )


def _vllm_command() -> list[str]:
    # Extend the installed, tested pipeline with preloaded profiles; vLLM owns scheduling.
    source = Path(importlib.util.find_spec("vllm_omni").origin).parent / "deploy/qwen3_tts.yaml"
    deploy = Path("/workspace/tts-deploy.yaml")
    base = source.read_text()
    base = base.replace("decode_batch_max_size: 4", "decode_batch_max_size: 4\n      decode_cudagraph_batch_sizes: [1, 2, 3, 4]")
    base = base.replace("custom_voice_dir: ", "custom_voice_dir: ")
    deploy.write_text("custom_voice_dir: " + json.dumps(str(ROOT / "voice_profiles")) + "\n" + base)
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID, "--omni", "--host", MODEL_SERVER_HOST, "--port", str(MODEL_SERVER_PORT),
        "--deploy-config", str(deploy), "--trust-remote-code",
        "--stage-overrides", '{"0":{"max_num_seqs":16},"1":{"max_num_seqs":16}}',
    ]


def _speech_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": body["input"], "voice": body["voice"], "task_type": "Base",
        "response_format": "wav", "language": body.get("language", "Auto"),
    }
    # Forward only documented vLLM-Omni speech parameters.
    for key in GENERATION_FIELDS:
        if key in body:
            payload[key] = body[key]
    return payload


async def _wait_for_vllm() -> None:
    assert _http_session is not None
    deadline = time.monotonic() + READY_TIMEOUT
    last_error = "not started"
    while time.monotonic() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            _write_log("VLLM_OMNI_FATAL")
            raise RuntimeError(f"vLLM-Omni exited with code {_server_process.returncode}; see {MODEL_LOG_FILE}")
        try:
            async with _http_session.get(f"{VLLM_URL}{MODEL_HEALTH_ENDPOINT}", timeout=10) as response:
                if response.status == 200:
                    return
                last_error = f"health HTTP {response.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        await asyncio.sleep(2)
    raise RuntimeError(f"vLLM-Omni did not become ready: {last_error}")


async def _warm_voice(name: str) -> None:
    assert _http_session is not None
    payload = {"input": WARMUP_TEXT, "voice": name, "task_type": "Base", "response_format": "wav", "language": "English", "max_new_tokens": 64}
    async with _http_session.post(f"{VLLM_URL}/v1/audio/speech", json=payload, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"voice warmup failed for {name}: HTTP {response.status}: {(await response.text())[:1000]}")
        validate_wav(await response.read(), expected_sample_rate=24_000)


async def _prepare_voices() -> None:
    assert _http_session is not None
    async with _http_session.get(f"{VLLM_URL}/v1/audio/voices", timeout=30) as response:
        # Omni also advertises a built-in `default` alias.
        if response.status != 200 or not set(VOICES).issubset((await response.json()).get("voices", [])):
            raise RuntimeError("vLLM-Omni did not preload all eight voice profiles")
    await asyncio.gather(*(_warm_voice(name) for name in VOICES))


async def _health(_request: aiohttp.web.Request) -> aiohttp.web.Response:
    global _fatal_logged
    healthy = _ready and _server_process is not None and _server_process.poll() is None
    if _ready and _server_process is not None and _server_process.poll() is not None and not _fatal_logged:
        _fatal_logged = True
        _write_log("VLLM_OMNI_FATAL")
    if healthy and _http_session is not None:
        try:
            async with _http_session.get(f"{VLLM_URL}{MODEL_HEALTH_ENDPOINT}", timeout=3) as response:
                healthy = response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            healthy = False
    return aiohttp.web.json_response({"status": "ok" if healthy else "loading"}, status=200 if healthy else 503)


async def _speech(request: aiohttp.web.Request) -> aiohttp.web.Response:
    try:
        try:
            body = validate_request(await request.json())
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
            "request_id": request_id, "voice": body["voice"], "mode": voice_mode(body["voice"]),
            "format": "wav", "sample_rate": metadata["sample_rate"], "duration": metadata["duration"],
            "audio_base64": base64.b64encode(audio).decode("ascii"),
        })
    except Exception:
        log.exception("TTS request failed")
        return aiohttp.web.json_response({"error": "TTS generation failed"}, status=500)


@asynccontextmanager
async def lifecycle():
    global _ready, _http_runner, _http_session, _server_process, _log_forwarder, _fatal_logged
    _rotate_model_log()
    for marker in _env_lines("MODEL_INFO_LOG_MSGS", ["VLLM_OMNI_LOADING"]):
        _write_log(marker)
    _startup_diagnostics()
    log_path = Path(MODEL_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    _fatal_logged = False
    _server_process = subprocess.Popen(
        _vllm_command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    _log_forwarder = threading.Thread(
        target=_forward_model_output, args=(_server_process.stdout, log_file),
        name="vllm-log-forwarder", daemon=True,
    )
    _log_forwarder.start()
    _http_session = aiohttp.ClientSession()
    try:
        await _wait_for_vllm()
        await _prepare_voices()
        app = aiohttp.web.Application()
        app.router.add_get("/health", _health)
        app.router.add_post("/v1/audio/speech", _speech)
        _http_runner = aiohttp.web.AppRunner(app)
        await _http_runner.setup()
        await aiohttp.web.TCPSite(_http_runner, "127.0.0.1", WRAPPER_PORT).start()
        _ready = True
        for marker in _env_lines("MODEL_LOAD_LOG_MSG", ["VLLM_OMNI_READY"]):
            _write_log(marker)
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
        if _log_forwarder:
            _log_forwarder.join(timeout=5)
        log_file.close()
        _write_log("VLLM_OMNI_STOPPED")


def workload(body):
    try:
        return estimate_cost(body.get("input", ""), body)
    except (TypeError, ValueError, OverflowError):
        return 1


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1", model_server_port=WRAPPER_PORT,
    model_log_file=MODEL_LOG_FILE, model_healthcheck_url="/health", lifecycle=lifecycle(),
    handlers=[HandlerConfig(
        route="/v1/audio/speech", allow_parallel_requests=True, max_queue_time=600.0,
        workload_calculator=workload,
        benchmark_config=BenchmarkConfig(
            dataset=[{"input": os.getenv("TTS_BENCHMARK_TEXT", "Ready for today's challenge?"),
                      "voice": os.getenv("TTS_BENCHMARK_VOICE", "alex"),
                      "response_format": "wav", "language": os.getenv("TTS_BENCHMARK_LANGUAGE", "English")}],
            runs=int(os.getenv("BENCHMARK_RUNS", "1")),
            concurrency=int(os.getenv("BENCHMARK_CONCURRENCY", "1")),
            do_warmup=os.getenv("BENCHMARK_WARMUP", "false").lower() == "true",
        ),
    )],
    log_action_config=LogActionConfig(
        on_load=_env_lines("MODEL_LOAD_LOG_MSG", ["VLLM_OMNI_READY"]),
        on_error=_env_lines("MODEL_ERROR_LOG_MSGS", ["VLLM_OMNI_FATAL_ERROR", "VLLM_OMNI_FATAL"]),
        on_info=_env_lines("MODEL_INFO_LOG_MSGS", ["VLLM_OMNI_LOADING"]),
    ),
)

if __name__ == "__main__":
    _configure_log_redaction()
    Worker(worker_config).run()
