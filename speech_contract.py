"""Shared request and audio metadata contract for worker and durable controller."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

from audio_validation import validate_wav

VOICES = json.loads(Path(__file__).with_name("voices.json").read_text())
LANGUAGES = {name.lower(): name for name in ("Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian")}
GENERATION_FIELDS = {"language", "max_new_tokens", "seed", "speed", "instructions", "non_streaming_mode", "extra_params"}


def voice_mode(voice: str) -> str:
    return "xvec" if voice.endswith("_xvec") else "icl"


def _number(value: Any, name: str, low: float, high: float, integer: bool = False):
    if type(value) not in ((int,) if integer else (int, float)) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")
    return value


def validate_request(body: Any) -> dict:
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    unknown = body.keys() - GENERATION_FIELDS - {"input", "voice", "mode", "response_format", "request_id"}
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
    text, voice = body.get("input"), body.get("voice", "alex")
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError("input must be a non-empty string of at most 4000 characters")
    if not isinstance(voice, str) or voice not in VOICES:
        raise ValueError("unsupported voice")
    if body.get("mode", voice_mode(voice)) != voice_mode(voice):
        raise ValueError(f"voice {voice} requires mode={voice_mode(voice)}")
    if body.get("response_format", "wav") not in ("wav", "wave"):
        raise ValueError("response_format must be wav")
    request_id = body.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128):
        raise ValueError("request_id must be a non-empty string of at most 128 characters")
    language = body.get("language", "Auto")
    if not isinstance(language, str) or language.lower() not in LANGUAGES:
        raise ValueError("unsupported language")
    result = dict(body, voice=voice, mode=voice_mode(voice), response_format="wav", language=LANGUAGES[language.lower()], seed=body.get("seed", 0), max_new_tokens=body.get("max_new_tokens", 2048), speed=body.get("speed", 1.0))
    _number(result["seed"], "seed", 0, 2**32 - 1, integer=True)
    _number(result["max_new_tokens"], "max_new_tokens", 1, 4096, integer=True)
    _number(result["speed"], "speed", 0.25, 4)
    if "instructions" in result and (not isinstance(result["instructions"], str) or len(result["instructions"]) > 2000):
        raise ValueError("instructions must be a string of at most 2000 characters")
    if "non_streaming_mode" in result and type(result["non_streaming_mode"]) is not bool:
        raise ValueError("non_streaming_mode must be a boolean")
    extra = result.get("extra_params", {})
    if not isinstance(extra, dict) or extra.keys() - {"temperature", "top_p", "top_k"}:
        raise ValueError("extra_params supports temperature, top_p, top_k only")
    for name, bounds in {"temperature": (0, 2), "top_p": (0.000001, 1), "top_k": (-1, 3072)}.items():
        if name in extra:
            _number(extra[name], name, *bounds, integer=name == "top_k")
            if name == "top_k" and extra[name] == 0:
                raise ValueError("top_k must be -1 or positive")
    return result


def estimate_cost(text: str, generation: dict) -> int:
    # Same units at routing and worker reporting; a text/output-budget estimate, not request count.
    return max(1, len(text) * 12 + int(generation.get("max_new_tokens", 2048)) // 8)


def validate_response(response: Any, request: dict) -> tuple[bytes, dict]:
    if not isinstance(response, dict) or any(response.get(key) != request[key] for key in ("request_id", "voice", "mode")):
        raise ValueError("worker metadata does not match the job")
    audio = base64.b64decode(response["audio_base64"], validate=True)
    metadata = validate_wav(audio)
    duration = response.get("duration")
    if response.get("format") != "wav" or response.get("sample_rate") != metadata["sample_rate"] or type(duration) not in (int, float) or not math.isfinite(duration) or abs(duration - metadata["duration"]) > 0.001:
        raise ValueError("worker metadata does not match WAV")
    return audio, metadata
