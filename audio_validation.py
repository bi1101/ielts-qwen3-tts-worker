"""Strict WAV validation shared by the worker and controller."""
from __future__ import annotations

import io
import math
import wave
from array import array


class AudioValidationError(ValueError):
    pass


def validate_wav(data: bytes, expected_sample_rate: int = 24_000) -> dict[str, int | float]:
    if not data or len(data) < 44:
        raise AudioValidationError("empty or truncated WAV payload")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioValidationError(f"invalid WAV: {exc}") from exc
    if channels != 1:
        raise AudioValidationError(f"expected mono WAV, got {channels} channels")
    if sample_width != 2:
        raise AudioValidationError(f"expected 16-bit PCM, got {sample_width * 8}-bit")
    if sample_rate != expected_sample_rate:
        raise AudioValidationError(f"expected {expected_sample_rate} Hz, got {sample_rate} Hz")
    if frames <= 0 or len(raw) != frames * channels * sample_width:
        raise AudioValidationError("WAV contains empty or truncated audio frames")
    samples = array("h")
    samples.frombytes(raw)
    if not any(sample != 0 for sample in samples):
        raise AudioValidationError("WAV contains only silence")
    if not all(math.isfinite(sample) for sample in samples):
        raise AudioValidationError("WAV contains non-finite samples")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration": frames / sample_rate,
    }
