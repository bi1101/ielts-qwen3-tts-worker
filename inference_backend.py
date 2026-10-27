"""Inference adapters. Direct Qwen is the production fallback; vLLM-Omni is probed explicitly."""
from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class TTSBackend(ABC):
    @abstractmethod
    def load(self, voices: dict[str, dict[str, Any]], root: Path) -> None: ...

    @abstractmethod
    def generate_batch(self, items: list[dict[str, Any]]) -> list[tuple[str, bytes, int]]: ...


class DirectQwenBackend(TTSBackend):
    """Qwen3TTSModel with one cached prompt per voice and one generation loop."""

    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base") -> None:
        self.model_id = model_id
        self.model = None
        self.prompts: dict[str, Any] = {}

    def load(self, voices: dict[str, dict[str, Any]], root: Path) -> None:
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        for voice, config in voices.items():
            ref_audio = root / config["file"]
            if not ref_audio.is_file():
                raise FileNotFoundError(ref_audio)
            self.prompts[voice] = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=config.get("ref_text"),
                x_vector_only_mode=config["x_vector_only_mode"],
            )
        log.info("cached %d Qwen voice clone prompts", len(self.prompts))

    def generate_batch(self, items: list[dict[str, Any]]) -> list[tuple[str, bytes, int]]:
        import io
        import soundfile as sf

        if not items:
            return []
        if self.model is None:
            raise RuntimeError("Qwen model is not loaded")
        texts = [item["input"] for item in items]
        languages = [item.get("language", "Auto") for item in items]
        # create_voice_clone_prompt returns a one-item list for one reference
        # recording; the batched API wants the prompt items themselves.
        prompts = [self.prompts[item["voice"]][0] for item in items]
        with torch.inference_mode():
            wavs, sample_rate = self.model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=prompts,
                max_new_tokens=max(int(item.get("max_new_tokens", 2048)) for item in items),
            )
        results = []
        for item, wav in zip(items, wavs, strict=True):
            output = io.BytesIO()
            sf.write(output, wav, sample_rate, format="WAV", subtype="PCM_16")
            results.append((item["request_id"], output.getvalue(), sample_rate))
        return results


def probe_vllm_omni() -> dict[str, Any]:
    """Report support without pretending that importability proves feature parity."""
    module = importlib.util.find_spec("vllm_omni")
    if module is None:
        return {"available": False, "reason": "vllm_omni is not installed"}
    return {
        "available": True,
        "reason": "module is installed; Qwen3-TTS Base clone API still requires a feature probe",
        "required_features": [
            "Qwen3-TTS Base",
            "ICL ref_audio/ref_text",
            "x_vector_only_mode",
            "reusable clone prompts",
            "batched generation",
        ],
    }
