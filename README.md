# IELTS Qwen3-TTS Vast worker

Custom Vast PyWorker for Qwen/Qwen3-TTS-12Hz-1.7B-Base voice cloning.

The worker exposes `POST /v1/audio/speech` with the OpenAI speech request shape (`input`, `voice`, `response_format`, optional `language`). It supports ICL voices (`alex`, `ben`, `jane`, `june`) and x-vector-only voices (`*_xvec`). The response is WAV audio.
