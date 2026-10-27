# Deployed endpoint concurrency check — 2026-10-27

Endpoint 42781 / group 53004, online vLLM-Omni 0.30.0, worker commit d9d67c2. Endpoint max_workers=8, cold_workers=0. Each engine stage is configured with max_num_seqs=8. No deployment settings were changed during the test.

128 jobs per level, all eight voices rotated, same 94-character English input, seed=0, max_new_tokens=256. Each result passed HTTP, request/voice/mode metadata and mono PCM16 24 kHz WAV validation. The script uses a fixed number of asynchronous dispatch loops; it retains no audio or credentials in its report.

| Concurrent client requests | Successes | Jobs/min | Median latency | p95 latency |
|---:|---:|---:|---:|---:|
| 8 | 128/128 | 160.14 | 2.835s | 3.507s |
| 16 | 128/128 | 238.38 | 3.677s | 4.535s |
| 32 | 128/128 | 240.36 | 7.277s | 8.281s |
| 64 | 128/128 | 241.87 | 14.307s | 15.716s |
| 128 | 128/128 | 237.25 | 16.883s | 30.894s |

**Use concurrency 16 for this single RTX 5090.** Throughput plateaus there. In this sweep, the highest tested in-flight concurrency was 128, with zero application-level failures; this is not a discovered hard API limit. Requests above the eight active generation sequences queue within the worker/engine.

The worker monitor observed only one replica throughout. Eight replicas would provide a configured ceiling of 64 active generation sequences in the talker stage, but aggregate throughput and autoscaling to eight workers have not been measured. A warm-worker resume preceding this sweep took about 68 seconds end to end. GPU samples during the sweep reached 13,299 MiB allocated and 80% utilization.

The input is `Ready for today's science challenge? Think carefully, and explain how you reached your answer.` Outputs were 4.08–7.92 seconds in the first two levels; this result must not be extrapolated to long-form narration. A preceding short-text burst reached 330 jobs/min at concurrency 64, but that was a different input and did not represent sustained capacity.

Fixed seeds did not produce byte-identical audio across repeated requests under continuous batching. The report stores audio hashes to expose this limitation. Persist and reuse the first successful artifact for a job; seed alone is not a byte-determinism guarantee. No voice-quality listening assessment or same-GPU direct-Qwen comparison was performed in this concurrency check.

Reproduce:

```bash
python -m unittest test_benchmark -v
python benchmark.py --sizes 8,16,32,64,128 --jobs 128 --output benchmarks/concurrency.json
```

`concurrency-2026-10-27.json` contains each validated request and percentile summaries. `concurrency-workers-2026-10-27.jsonl` records worker snapshots; `concurrency-gpu-2026-10-27.json` records retained GPU samples.

The subsequent [production rollout report](final-2026-10-27.md) includes 256/512-in-flight tests and observed multiworker processing.
