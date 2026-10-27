"""SQLite queue and bounded native Vast dispatch. One dispatcher owns each local DB."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import fcntl
import hashlib
import json
import os
import random
import re
import sqlite3
import time
from pathlib import Path

from audio_validation import validate_wav
from speech_contract import GENERATION_FIELDS, VOICES, estimate_cost, validate_request, validate_response

_current_job = contextvars.ContextVar("tts_job", default=None)


class WorkerHTTPError(RuntimeError):
    def __init__(self, status):
        self.status = int(status) if status is not None else None
        super().__init__(f"worker HTTP {self.status}")


def retryable(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if status is None:
        match = re.search(r"(?:HTTP|status)\s*[:=]?\s*(\d{3})", str(exc), re.I)
        status = int(match[1]) if match else None
    return status is None or status in (408, 429) or 500 <= status < 600


def canonical_job(text: str, voice: str, mode: str, generation: dict) -> tuple[str, str]:
    identity = json.dumps({"text": text, "voice": voice, "mode": mode, "generation": generation}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(identity.encode()).hexdigest(), json.dumps(generation, sort_keys=True, separators=(",", ":"), allow_nan=False)


def durable_write(target: Path, audio: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(audio)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class JobStore:
    def __init__(self, path: str | Path, output_dir: str | Path = "tts-output"):
        self.path = str(Path(path).resolve())
        self.output_dir = Path(output_dir).resolve()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, dedupe_key TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL, voice TEXT NOT NULL, mode TEXT NOT NULL,
            generation_json TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, started_at REAL,
            completed_at REAL, endpoint_request_id TEXT, output_path TEXT, error TEXT
          );
          CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, updated_at);
        """)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(jobs)")}
        for name, declaration in (("next_attempt_at", "REAL NOT NULL DEFAULT 0"), ("metadata_json", "TEXT")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    def close(self) -> None:
        self.db.close()

    def recover(self) -> None:
        # Only call while holding the dispatcher lock; enqueuing must never reset live jobs.
        for row in self.db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
            target = Path(row["output_path"])
            if target.exists():
                metadata = validate_wav(target.read_bytes())
                self.mark_success(row, {**metadata, "recovered_artifact": True})
        self.db.execute("UPDATE jobs SET status='retry',updated_at=?,next_attempt_at=0,error='recovered after dispatcher restart' WHERE status='running'", (time.time(),))

    def enqueue(self, text: str, voice: str, mode: str | None, generation: dict, output_path: str | None = None) -> tuple[str, bool]:
        if not isinstance(generation, dict) or generation.keys() - GENERATION_FIELDS:
            raise ValueError("generation must contain supported generation parameters only")
        request = validate_request({"input": text, "voice": voice, **({"mode": mode} if mode is not None else {}), **generation})
        mode = request["mode"]
        params = {key: value for key, value in request.items() if key in GENERATION_FIELDS}
        dedupe_key, generation_json = canonical_job(text, voice, mode, params)
        job_id = dedupe_key[:32]
        path = str(Path(output_path).resolve() if output_path else self.output_dir / f"{job_id}.wav")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT job_id FROM jobs WHERE dedupe_key=?", (dedupe_key,)).fetchone()
            if existing:
                self.db.execute("COMMIT")
                return existing["job_id"], False
            if self.db.execute("SELECT 1 FROM jobs WHERE output_path=?", (path,)).fetchone() or Path(path).exists():
                raise ValueError("output path already belongs to another artifact")
            stamp = time.time()
            self.db.execute("INSERT INTO jobs(job_id,dedupe_key,text,voice,mode,generation_json,status,created_at,updated_at,output_path) VALUES(?,?,?,?,?,?,'queued',?,?,?)", (job_id, dedupe_key, text, voice, mode, generation_json, stamp, stamp, path))
            self.db.execute("COMMIT")
            return job_id, True
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def claim(self, max_attempts: int = 5):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("UPDATE jobs SET status='failed',completed_at=?,updated_at=?,error='attempt limit reached after recovery' WHERE status IN ('queued','retry') AND attempt_count>=?", (time.time(), time.time(), max_attempts))
            row = self.db.execute("SELECT * FROM jobs WHERE status IN ('queued','retry') AND next_attempt_at<=? ORDER BY created_at LIMIT 1", (time.time(),)).fetchone()
            if row:
                stamp = time.time()
                self.db.execute("UPDATE jobs SET status='running',attempt_count=attempt_count+1,started_at=?,updated_at=?,error=NULL WHERE job_id=?", (stamp, stamp, row["job_id"]))
                row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
            self.db.execute("COMMIT")
            return row
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def record_request(self, job_id: str, request_idx) -> None:
        if request_idx:
            self.db.execute("UPDATE jobs SET endpoint_request_id=?,updated_at=? WHERE job_id=? AND endpoint_request_id IS NOT ?", (str(request_idx), time.time(), job_id, str(request_idx)))

    def mark_success(self, row, metadata: dict) -> None:
        self.db.execute("UPDATE jobs SET status='succeeded',completed_at=?,updated_at=?,metadata_json=?,error=NULL WHERE job_id=?", (time.time(), time.time(), json.dumps(metadata, sort_keys=True), row["job_id"]))

    def mark_error(self, row, exc: Exception, max_attempts: int) -> bool:
        retry = retryable(exc) and row["attempt_count"] < max_attempts
        delay = min(60, 2**min(row["attempt_count"], 6)) + random.uniform(0, 1) if retry else 0
        # SDK exceptions can contain authentication mappings; persist only safe diagnostics.
        detail = str(exc) if isinstance(exc, WorkerHTTPError) else type(exc).__name__
        self.db.execute("UPDATE jobs SET status=?,updated_at=?,next_attempt_at=?,completed_at=?,error=? WHERE job_id=?", ("retry" if retry else "failed", time.time(), time.time() + delay, None if retry else time.time(), detail, row["job_id"]))
        return retry

    def counts(self) -> dict[str, int]:
        return {row["status"]: row["count"] for row in self.db.execute("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status")}


def tracked_endpoint(endpoint, store):
    from vastai.serverless.client.endpoint import Endpoint_

    class TrackedEndpoint(Endpoint_):
        async def _route(self, **kwargs):
            route = await super()._route(**kwargs)
            if (job_id := _current_job.get()) is not None:
                store.record_request(job_id, route.request_idx)
            return route

    # SDK 1.9.1 exposes no route callback; override this one seam to persist IDs before GPU work.
    return TrackedEndpoint(endpoint.client, data=endpoint.data)


class Dispatcher:
    def __init__(self, store, endpoint_name="ielts-qwen3-tts", concurrency=16, max_concurrency=128, max_attempts=5, request_timeout=900,
                 jobs_per_worker=16, queue_threshold_seconds=10.0, queue_persistence_samples=2, worker_poll_seconds=5.0,
                 bootstrap_concurrency=1):
        if not 1 <= concurrency <= max_concurrency <= 512 or max_attempts < 1 or request_timeout <= 0:
            raise ValueError("require 1 <= concurrency <= max_concurrency <= 512, positive attempts and timeout")
        if jobs_per_worker < 1 or queue_threshold_seconds <= 0 or queue_persistence_samples < 1 or worker_poll_seconds <= 0:
            raise ValueError("require positive worker capacity, queue threshold, persistence samples and poll interval")
        if not 1 <= bootstrap_concurrency <= concurrency:
            raise ValueError("bootstrap concurrency must be between 1 and concurrency")
        self.store, self.endpoint_name = store, endpoint_name
        self.target, self.max_target = concurrency, max_concurrency
        self.max_attempts, self.request_timeout = max_attempts, request_timeout
        self.jobs_per_worker = jobs_per_worker
        self.queue_threshold_seconds = queue_threshold_seconds
        self.queue_persistence_samples = queue_persistence_samples
        self.worker_poll_seconds = worker_poll_seconds
        self.ready_workers = 0
        self.estimated_queue_seconds = 0.0
        self._queue_high_streak = 0
        self._next_worker_poll = 0.0
        self.success_streak = 0
        self.bootstrap_concurrency = bootstrap_concurrency
        self._waiting_for_ready = False

    async def _refresh_capacity(self, endpoint, force=False) -> None:
        """Follow Vast's ready-worker capacity without creating idle client work."""
        now = time.monotonic()
        if not force and now < self._next_worker_poll:
            return
        self._next_worker_poll = now + self.worker_poll_seconds
        get_workers = getattr(endpoint, "get_workers", None)
        if get_workers is None:
            # Unit-test doubles and older SDKs cannot expose worker state.
            self.ready_workers = 0
            return
        try:
            workers = await get_workers()
        except Exception:
            # Routing remains safe if the observability call is temporarily unavailable.
            self.ready_workers = 0
            self.target = min(self.target, self.bootstrap_concurrency)
            self._waiting_for_ready = True
            self.estimated_queue_seconds = float("inf") if self.target else 0.0
            return
        ready_states = {"idle", "ready", "working", "busy"}
        ready = [worker for worker in workers if str(getattr(worker, "status", "")).lower() in ready_states]
        self.ready_workers = len(ready)
        if not ready:
            # One request wakes Vast's autoscaler; queueing the whole bootstrap
            # target only creates avoidable no-ready-worker retries.
            self.target = min(self.target, self.bootstrap_concurrency)
            self._waiting_for_ready = True
        elif self._waiting_for_ready:
            self.target = min(self.max_target, max(self.jobs_per_worker, self.jobs_per_worker * self.ready_workers))
            self._waiting_for_ready = False
        queue_times = []
        for worker in ready:
            load = max(float(getattr(worker, "cur_load", 0.0)), 0.0)
            throughput = max(
                float(getattr(worker, "cur_perf", 0.0)),
                float(getattr(worker, "measured_perf", 0.0)),
                float(getattr(worker, "perf", 0.0)),
                0.0,
            )
            queue_times.append(load / throughput if throughput else (float("inf") if load else 0.0))
        self.estimated_queue_seconds = max(queue_times, default=(float("inf") if self.target and not ready else 0.0))
        if self.estimated_queue_seconds > self.queue_threshold_seconds:
            self._queue_high_streak += 1
        else:
            self._queue_high_streak = 0
        capacity_target = min(self.max_target, max(self.jobs_per_worker, self.jobs_per_worker * self.ready_workers))
        # Do not fill newly available GPUs until Vast's queue estimate is persistently high.
        if capacity_target > self.target and self._queue_high_streak >= self.queue_persistence_samples:
            self.target = capacity_target
        elif capacity_target < self.target:
            self.target = capacity_target

    async def run(self, once=False, poll_seconds=1.0, endpoint=None) -> None:
        client = None
        active = set()
        with open(self.store.path + ".dispatcher.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store.recover()
            try:
                if endpoint is None:
                    from vastai import CoroutineServerless
                    client = CoroutineServerless(api_key=os.environ["VAST_API_KEY"], default_request_timeout=self.request_timeout)
                    endpoint = tracked_endpoint(await client.get_endpoint(name=self.endpoint_name), self.store)
                while True:
                    await self._refresh_capacity(endpoint)
                    while len(active) < self.target and (row := self.store.claim(self.max_attempts)) is not None:
                        active.add(asyncio.create_task(self._process(endpoint, row)))
                    if active:
                        done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            outcome = await task
                            if outcome == "retry":
                                self.target = max(1, self.target // 2)
                                self.success_streak = 0
                            elif outcome == "success":
                                self.success_streak += 1
                                if self.success_streak >= self.target:
                                    self.target = min(self.max_target, self.target + 1)
                                    self.success_streak = 0
                    else:
                        counts = self.store.counts()
                        if once and not (counts.get("queued", 0) or counts.get("retry", 0)):
                            return
                        await asyncio.sleep(poll_seconds)
            finally:
                for task in active:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                if client is not None:
                    await client.close()

    async def _process(self, endpoint, row) -> str:
        token = _current_job.set(row["job_id"])
        try:
            generation = json.loads(row["generation_json"])
            # Read the pre-normalization queue format without losing existing jobs.
            if "generation" in generation:
                generation = generation["generation"]
            try:
                payload = validate_request({"input": row["text"], "voice": row["voice"], "mode": row["mode"], "request_id": row["job_id"], **generation})
            except (ValueError, TypeError):
                raise WorkerHTTPError(400) from None
            target = Path(row["output_path"])
            if target.exists():
                metadata = validate_wav(target.read_bytes())
                self.store.mark_success(row, {**metadata, "recovered_artifact": True})
                return "success"
            # Outer timeout also bounds SDK rerouting and cold-start polling.
            async with asyncio.timeout(self.request_timeout):
                result = await endpoint.request("/v1/audio/speech", payload, cost=estimate_cost(payload["input"], payload), retry=False, timeout=self.request_timeout)
            self.store.record_request(row["job_id"], result.get("request_idx"))
            if not result.get("ok"):
                raise WorkerHTTPError(result.get("status"))
            audio, metadata = validate_response(result.get("response"), payload)
            durable_write(target, audio)
            self.store.mark_success(row, {**metadata, "voice": row["voice"], "mode": row["mode"], "request_id": row["job_id"], "sha256": hashlib.sha256(audio).hexdigest()})
            return "success"
        except asyncio.CancelledError:
            self.store.mark_error(row, TimeoutError(), self.max_attempts)
            raise
        except Exception as exc:
            return "retry" if self.store.mark_error(row, exc, self.max_attempts) else "failed"
        finally:
            _current_job.reset(token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="tts.sqlite3")
    parser.add_argument("--output-dir", default="tts-output")
    sub = parser.add_subparsers(dest="command", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--text", required=True)
    enqueue.add_argument("--voice", required=True, choices=sorted(VOICES))
    enqueue.add_argument("--mode", choices=["icl", "xvec"])
    enqueue.add_argument("--language", default="Auto")
    enqueue.add_argument("--max-new-tokens", type=int, default=2048)
    enqueue.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run")
    run.add_argument("--endpoint", default="ielts-qwen3-tts")
    run.add_argument("--concurrency", type=int, default=16, help="Bootstrap in-flight jobs when no worker is ready")
    run.add_argument("--max-concurrency", type=int, default=128, help="Hard cap; default is 16 jobs per eight-worker pool")
    run.add_argument("--max-attempts", type=int, default=5)
    run.add_argument("--request-timeout", type=float, default=900)
    run.add_argument("--jobs-per-ready-worker", type=int, default=16)
    run.add_argument("--bootstrap-concurrency", type=int, default=1, help="In-flight jobs used to wake a cold endpoint")
    run.add_argument("--queue-threshold-seconds", type=float, default=10.0)
    run.add_argument("--queue-persistence-samples", type=int, default=2)
    run.add_argument("--worker-poll-seconds", type=float, default=5.0)
    run.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    store = JobStore(args.db, args.output_dir)
    try:
        if args.command == "enqueue":
            job_id, created = store.enqueue(args.text, args.voice, args.mode, {"language": args.language, "max_new_tokens": args.max_new_tokens, "seed": args.seed})
            print(json.dumps({"job_id": job_id, "created": created}))
        else:
            asyncio.run(Dispatcher(
                store, args.endpoint, args.concurrency, args.max_concurrency,
                args.max_attempts, args.request_timeout, args.jobs_per_ready_worker,
                args.queue_threshold_seconds, args.queue_persistence_samples,
                args.worker_poll_seconds, args.bootstrap_concurrency,
            ).run(once=not args.continuous))
            print(json.dumps(store.counts(), sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
