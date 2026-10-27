"""Durable, bounded dispatcher for Vast Serverless Qwen3-TTS jobs.

Usage:
  python tts_controller.py enqueue --text 'Hello' --voice alex
  python tts_controller.py run --db tts.sqlite3 --endpoint ielts-qwen3-tts --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import sqlite3
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from audio_validation import AudioValidationError, validate_wav

VOICES = {
    "alex": "icl", "ben": "icl", "jane": "icl", "june": "icl",
    "alex_xvec": "xvec", "ben_xvec": "xvec", "jane_xvec": "xvec", "june_xvec": "xvec",
}
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def now() -> float:
    return time.time()


def estimate_cost(text: str, generation: dict[str, Any]) -> int:
    # Qwen audio-token cost is not exposed before generation; this keeps autoscaling proportional to text and output budget.
    output_budget = int(generation.get("max_new_tokens", 2048))
    return max(1, math.ceil(len(text) / 4) * 12 + min(output_budget, 4096) // 8)


def canonical_job(text: str, voice: str, mode: str, generation: dict[str, Any]) -> tuple[str, str]:
    payload = json.dumps({"text": text, "voice": voice, "mode": mode, "generation": generation}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), payload


class JobStore:
    def __init__(self, path: str | Path, output_dir: str | Path = "tts-output"):
        self.path = str(path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          dedupe_key TEXT NOT NULL UNIQUE,
          text TEXT NOT NULL,
          voice TEXT NOT NULL,
          mode TEXT NOT NULL,
          generation_json TEXT NOT NULL,
          status TEXT NOT NULL,
          attempt_count INTEGER NOT NULL DEFAULT 0,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          started_at REAL,
          completed_at REAL,
          endpoint_request_id TEXT,
          output_path TEXT,
          error TEXT
        );
        CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, updated_at);
        """)

    def close(self) -> None:
        self.db.close()

    def enqueue(self, text: str, voice: str, mode: str | None, generation: dict[str, Any], output_path: str | None = None) -> tuple[str, bool]:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("text must be 1-4000 characters")
        if voice not in VOICES:
            raise ValueError(f"unsupported voice: {voice}")
        expected_mode = VOICES[voice]
        mode = mode or expected_mode
        if mode != expected_mode:
            raise ValueError(f"voice {voice} requires mode={expected_mode}")
        dedupe_key, generation_json = canonical_job(text, voice, mode, generation)
        existing = self.db.execute("SELECT job_id FROM jobs WHERE dedupe_key=?", (dedupe_key,)).fetchone()
        if existing:
            return existing["job_id"], False
        job_id = hashlib.sha256(f"{dedupe_key}:{uuid.uuid4().hex}".encode()).hexdigest()[:24]
        path = output_path or str(self.output_dir / f"{job_id}.wav")
        timestamp = now()
        self.db.execute(
            "INSERT INTO jobs(job_id,dedupe_key,text,voice,mode,generation_json,status,created_at,updated_at,output_path) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (job_id, dedupe_key, text, voice, mode, generation_json, "queued", timestamp, timestamp, path),
        )
        return job_id, True

    def claim(self) -> sqlite3.Row | None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM jobs WHERE status IN ('queued','retry') ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                self.db.execute("COMMIT")
                return None
            stamp = now()
            self.db.execute("UPDATE jobs SET status='running',attempt_count=attempt_count+1,started_at=?,updated_at=?,error=NULL WHERE job_id=?", (stamp, stamp, row["job_id"]))
            self.db.execute("COMMIT")
            return self.db.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def mark_success(self, job_id: str, endpoint_request_id: str, output_path: str, metadata: dict[str, Any]) -> None:
        self.db.execute("UPDATE jobs SET status='succeeded',completed_at=?,updated_at=?,endpoint_request_id=?,output_path=?,error=? WHERE job_id=?", (now(), now(), endpoint_request_id, output_path, json.dumps(metadata, sort_keys=True), job_id))

    def mark_retry(self, job_id: str, error: str) -> None:
        self.db.execute("UPDATE jobs SET status='retry',updated_at=?,error=? WHERE job_id=?", (now(), error[:4000], job_id))

    def mark_failed(self, job_id: str, error: str) -> None:
        self.db.execute("UPDATE jobs SET status='failed',completed_at=?,updated_at=?,error=? WHERE job_id=?", (now(), now(), error[:4000], job_id))

    def counts(self) -> dict[str, int]:
        return {row["status"]: row["count"] for row in self.db.execute("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status")}


class Dispatcher:
    def __init__(self, store: JobStore, endpoint_name: str, concurrency: int, max_concurrency: int, max_attempts: int, request_timeout: float):
        self.store = store
        self.endpoint_name = endpoint_name
        self.target = max(1, concurrency)
        self.max_target = max(self.target, max_concurrency)
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.success_streak = 0

    async def run(self, once: bool = False, poll_seconds: float = 1.0) -> None:
        from vastai import CoroutineServerless
        client = CoroutineServerless(api_key=os.environ["VAST_API_KEY"], default_request_timeout=self.request_timeout)
        try:
            endpoint = await client.get_endpoint(name=self.endpoint_name)
            active: set[asyncio.Task] = set()
            while True:
                while len(active) < self.target:
                    row = self.store.claim()
                    if row is None:
                        break
                    active.add(asyncio.create_task(self._process(client, endpoint, row)))
                if active:
                    done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        transient = await task
                        if transient:
                            self.target = max(1, self.target - 1)
                            self.success_streak = 0
                        else:
                            self.success_streak += 1
                            if self.success_streak >= 8 and self.target < self.max_target:
                                self.target += 1
                                self.success_streak = 0
                    continue
                if once or self.store.counts().get("queued", 0) + self.store.counts().get("retry", 0) == 0:
                    return
                await asyncio.sleep(poll_seconds)
        finally:
            await client.close()

    async def _process(self, client, endpoint, row: sqlite3.Row) -> bool:
        generation = json.loads(row["generation_json"])
        payload = {"input": row["text"], "voice": row["voice"], "mode": row["mode"], "response_format": "wav", "language": generation.pop("language", "Auto"), "request_id": row["job_id"], **generation}
        try:
            result = await endpoint.request("/v1/audio/speech", payload, cost=estimate_cost(row["text"], generation), timeout=self.request_timeout)
            status = result.get("status")
            if not result.get("ok"):
                raise RuntimeError(f"worker HTTP {status}: {result.get('text', '')[:500]}")
            response = result.get("response") or {}
            if response.get("request_id") != row["job_id"] or response.get("voice") != row["voice"] or response.get("mode") != row["mode"]:
                raise ValueError("worker metadata does not match the job")
            audio = base64.b64decode(response["audio_base64"], validate=True)
            metadata = validate_wav(audio)
            target = Path(row["output_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_bytes(audio)
            temp.replace(target)
            metadata.update({"voice": row["voice"], "mode": row["mode"], "request_id": row["job_id"]})
            self.store.mark_success(row["job_id"], str(result.get("request_idx", "")), str(target), metadata)
            return False
        except Exception as exc:
            message = str(exc)
            status = _status_from_error(message)
            permanent = isinstance(exc, (ValueError, AudioValidationError)) or status == 400
            if permanent or row["attempt_count"] >= self.max_attempts:
                self.store.mark_failed(row["job_id"], message)
                return False
            self.store.mark_retry(row["job_id"], message)
            return True


def _status_from_error(error: str) -> int | None:
    import re
    match = re.search(r"(?:HTTP|status)\s*(\d{3})", error, re.I)
    return int(match.group(1)) if match else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="tts.sqlite3")
    parser.add_argument("--output-dir", default="tts-output")
    sub = parser.add_subparsers(dest="command", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--text", required=True)
    enqueue.add_argument("--voice", required=True, choices=sorted(VOICES))
    enqueue.add_argument("--mode", choices=["icl", "xvec"])
    enqueue.add_argument("--language", default="Auto")
    enqueue.add_argument("--max-new-tokens", type=int, default=2048)
    run = sub.add_parser("run")
    run.add_argument("--endpoint", default="ielts-qwen3-tts")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--max-concurrency", type=int, default=8)
    run.add_argument("--max-attempts", type=int, default=5)
    run.add_argument("--request-timeout", type=float, default=900)
    run.add_argument("--continuous", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = JobStore(args.db, args.output_dir)
    try:
        if args.command == "enqueue":
            job_id, created = store.enqueue(args.text, args.voice, args.mode, {"language": args.language, "max_new_tokens": args.max_new_tokens})
            print(json.dumps({"job_id": job_id, "created": created}))
        else:
            asyncio.run(Dispatcher(store, args.endpoint, args.concurrency, args.max_concurrency, args.max_attempts, args.request_timeout).run(once=not args.continuous))
            print(json.dumps(store.counts(), sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
