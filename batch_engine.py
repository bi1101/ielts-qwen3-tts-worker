"""One-generation-loop microbatch accumulator for the internal worker HTTP server."""
from __future__ import annotations

import asyncio
import collections
import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class BatchItem:
    payload: dict[str, Any]
    future: asyncio.Future

    @property
    def group(self) -> tuple[str, str, int]:
        text_len = len(self.payload.get("input", ""))
        bucket = min(8, max(0, int(math.log2(max(1, text_len // 24 + 1)))))
        return (
            self.payload["voice"].removesuffix("_xvec"),
            "xvec" if self.payload["voice"].endswith("_xvec") else "icl",
            bucket,
        )


class BatchEngine:
    def __init__(self, backend, max_batch_size: int = 4, max_wait_ms: int = 35):
        self.backend = backend
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_s = max(0.001, max_wait_ms / 1000)
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self.pending: collections.deque[BatchItem] = collections.deque()
        self.task: asyncio.Task | None = None
        self.closed = False

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="tts-batch-loop")

    async def close(self) -> None:
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(BatchItem(payload, future))
        return await future

    async def _next(self) -> BatchItem:
        if self.pending:
            return self.pending.popleft()
        return await self.queue.get()

    async def _run(self) -> None:
        while not self.closed:
            first = await self._next()
            batch = [first]
            group = first.group
            deadline = asyncio.get_running_loop().time() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if candidate.group == group:
                    batch.append(candidate)
                else:
                    self.pending.append(candidate)
            await self._generate(batch)

    async def _generate(self, batch: list[BatchItem]) -> None:
        payloads = [item.payload for item in batch]
        try:
            results = await asyncio.to_thread(self.backend.generate_batch, payloads)
            by_id = {request_id: (audio, rate) for request_id, audio, rate in results}
            for item in batch:
                result = by_id.get(item.payload["request_id"])
                if result is None:
                    item.future.set_exception(RuntimeError("backend omitted a batch result"))
                else:
                    item.future.set_result({"audio": result[0], "sample_rate": result[1]})
        except Exception as exc:
            log.exception("batch generation failed; retrying items individually")
            for item in batch:
                try:
                    result = await asyncio.to_thread(self.backend.generate_batch, [item.payload])
                    item.future.set_result({"audio": result[0][1], "sample_rate": result[0][2]})
                except Exception as item_exc:
                    item.future.set_exception(item_exc if item_exc else exc)
