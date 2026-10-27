import asyncio
import base64
import io
import unittest
import wave
from unittest.mock import AsyncMock, patch

from benchmark import monitor_workers, percentile, run_batch


class Endpoint:
    def __init__(self, bad_metadata=False):
        self.active = self.peak = 0
        self.bad_metadata = bad_metadata
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * 240)
        self.audio = base64.b64encode(stream.getvalue()).decode()

    async def request(self, route, payload, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        return {"ok": True, "status": 200, "latency": 0.001, "response": {
            "request_id": "wrong" if self.bad_metadata else payload["request_id"],
            "voice": payload["voice"], "mode": payload["mode"], "format": "wav",
            "sample_rate": 24_000, "duration": 0.01, "audio_base64": self.audio,
        }}


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_requests_and_all_voices(self):
        endpoint = Endpoint()
        result = await run_batch(endpoint, 3, 24)
        self.assertEqual(endpoint.peak, 3)
        self.assertEqual(result["succeeded"], 24)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(len({r["voice"] for r in result["requests"]}), 8)
        self.assertEqual(len({r["request_id"] for r in result["requests"]}), 24)

    async def test_rejected_metadata_counts_failure(self):
        result = await run_batch(Endpoint(bad_metadata=True), 2, 8)
        self.assertEqual(result["failed"], 8)
        self.assertEqual(result["jobs_per_minute"], 0)
        self.assertIsNone(result["latency_p95_seconds"])

    def test_percentile(self):
        self.assertEqual(percentile([8, 2, 1, 4], 0.5), 2)
        self.assertEqual(percentile([8, 2, 1, 4], 0.95), 8)

    async def test_monitor_retains_only_allowlisted_fields(self):
        stopped = asyncio.Event()
        process = AsyncMock(returncode=0)
        async def communicate():
            stopped.set()
            return b'[{"id":1,"status":"ready","api_key":"private"}]', b''
        process.communicate.side_effect = communicate
        samples = []
        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=process)):
            await monitor_workers(42781, samples, stopped)
        self.assertEqual(samples[0]['workers'][0]['id'], 1)
        self.assertNotIn('api_key', samples[0]['workers'][0])
