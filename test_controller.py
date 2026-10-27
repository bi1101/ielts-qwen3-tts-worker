import asyncio
import base64
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from test_benchmark import Endpoint
from tts_controller import Dispatcher, JobStore, WorkerHTTPError, durable_write, retryable
from speech_contract import validate_request


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.temp.name) / 'jobs.sqlite3', Path(self.temp.name) / 'out')

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_idempotency_and_producer_does_not_recover_running(self):
        job, created = self.store.enqueue('Hello', 'alex', None, {})
        again, created_again = self.store.enqueue('Hello', 'alex', 'icl', {'seed': 0, 'language': 'Auto', 'max_new_tokens': 2048, 'speed': 1.0})
        self.assertEqual(job, again)
        self.assertFalse(created_again)
        row = self.store.claim()
        other = JobStore(self.store.path, self.store.output_dir)
        other.enqueue('new', 'ben_xvec', None, {})
        self.assertEqual(other.counts()['running'], 1)
        other.close()
        self.assertNotIn('generation', json.loads(row['generation_json']))
        with self.assertRaises(ValueError):
            self.store.enqueue('Different', 'ben', None, {}, row['output_path'])

    def test_retry_status_and_backoff(self):
        for status in (408, 429, 500, 501, 502, 503, 504, 599):
            self.assertTrue(retryable(WorkerHTTPError(status)))
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(retryable(WorkerHTTPError(status)))
        self.assertTrue(retryable(TimeoutError()))
        self.store.enqueue('Hello', 'alex', None, {})
        row = self.store.claim()
        self.store.record_request(row['job_id'], 42)
        self.assertTrue(self.store.mark_error(row, WorkerHTTPError(503), 5))
        saved = self.store.db.execute('SELECT * FROM jobs').fetchone()
        self.assertGreater(saved['next_attempt_at'], time.time())
        self.assertEqual(saved['endpoint_request_id'], '42')
        self.assertIsNone(self.store.claim())

    def test_recover_artifact_after_crash_at_final_attempt(self):
        self.store.enqueue('Hello', 'alex', None, {})
        row = self.store.claim()
        durable_write(Path(row['output_path']), base64.b64decode(Endpoint().audio))
        self.store.db.execute('UPDATE jobs SET attempt_count=5')
        self.store.recover()
        self.assertEqual(self.store.counts(), {'succeeded': 1})
        self.assertIsNone(self.store.claim())

    def test_validation(self):
        for data in ({'voice': []}, {'seed': True}, {'max_new_tokens': '4'}, {'speed': float('nan')}, {'extra_params': {'unsafe': 1}}, {'language': 'unknown'}, {'mode': 'xvec'}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_request({'input': 'Hello', **data})


class AsyncQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.temp.name) / 'queue.sqlite3', Path(self.temp.name) / 'out')

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    async def test_dispatch_writes_validated_results_and_is_bounded(self):
        for i in range(9):
            self.store.enqueue(f'Hello {i}', 'alex', None, {})
        endpoint = Endpoint()
        await Dispatcher(self.store, concurrency=2, max_concurrency=3).run(once=True, endpoint=endpoint)
        self.assertLessEqual(endpoint.peak, 3)
        self.assertEqual(self.store.counts(), {'succeeded': 9})
        for row in self.store.db.execute('SELECT * FROM jobs'):
            self.assertIsNone(row['error'])
            self.assertTrue(Path(row['output_path']).exists())
            self.assertIn('sha256', json.loads(row['metadata_json']))

    async def test_delayed_retry_is_not_dropped_when_draining(self):
        self.store.enqueue('Hello', 'alex', None, {})
        base = Endpoint()
        class RetryEndpoint:
            calls = 0
            async def request(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {'ok': False, 'status': 503, 'request_idx': 7}
                return await base.request(*args, **kwargs)
        endpoint = RetryEndpoint()
        await Dispatcher(self.store, concurrency=1, max_concurrency=1).run(once=True, poll_seconds=.05, endpoint=endpoint)
        self.assertEqual(endpoint.calls, 2)
        self.assertEqual(self.store.counts(), {'succeeded': 1})
        self.assertEqual(self.store.db.execute('SELECT attempt_count FROM jobs').fetchone()[0], 2)

    async def test_continuous_accepts_new_jobs_and_prevents_second_dispatcher(self):
        endpoint = Endpoint()
        task = asyncio.create_task(Dispatcher(self.store).run(endpoint=endpoint, poll_seconds=.01))
        await asyncio.sleep(.03)
        self.assertFalse(task.done())
        other = JobStore(self.store.path, self.store.output_dir)
        with self.assertRaises(BlockingIOError):
            await Dispatcher(other).run(endpoint=endpoint)
        other.enqueue('Arriving later', 'jane', None, {})
        other.close()
        for _ in range(100):
            if self.store.counts().get('succeeded'):
                break
            await asyncio.sleep(.01)
        self.assertEqual(self.store.counts(), {'succeeded': 1})
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
