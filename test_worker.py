import unittest
import io
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import worker


class VoiceReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_alias_is_allowed_but_missing_clone_is_rejected(self):
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={"voices": [*worker.VOICES, "default"]})
        session = MagicMock()
        session.get.return_value.__aenter__ = AsyncMock(return_value=response)
        session.get.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch.object(worker, "_http_session", session), patch.object(worker, "_warm_voice", new_callable=AsyncMock) as warm:
            await worker._prepare_voices()
            self.assertEqual(warm.await_count, 8)
            response.json.return_value = {"voices": ["default", "alex"]}
            with self.assertRaisesRegex(RuntimeError, "preload"):
                await worker._prepare_voices()
            self.assertEqual(warm.await_count, 8)


class LoggingTests(unittest.TestCase):
    def test_model_output_is_saved_and_forwarded(self):
        saved = io.StringIO()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            worker._forward_model_output(io.StringIO("ready\nfailed\n"), saved)
        self.assertEqual(saved.getvalue(), "ready\nfailed\n")
        self.assertIn("[vllm] ready", stdout.getvalue())
        self.assertIn("[vllm] failed", stdout.getvalue())
