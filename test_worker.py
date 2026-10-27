import unittest
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
