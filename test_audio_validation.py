import base64
import unittest
from audio_validation import AudioValidationError, validate_wav
from test_benchmark import Endpoint


class AudioTests(unittest.TestCase):
    def test_container_is_complete_and_expected_sample_rate(self):
        data = base64.b64decode(Endpoint().audio)
        self.assertEqual(validate_wav(data)['frames'], 240)
        for bad in (b'', b'not a WAV' * 20, data[:-2]):
            with self.assertRaises(AudioValidationError):
                validate_wav(bad)
        with self.assertRaises(AudioValidationError):
            validate_wav(data, expected_sample_rate=16000)
