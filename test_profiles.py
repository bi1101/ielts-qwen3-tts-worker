import hashlib
import json
import struct
import unittest
from pathlib import Path

from speech_contract import VOICES, voice_mode

ROOT = Path(__file__).resolve().parent


def tensors(path):
    raw = path.read_bytes()
    header_size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + header_size])
    data = raw[8 + header_size:]
    return {key: (entry['shape'], data[slice(*entry['data_offsets'])]) for key, entry in header.items() if key != '__metadata__'}


class ProfileTests(unittest.TestCase):
    def test_eight_profiles_and_shared_embeddings(self):
        root = ROOT / 'voice_profiles'
        manifest = json.loads((root / 'custom_voice_manifest.json').read_text())
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(set(manifest['voices']), set(VOICES))
        for name, entry in manifest['voices'].items():
            with self.subTest(voice=name):
                self.assertEqual(entry['mode'], voice_mode(name))
                values = tensors(root / entry['file'])
                self.assertEqual(values['speaker_embedding'][0], [entry['embedding_dim']])
                self.assertEqual(entry['embedding_dim'], 2048)
                if entry['mode'] == 'icl':
                    self.assertEqual(values['ref_code'][0], [entry['ref_code_length'], 16])
                    self.assertEqual(entry['ref_text'], VOICES[name]['ref_text'])
                    xvec = tensors(root / f'{name}_xvec.safetensors')
                    self.assertEqual(values['speaker_embedding'], xvec['speaker_embedding'])
                    digest = hashlib.sha256((ROOT / 'voices' / VOICES[name]['file']).read_bytes()).hexdigest()
                    self.assertEqual(provenance['voices'][name]['audio_sha256'], digest)
                else:
                    self.assertEqual(set(values), {'speaker_embedding'})
