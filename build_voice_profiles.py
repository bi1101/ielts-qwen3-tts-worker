"""Build eight reusable profiles using the precompute tool shipped with vLLM-Omni 0.30.0."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--characters', type=Path, help='Optional source character directory to verify against bundled WAV/text')
    parser.add_argument('--output', type=Path, default=ROOT / 'voice_profiles')
    args = parser.parse_args()
    from safetensors.torch import load_file, save_file
    import torch
    tool = Path('/app/vllm-omni/examples/online_serving/text_to_speech/qwen3_tts/precompute_custom_voice.py')
    spec = importlib.util.spec_from_file_location('omni_precompute', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    voices = json.loads((ROOT / 'voices.json').read_text())
    provenance = {}
    for name in ('alex', 'ben', 'jane', 'june'):
        config = voices[name]
        audio = ROOT / 'voices' / config['file']
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        if args.characters:
            source = args.characters / Path(config['source']).relative_to('characters')
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f'{name}: reference WAV differs from character source')
            source_text = (args.characters / Path(config['source_text']).relative_to('characters')).read_text()
            if ' '.join(source_text.replace('"', '').split()) != ' '.join(config['ref_text'].split()):
                raise ValueError(f'{name}: reference transcript differs from character source')
        module._write_voice(model='Qwen/Qwen3-TTS-12Hz-1.7B-Base', output_dir=args.output, voice_name=name, ref_audio=str(audio), ref_text=config['ref_text'], mode='icl', speaker_description=None, device=torch.device('cuda'))
        tensors = load_file(str(args.output / f'{name}.safetensors'))
        save_file({'speaker_embedding': tensors['speaker_embedding']}, str(args.output / f'{name}_xvec.safetensors'))
        manifest_path = args.output / module.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest['voices'][f'{name}_xvec'] = {'name': f'{name}_xvec', 'file': f'{name}_xvec.safetensors', 'mode': 'xvec', 'embedding_dim': int(tensors['speaker_embedding'].numel())}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        provenance[name] = {'audio_sha256': digest, 'text_sha256': hashlib.sha256(config['ref_text'].encode()).hexdigest(), 'source': config['source'], 'source_text': config.get('source_text')}
    (args.output / 'provenance.json').write_text(json.dumps({'vllm_omni': '0.30.0', 'voices': provenance}, indent=2) + '\n')


if __name__ == '__main__':
    main()
