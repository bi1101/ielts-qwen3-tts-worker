"""Real eight-voice native SDK acceptance plus explicit controller fault injection."""
import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from tts_controller import Dispatcher, JobStore, tracked_endpoint
from speech_contract import VOICES, estimate_cost


async def main():
    import os
    from vastai import CoroutineServerless
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='ielts-qwen3-tts')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/acceptance.json'))
    args = parser.parse_args()
    client = CoroutineServerless(api_key=os.environ['VAST_API_KEY'], default_request_timeout=1200)
    try:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / 'queue.sqlite3', Path(directory) / 'audio')
            original = await client.get_endpoint(name=args.endpoint)
            endpoint = tracked_endpoint(original, store)
            for voice in VOICES:
                store.enqueue('Ready for the science challenge?', voice, None, {'language': 'English', 'max_new_tokens': 128, 'seed': 0})
            started = time.perf_counter()
            await Dispatcher(store, args.endpoint, 8, 8, 5, 1200).run(once=True, endpoint=endpoint)
            result = {'endpoint': args.endpoint, 'eight_voice_elapsed_seconds': round(time.perf_counter() - started, 3), 'counts': store.counts(), 'voices': []}
            for row in store.db.execute('SELECT * FROM jobs ORDER BY voice'):
                metadata = json.loads(row['metadata_json']) if row['metadata_json'] else {}
                result['voices'].append({'voice': row['voice'], 'mode': row['mode'], 'status': row['status'], 'attempts': row['attempt_count'], 'endpoint_request_id': row['endpoint_request_id'], **metadata})
            result['malformed'] = []
            for body in ({'input': 'Hi', 'voice': 'unsupported'}, {'input': 'Hi', 'voice': 'alex', 'mode': 'xvec'}, {'input': 'Hi', 'voice': 'alex', 'seed': True}):
                response = await original.request('/v1/audio/speech', body, cost=100, retry=False, timeout=120)
                result['malformed'].append({'status': response.get('status'), 'passed': response.get('status') == 400})
            class FaultEndpoint:
                def __init__(self):
                    self.attempts = 0
                async def request(self, *a, **kw):
                    self.attempts += 1
                    if self.attempts == 1:
                        return {'ok': False, 'status': 503}
                    if self.attempts == 2:
                        raise TimeoutError('injected cold-start timeout')
                    return await endpoint.request(*a, **kw)
            job_id, _ = store.enqueue('This job retries safely and completes.', 'alex', None, {'language': 'English', 'max_new_tokens': 128})
            fault = FaultEndpoint()
            await Dispatcher(store, args.endpoint, 1, 1, 5, 1200).run(once=True, endpoint=fault, poll_seconds=.1)
            row = store.db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            result['retry_injection'] = {'injected': ['HTTP 503', 'TimeoutError'], 'status': row['status'], 'attempts': row['attempt_count'], 'endpoint_request_id': row['endpoint_request_id'], 'final_response_from_real_endpoint': True}
            result['passed'] = result['counts'] == {'succeeded': 8} and all(x['passed'] for x in result['malformed']) and row['status'] == 'succeeded' and row['attempt_count'] == 3 and all(x['endpoint_request_id'] for x in result['voices'])
            store.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
            if not result['passed']:
                raise SystemExit('acceptance failed; see JSON evidence')
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())
