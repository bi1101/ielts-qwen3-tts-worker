FROM vllm/vllm-omni@sha256:b09c9eab7d2e30817208de92ccb482756d13d4339377d7cc4c4ce9f47bf04174
# The engine/CUDA layer is immutable. Bake the tested PyWorker SDK, rather than downloading it on every cold start.
RUN python3 -m pip install --no-cache-dir vastai==1.9.1 aiohttp==3.14.3 soundfile==0.14.0 \
 && python3 -c "import vllm, vllm_omni; from vastai import Worker, WorkerConfig; assert vllm.__version__ == '0.30.0'"
