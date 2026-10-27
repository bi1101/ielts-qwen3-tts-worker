#!/bin/bash
set -euo pipefail
# The base image can leave SSH keys owned by its build user.
chown -R root:root /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
export USE_SYSTEM_PYTHON=true SDK_VERSION=1.9.1
export PYWORKER_REPO=https://github.com/bi1101/ielts-qwen3-tts-worker
mkdir -p /workspace
if [ ! -d /workspace/vast-pyworker/.git ]; then
  git clone "$PYWORKER_REPO" /workspace/vast-pyworker
fi
cd /workspace/vast-pyworker
git fetch origin master
git checkout "${PYWORKER_REF:-origin/master}"
python3 -m pip install --no-cache-dir -r requirements.txt
curl -fsSL https://raw.githubusercontent.com/vast-ai/pyworker/60cfeca889f979fd73cf9e00adcd7b0ebc016fdc/start_server.sh -o /workspace/start_server.sh
exec bash /workspace/start_server.sh
