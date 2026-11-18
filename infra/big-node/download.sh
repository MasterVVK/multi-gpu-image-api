#!/usr/bin/env bash
# Загрузка весов MiniMax-H3-Image на узел временным контейнером.
# На хост ничего не ставится; файлы пишутся от пользователя, запустившего скрипт.
# Варианты transformer/ и vae/ (родной формат, 68 ГБ) не нужны: пайплайн diffusers
# читает transformer_diffusers/ и vae_diffusers/.
# --exclude повторяется на каждый шаблон: несколько шаблонов после одного флага
# hf принимает за имена файлов и качает только их.
set -euo pipefail
# Каталог комплекта — тот, где лежит скрипт: веса пишутся в ./hf рядом с compose.yml.
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/hf"
docker rm -f minimax-h3-dl >/dev/null 2>&1 || true
docker run -d --name minimax-h3-dl --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$ROOT/hf:/hf" python:3.12-slim \
  sh -c "pip install -q --user huggingface_hub && until /tmp/.local/bin/hf download MiniMaxAI/MiniMax-H3-Image --exclude 'transformer/*' --exclude 'vae/*' --exclude 'assets/*' --max-workers 4; do echo retry; sleep 10; done; echo DOWNLOAD_DONE"
