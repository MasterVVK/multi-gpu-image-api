#!/usr/bin/env bash
# Загрузка весов FLUX.3 Image Klein base на узел временным контейнером (на хост ничего не ставится).
# Репозиторий закрыт лицензией (FLUX Kommunity License): аккаунт Hugging Face должен принять её
# на странице модели, а токен этого аккаунта передаётся через окружение и только в контейнер:
#   read -rs HF_TOKEN; export HF_TOKEN; ./download-flux3.sh
# Качаются только трансформер, энкодер и VAE в формате diffusers (~20 ГБ): без однофайлового
# дубля, fp8/nvfp4-вариантов и апсемплера промпта — бэкенд собирает индекс пайплайна сам.
set -euo pipefail
: "${HF_TOKEN:?задайте HF_TOKEN (токен аккаунта, принявшего лицензию модели)}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
docker rm -f flux3-dl >/dev/null 2>&1 || true
# --rm: контейнер вместе с токеном в его окружении исчезает сразу после загрузки.
docker run --rm --name flux3-dl --user "$(id -u):$(id -g)" -e HF_TOKEN \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$ROOT/hf:/hf" python:3.12-slim \
  sh -c 'pip install -q --user huggingface_hub && until /tmp/.local/bin/hf download black-forest-labs/flux-3-image-klein-base --include "transformer/*" "text_encoder/*" "vae/*" LICENSE.md NOTICE.md README.md --max-workers 4; do echo retry; sleep 10; done; echo DOWNLOAD_DONE'
