# Воркер images-api для MiniMax-H3-Image (узел с картой на 96 ГБ).
# В образе только стек и код воркера; веса монтируются снаружи только на чтение.
#
# Сборка за прокси: docker compose build --build-arg https_proxy=http://host:port
# (https_proxy — предопределённый аргумент Docker: объявлять его не нужно, и в
# историю образа он не попадает).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

# torch — с сервера PyTorch: сборка cu128 знает Blackwell (RTX PRO 6000), а CUDA-библиотеки
# приезжают колёсами nvidia-*-cu12, поэтому базовый образ обычный, без CUDA.
# torchvision нужен процессору Qwen3-VL (Qwen2VLImageProcessorFast).
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY worker /opt/images-worker
# Проверка узла без обращения к CUDA: веса, кэш, nvidia-smi.
COPY check_node.py /opt/check_node.py
# Байт-код собираем при сборке: процесс работает от uid 1000 и писать в /opt не может.
RUN python -m compileall -q /opt/images-worker/images_worker

# HF_HUB_CACHE — каталог с весами, смонтированный только на чтение. HF_HOME намеренно
# указывает в другое, записываемое место: это служебный каталог huggingface_hub (токен,
# xet, .agent_harnesses.json). В офлайн-режиме воркер туда не пишет (проверено: генерация
# проходит и с HF_HOME на томе только для чтения), разделение — страховка на случай, когда
# библиотека всё же захочет что-то записать.
# HF_HUB_OFFLINE=1: ни одного обращения в сеть, всё только из кэша.
ENV PYTHONPATH=/opt/images-worker \
    HOME=/tmp \
    HF_HOME=/tmp/hf-home \
    HF_HUB_CACHE=/hf/hub \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    IMG_WORKER_BACKEND=minimax \
    IMG_WORKER_HOST=0.0.0.0 \
    IMG_WORKER_PORT=8107

USER 1000:1000
EXPOSE 8107
CMD ["python", "-m", "images_worker"]
