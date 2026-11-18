#!/usr/bin/env python3
"""Проверка узла перед первой генерацией: веса, кэш, nvidia-smi. Карту не занимает.

    docker compose run --rm worker python /opt/check_node.py

torch.cuda здесь не трогается: CUDA-контекст не создаётся, VRAM не расходуется.
Код выхода 0 — всё на месте, 1 — есть ошибки (они помечены FAIL).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from images_worker.backends.base import _hf_cache_dir, weights_present
from images_worker.registry import get_spec

MODEL = "minimax-h3-image"
SHARDED = {"text_encoder": "model.safetensors.index.json", "transformer_diffusers": "diffusion_pytorch_model.safetensors.index.json"}
SMALL = [
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "processor/preprocessor_config.json",
    "processor/tokenizer.json",
    "text_encoder/config.json",
    "transformer_diffusers/config.json",
    "vae_diffusers/config.json",
]
failed = False


def report(ok: bool, text: str) -> None:
    global failed
    failed = failed or not ok
    print("%s  %s" % ("ok  " if ok else "FAIL", text))


def main() -> int:
    spec = get_spec(MODEL)
    print("cache: %s (HF_HUB_OFFLINE=%s)" % (_hf_cache_dir(), os.environ.get("HF_HUB_OFFLINE", "")))

    try:
        from images_worker.backends.minimax import resolve_snapshot

        root = resolve_snapshot(spec)
        report(True, "snapshot %s" % root)
    except Exception as exc:
        report(False, "snapshot: %s" % exc)
        return 1
    report(weights_present(spec), "weights_present() — модель попадёт в supported_models воркера")

    for rel in SMALL:
        report(os.path.isfile(os.path.join(root, rel)), rel)

    for folder, index_name in SHARDED.items():
        try:
            with open(os.path.join(root, folder, index_name)) as fh:
                index = json.load(fh)
        except (OSError, ValueError) as exc:
            report(False, "%s/%s: %s" % (folder, index_name, exc))
            continue
        shards = sorted(set(index["weight_map"].values()))
        missing = [name for name in shards if not os.path.isfile(os.path.join(root, folder, name))]
        size = sum(os.path.getsize(os.path.join(root, folder, name)) for name in shards if name not in missing)
        expected = int(index.get("metadata", {}).get("total_size", 0))
        # Файл шарда чуть больше суммы тензоров (заголовок safetensors), меньше быть не может.
        ok = not missing and size >= expected
        report(ok, "%s: %d/%d шардов, %.1f ГБ на диске при %.1f ГБ тензоров%s" % (
            folder, len(shards) - len(missing), len(shards), size / 1e9, expected / 1e9,
            "; нет: " + ", ".join(missing) if missing else ""))

    vae_dir = os.path.join(root, "vae_diffusers")
    vae = [n for n in (os.listdir(vae_dir) if os.path.isdir(vae_dir) else []) if n.endswith(".safetensors")]
    report(bool(vae), "vae_diffusers: %s" % (", ".join(vae) or "нет файла весов"))

    probe = os.path.join(_hf_cache_dir(), ".write-probe")
    try:
        open(probe, "w").close()
        os.unlink(probe)
        print("note  каталог с весами доступен на запись — ожидался том только для чтения (:ro)")
    except OSError:
        report(True, "каталог с весами только для чтения")

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()
        report(len(out) == 1, "nvidia-smi видит карт: %d (нужна ровно одна, с индексом 0)" % len(out))
        for line in out:
            print("      " + line)
        if len(out) == 1:
            free = int(out[0].split(",")[-1])
            enough = free >= spec.peak_vram_mb
            print("%s  свободно %d МБ, шлюз потребует %d МБ + резерв узла" % ("ok  " if enough else "note", free, spec.peak_vram_mb))
    except FileNotFoundError:
        report(False, "nvidia-smi не найден: контейнер запущен без GPU или без capability `utility`")
    except (subprocess.SubprocessError, ValueError) as exc:
        report(False, "nvidia-smi: %s" % exc)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
