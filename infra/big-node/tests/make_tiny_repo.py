"""Собрать поддельный кэш HuggingFace с крошечной MiniMax-H3-Image.

Раскладка — ровно как у настоящего кэша: <HF_HOME>/hub/models--MiniMaxAI--MiniMax-H3-Image/
{refs/main, snapshots/<rev>/...}. Мелкие конфиги (modular_model_index.json, scheduler, tokenizer,
processor) копируются из настоящего локального кэша как есть, а три компонента с весами
(text_encoder, transformer_diffusers, vae_diffusers) создаются настоящими классами с крошечными
конфигами и случайными весами. Так бэкенд проходит весь путь — блоки, имена состояния, загрузку
с диска, порядок стадий — на CPU за секунды.

Запуск: python make_tiny_repo.py <каталог HF_HOME>
"""

from __future__ import annotations

import os
import shutil
import sys

import torch

REPO = "MiniMaxAI/MiniMax-H3-Image"
REV = "8ced60068623def5b74b4d10558117cb61d6482d"
LATENT_CHANNELS = 4
TEXT_DIM = 32


def real_snapshot() -> str:
    root = os.path.expanduser("~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots")
    return os.path.join(root, sorted(os.listdir(root))[0])


def build(hf_home: str) -> str:
    from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    repo_dir = os.path.join(hf_home, "hub", "models--" + REPO.replace("/", "--"))
    snap = os.path.join(repo_dir, "snapshots", REV)
    if os.path.isdir(repo_dir):
        os.system("chmod -R u+w %s" % repo_dir)
        shutil.rmtree(repo_dir)
    os.makedirs(snap)
    os.makedirs(os.path.join(repo_dir, "refs"))
    with open(os.path.join(repo_dir, "refs", "main"), "w") as fh:
        fh.write(REV)

    src = real_snapshot()
    for name in ("model_index.json", "modular_model_index.json"):
        shutil.copy(os.path.join(src, name), os.path.join(snap, name))
    for name in ("scheduler", "tokenizer", "processor"):
        shutil.copytree(os.path.join(src, name), os.path.join(snap, name))  # symlinks -> файлы

    torch.manual_seed(0)

    transformer = MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=LATENT_CHANNELS,
        patch_size=(1, 2, 2),
        text_dim=TEXT_DIM,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
        num_pdd_heads=6,
    )
    transformer.save_pretrained(os.path.join(snap, "transformer_diffusers"))

    vae = AutoencoderKLMiniMaxH3Image(
        latent_channels=LATENT_CHANNELS,
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        latents_mean=(0.0,) * LATENT_CHANNELS,
        latents_std=(1.0,) * LATENT_CHANNELS,
    )
    vae.save_pretrained(os.path.join(snap, "vae_diffusers"))

    # Блок кодирования читает hidden_states[50], поэтому слоёв должно быть больше 50.
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": TEXT_DIM,
            "intermediate_size": 64,
            "num_hidden_layers": 51,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 151936,
            "max_position_embeddings": 8192,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": TEXT_DIM,
            "deepstack_visual_indexes": [0, 1],
            "num_position_embeddings": 64,
        },
    )
    text_encoder = Qwen3VLForConditionalGeneration(config)
    text_encoder.save_pretrained(os.path.join(snap, "text_encoder"))

    return snap


if __name__ == "__main__":
    print(build(sys.argv[1]))
