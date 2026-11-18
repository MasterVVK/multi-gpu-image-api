"""Замер: куда ложатся веса при загрузке на CPU — в анонимную память или в файловый mmap.

Собирает компоненты среднего размера (сотни МБ) с весами в bf16, как у настоящего чекпойнта,
и смотрит RssAnon/RssFile процесса после from_pretrained. Каждый вариант — в отдельном процессе.
"""
import os, sys, subprocess, json
import torch

ROOT = sys.argv[1]

def rss():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("RssAnon", "RssFile", "RssShmem")):
            k, v = line.split(":"); out[k] = int(v.split()[0]) // 1024
    return out

def build():
    from diffusers import MiniMaxH3ImageTransformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    t = MiniMaxH3ImageTransformer2DModel(num_attention_heads=8, attention_head_dim=128, hidden_size=1024, num_layers=8,
        num_refiner_layers=1, ffn_dim=4096, in_channels=4, text_dim=512, freq_dim=64, time_embed_hidden_dim=1024, time_embed_dim=512)
    t.to(torch.bfloat16)
    for name in t._keep_in_fp32_modules:
        getattr(t, name).to(torch.float32)
    t.save_pretrained(os.path.join(ROOT, "transformer"))
    cfg = Qwen3VLConfig(text_config={"hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 51, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 151936, "dtype": "bfloat16",
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [16, 8, 8], "rope_type": "default"}},
        vision_config={"depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 2, "out_hidden_size": 512,
        "deepstack_visual_indexes": [0, 1], "num_position_embeddings": 64})
    m = Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16)
    m.save_pretrained(os.path.join(ROOT, "text_encoder"))

def probe(which, mode):
    before = rss()
    kw = {"dtype": torch.bfloat16}
    if mode == "device_map":
        kw["device_map"] = "cpu"
    if which == "transformer":
        from diffusers import MiniMaxH3ImageTransformer2DModel as C
    else:
        from transformers import Qwen3VLForConditionalGeneration as C
    base = rss()
    m = C.from_pretrained(os.path.join(ROOT, which), **kw)
    loaded = rss()
    s = sum(float(p.float().abs().sum()) for p in list(m.parameters())[:2000])  # трогаем все страницы
    touched = rss()
    size = sum(p.numel() * p.element_size() for p in m.parameters()) // 2**20
    hooks = sum(1 for mod in m.modules() if hasattr(mod, "_hf_hook"))
    print(json.dumps({"which": which, "mode": mode, "weights_mb": size, "anon_delta_loaded": loaded["RssAnon"] - base["RssAnon"],
        "file_delta_loaded": loaded["RssFile"] - base["RssFile"], "anon_delta_touched": touched["RssAnon"] - base["RssAnon"],
        "file_delta_touched": touched["RssFile"] - base["RssFile"], "accelerate_hooks": hooks, "dtype": str(m.dtype)}))

if len(sys.argv) == 2:
    if not os.path.isdir(os.path.join(ROOT, "text_encoder")):
        build()
    for which in ("text_encoder", "transformer"):
        for mode in ("plain", "device_map"):
            subprocess.run([sys.executable, __file__, ROOT, which, mode], check=True)
else:
    probe(sys.argv[2], sys.argv[3])
