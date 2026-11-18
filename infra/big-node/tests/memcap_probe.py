"""Под лимитом памяти контейнера меньше размера весов: загрузить компонент (mmap) и прочитать
все веса последовательно — как это делает перенос на карту. Если процесс выжил, значит страницы
весов файловые и вытесняются в пределах лимита, а не копятся в анонимной памяти."""
import sys

import torch

which = sys.argv[1]
if which == "text_encoder":
    from transformers import Qwen3VLForConditionalGeneration as C
else:
    from diffusers import MiniMaxH3ImageTransformer2DModel as C
m = C.from_pretrained("/repo/" + which, dtype=torch.bfloat16)
total = 0
with torch.no_grad():
    for p in m.parameters():
        total += p.numel() * p.element_size()
        p.to(torch.bfloat16).view(-1)[:: 1024].sum()   # трогаем каждую страницу
        _ = p.clone()                                   # и полная копия тензора, как .to(device)
        del _
def read(path):
    try:
        return open(path).read().strip().replace("\n", " ")
    except OSError as e:
        return str(e)
with open("/proc/self/status") as fh:
    status = {line.split(":")[0]: int(line.split()[1]) // 1024 for line in fh if line.startswith(("RssAnon", "RssFile", "VmHWM"))}

with open("/sys/fs/cgroup/memory.stat") as fh:
    raw = fh.read().split("\n")
stat = {line.split()[0]: int(line.split()[1]) // 2**20 for line in raw if line.split() and line.split()[0] in ("anon", "file", "file_mapped")}
print(which, "weights_mb", total // 2**20, "rss_mb", status, "cgroup_max", read("/sys/fs/cgroup/memory.max"), "cgroup_stat_mb", stat, "events:", read("/sys/fs/cgroup/memory.events"))
