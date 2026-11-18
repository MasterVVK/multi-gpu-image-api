"""Под лимитом памяти контейнера меньше размера весов: загрузить компонент (mmap) и прочитать
все веса последовательно — как это делает перенос на карту. Если процесс выжил, значит страницы
весов файловые и вытесняются в пределах лимита, а не копятся в анонимной памяти."""
import sys, torch
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
    try: return open(path).read().strip().replace("\n", " ")
    except OSError as e: return str(e)
status = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith(("RssAnon", "RssFile", "VmHWM"))}

raw = open("/sys/fs/cgroup/memory.stat").read().split("\n")
stat = {l.split()[0]: int(l.split()[1]) // 2**20 for l in raw if l.split() and l.split()[0] in ("anon", "file", "file_mapped")}
print(which, "weights_mb", total // 2**20, "rss_mb", status, "cgroup_max", read("/sys/fs/cgroup/memory.max"), "cgroup_stat_mb", stat, "events:", read("/sys/fs/cgroup/memory.events"))
