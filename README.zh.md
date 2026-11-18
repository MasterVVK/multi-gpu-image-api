[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md)

# multi-gpu-image-api

跨多个 GPU 节点的统一文生图 HTTP API，为**并非独占**的显卡提供显存仲裁。

它用来替代那种"每个模型一个 FastAPI 服务"的做法——每个服务都默认自己独占 GPU。
在我们的场景里是四个服务共用一张 RTX 3090，日志里的结果毫不意外：
`CUDA out of memory`，以及 OOM-killer 顺手杀掉了邻近项目。

它带来的能力：

* **显存仲裁**——每张卡同时只跑一个生成任务，按剩余资源原则等待空闲显存，
  宁可干净地拒绝，也不要 OOM；
* **泄漏看门狗**——任务失败、被取消或网关崩溃之后，显存都不会被继续占着；
* 一个统一契约，取代多个各自演化的契约；
* 队列和任务历史可以熬过重启（以前只存在进程内的字典里）；
* API 密钥鉴权，并修补了图片下发中的路径穿越漏洞；
* 增加 GPU 节点无需改动代码；
* 提供 OpenAI 兼容端点，现成的客户端和 UI 可直接使用。

## 为什么是"网关 + 工作进程"，而不是单一进程

我们需要的这些模型依赖栈互不兼容：

| 后端 | torch | transformers | diffusers |
|---|---|---|---|
| Qwen-Image | 2.9.1 | 4.57.3 | 0.36.0 |
| Z-Image | 2.10.0 | **5.1.0** | 0.36.0 |
| ERNIE-Image | 2.5.1+cu121 | **5.5.4** | main @ `c8c8401` |
| HiDream-O1 | 2.5.1+cu121 | **4.57.1**（硬锁定） | 不使用，自带代码 |

它们无法共存于同一个虚拟环境。所以"统一服务"的含义是：一个从不 import torch 的
公开进程，加上若干轻量工作进程，每个都从自己的 venv 启动：

```
客户端 ──8010──▶ 网关  (独立 .venv，不含 torch)
                  FastAPI + SQLite + 显存仲裁器 + Prompt Enhancer
                        │ httpx                    │ httpx 经 ssh -L
              127.0.0.1:8101..8106          127.0.0.1:18101..18106
              ┌──────────────────┐          ┌──────────────────┐
              │ 节点 node-a      │          │ 节点 node-b      │
              │ 8101 qwen        │          │ 18101 qwen       │
              │ 8102 zimage      │          │ 18102 zimage     │
              │ 8103 ernie       │          │ 18104 hidream    │
              │ 8104 hidream     │          │ 18105 qwen21     │
              │ 8105 qwen21      │          │ 18106 moss       │
              │ 8106 moss        │          └──────────────────┘
              └──────────────────┘
```

工作进程包是用 `.pth` 文件装进各个外部 venv 的，而不是用 pip。`.pth` 只是一行
路径文本——它在结构上就不可能改动任何人的版本锁定，并且一条命令即可移除。

工作进程把生成好的 PNG 以字节形式放在 HTTP 响应里返回，网关是唯一写磁盘的一方。
因此远程节点不需要共享文件系统，而 `completed` 状态的含义正是"文件已在网关上"。

## 模型

| id | 步数 | CFG | 预设分辨率 | 任意尺寸 | 显存峰值 |
|---|---|---|---|---|---|
| `z-image-turbo` | 9 | 0.0 | 1024² 及另外 4 种 | ×32 | 13.6 GB（实测） |
| `z-image-base` | 30 | 5.0 | 同上 | ×32 | 14.3 GB（估算） |
| `qwen-image` | 50 | 4.0（`true_cfg_scale`） | 1328² 及另外 4 种 | ×16 | 21.5 GB（实测） |
| `qwen-image-2.1` | 40 | 1.0（默认关闭 CFG） | 2048² 及另外 6 种，最大 2752×1536 | ×32 | 10.7 GB（实测） |
| `qwen-image-2.1-bf16` | 40 | 1.0（默认关闭 CFG） | 同上 | ×32 | 17.7 GB（实测） |
| `qwen-image-2.1-turbo` | 8（固定） | 无 | 同上 | ×32 | 10.7 GB（沿用 base 的实测值） |
| `moss-image` | 50 | 3.0 | 1408² 及另外 6 种，约 2 MP | ×16 | 11.7 GB（按 torch 计数器实测） |
| `ernie-image-turbo` | 8 | 1.0 | 7 种预设 | 否 | 17.2 GB（实测） |
| `ernie-image` | 50 | 4.0 | 同上 | 否 | 17.2 GB（沿用 turbo 实测） |
| `hidream-o1` | 50 | 5.0，`shift` 3.0 | 11 种预设，最高 3104×1312 | ×64 | 19.3 GB（实测） |

Qwen-Image 2.1 运行在独立的 `qwen21` 后端（单独的 venv：需要 diffusers main 分支的
`QwenImage21Pipeline` 和 transformers ≥ 5.17）。为了挤进共享的 24 GB 显卡：文本编码器
（17.5 GB）留在 CPU 上（每个 prompt 约 10–20 秒），transformer 权重以 float8 存储、逐层
转换为 bf16，VAE 以 1024 px 分块解码（整张 2K 图在 24 GB 内根本无法解码）。在 RTX 3090
上实测 2752×1536、40 步：峰值 10.1 GB、277 秒；全部放在显卡上用 bf16 则是 17.2 GB、
约 300 秒。CFG 默认关闭（`guidance_scale=1.0`），`negative_prompt` 仅在
`guidance_scale > 1` 时生效。Turbo 的步数和 guidance 固化在检查点里：其他步数或
`negative_prompt` 会被 422 拒绝。2.1 的权重采用 Qwen Research License（仅限非商业用途）。

`qwen-image-2.1-bf16` 是同一检查点的完整 transformer 权重版本：参考级质量，峰值 16.8 GB。
在共享显卡上只有邻居进程离开后才能启动，因此更看重结果的客户端应优先请求它，遇到
`vram_unavailable` 时再用 `qwen-image-2.1` 重试。

MOSS-Image（OpenMOSS，Apache-2.0）是 `moss` 后端：与 `qwen21` 共用同一个 venv，但进程独立。
该模型自带 pipeline（权重仓库中的 `pipeline_moss_image.py`，需要 `trust_remote_code`），因此
worker 以 `HF_HUB_OFFLINE=1` 运行，代码取自固定的快照。文本编码器同样留在 CPU 上。除 1.0 以外
的任何 `guidance_scale` 都会启用 CFG；为 1.0 时带 `negative_prompt` 的请求会被 422 拒绝。

这两个后端都需要 Python ≥ 3.11（transformers ≥ 5.17）：系统 Python 为 3.10 的节点请为
`infra/remote-node/bootstrap.sh` 设置 `PYTHON=`。

"实测"指用 `scripts/measure_vram.py` 在默认预设下测量，扣除同卡邻居的背景占用，
再加 5% 余量。校准不是走过场：`qwen-image` 从项目文档里抄来的数字（16 GB）比真实
峰值 20.4 GB **低了 4.4 GB**。在一张紧张的卡上，这个差距换来的是 OOM，而不是一次
诚实的拒绝。

ERNIE 后端会在加载时丢弃管线内置的 prompt enhancer（Ministral-3.3B）。
`use_pe=False` 并不能移除它——它是管线的一个*组件*，而不是 `__call__` 里的分支，
因此总会被加载。丢弃之后，实测峰值从超过 23.5 GB（模型在 24 GB 卡上因此 OOM）
降到了 16.5 GB。

实时目录见 `GET /api/v1/models`，定义在 `worker/images_worker/registry.py`。

新增一个模型只需要一个 `ModelSpec` 和一个后端类——网关、调度器和 API 都不用改。

## 显存策略

设计前提是：GPU **与你无法控制的其他工作负载共享**——在我们这里是 ollama、
一套 RAG、TTS 和 whisper。由此得出的规则：

1. **独占。** 每个节点同一时刻只跑一个生成任务，其余排队。
2. **每个节点只常驻一个模型。** 加载另一个之前先卸载前一个。
3. **用 `nvidia-smi` 做预检**，绝不用 `torch.cuda.memory_allocated()`：开启
   `enable_model_cpu_offload()` 后，torch 对已加载的模型报告约 0，因为权重待在
   系统内存里，按模块逐个搬上卡。
4. **最多等 10 分钟。** 显存不够时任务进入 `waiting_vram`，网关每 5 秒轮询所有
   节点。腾出来了就执行；`IMG_VRAM_WAIT_TIMEOUT_S` 到期则任务以
   `error_code=vram_unavailable` 结束，并附上究竟是谁占着卡：

   ```
   等待空闲显存 600 秒未果 —— node-a: 空闲 11781 MB，预留 1024 MB，
   需要 21500 MB，ollama 占用 qwen3-embedding:8b (6.3 GB)、
   NuExtract3-GGUF:Q4_K_M (3.9 GB)、qwen2.5:0.5b (0.7 GB)
   ```

5. **绝不驱逐邻居。** 其他工作负载优先级更高。本服务只会等待和拒绝。

排在*自己*的任务后面并不算显存不足：这种情况下 10 分钟倒计时不会启动，
否则凡是排在第二位的任务都会被误杀。

### 让显存不会在故障后卡住

* 每一条退出路径都在 `finally` 里卸载——成功、报错、超时、取消；
* 看门狗每 30 秒检查没有租约的节点：空闲时仍常驻的模型会被卸载；
  报告"忙碌"却没有我们租约的工作进程（网关崩溃留下的孤儿任务）会被取消；
  占用超过 `IMG_STUCK_VRAM_SECONDS` 则升级为重启该工作进程；
* 工作进程自带 idle-unloader，用来兜住"崩掉的恰恰是网关本身"这种情况；
* `IMG_WORKER_IDLE_EXIT_SECONDS` 在长时间空闲后让进程退出——这是交还
  CUDA 上下文（每个工作进程约 340 MB）的唯一办法；
* 网关启动时，会先卸载所有工作进程上的一切，然后才开始接活；
* 每一次干预都记入 `vram_incidents` 表，可通过
  `GET /api/v1/nodes/{name}/incidents` 查看。我们实际跑出来的记录：
  `unload hidream-o1 — 释放 16802 MB`、`cancel_orphan`。

## API

### 原生接口

```
POST   /api/v1/generate              {model, prompt, aspect_ratio|width+height, ...}
GET    /api/v1/tasks/{id}
GET    /api/v1/tasks/{id}/wait?timeout=300     阻塞等待，免去轮询（最大 1800）
DELETE /api/v1/tasks/{id}            取消
GET    /api/v1/tasks?limit=&status=
GET    /api/v1/images/{path}         ?sig=... — 签名链接，可直接用于 <img>
GET    /api/v1/models
GET    /api/v1/resolutions?model=
GET    /api/v1/queue                 谁在跑、谁在等、还剩多久
GET    /api/v1/nodes                 GPU、常驻模型、租约
GET    /api/v1/pe/status             Prompt Enhancer 链路状态
GET    /healthz  /readyz             免密钥；无可用节点时 /readyz 返回 503
```

```bash
KEY=$(grep '^IMG_API_KEYS=' .env | cut -d= -f2- | cut -d, -f1)

TID=$(curl -sS -X POST localhost:8010/api/v1/generate -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-image-turbo","prompt":"a red cube on a white table","aspect_ratio":"1:1"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["task_id"])')

curl -sS "localhost:8010/api/v1/tasks/$TID/wait?timeout=600" -H "X-API-Key: $KEY"
```

### OpenAI 兼容接口

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8010/v1", api_key=KEY)
r = c.images.generate(model="hidream-o1", prompt="blue whale over a city", size="2048x2048")
print(r.data[0].url)
```

`gpt-image-1` / `dall-e-*` 会映射到 `IMG_OPENAI_DEFAULT_MODEL`；`quality` 是步数
倍率；`n` 受 `IMG_OPENAI_MAX_N` 限制——在独占租约下，四个 50 步任务意味着一个
约 10 分钟的阻塞 HTTP 请求，批量请求请改用原生异步 API。超时时该层返回 504，
并**附带 `task_ids`**，这样已经跑起来的工作不会白费。

错误信封故意做成两套：`/api/v1` 用 `{"error":{code,message,request_id}}`，
`/v1` 用 `{"error":{message,type,param,code}}`——否则 OpenAI SDK 认不出来。

## Prompt Enhancer

它住在网关里，在派发之前只调用一次。对端点只有一条要求：不得占用本服务正要保护
的那些卡的显存——也就是说，用远程 LLM 主机或走云端的模型，而不是本地模型。
`IMG_PE_ENDPOINTS` 里的链路按顺序尝试；全部失败时，任务用原始提示词继续执行，
并标记 `pe_status=failed`。提示词增强是锦上添花，不是依赖项。

ERNIE 管线内置的增强器不只是被关闭，而是在加载时就从管线里移除了——详见「模型」一节的说明。

## 安装

```bash
uv sync                                    # 网关 venv（不含 torch）
cp .env.example .env                       # 设置 IMG_API_KEYS、IMG_WORKER_TOKEN
./scripts/install_worker_into_venvs.sh     # 向每个模型 venv 写入 .pth
./scripts/install_systemd.sh               # 单元文件与各工作进程的 env 文件
sudo systemctl enable --now images-worker@{qwen,zimage,ernie,hidream,qwen21,moss} images-api
```

模型项目里除了每个 venv 一个 `.pth` 文件之外什么都没改，用
`./scripts/install_worker_into_venvs.sh --uninstall` 即可移除。

### 接入第二个节点

```bash
# 在主节点：固化版本，然后通过局域网发送代码和权重
./scripts/capture_venv_locks.sh
REMOTE=user@node-b MODELS="hidream zimage" ./scripts/sync_to_node.sh

# 在第二个节点：按同样的版本重建 venv
cd images-api/infra/remote-node
BACKENDS="hidream zimage" WORKER_TOKEN=<与网关相同> ./bootstrap.sh
sudo systemctl enable --now images-worker@hidream images-worker@zimage

# 回到主节点：建隧道，然后注册节点——无需重启网关
sudo systemctl enable --now images-tunnel@node-b
curl -sS -X POST localhost:8010/api/v1/nodes -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{
    "name":"node-b","kind":"remote",
    "base_urls":{"hidream":"http://127.0.0.1:18104","zimage":"http://127.0.0.1:18102"},
    "vram_reserve_mb":4096,"enabled":false}'
curl -sS -X POST localhost:8010/api/v1/nodes/node-b/refresh -H "X-API-Key: $KEY"
curl -sS -X PATCH localhost:8010/api/v1/nodes/node-b -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"enabled":true}'
```

哪个节点上有哪些模型，网关会从每个工作进程的 `/status` 自行发现，所以给节点新增
一个模型只是一次 rsync 加一次 `bootstrap.sh`——不改代码，不重启。
让节点退出轮转而不丢掉当前任务：`PATCH /api/v1/nodes/<name> {"draining": true}`。

用 `rsync -aH` 而不是 `tar`：HuggingFace 缓存建立在指向 blob 存储的符号链接和
硬链接之上，rsync 既能原样保留，又能在中断后续传——每个模型 30–60 GB 时这一点很关键。

**跨 Python 版本的版本锁定。** 在一台机器上固化的锁定文件，如果目标机 Python 更
旧就无法照搬：我们撞上了 `numpy==2.4.3` 和 `networkx==3.6.1`，两者都要求 3.11+。
因此 `bootstrap.sh` 只硬锁定决定模型行为的那些包——torch、transformers、
diffusers、accelerate、tokenizers、safetensors、huggingface-hub——其余交给 pip
按本地解释器解析。当硬锁定版本不可用时，脚本会在主版本号范围内回退并打印醒目
警告：悄悄替换依赖栈版本是不行的，那会改变出图结果。在我们的第二个节点上，
torch 和 transformers 完全一致，只有 numpy 有出入。

`flash-attn` 需要从源码编译，且构建环境中必须已有 torch，因此单独放在最后安装。
如果编译失败，脚本会在 HiDream 的管线里关闭它——这是上游文档给出的标准绕行方案。

## 验证

```bash
./scripts/smoke.sh                                   # 所有模型端到端
.venv/bin/python scripts/test_exclusive_lock.py      # 压力下的独占性
.venv/bin/python scripts/test_vram_backpressure.py   # 先等待、后拒绝的行为
./scripts/test_failure_vram.sh                       # 崩溃后的显存
.venv/bin/python scripts/measure_vram.py             # 校准峰值
.venv/bin/python -m pytest tests/ -q                 # 单元测试
```

`test_exclusive_lock.py` 断言任何节点都不会同时持有两个任务，并且总耗时不低于
节点数所允许的下限——独占的含义是每张**卡**一个任务，因此两个节点并行是合法的。

`test_failure_vram.sh` 会在生成过程中 SIGKILL 工作进程、在生成过程中取消任务、
在生成过程中重启网关，然后检查显存是否归还、任务是否被悄无声息地弄丢。

## 相对于"每模型一服务"的破坏性变更

* **扁平的模型 id。** `{"model": "turbo"}` 曾在三个服务里都合法且含义各不相同；
  不存在安全的自动映射，因此现在返回 422 并列出候选项。
* **`qwen-image` 的 `cfg_scale` 现在真的生效了。** 旧代码接收了这个参数却从未
  传给管线，管线始终按自己的默认值 `true_cfg_scale=4.0` 运行。同一个 seed 配合
  `cfg_scale != 4.0` 现在会得到不同的图。
* **`qwen` 的 4:3 预设是 1472×1136**，而不是 1472×1140——1140 不能被 8 整除，
  diffusers 会静默地重新取整。
* 当服务绑定到非 loopback 地址时，**API 密钥是强制的**。
* `hidream-o1` 会把任意尺寸吸附到最接近的原生分辨率（请求 1024×1024 会得到
  2048×2048）。响应里给出的始终是实际尺寸。

## 配置

全部在 `.env` 中，`.env.example` 带有注释。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `IMG_API_KEYS` | — | 逗号分隔；第一个是管理员密钥 |
| `IMG_VRAM_WAIT_TIMEOUT_S` | 600 | 拒绝之前等待空闲显存的时长 |
| `IMG_IDLE_UNLOAD_SECONDS` | 180 | 节点空闲多久后卸载常驻模型 |
| `IMG_STUCK_VRAM_SECONDS` | 300 | 空闲却仍被占用超过此值即记为事件 |
| `IMG_ALLOW_WORKER_RESTART` | false | 允许看门狗 `systemctl restart` 工作进程 |
| `IMG_NODES_JSON` | 一个本地节点 | 首次启动时的节点定义 |
| `IMG_PE_ENDPOINTS` | — | Prompt Enhancer 链路，按顺序尝试 |
| `IMG_OPENAI_MAX_N` | 2 | OpenAI 层中 `n` 的上限 |
| `IMG_PROJECTS_ROOT` | `$HOME` | 模型项目目录所在位置 |

## 环境要求

网关需要 Python 3.12，工作进程需要 3.10+。需要带 `nvidia-smi` 的 NVIDIA 驱动。
不需要 Redis、消息代理或反向代理——WAL 模式的 SQLite 加一个 `--workers 1` 的单进程
即可，因为独占保证依赖的是进程内锁，第二个 uvicorn worker 会悄悄破坏它。
当 `WEB_CONCURRENCY != 1` 时应用会拒绝启动。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
