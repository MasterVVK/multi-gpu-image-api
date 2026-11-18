[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md)

# multi-gpu-image-api

One HTTP API for text-to-image generation across several GPU nodes, with VRAM
arbitration for cards you do **not** own exclusively.

It replaces a set of per-model FastAPI services that each assumed they owned the
GPU. In our case that was four services on one RTX 3090 — with predictable
results in the logs: `CUDA out of memory`, and an OOM-killer taking down
neighbouring projects.

What it adds:

* **VRAM arbitration** — one generation per card, wait for free memory on a
  best-effort basis, refuse cleanly instead of OOM-ing;
* **leak watchdog** — memory is never left occupied after a failure, a
  cancellation, or a gateway crash;
* a single contract instead of several diverging ones;
* queue and task history survive a restart (they used to live in a process dict);
* API-key auth, and a closed path-traversal hole in image serving;
* several GPU nodes with no code changes;
* an OpenAI-compatible endpoint, so existing clients and UIs work as-is.

## Why a gateway plus workers, not one process

The models we needed sit on mutually incompatible stacks:

| backend | torch | transformers | diffusers |
|---|---|---|---|
| Qwen-Image | 2.9.1 | 4.57.3 | 0.36.0 |
| Z-Image | 2.10.0 | **5.1.0** | 0.36.0 |
| ERNIE-Image | 2.5.1+cu121 | **5.5.4** | main @ `c8c8401` |

They cannot share a virtualenv. So "one service" means one public process that
never imports torch, plus thin workers, each launched from its own venv:

```
clients ──8010──▶ gateway  (own .venv, NO torch)
                  FastAPI + SQLite + VRAM arbiter + Prompt Enhancer
                        │ httpx                    │ httpx over ssh -L
              127.0.0.1:8101..8106          127.0.0.1:18101..18106
              ┌──────────────────┐          ┌──────────────────┐
              │ node-a           │          │ node-b           │
              │ 8101 qwen        │          │ 18101 qwen       │
              │ 8102 zimage      │          │ 18102 zimage     │
              │ 8103 ernie       │          │ 18105 qwen21     │
              │ 8105 qwen21      │          │ 18106 moss       │
              │ 8106 moss        │          └──────────────────┘
              └──────────────────┘
```

The worker package is installed into each foreign venv with a `.pth` file rather
than pip. A `.pth` is a single line containing a path — it is structurally
incapable of moving anyone's pins, and it is removed with one command.

Workers return the finished PNG as bytes in the HTTP response; the gateway is the
only writer to disk. A remote node therefore needs no shared filesystem, and
`completed` means exactly "the file is on the gateway".

## Models

| id | steps | CFG | presets | arbitrary size | peak VRAM |
|---|---|---|---|---|---|
| `z-image-turbo` | 9 | 0.0 | 1024² + 4 more | ×32 | 13.6 GB (measured) |
| `z-image-base` | 30 | 5.0 | same | ×32 | 14.3 GB (estimated) |
| `qwen-image` | 50 | 4.0 (`true_cfg_scale`) | 1328² + 4 more | ×16 | 21.5 GB (measured) |
| `qwen-image-2.1` | 40 | 1.0 — CFG off | 2048² + 6 more, up to 2752×1536 | ×32 | 10.7 GB (measured) |
| `qwen-image-2.1-bf16` | 40 | 1.0 — CFG off | same | ×32 | 17.7 GB (measured) |
| `qwen-image-2.1-turbo` | 8 (fixed) | none | same | ×32 | 10.7 GB (carried over from base) |
| `moss-image` | 50 | 3.0 | 1408² + 6 more, ~2 MP | ×16 | 11.7 GB (measured via torch counters) |
| `ernie-image-turbo` | 8 | 1.0 | 7 presets | no | 17.2 GB (measured) |
| `ernie-image` | 50 | 4.0 | same | no | 17.2 GB (inferred) |
| `flux3-klein-full` | 50 | 3.5 | 2048² + 8 more, up to 3072×1312 | ×16 | 36.5 GB (measured; 96 GB card only) |

Qwen-Image 2.1 runs in its own `qwen21` backend (separate venv: it needs
`QwenImage21Pipeline` from diffusers main and transformers ≥ 5.17). To fit a shared
24 GB card the text encoder (17.5 GB) stays on the CPU (10–20 s per prompt), the
transformer weights are stored as float8 and cast to bf16 layer by layer, and the
VAE decodes in 1024 px tiles (a whole 2K image does not decode in 24 GB at all).
Measured on an RTX 3090 at 2752×1536, 40 steps: 10.1 GB peak and 277 s, against
17.2 GB and ~300 s with everything on the card in bf16. CFG is off by default
(`guidance_scale=1.0`); `negative_prompt` only applies when `guidance_scale > 1`.
Turbo has its step count and guidance baked into the checkpoint: a different step
count or a `negative_prompt` is rejected with 422. The 2.1 weights are under the
Qwen Research License (non-commercial use).

`qwen-image-2.1-bf16` is the same checkpoint with full transformer weights: reference quality
at 16.8 GB. On a shared card it only starts once the neighbours are gone, so a client that
values the result over time should ask for it first and retry with `qwen-image-2.1` on
`vram_unavailable`.

MOSS-Image (OpenMOSS, Apache-2.0) is the `moss` backend: same venv as `qwen21`, its own
process. The model ships its own pipeline (`pipeline_moss_image.py` in the weights repo,
`trust_remote_code`), so the worker runs with `HF_HUB_OFFLINE=1` and reads the code from the
pinned snapshot. The text encoder stays on the CPU here too. CFG is on for any
`guidance_scale` except 1.0; at 1.0 a `negative_prompt` is rejected with 422.

Both backends need Python ≥ 3.11 (transformers ≥ 5.17): on a node whose system Python is
3.10, set `PYTHON=` for `infra/remote-node/bootstrap.sh`.

FLUX.3 Image Klein (`black-forest-labs/flux-3-image-klein-base`, gated) is the `flux3`
backend: the base, non-distilled checkpoint with every component resident on the card —
20.5 GB of weights and a 36.5 GB peak (measured at 2560×1440: ~32 GiB allocated by torch,
plus the CUDA context and a margin). That does not fit a 24 GB node, so `flux3-klein-full`
is served only by the big-card node (`node-c` in the examples): a docker container from
`infra/big-node`, worker port 8104, tunnelled to 18304 on the gateway side. About 113 s per
2560×1440 image at 50 steps. `negative_prompt` only applies when `guidance_scale > 1`;
arbitrary sizes must be multiples of 16, up to 3072 per side. FLUX.3's own 9B prompt
upsampler is not used: clients are expected to send full scene descriptions. The weights are
under the FLUX Kommunity License v1.0 (non-commercial use, plus commercial use of outputs for
users under US$5M in revenue).

"Measured" means `scripts/measure_vram.py`, at the default preset, minus the
background from co-tenants, plus 5%. Calibration is not a formality: for
`qwen-image` the figure taken from project docs (16 GB) turned out to be
**4.4 GB below** the real 20.4 GB peak. On a tight card that gap buys you an OOM
instead of an honest refusal.

The ERNIE backend drops the pipeline's built-in prompt enhancer (Ministral-3.3B)
at load time. `use_pe=False` does not remove it — it is a pipeline *component*,
not a branch in `__call__`, so it loads regardless. Dropping it took the measured
peak from over 23.5 GB, where the model OOM-ed on a 24 GB card, down to 16.5 GB.

Live catalogue: `GET /api/v1/models`. Definitions: `worker/images_worker/registry.py`.

Adding a model means adding one `ModelSpec` and one backend class — the gateway,
the scheduler and the API need no changes.

## VRAM policy

The design assumption is that the GPU is **shared with workloads you don't
control** — in our case ollama, a RAG stack, TTS and whisper. Hence:

1. **Exclusive.** One generation per node at a time. The rest queue.
2. **One resident model per node.** Loading another evicts the previous one first.
3. **Preflight via `nvidia-smi`**, never `torch.cuda.memory_allocated()`: with
   `enable_model_cpu_offload()` torch reports ~0 for a loaded model, because the
   weights live in system RAM and stream in per module.
4. **Wait up to 10 minutes.** Short of memory, the task moves to `waiting_vram`
   and the gateway polls every node every 5 seconds. If memory frees up, it runs.
   If `IMG_VRAM_WAIT_TIMEOUT_S` expires, the task ends with
   `error_code=vram_unavailable` and a message naming who actually holds the card:

   ```
   waited 600s for free VRAM — node-a: free 11781 MB, reserve 1024 MB,
   need 21500 MB, ollama holds qwen3-embedding:8b (6.3 GB),
   NuExtract3-GGUF:Q4_K_M (3.9 GB), qwen2.5:0.5b (0.7 GB)
   ```

5. **Never evict the neighbours.** Other workloads have priority. This service
   can only wait and decline.

Queuing behind *our own* task is not a VRAM shortage: the 10-minute countdown does
not start in that case, otherwise everything that happened to be second in line
would be killed.

### Not leaking VRAM on failures

* unload in `finally` on every exit path — success, error, timeout, cancellation;
* a watchdog checks unleased nodes every 30 s: a resident model while idle gets
  unloaded; a worker reporting "busy" without our lease (an orphan left by a
  gateway crash) gets cancelled; memory held longer than
  `IMG_STUCK_VRAM_SECONDS` escalates to a worker restart;
* the worker runs its own idle-unloader, as insurance for the case where the
  gateway itself is the thing that died;
* `IMG_WORKER_IDLE_EXIT_SECONDS` exits the process after a long idle period —
  the only way to give back the CUDA context (~340 MB per worker);
* on startup the gateway unloads everything on every worker before accepting work;
* every intervention is recorded in a `vram_incidents` table, readable at
  `GET /api/v1/nodes/{name}/incidents`. Entries look like
  `unload <model> — freed <N> MB` or `cancel_orphan`.

## API

### Native

```
POST   /api/v1/generate              {model, prompt, aspect_ratio|width+height, ...}
GET    /api/v1/tasks/{id}
GET    /api/v1/tasks/{id}/wait?timeout=300     block instead of polling (max 1800)
DELETE /api/v1/tasks/{id}            cancel
GET    /api/v1/tasks?limit=&status=
GET    /api/v1/images/{path}         ?sig=... — signed link, works in <img>
GET    /api/v1/models
GET    /api/v1/resolutions?model=
GET    /api/v1/queue                 who runs, who waits, how long is left
GET    /api/v1/nodes                 GPU, resident model, lease
GET    /api/v1/pe/status             Prompt Enhancer chain health
GET    /healthz  /readyz             no key; /readyz returns 503 if no node is ready
```

```bash
KEY=$(grep '^IMG_API_KEYS=' .env | cut -d= -f2- | cut -d, -f1)

TID=$(curl -sS -X POST localhost:8010/api/v1/generate -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-image-turbo","prompt":"a red cube on a white table","aspect_ratio":"1:1"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["task_id"])')

curl -sS "localhost:8010/api/v1/tasks/$TID/wait?timeout=600" -H "X-API-Key: $KEY"
```

### OpenAI-compatible

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8010/v1", api_key=KEY)
r = c.images.generate(model="qwen-image-2.1", prompt="blue whale over a city", size="2048x2048")
print(r.data[0].url)
```

`gpt-image-1` / `dall-e-*` alias to `IMG_OPENAI_DEFAULT_MODEL`; `quality` is a
step multiplier; `n` is capped by `IMG_OPENAI_MAX_N`, because under an exclusive
lease four 50-step jobs is a ~10-minute blocking HTTP request — use the native
async API for batches. On timeout the layer returns 504 **together with the
`task_ids`**, so work already in flight is not lost.

Errors use two envelopes on purpose: `{"error":{code,message,request_id}}` for
`/api/v1`, and `{"error":{message,type,param,code}}` for `/v1`, because the
OpenAI SDK will not recognise anything else.

## Prompt Enhancer

Lives in the gateway and runs once, before dispatch. One rule for endpoints:
they must not consume VRAM on the cards this service is trying to protect —
so a remote LLM host or a cloud-routed model, never a local one. The
`IMG_PE_ENDPOINTS` chain is tried in order; if all fail, the task runs with the
original prompt and `pe_status=failed`. Enhancement is an improvement, not a
dependency.

ERNIE's in-pipeline enhancer is not merely disabled but removed from the
pipeline at load time — see the note under Models.

## Install

```bash
uv sync                                    # gateway venv (no torch)
cp .env.example .env                       # set IMG_API_KEYS, IMG_WORKER_TOKEN
./scripts/install_worker_into_venvs.sh     # .pth into each model venv
./scripts/install_systemd.sh               # units and per-worker env files
sudo systemctl enable --now images-worker@{qwen,zimage,ernie,qwen21,moss} images-api
```

Nothing in the model projects is modified except one `.pth` file per venv,
removable with `./scripts/install_worker_into_venvs.sh --uninstall`.

### Adding a second node

```bash
# on the primary: capture versions, then ship code and weights over the LAN
./scripts/capture_venv_locks.sh
REMOTE=user@node-b MODELS="zimage moss" ./scripts/sync_to_node.sh

# on the second node: rebuild the venvs from those versions
cd images-api/infra/remote-node
BACKENDS="zimage moss" WORKER_TOKEN=<same as gateway> ./bootstrap.sh
sudo systemctl enable --now images-worker@zimage images-worker@moss

# back on the primary: tunnel, then register the node — no gateway restart
sudo systemctl enable --now images-tunnel@node-b
curl -sS -X POST localhost:8010/api/v1/nodes -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{
    "name":"node-b","kind":"remote",
    "base_urls":{"zimage":"http://127.0.0.1:18102","moss":"http://127.0.0.1:18106"},
    "vram_reserve_mb":4096,"enabled":false}'
curl -sS -X POST localhost:8010/api/v1/nodes/node-b/refresh -H "X-API-Key: $KEY"
curl -sS -X PATCH localhost:8010/api/v1/nodes/node-b -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"enabled":true}'
```

Which models live where is discovered from each worker's `/status`, so adding a
model to a node is an rsync plus `bootstrap.sh` — no code change, no restart.
Take a node out of rotation without losing its current task with
`PATCH /api/v1/nodes/<name> {"draining": true}`.

`rsync -aH`, not `tar`: the HuggingFace cache is built on symlinks and hardlinks
into a blob store, and rsync both preserves that and resumes after a break — which
matters at 30–60 GB per model.

**Pinning across Python versions.** Locks captured on one machine do not
transplant if the target runs an older Python: we hit `numpy==2.4.3` and
`networkx==3.6.1`, both requiring 3.11+. `bootstrap.sh` therefore hard-pins only
what determines model behaviour — torch, transformers, diffusers, accelerate,
tokenizers, safetensors, huggingface-hub — and lets pip resolve the rest for the
local interpreter. When a hard pin is unavailable it falls back within the major
version and prints a loud warning: silently substituting a stack version changes
the images. On our second node torch and transformers matched exactly; only numpy
differed.

## Verification

```bash
./scripts/smoke.sh                                   # every model end to end
.venv/bin/python scripts/test_exclusive_lock.py      # exclusivity under load
.venv/bin/python scripts/test_vram_backpressure.py   # wait-then-refuse behaviour
./scripts/test_failure_vram.sh                       # VRAM after crashes
.venv/bin/python scripts/measure_vram.py             # calibrate peaks
.venv/bin/python -m pytest tests/ -q                 # unit tests
```

`test_exclusive_lock.py` asserts that no node ever holds two tasks and that total
wall time is not lower than the node count allows — exclusivity means one task per
**card**, so two nodes legitimately run in parallel.

`test_failure_vram.sh` SIGKILLs a worker mid-generation, cancels mid-generation,
and restarts the gateway mid-generation, then checks that memory comes back and
that the task is not silently lost.

## Breaking changes vs. per-model services

* **Flat model ids.** `{"model": "turbo"}` was valid in three services with three
  different meanings; there is no safe automatic mapping, so it returns 422
  listing the candidates.
* **`cfg_scale` on `qwen-image` now actually works.** The old code accepted the
  parameter and never passed it to the pipeline, which always ran on its own
  `true_cfg_scale=4.0` default. The same seed with `cfg_scale != 4.0` now produces
  a different image.
* **`qwen` 4:3 preset is 1472×1136**, not 1472×1140 — 1140 is not divisible by 8
  and diffusers silently re-rounded it.
* **An API key is mandatory** when the service binds to anything but loopback.

## Configuration

Everything is in `.env`; `.env.example` is annotated.

| variable | default | meaning |
|---|---|---|
| `IMG_API_KEYS` | — | comma-separated; the first one is the admin key |
| `IMG_VRAM_WAIT_TIMEOUT_S` | 600 | how long to wait for a free card before refusing |
| `IMG_IDLE_UNLOAD_SECONDS` | 180 | unload the resident model after node idle |
| `IMG_STUCK_VRAM_SECONDS` | 300 | held-but-idle memory past this is an incident |
| `IMG_ALLOW_WORKER_RESTART` | false | let the watchdog `systemctl restart` a worker |
| `IMG_NODES_JSON` | one local node | node definitions for the first boot |
| `IMG_PE_ENDPOINTS` | — | Prompt Enhancer chain, tried in order |
| `IMG_OPENAI_MAX_N` | 2 | cap on `n` in the OpenAI layer |
| `IMG_PROJECTS_ROOT` | `$HOME` | where the model project directories live |

## Requirements

Python 3.12 for the gateway, 3.10+ for workers. NVIDIA driver with `nvidia-smi`.
No Redis, no message broker, no reverse proxy — SQLite in WAL mode and a single
`--workers 1` process, because the exclusivity guarantee is an in-process lock and
a second uvicorn worker would silently break it. The app refuses to start if
`WEB_CONCURRENCY != 1`.

## Licence

MIT — see [LICENSE](LICENSE).
