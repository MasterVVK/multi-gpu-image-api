# MiniMax-H3-Image: воркер images-api в контейнере

Узел шлюза images-api для модели `MiniMaxAI/MiniMax-H3-Image` на чужом боевом хосте:
карта RTX PRO 6000 (96 ГБ), 125 ГБ RAM. Бэкенд `minimax`, модель `minimax-h3-image`,
порт воркера 8107.

Состояние: бэкенд и контейнер проверены на CPU с крошечными весами (см. «Что проверено»).
**На настоящей карте и с настоящими весами не запускалось ни разу** — первый запуск
делается руками по разделу 4, с `nvidia-smi` и `free` перед глазами.

## 1. Как это устроено

Три требования владельца хоста и чем они закрыты:

| Требование | Чем закрыто |
|---|---|
| Не стартовать без свободной VRAM | Шлюз сверяет `free_mb` из `nvidia-smi` с `peak_vram_mb=71800` (замер 68.4 ГБ на 16:9 + 5 %) + резерв узла. Прямой вызов воркера (раздел 4) эту проверку обходит — там она делается руками. |
| После картинки VRAM свободна полностью | Веса живут на карте только внутри задачи. Через 120 с после задачи процесс завершается (`os._exit(0)`) — так возвращается и CUDA-контекст; контейнер поднимается обратно и до следующей задачи карту не трогает. |
| Не забивать RAM хоста | Энкодер и трансформер (по ~62 ГБ) на карте по очереди, с диска на каждую задачу. Веса читаются через mmap (страницы файловые, в анонимную память не копируются), `mem_limit: 32g` ограничивает и файловый кэш контейнера. |

Ход одной задачи: энкодер Qwen3-VL с диска на карту → эмбеддинги промпта (на CPU) →
энкодер выгружен → трансформер и VAE на карту → 6 шагов денойза → декод → выгрузка.
Чтение ~130 ГБ с диска входит во время генерации: минуты на картинку, тайм-аут шлюза 20,5 мин.

У модели нет числа шагов, CFG и negative_prompt — шлюз их для неё не принимает.
Размеры: пресеты ~16,8 Мп через `aspect_ratio` (`1:1` 4096×4096, `4:3` 4736×3552,
`3:2` 5024×3360, `16:9` 5472×3072, `21:9` 6272×2688 и портретные), либо явные
`width`/`height` от 1024 до 4096, кратные 32.

## 2. Состав каталога

Бэкенд и запись модели — в самом репозитории images-api (`worker/images_worker/backends/minimax.py`,
`worker/images_worker/registry.py`); здесь только то, что нужно узлу.

| Файл | Что это |
|---|---|
| `worker/` | Пакет воркера для образа — копия `../../worker` из корня репозитория. В git не хранится: кладётся перед сборкой (3.1) |
| `Dockerfile`, `requirements.txt`, `.dockerignore` | Образ: python 3.12, torch 2.9.1+cu128, diffusers d9586a3, transformers 5.23.0 |
| `Dockerfile.hidream`, `requirements-hidream.txt`, `hidream_vendor/` | Отдельный образ для HiDream-O1-Image (transformers 4.57 и вендорный код) |
| `compose.yml`, `worker.env.example`, `.env.example` | Запуск на узле: токен и имя узла — в `worker.env`, UUID карты — в `.env` |
| `check_node.py` | Проверка узла без обращения к CUDA (в образе — `/opt/check_node.py`) |
| `tests/` | Тесты на CPU: `run_tests.sh`, пробники памяти `rss_probe.py`, `memcap_probe.py`. Поддельный кэш весов `tests/fake_hf` собирается при первом запуске и в git не хранится |
| `download.sh`, `download-*.sh` | Загрузка весов в `./hf` каталога, где лежит скрипт (у MiniMax уже выполнена) |
| `gpu-watchdog.sh`, `images-big-node-gpu-watchdog.{service,timer}` | Сторож GPU контейнеров, ставится на машину шлюза (раздел 5) |

## 3. Развёртывание

### 3.1. Собрать комплект (локально)

Контекст сборки образа — этот каталог, а пакет воркера в нём (`worker/`) — копия
`../../worker` из корня репозитория. В git её нет: положить перед первой сборкой и
обновлять после любых изменений в `worker/` репозитория:

```bash
cd infra/big-node
rm -rf worker && cp -r ../../worker worker   # пакет воркера в контекст сборки
./tests/run_tests.sh                          # ~3,5 минуты на CPU
```

### 3.2. Отправить на хост

```bash
HOST=user@<хост>
DIR=minimax-h3-image        # каталог комплекта на хосте, путь от домашнего каталога
rsync -av Dockerfile Dockerfile.hidream compose.yml requirements.txt requirements-hidream.txt \
      .dockerignore check_node.py worker.env.example .env.example download*.sh "$HOST:$DIR/"
rsync -av --delete --exclude __pycache__ worker/ "$HOST:$DIR/worker/"
rsync -av --delete hidream_vendor/ "$HOST:$DIR/hidream_vendor/"
```

Каталог `hf/` с весами на хосте этими командами не затрагивается.

### 3.3. На хосте: проверить окружение

```bash
cd ~/minimax-h3-image        # каталог комплекта

# веса: снапшот на месте, ~135 ГБ
ls hf/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/*/
du -sh hf

# карта: её UUID пойдёт в .env (3.4)
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv

# docker видит рантайм nvidia; память хоста
docker info | grep -i runtime
free -g
```

Если uid пользователя на хосте не 1000 — убедиться, что `hf/` читается остальными
(`chmod -R o+rX hf`): процесс в контейнере работает от uid 1000.

### 3.4. Токен, имя узла и карта

```bash
cp worker.env.example worker.env && chmod 600 worker.env
# вписать IMG_WORKER_TOKEN=<тот же, что получит шлюз> и IMG_WORKER_NODE=<имя узла в шлюзе>

cp .env.example .env
# вписать IMG_GPU_UUID=<UUID карты из nvidia-smi выше>
docker compose config --quiet && echo ok   # пока UUID не задан — ошибка с подсказкой
```

UUID подставляется в `device_ids` всех сервисов `compose.yml` (`${IMG_GPU_UUID:?...}`):
контейнеры видят только эту карту, а без значения compose не выполнит ни одной команды,
вместо того чтобы молча занять не ту карту. `.env` docker compose читает сам из каталога
с `compose.yml`, в контейнеры он не передаётся.

### 3.5. Сборка

```bash
docker compose build
# за прокси: docker compose build --build-arg https_proxy=http://host:port
```

Образ 7,6 ГБ; локально (через прокси) сборка заняла около 15 минут, из них 11 — слой с torch.

### 3.6. Проверка узла, карту не занимает

```bash
docker compose run --rm worker python /opt/check_node.py
```

Ожидается `ok` во всех строках и код выхода 0: снапшот найден, все шарды `text_encoder`
(14) и `transformer_diffusers` (12) на месте и нужного размера, каталог с весами только
для чтения, `nvidia-smi` видит ровно одну карту. `FAIL nvidia-smi не найден` означает, что
контейнер не получил GPU: проверить nvidia-container-toolkit и блок `deploy` в `compose.yml`.

### 3.7. Запуск

```bash
docker compose up -d
docker compose ps                              # через минуту — (healthy)
curl -s 127.0.0.1:8107/healthz

set -a; . ./worker.env; set +a
curl -s -H "X-Worker-Token: $IMG_WORKER_TOKEN" 127.0.0.1:8107/status | python3 -m json.tool
```

В `/status` должно быть: `"supported_models": ["minimax-h3-image"]`, `"gpu": {"available": true,
"name": "... RTX PRO 6000 ...", "total_mb": ~97000}`, `"resident_model": null`.
В `nvidia-smi` на хосте процесса контейнера быть не должно: до первой задачи воркер
CUDA не трогает.

## 4. Первая генерация — вручную, мимо шлюза

Прямой вызов воркера **не проверяет свободную VRAM** — это делает только шлюз. Поэтому сначала:

```bash
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits   # нужно >= 72000 у PRO 6000
free -g                                                                  # сколько свободно до запуска
```

В двух соседних терминалах:

```bash
watch -n 2 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; free -g | head -2'
docker compose logs -f worker
```

Запуск (сначала 2048×2048 — те же веса и то же чтение с диска, но меньше активаций;
потом повторить с 4096×4096):

```bash
H="X-Worker-Token: $IMG_WORKER_TOKEN"; J='Content-Type: application/json'
curl -s -H "$H" -H "$J" -d '{"model_id":"minimax-h3-image"}' 127.0.0.1:8107/load

curl -s --max-time 1800 -D headers.txt -o first.png -w 'http=%{http_code} %{size_download} байт за %{time_total} с\n' \
  -H "$H" -H "$J" 127.0.0.1:8107/generate -d '{
    "task_id":"manual-1","model_id":"minimax-h3-image",
    "prompt":"An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts.",
    "width":2048,"height":2048,"num_inference_steps":6,"guidance_scale":1.0,"seed":42}'
```

Отмена из третьего терминала: `curl -s -H "$H" -H "$J" -d '{"task_id":"manual-1"}' 127.0.0.1:8107/cancel`.
Она срабатывает между стадиями и между шагами денойза; чтение весов с диска и идущий шаг не прерываются.
Аварийно: `docker compose kill worker` — память карты освобождается сразу.

Что должно быть в журнале и на что смотреть:

| Строка журнала | Что записать |
|---|---|
| `text stage done in …s, peak allocated … MiB, reserved … MiB, left on card 0 MiB` | время и пик стадии энкодера |
| `denoise step N/6 done in …s` | секунды на шаг |
| `image stage done in …s, peak allocated … MiB, reserved … MiB, left on card 0 MiB` | время и пик стадии трансформера и VAE |
| `exiting 120s after the task to release the CUDA context` | через 2 минуты после задачи |

И вне журнала: максимум `memory.used` по `nvidia-smi`; `used` и `buff/cache` по `free` во
время чтения весов; после выхода процесса — `memory.used` карты вернулся к значению до
запуска, `docker inspect -f '{{.RestartCount}}' images-worker-minimax` вырос на 1.

`left on card` больше нуля или строка `task … left … MiB allocated on the card` — утечка:
память всё равно вернётся при выходе процесса, но причину надо искать.

По итогам поправить в реестре images-api: `peak_vram_mb` (максимум `nvidia-smi` во время
задачи на 4096×4096, +5 %) и `seconds_per_step_hint` (тайм-аут шлюза = 240 + 18 × hint;
держать его в 2–3 раза больше реального времени задачи).

## 5. Подключение к шлюзу

Выполняется на машине шлюза. Команды составлены по README шлюза, с этой моделью не проверялись.

Имя узла в примерах — `node-c`; оно же должно стоять в `IMG_WORKER_NODE` в `worker.env`.

```bash
# 1. бэкенд minimax и его запись уже в реестре images-api; проверка — тесты шлюза
cd /путь/к/images-api && pytest tests/test_registry.py tests/test_resolve.py

# 2. туннель: в /etc/ssh-tunnels/node-c.env форварды на порты воркеров узла
#    (-L 18307:127.0.0.1:8107 — minimax; qwen21, ernie, hidream, zimage, moss — 18305, 18303, 18304, 18302, 18306)
sudo systemctl enable --now images-tunnel@node-c
curl -s 127.0.0.1:18307/healthz

# 3. узел: сначала выключенным, затем опрос и включение
curl -sS -X POST localhost:8010/api/v1/nodes -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d '{
  "name":"node-c","kind":"remote","base_urls":{"minimax":"http://127.0.0.1:18307"},
  "worker_token":"<токен из worker.env>","vram_reserve_mb":4096,"enabled":false}'
curl -sS -X POST localhost:8010/api/v1/nodes/node-c/refresh -H "X-API-Key: $KEY"
curl -sS -X PATCH localhost:8010/api/v1/nodes/node-c -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"enabled":true}'

# 4. задача через шлюз
curl -sS -X POST localhost:8010/api/v1/generate -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3-image","prompt":"...","aspect_ratio":"16:9"}'
curl -sS "localhost:8010/api/v1/tasks/<id>/wait?timeout=1500" -H "X-API-Key: $KEY"

# 5. сторож GPU контейнеров (gpu-watchdog.sh): установка — в шапке
#    images-big-node-gpu-watchdog.service, настройки — в шапке скрипта
#    (IMG_WORKER_TOKEN, REMOTE_SSH, REMOTE_DIR, при других портах туннеля — SERVICES)
```

Учесть на стороне клиентов: задача идёт минуты, а не секунды, — тайм-ауты ожидания
(`/wait` максимум 1800 с) должны это покрывать. Картинка 16,8 Мп в PNG — до ~50 МБ;
лимит шлюза `IMG_MAX_IMAGE_BYTES` по умолчанию 64 МБ, этого хватает.

## 6. Настройки

Заданы в `compose.yml`; токен и имя узла — в `worker.env`. UUID карты (`IMG_GPU_UUID`) —
не переменная контейнера, а подстановка compose: он задаётся в `.env` рядом с `compose.yml`.

| Переменная | Значение | Зачем |
|---|---|---|
| `IMG_WORKER_BACKEND` | `minimax` | какой бэкенд обслуживает процесс |
| `IMG_WORKER_HOST` | `0.0.0.0` | внутри контейнера — все интерфейсы, иначе проброс порта не работает; на хосте порт открыт только на `127.0.0.1` |
| `IMG_WORKER_PORT` | `8107` | порт воркера |
| `IMG_WORKER_NODE` | `node-c` (в `worker.env`) | имя узла в ответах воркера; общее для всех сервисов комплекта |
| `IMG_WORKER_TOKEN` | секрет | обязателен: без него воркер доступен любому контейнеру той же docker-сети |
| `IMG_WORKER_IDLE_EXIT_S` | `0` | общий выход по простою выключен: он перезапускал бы контейнер по кругу, даже когда карта не тронута |
| `IMG_WORKER_IDLE_UNLOAD_S` | `180` | страховочная «выгрузка»; у этого бэкенда VRAM не затрагивает |
| `IMG_MINIMAX_EXIT_AFTER_S` | `120` | выход из процесса после задачи; `0` — не выходить (CUDA-контекст останется на карте). Увеличить, если ответ с картинкой не успевает уйти через туннель |
| `IMG_MINIMAX_PLACEMENT` | `to` | `device_map` — читать веса сразу на карту; запасной путь, см. раздел 8 |
| `HF_HUB_CACHE` | `/hf/hub` | веса, том только для чтения |
| `HF_HOME` | `/tmp/hf-home` | служебные файлы huggingface_hub — вне тома с весами |
| `HF_HUB_OFFLINE` | `1` | ни одного обращения в сеть |

## 7. Остановка, откат, удаление

```bash
cd ~/minimax-h3-image        # каталог комплекта

docker compose stop                 # остановить; сам не поднимется (unless-stopped)
docker compose start                # вернуть

docker compose down                 # удалить контейнер и сеть проекта
docker image rm images-worker-minimax:1
rm -f worker.env                    # токен
```

Откат на предыдущую сборку: перед новой сборкой сохранить текущий образ под другим тегом
(`docker tag images-worker-minimax:1 images-worker-minimax:prev`), для отката — вернуть
тег обратно и выполнить `docker compose up -d`.

Полное удаление с хоста: команды выше, затем `rm -rf ~/minimax-h3-image` — это
удалит и веса (~135 ГБ). На хост вне этого каталога ничего не устанавливалось; `docker
builder prune` и `docker system prune` на чужом хосте не запускать — они затронут не только
наш кэш.

На стороне шлюза: `PATCH /api/v1/nodes/node-c {"draining":true}` → дождаться конца задачи →
`DELETE /api/v1/nodes/node-c`, остановить туннель (`systemctl disable --now images-tunnel@node-c`)
и сторож GPU (`systemctl disable --now images-big-node-gpu-watchdog.timer`). Убирать модель
из реестра (`MINIMAX_H3_IMAGE` в `registry.py`, бэкенд в `backends/__init__.py` и
`backends/minimax.py`) не обязательно: без узла, который её объявляет, запрос к ней
завершается `model_unavailable`, а не висит в очереди.

## 8. Если что-то пошло не так

| Симптом | Причина и действие |
|---|---|
| `failed to load text_encoder … out of memory`, шлюз получает `vram_oom` | карта занята соседями сильнее, чем обещал преflight, или занижен `peak_vram_mb` |
| `failed to load processor …` | неполный снапшот: `check_node.py` покажет, чего нет |
| контейнер убит, `docker inspect -f '{{.State.OOMKilled}}'` = true | упёрлись в `mem_limit`. Попробовать `IMG_MINIMAX_PLACEMENT=device_map`; если не помогло — поднять лимит и разобраться, что заняло анонимную память |
| `/status`: `"gpu": {"available": false, "error": "nvidia-smi not found"}` | контейнер без GPU или без capability `utility` |
| задача обрывается ровно через 20,5 мин | медленный диск: увеличить `seconds_per_step_hint` в реестре |
| ответ с картинкой не доходит, воркер перезапустился | туннель медленнее 120 с на ответ: увеличить `IMG_MINIMAX_EXIT_AFTER_S` |

## 9. Что проверено

Всё — локально, на CPU, без карты.

- **Тесты** (`tests/run_tests.sh`, 33 теста): настоящие блоки `MiniMaxH3ImageBlocks` и
  настоящие классы трансформера, VAE, Qwen3-VL с крошечными случайными весами, загрузка из
  кэша «только на чтение», офлайн. Порядок стадий; имена состояния и `output=`; к началу
  загрузки второй стадии компоненты первой уничтожены; после задачи, отмены и ошибки не
  живёт ни один тяжёлый компонент; отмена между стадиями, внутри денойза и перед декодом;
  `image_processor` создаётся сам; воспроизводимость по сиду; правила реестра и шлюза.
- **Контейнер**: образ собран, воркер в нём стартует от uid 1000, слушает `0.0.0.0:8107`,
  токен проверяется, генерация и отмена через HTTP проходят, том с весами `:ro`, в кэш
  ничего не пишется, healthcheck зелёный, после `os._exit(0)` контейнер перезапускается.
- **Память**: при загрузке на CPU веса остаются файловым mmap (RssAnon +7 МБ на 668 МБ
  весов энкодера, +1 МБ на 438 МБ трансформера); под лимитом памяти контейнера меньше размера весов процесс не убит — страницы
  вытесняются в пределах лимита.

Не проверено — негде: всё, что связано с картой и настоящими весами. Перенос на `cuda`,
реальный пик VRAM, время чтения 130 ГБ и шага денойза, поведение `mem_limit` при переносе
на карту, `nvidia-smi` внутри контейнера, выход из процесса после реальной задачи, качество
картинки, работа через шлюз и туннель.
