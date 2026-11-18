# Узел с большой картой: контейнерный комплект

Комплект, по которому 17.11.2026 развёрнут узел с большой картой (в примерах — `node-c`:
RTX PRO 6000 96 ГБ, внешний боевой сервер). Подробная инструкция — `README.md` рядом
(написана для MiniMax-H3-Image; сервис `qwen21` в `compose.yml` устроен так же).

Отличия от остальных узлов:
* воркеры работают в docker, а не в systemd с venv; образ `images-worker-minimax` общий
  у minimax, qwen21 и ernie, у hidream свой (`Dockerfile.hidream`);
* сборочный контекст — каталог с `worker/` внутри: перед сборкой скопировать сюда
  `../../worker` (`rm -rf worker && cp -r ../../worker worker`); копия в git не хранится;
* карта задаётся UUID в `.env` рядом с `compose.yml` (`IMG_GPU_UUID`, см. `.env.example`),
  токен и имя узла — в `worker.env` (`IMG_WORKER_TOKEN`, `IMG_WORKER_NODE`);
* модели только «целиком на карте»: у `qwen21` объявлена одна `qwen-image-2.1-full`
  (`IMG_WORKER_MODELS`), чтобы шлюз не послал туда варианты, держащие энкодер в RAM хоста;
* веса лежат в `./hf` на узле, том смонтирован только на чтение, `HF_HUB_OFFLINE=1`;
* туннель на машине шлюза — инстанс `images-tunnel@node-c` (`/etc/ssh-tunnels/node-c.env`),
  18303 → ernie, 18304 → hidream, 18305 → qwen21, 18307 → minimax;
* сторож GPU контейнеров — `gpu-watchdog.sh` и `images-big-node-gpu-watchdog.{service,timer}`
  на машине шлюза (установка — в шапке `.service`, настройки — в шапке скрипта).

Тесты комплекта (`tests/run_tests.sh`) создают крошечный поддельный репозиторий весов
(`make_tiny_repo.py`, в `tests/fake_hf`, в git не хранится) и гоняют настоящий модульный
пайплайн diffusers на CPU. `registry_patch.py` из исходного комплекта больше не нужен:
бэкенд `minimax` и его запись уже в реестре.
