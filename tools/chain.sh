#!/system/bin/sh
# chain.sh [tag] -- 链式分段：末段(22..32, 持有采样器) spawn 前驱(11..22)，后者再 spawn (0..11)。
# 三进程全驻留（MAPBUDGET=1500M / RPCMEM_FREE=0），段间走 socketpair 乒乓。
# VAE 默认走 HTP0（CPU 58~67 s → HTP 18.8 s）。
#
# 可由环境变量覆盖（webui 用这套）：
#   PROMPT / STEPS / SEED / W / H / SD_DIT_DUMP_DIR
TAG=${1:-chain}

cd /data/local/tmp/qisplit || exit 1

PROMPT="${PROMPT:-cat}"
STEPS="${STEPS:-8}"
SEED="${SEED:-42}"
W="${W:-256}"
H="${H:-256}"
# 留空 = 用 sd.cpp 默认（物理核数）。手机上大小核混用，显式给小值往往更快。
THREADS="${THREADS:-}"
# VAE 跑哪个后端。HTP0 是定版：CPU 58~67 s → HTP 18.8 s，像素 MAE 0.030/255。
# VAE 上 HTP 占 644 MB IOVA，所以 DiT 的预算要压到 1500M（1900M 装不下会挂）。
# 回退到纯 CPU VAE：VAE_BACKEND=CPU GGML_HEXAGON_MAPBUDGET=1900M
VAE_BACKEND="${VAE_BACKEND:-HTP0}"
VAE_THREADS=""

export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF=128M,128M
export GGML_HEXAGON_MAPBUDGET="${GGML_HEXAGON_MAPBUDGET:-1500M}"
export GGML_HEXAGON_RPCMEM_FREE="${GGML_HEXAGON_RPCMEM_FREE:-0}"

export SD_DIT_SPLIT_PLAN="${SD_DIT_SPLIT_PLAN:-11,22}"
export SD_DIT_LAYER_FIRST="${SD_DIT_LAYER_FIRST:-22}"
export SD_DIT_LAYER_LAST="${SD_DIT_LAYER_LAST:-32}"
# 默认不落 joint/vel（webui 正常出图时不需要，落盘会多写 8 x 4.3 MB 并拖慢）
export SD_DIT_DUMP_DIR="${SD_DIT_DUMP_DIR:-}"
export SD_DIT_SPLIT_STEPS="$STEPS"
export SD_DIT_CTX_FILE="${SD_DIT_CTX_FILE:-/data/local/tmp/qisplit/qs-split-ctx.bin}"
# TE（Qwen3-VL-8B Q4_K_M，4.4 GB）编码完就释放：末段 RSS 4.48 GB -> 24 MB，代价约 +6 s。
# 只在「一次性跑」下安全（同进程不会再编码新 prompt）；常驻服务模式必须关掉。
export SD_DIT_RELEASE_TE="${SD_DIT_RELEASE_TE:-1}"

[ -n "$SD_DIT_DUMP_DIR" ] && mkdir -p "$SD_DIT_DUMP_DIR"

./sd-cli-hex \
  --diffusion-model models/dit.gguf \
  --llm models/te.gguf \
  --vae models/vae.safetensors \
  -p "$PROMPT" --sigmas qwen-image-2.1-turbo --steps "$STEPS" --cfg-scale 1.0 --seed "$SEED" \
  -H "$H" -W "$W" ${THREADS:+-t "$THREADS"} \
  --backend "diffusion=HTP0,vae=$VAE_BACKEND,te=CPU" \
  -o "out_$TAG.png"

echo "CHAIN exit=$?"
