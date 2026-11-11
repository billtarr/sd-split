#!/system/bin/sh
# chain.sh [tag] -- 链式分段：末段(22..32, 持有采样器) spawn 前驱(11..22)，后者再 spawn (0..11)。
# 三进程全驻留（每段 11 层 ≈ 1450 MiB IOVA，一次 map 到底），段间走 socketpair 乒乓。
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
# VAE 上 HTP 占 644 MB IOVA，与 DiT 的 1450 MiB 合计 2094 MiB，仍在 2304 MiB 窗口内。
# 回退到纯 CPU VAE：VAE_BACKEND=CPU
VAE_BACKEND="${VAE_BACKEND:-HTP0}"
VAE_THREADS=""

export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF=128M,128M

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
# 原生 M-RoPE 内核：采样 5.07 -> 1.96 s/it，出图与 pe 路径 MAE 0.45/255（PSNR 48.9 dB）。默认开。
# 回退到通用的「转置+repeat+mul」分解：SD_ROPE_NATIVE=0
export SD_ROPE_NATIVE="${SD_ROPE_NATIVE:-1}"
# VAE 的卷积走哪条路。默认 "auto,vae=on" = 强制 VAE 用 HTP 原生 CONV_2D。
# 原因：sd.cpp 的 auto 要求后端有 implicit GEMM，HTP0 上一律判 off → conv 退化成 IM2COL，
# 而 im2col 产物在 256² 就有 324 MB，被尺寸门槛拒掉后整段回退 CPU（图被切成 31 段 HTP0/CPU 交替）。
# 实测 VAE 解码 17.43 s → 1.55 s（11×），出图 MAE 0.57/255。回退：CONV_DIRECT=auto
CONV_DIRECT="${CONV_DIRECT:-auto,vae=on}"
# flash attention。默认强制 diffusion 开：sd.cpp 的 auto 探测在 HTP 上会以
# "insufficient probe memory" 退出（它读 ggml_backend_dev_memory()，HTP 只上报 mbuf 总量），
# 所以 auto 恒等于 off。
# 注意 HTP 的 HMX 内核在 KV 很长时数值会漂，`ggml_extend.cpp` 里有按 token 数的门槛
# （SD_FLASH_ATTN_MAX_KV，默认 1024）：256²(265 token) 走 FA，512²(4105) 自动退回 f32 三段式。
# 完全关掉：FLASH_ATTN=off
FLASH_ATTN="${FLASH_ATTN-off,diffusion=on}"

[ -n "$SD_DIT_DUMP_DIR" ] && mkdir -p "$SD_DIT_DUMP_DIR"

./sd-cli-hex \
  --diffusion-model models/dit.gguf \
  --llm models/te.gguf \
  --vae models/vae.safetensors \
  -p "$PROMPT" --sigmas qwen-image-2.1-turbo --steps "$STEPS" --cfg-scale 1.0 --seed "$SEED" \
  -H "$H" -W "$W" ${THREADS:+-t "$THREADS"} \
  --backend "diffusion=HTP0,vae=$VAE_BACKEND,te=CPU" \
  ${CONV_DIRECT:+--conv-direct "$CONV_DIRECT"} \
  ${FLASH_ATTN:+--flash-attn "$FLASH_ATTN"} \
  -o "out_$TAG.png"

echo "CHAIN exit=$?"
