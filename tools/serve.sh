#!/system/bin/sh
# 设备侧：分段常驻服务（末段持有采样器 + 采样器/HTTP，前两段是 worker）。
#
#   sh serve.sh [port]        # 默认 18080，PC 侧 adb forward tcp:18080 tcp:18080
#
# 与 chain.sh 的区别：
#   * SD_DIT_SPLIT_EXIT=0 —— worker 不早退，常驻等下一个请求；
#   * SD_DIT_SPLIT_STEPS=8 —— 末段步号回绕，worker 靠 step 回到 0 判定「新请求」并重读 ctx.bin；
#   * 末段（22..32）就是 sd-server 本身，HTTP 由它提供。
#
# 内存：末段 ~4.4 GB（TE+VAE+自己那 10 层），两个 worker 各 ~66 MB（不加载 TE/VAE）。
PORT="${1:-18080}"
cd /data/local/tmp/qisplit || exit 1

export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF=128M,128M
export GGML_HEXAGON_MAPBUDGET=1900M
export GGML_HEXAGON_RPCMEM_FREE=0

export SD_DIT_SPLIT_PLAN="${SD_DIT_SPLIT_PLAN:-11,22}"
export SD_DIT_LAYER_FIRST="${SD_DIT_LAYER_FIRST:-22}"
export SD_DIT_LAYER_LAST="${SD_DIT_LAYER_LAST:-32}"
export SD_DIT_SPLIT_STEPS="${SD_DIT_SPLIT_STEPS:-8}"
export SD_DIT_SPLIT_EXIT=0
export SD_DIT_CTX_FILE="${SD_DIT_CTX_FILE:-/data/local/tmp/qisplit/qs-split-ctx.bin}"
unset SD_DIT_DUMP_DIR SD_DIT_JOINT_IN SD_DIT_X_IN

exec ./sd-server-hex \
    --diffusion-model models/dit.gguf \
    --llm models/te.gguf \
    --vae models/vae.safetensors \
    --sigmas qwen-image-2.1-turbo \
    --steps 8 --cfg-scale 1.0 \
    -W "${W:-256}" -H "${H:-256}" \
    --backend diffusion=HTP0,vae=CPU,te=CPU \
    --listen-ip 127.0.0.1 --listen-port "$PORT"
