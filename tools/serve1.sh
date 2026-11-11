#!/system/bin/sh
# 设备侧：单进程常驻服务（**不切分**，权重走 IOVA 流式）。
#
#   sh serve1.sh [port]        # 默认 18080；PC 侧 adb forward tcp:18080 tcp:18080
#
# 为什么先给这个版本：分段链在服务模式下有个已知问题 —— worker 会继承 sd-server 的 argv，
# 自己也去 listen 同一个端口（见 docs/npu-split.md §9）。单进程版没有 worker，最稳。
# 代价：DiT 4 GB > 2.3 GB VA 窗口，只能流式换入换出，256²/8 步约 130 s（分段链约 125 s，差的不多）。
#
# 内存 ~4.4 GB（TE + DiT 流式 + VAE），实测 MemAvailable 峰值下降 ~4.7 GB。
PORT="${1:-18080}"
cd /data/local/tmp/qisplit || exit 1

export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF=128M,128M
export GGML_HEXAGON_MAPBUDGET="${GGML_HEXAGON_MAPBUDGET:-1400M}"
export GGML_HEXAGON_RPCMEM_FREE=1

# 明确关掉分段（否则 worker 会继承 argv 再起一个 server）
unset SD_DIT_SPLIT_PLAN SD_DIT_LAYER_FIRST SD_DIT_LAYER_LAST SD_DIT_SPLIT_STEPS
unset SD_DIT_SPLIT_FD SD_DIT_SPLIT_WORKER SD_DIT_SPLIT_EXIT SD_DIT_CTX_FILE
unset SD_DIT_DUMP_DIR SD_DIT_JOINT_IN SD_DIT_X_IN

exec ./sd-server-hex \
    --diffusion-model models/dit.gguf \
    --llm models/te.gguf \
    --vae models/vae.safetensors \
    --sigmas qwen-image-2.1-turbo \
    --steps 8 --cfg-scale 1.0 \
    -W "${W:-256}" -H "${H:-256}" \
    --backend diffusion=HTP0,vae=HTP0,te=CPU \
    --listen-ip "${LISTEN_IP:-0.0.0.0}" --listen-port "$PORT"
