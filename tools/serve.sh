#!/system/bin/sh
# 设备侧：分段常驻服务（末段持有采样器 + 采样器/HTTP，前两段是 worker）。
#
#   sh serve.sh [port]        # 默认 18080，默认监听 0.0.0.0（PC 直连 http://<手机IP>:18080）
#   只想本机：LISTEN_IP=127.0.0.1 sh serve.sh，然后 adb forward tcp:18080 tcp:18080
#
# 与 chain.sh 的区别：
#   * SD_DIT_SPLIT_EXIT=0 —— worker 不早退，常驻等下一个请求；
#   * SD_DIT_SPLIT_STEPS=8 —— 末段步号回绕，worker 靠 step 回到 0 判定「新请求」并重读 ctx.bin；
#   * 末段（22..32）就是 sd-server 本身，HTTP 由它提供。
#
# 内存：两个 worker 各 ~66 MB；末段 = 1450 MiB DiT + 4.4 GB TE + 644 MiB VAE。
PORT="${1:-18080}"
cd /data/local/tmp/qisplit || exit 1

export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF=128M,128M

export SD_DIT_SPLIT_PLAN="${SD_DIT_SPLIT_PLAN:-11,22}"
export SD_DIT_LAYER_FIRST="${SD_DIT_LAYER_FIRST:-22}"
export SD_DIT_LAYER_LAST="${SD_DIT_LAYER_LAST:-32}"
export SD_DIT_SPLIT_STEPS="${SD_DIT_SPLIT_STEPS:-8}"
export SD_DIT_SPLIT_EXIT=0
export SD_DIT_CTX_FILE="${SD_DIT_CTX_FILE:-/data/local/tmp/qisplit/qs-split-ctx.bin}"
unset SD_DIT_DUMP_DIR SD_DIT_JOINT_IN SD_DIT_X_IN

# 常驻服务不能释放 TE（后面的请求还要编码 prompt），末段因此常驻 4.4 GB。
export SD_DIT_RELEASE_TE=0
# 原生 M-RoPE（采样 5.07 -> 1.45 s/it）
export SD_ROPE_NATIVE="${SD_ROPE_NATIVE:-1}"
# 放开 flash attention 的 token 门槛：512²（4105 token）也走 HMX 内核。
# ⚠️ HMX 在 KV 很长时数值会漂（512² 与不开 FA 的 MAE 24.6/255，会换一只毛色不同的猫），
#    想稳妥就设 SD_FLASH_ATTN_MAX_KV=1024（默认值，512² 自动退回 f32 三段式）。
export SD_FLASH_ATTN_MAX_KV="${SD_FLASH_ATTN_MAX_KV-0}"

# VAE 分块解码（与 chain.sh 同源，见 I-049）：512² 的最后一档 conv（512×512 输出）
# kparams 在 VTCM 约束下求解失败 → 整档落 CPU，VAE 解码 78.5 s；切成 320 tile 后进 DSP，6.8 s。
# 320² 单 tile 走同一路径，只多 0.23 s，所以默认全尺寸开。
# 关掉：VAE_TILING=  sh serve.sh
VAE_TILING="${VAE_TILING---vae-tiling --vae-tile-size 320}"

# CPU 拓扑（v73）：cpu0-2 = A510 2.0 GHz，cpu3-6 = A715 2.8 GHz，cpu7 = X3 3.19 GHz。
# TE（Qwen3-VL-8B Q4_K_M）在 CPU 上只做 10~16 token 的 GEMV，本质是「把 4.8 GB 权重扫一遍」，
# 8 线程时 3 个小核各领一份行、成为同步瓶颈（实测编码时 cpu0 1.67 / cpu4 1.92 / cpu7 1.6 GHz，
# 大核根本没吃满）。THREADS/AFFINITY 用来把它压到大核上。
set -- --diffusion-model models/dit.gguf \
    --llm models/te.gguf \
    --vae models/vae.safetensors \
    --sigmas qwen-image-2.1-turbo \
    --steps 8 --cfg-scale 1.0 \
    -W "${W:-256}" -H "${H:-256}" \
    --backend "diffusion=HTP0,vae=HTP0,te=CPU" \
    --conv-direct "auto,vae=on" \
    --flash-attn "off,diffusion=on" \
    ${VAE_TILING:+$VAE_TILING} \
    --listen-ip "${LISTEN_IP:-0.0.0.0}" --listen-port "$PORT"

# 默认值：5 线程 + cpu3~7（4 个 A715 + X3）。实测（v73，256²/8 步）：
#   默认 8 线程/不绑核：预热 27.8 s、TE 4.6 s、sampling 39.4 s
#   f8 + 5 线程      ：预热 12.6 s、TE 4.2 s、sampling 37.6 s   ← 定版
#   f0 + 4 线程      ：预热 15.2 s、TE 4.6 s、sampling 38.2 s
# 想回退：AFFINITY= THREADS=  sh serve.sh
THREADS="${THREADS-5}"
AFFINITY="${AFFINITY-f8}"
[ -n "${THREADS:-}" ] && set -- "$@" -t "$THREADS"

# AFFINITY 是十六进制 CPU 掩码（toybox 的 taskset 不认 -c/list 形式，也不要 0x 前缀）：
#   f8 = cpu3~7（4 个 A715 + X3）；f0 = cpu4~7；80 = 只 X3。
if [ -n "${AFFINITY:-}" ]; then
    exec taskset "$AFFINITY" ./sd-server-hex "$@"
fi
exec ./sd-server-hex "$@"
