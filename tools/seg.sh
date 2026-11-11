#!/system/bin/sh
# seg.sh <tag> <first> <last> <joint_in_dir|-> <x_in_dir|->
# Runs one DiT segment for the layer-split numeric check (see docs/npu-split.md §3).
TAG=$1
FIRST=$2
LAST=$3
JOINT=$4
XIN=$5

cd /data/local/tmp/qisplit || exit 1

export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF="${GGML_HEXAGON_MBUF:-128M,128M}"
export GGML_HEXAGON_MAPBUDGET="${GGML_HEXAGON_MAPBUDGET:-1400M}"
export GGML_HEXAGON_RPCMEM_FREE="${GGML_HEXAGON_RPCMEM_FREE:-1}"

export SD_DIT_LAYER_FIRST="$FIRST"
export SD_DIT_LAYER_LAST="$LAST"
export SD_DIT_DUMP_DIR="dump_$TAG"
[ "$JOINT" != "-" ] && export SD_DIT_JOINT_IN="$JOINT"
[ "$XIN" != "-" ] && export SD_DIT_X_IN="$XIN"

mkdir -p "dump_$TAG"

./sd-cli-hex \
  --diffusion-model models/dit.gguf \
  --llm models/te.gguf \
  --vae models/vae.safetensors \
  -p cat --sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0 --seed 42 \
  -H 256 -W 256 \
  --backend diffusion=HTP0,vae=CPU,te=CPU \
  -o "out_$TAG.png"

echo "SEG $TAG exit=$?"
