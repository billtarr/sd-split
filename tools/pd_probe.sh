#!/system/bin/sh
# Two concurrent sd-cli-hex processes -> tests whether two fastrpc PDs can coexist.
# Usage: pd_probe.sh <steps> <mapbudget> <tag>
STEPS=${1:-4}
BUDGET=${2:-1400M}
TAG=${3:-pd}

cd /data/local/tmp/qisplit || exit 1
export LD_LIBRARY_PATH=lib
export ADSP_LIBRARY_PATH=lib
export GGML_HEXAGON_OPQUEUE=1
export GGML_HEXAGON_MBUF=128M,128M
export GGML_HEXAGON_MAPBUDGET=$BUDGET
export GGML_HEXAGON_RPCMEM_FREE=1

for i in A B; do
  ./sd-cli-hex \
    --diffusion-model models/dit.gguf \
    --llm models/te.gguf \
    --vae models/vae.safetensors \
    -p cat --steps "$STEPS" --cfg-scale 1.0 \
    -H 256 -W 256 \
    --backend diffusion=HTP0,vae=CPU,te=CPU \
    -o "${TAG}${i}.png" > "${TAG}${i}.log" 2>&1 &
  if [ "$i" = "A" ]; then PA=$!; else PB=$!; fi
done

echo "started A=$PA B=$PB"
wait $PA; echo "A exit=$?"
wait $PB; echo "B exit=$?"
echo "--- done ---"
