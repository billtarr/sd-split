#!/usr/bin/env bash
# build.sh -- 交叉编译 sd-cli 到 Android arm64（NDK r26d + SDK cmake 3.22.1）
#
#   ./build.sh              # hexagon（NPU）—— 本项目的目标形态
#   ./build.sh cpu          # 纯 CPU 后端（对照用）
#   ./build.sh opencl       # OpenCL 后端（Adreno；注意 VAE 数值问题，见 docs）
#
# 流程：fetch_deps.sh 拉上游 → apply_overlay 覆盖本项目改动 → cmake+ninja。
# 产物落在 $BUILD/bin/sd-cli，strip 后 push 到 /data/local/tmp/qisplit/ 直接跑
# （不要放进 app 私有目录，targetSdk 35 禁 exec）。
set -euo pipefail

SD_SRC="${SD_SRC:-stable-diffusion.cpp-master}"
BUILD="${BUILD:-build-hex}"
NDK="${NDK:-D:/ndk/android-ndk-r26d}"
CMAKE="${CMAKE:-D:/android-tools/sdk/cmake/3.22.1/bin/cmake.exe}"
NINJA_DIR="$(dirname "$CMAKE")"
NINJA="$NINJA_DIR/ninja.exe"
export PATH="$NINJA_DIR:$PATH"

OPENCL_LIB="${OPENCL_LIB:-D:/pocket-model-studio/app/src/main/jniLibs/arm64-v8a/libOpenCL.so}"
# cmake.exe 是 Windows 程序，给它的路径必须是 D:/... 形式（/d/... 会被当成相对路径）
SD_ROOT="$(pwd -W 2>/dev/null || pwd)"

BACKEND="${1:-hexagon}"
EXTRA=()
case "$BACKEND" in
  cpu)    ;;
  opencl)
    # 交叉编译时 FindOpenCL 一定找不到：头文件用 Khronos 官方那份，
    # 库用设备上 pull 出来的 /vendor/lib64/libOpenCL.so（厂商 Adreno loader，不能自己编）
    # 变量名是 FindOpenCL 的混合大小写形式（OpenCL_INCLUDE_DIR），给全大写它不认
    EXTRA+=(-DSD_OPENCL=ON
            -DOpenCL_INCLUDE_DIR="$SD_ROOT/cl-headers"
            -DOpenCL_LIBRARY="$OPENCL_LIB") ;;
  hexagon)
    # ggml-hexagon 后端（NPU）：host 侧从 llama.cpp 移植，DSP applet 用 ref/hexagon/ 里那份
    # 预编译的 libggml-htp-v73.so（不需要 Hexagon SDK，sdk-shim/ 顶替）。
    # 编译完还要把 applet 推到设备 lib/，否则 ADSP_LIBRARY_PATH 找不到 skel（报 0x80000406）。
    EXTRA+=(-DGGML_HEXAGON=ON) ;;
  *) echo "用法: ./build.sh [hexagon|cpu|opencl]" >&2; exit 1 ;;
esac

if [ ! -d "$SD_SRC" ]; then
  echo "缺源码目录 $SD_SRC —— 先跑 ./fetch_deps.sh" >&2
  exit 1
fi

# ---------------------------------------------------------------------------- overlay
# 本项目对上游的全部改动都放在 overlay/ 下，按相对路径原样覆盖。
# 为什么不用 patch：ggml-hexagon 是「整个目录换成 llama.cpp 的新版」，
# 跟上游 diff 会产生 130+ 个文件、几十 MB 的补丁，反而不可读。
apply_overlay() {
  echo "→ 应用 overlay（相对上游的改动）"
  local n=0 rel
  while IFS= read -r rel; do
    mkdir -p "$SD_SRC/$(dirname "$rel")"
    cp "overlay/$rel" "$SD_SRC/$rel"
    n=$((n + 1))
  done < <(cd overlay && find . -type f | sed 's|^\./||')
  echo "  覆盖 $n 个文件"
}
apply_overlay

"$CMAKE" -S "$SD_SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_BUILD_EXAMPLES=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  "${EXTRA[@]}"

# sd-cli 是主力；sd-server 给常驻服务用（tools/serve.sh）。
"$CMAKE" --build "$BUILD" --target sd-cli sd-server -j "${JOBS:-$(nproc 2>/dev/null || echo 8)}"

ls -lh "$BUILD/bin/sd-cli"
echo "✓ $BUILD/bin/sd-cli"
if [ -f "$BUILD/bin/sd-server" ]; then
  ls -lh "$BUILD/bin/sd-server"
  echo "✓ $BUILD/bin/sd-server"
fi
if [ "$BACKEND" = "hexagon" ]; then
  cat <<'EOF'

下一步（推设备）：
  STRIP=D:/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe
  cp build-hex/bin/sd-cli /tmp/sd-cli-hex && "$STRIP" /tmp/sd-cli-hex
  adb push /tmp/sd-cli-hex /data/local/tmp/qisplit/sd-cli-hex
  cp build-hex/bin/sd-server /tmp/sd-server-hex && "$STRIP" /tmp/sd-server-hex
  adb push /tmp/sd-server-hex /data/local/tmp/qisplit/sd-server-hex
  adb push ref/hexagon/libggml-htp-v73.so /data/local/tmp/qisplit/lib/
EOF
fi
