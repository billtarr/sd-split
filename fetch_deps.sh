#!/usr/bin/env bash
# fetch_deps.sh -- 拉取编译所需的第三方源码（不提交进 git）
#
#   ./fetch_deps.sh
#
# 三件事：
#   1. stable-diffusion.cpp 源码（master，含 qwen_image_2_1 支持）
#   2. 与它**配套的** patched ggml —— 必须取 .gitmodules 里 pin 住的那个 commit，
#      随手拉 leejet/ggml 的 master 会编不过（master 是上游 ggml，缺 ggml-graph.h，
#      sd.cpp 的 compute_workspace 直接 #error）。
#   3. Khronos OpenCL 头文件（只在 ./build.sh opencl 时需要）
#
# 网络：GitHub 直连 TLS 会失败，这里统一走 codeload（实测 5 MB/s 可用）；
# GitHub API 反而能直连，用来查 submodule 的 sha。
set -euo pipefail
cd "$(dirname "$0")"

SD_DIR="stable-diffusion.cpp-master"
SD_REPO="leejet/stable-diffusion.cpp"
CLONE_HOST="${CLONE_HOST:-https://codeload.github.com}"

fetch_tar() { # <repo> <ref> <outdir>
  local repo="$1" ref="$2" out="$3"
  echo "→ $repo@$ref"
  curl -sL --noproxy '*' --retry 3 -o /tmp/_deps.tgz "$CLONE_HOST/$repo/tar.gz/$ref"
  rm -rf "$out" && mkdir -p "$out"
  # Windows 上解不出符号链接（.pi/ 那类），只需源码，忽略即可
  tar xzf /tmp/_deps.tgz --strip-components=1 -C "$out" --exclude='.pi' 2>/dev/null || true
}

if [ ! -d "$SD_DIR" ]; then
  # master 的 tarball 不含 .git，拿不到子模块指针 → 先问 API
  fetch_tar "$SD_REPO" "refs/heads/master" "$SD_DIR"
fi

if [ ! -f "$SD_DIR/ggml/include/ggml-graph.h" ]; then
  SHA=$(curl -sL --noproxy '*' --max-time 30 \
        "https://api.github.com/repos/$SD_REPO/contents/ggml?ref=master" \
        | tr -d ' \n' | sed -n 's/.*"sha":"\([0-9a-f]\{40\}\)".*/\1/p')
  if [ -z "${SHA:-}" ]; then
    echo "取不到 ggml 子模块的 sha，检查网络" >&2
    exit 1
  fi
  echo "ggml pin = $SHA"
  fetch_tar "leejet/ggml" "$SHA" "$SD_DIR/ggml"
fi

if [ ! -f cl-headers/CL/cl.h ]; then
  echo "→ OpenCL-Headers（只有 opencl 后端需要）"
  curl -sL --noproxy '*' --retry 3 -o /tmp/_cl.tgz \
    "$CLONE_HOST/KhronosGroup/OpenCL-Headers/tar.gz/refs/heads/main"
  rm -rf cl-headers && mkdir -p cl-headers
  tar xzf /tmp/_cl.tgz --strip-components=1 -C cl-headers
fi

echo "✓ 依赖就绪：$SD_DIR（+ ggml）, cl-headers"
