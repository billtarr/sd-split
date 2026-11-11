# sd-split

把 **Qwen-Image 2.1 Turbo** 按层切成多段、跑在骁龙 8 Gen 2（SM8550 / v73）的 Hexagon NPU 上，
用 `stable-diffusion.cpp` 作底座、`ggml-hexagon` 作后端。

目标只有一个：**让 4 GB 的 DiT 权重全部走 NPU，且一次都不搬运。**
v73 每个 fastrpc PD 只有 **2304 MB** 的 IOVA 窗口，装不下 3.91 GiB 的 DiT，
所以按层切成 3 段（11 / 11 / 10 层），每段 ≈ 1.45 GB，一次 map 到底、全程常驻。

## 实测（256² / 8 步 / seed 42，一加 Ace 2 Pro）

| 形态 | 采样 | VAE 解码 | 端到端 | 出图 |
|---|---|---|---|---|
| CPU 单进程 | 48.5 s/it | 58 s（CPU） | 437 s | 基线 |
| NPU 单进程 + 权重流式（已移除） | 7.73 s/it | 58 s（CPU） | 133.8 s | `39595a07…` |
| **NPU 三段 + VAE 上 NPU** | **5.05 s/it** | **19.2 s** | **85.1 s** | `1ceb0cc0…` |
| 上面再加 TE 用完即释放 | 5.10 s/it | 17.8 s | 91.7~96.6 s | `1ceb0cc0…` |

- 三段相对整模型：采样 **−34%**。全程 `mapped 1450 MiB` 常驻、**权重零搬运**——每段一次 map 到底，
  不做任何换入换出（流式代码已整体移除，见「已知限制」）。
- VAE 上 NPU：58 s → 19 s（**−67%**），与 CPU 结果比 `MAE 0.030/255`、`PSNR 63.2 dB`。
- TE 用完即释放：末段 RSS **4.48 GB → 24 MB**，代价 +7~13% 时间。
- 512² / 8 步：**437.7 s**（采样 8×32.0 s、VAE 152 s）。全程零搬运，慢是 token 数（265 → 4105）。

数值判据：分段结果与整模型**逐位相同**（`vel` 8/8 步 `max|d| = 0`，出图 md5 一致）。

## 快速开始

```bash
./fetch_deps.sh                # 拉 stable-diffusion.cpp master + 配套 ggml pin + OpenCL 头
./build.sh                     # hexagon 后端（默认），产物 build-hex/bin/sd-cli
./build.sh cpu                 # 纯 CPU 对照
```

推设备与运行：

```bash
export MSYS_NO_PATHCONV=1; A=D:/android-tools/sdk/platform-tools/adb.exe
STRIP=D:/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe
cp build-hex/bin/sd-cli /tmp/sd-cli-hex && "$STRIP" /tmp/sd-cli-hex
"$A" push /tmp/sd-cli-hex                 /data/local/tmp/qisplit/sd-cli-hex
"$A" push ref/hexagon/libggml-htp-v73.so  /data/local/tmp/qisplit/lib/
"$A" push tools/chain.sh                  /data/local/tmp/qisplit/
"$A" shell "chmod 755 /data/local/tmp/qisplit/sd-cli-hex /data/local/tmp/qisplit/chain.sh"

"$A" shell "cd /data/local/tmp/qisplit && PROMPT='a red fox in snow' W=256 H=256 sh chain.sh t1"
```

权重放在设备 `/data/local/tmp/qisplit/models/`：

| 文件 | 大小 |
|---|---|
| `dit.gguf` (Qwen-Image 2.1 Turbo, Q4_K_M) | 3.91 GiB |
| `te.gguf` (Qwen3-VL-8B, Q4_K_M) | 4.68 GiB |
| `vae.safetensors` | 644 MiB |

## 架构

```
段 0 (层 0..11)      段 1 (层 11..22)     段 2 (层 22..32 + head + 采样器 + VAE)
  prelude             接 joint 续算          接 joint、算 velocity、更新 x
    ↑ joint 4.34 MB ←──┘   ↑ joint ←─────────┘
    └──── x 64 KB ──────────┴──── x ──────────┘        每步一轮，全同步
```

- **采样器留在末段**（ER-SDE 多阶段，不能在外部 driver 里重写，否则数值必分叉）。
- 段间走 `socketpair(AF_UNIX)`，帧格式 `int32 step | int32 ndim | int64 dims[] | float32 data[]`；
  `step = -1` 是控制帧，送 context（147 KB，每请求一次）。
- 每段只注册自己区间的 block，**越界层根本不向 gguf 要张量**。
- worker 不加载 TE / VAE（否则三份 4.8 GB 会把整机打进 OOM，见 `docs/npu-split.md §8`）。

## 目录

```
overlay/            对上游的全部改动，按相对路径覆盖（build.sh 自动 apply）
  src/model/diffusion/qwen_image_2_1.hpp   层区间切分 + IPC + TE 释放
  src/pipeline/diffusion_engine.{h,cpp}    release_conditioner() + worker 跳过加载
  src/pipeline/image.cpp                   编码后释放 TE
  ggml/include/ggml.h、ggml/src/ggml.c     后端注册
  ggml/src/ggml-hexagon/                   从 llama.cpp 移植的 HTP 后端（含 sdk-shim）
tools/              chain.sh（三段）/ serve.sh（分段常驻服务）/ webui.py / memwatch.sh / mempeak.sh
docs/               架构、运行手册、移植踩坑与实测数据
ref/                libggml-htp-v73.so（DSP applet）、libcdsprpc.so（ABI 常量参考）
```

## 关键环境变量

| 变量 | 作用 |
|---|---|
| `GGML_HEXAGON_OPSTAT` | `1` = 退出时打印按算子/按形状聚合的 DSP 耗时表（见 `docs/npu-split.md §14`） |
| `GGML_HEXAGON_PROFILE` | `1`/`2`/`3` = 逐 op 原始 dump（量大，只在排查单点时用） |
| `SD_DIT_SPLIT_PLAN` | 切点，如 `11,22` |
| `SD_DIT_LAYER_FIRST/LAST` | 本段层区间 |
| `SD_DIT_RELEASE_TE` | `1` = 编码完释放 TE（只适合一次性跑；常驻服务必须关） |
| `SD_DIT_DUMP_DIR` | 落 `x_/joint_/vel_<step>.bin`，用于对账 |

## 已知限制

- **v73 单 CDSP 核**：两进程同时算会算力对折，多进程只扩容不加速。
- **权重流式已彻底移除**（`relinquish/restore`、`MAPBUDGET`、`RPCMEM_FREE` 全部删除）。
  它能把 3.9 GiB 塞进 2304 MB 窗口，但 256² 每步要搬 6.9 GB、8 步共 54.9 GiB；分段之后完全不需要。
  代价：**单进程跑整模型这条路没有了**——权重 > 2304 MB 会直接 map 失败，必须走分段。
- **512² 的 VAE 152 s**（占端到端 35%）：疑似中间层激活超 VTCM（8 MB）落 DDR，未深挖。
- `--steps` 会被 `--sigmas <preset>` 覆盖（Turbo 的 preset 写死 8 步）。
- 没有 root、没有 udmabuf，VAE 只能走 CPU 或 HTP（本仓库只有这两个后端编进来）。

## 硬件/工具链

骁龙 8 Gen 2（v73）· 一加 Ace 2 Pro（PJA110）· NDK r26d · cmake 3.22.1 · JDK 17（不需要，纯 native）

## 相关

姊妹项目 [llamasplit](../llamasplit)：llama.cpp 的按层切分，同样的 IOVA 窗口约束，
那边每段 1.15~1.5 GB。
