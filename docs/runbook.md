# Runbook

> 本文档只写**当前可用**的流程。历史上被推翻的做法（单进程权重流式、OpenCL 出图、
> 手写 sd-cli 长命令带流式环境变量等）收在文末「附录：已废弃做法」，仅作背景，
> 不要照着执行。

全部命令在 **`D:/sd-split`** 下执行（Windows 侧），设备侧工作目录是
`/data/local/tmp/qisplit`（权重、二进制、applet 都在这棵树里）。

```bash
ADB=D:/android-tools/sdk/platform-tools/adb.exe
export MSYS_NO_PATHCONV=1
```

> adb.exe 是 Windows 程序：**本地路径必须写 `D:/...`**，`/d/...` 会 `cannot stat`。
> 推上去的二进制要 `chmod 755`，否则报 `CHAIN exit=126`。

## 1. 构建

```bash
./fetch_deps.sh                       # 拉 sd.cpp master + pin 住的 ggml + OpenCL 头
BUILD=build-hex ./build.sh hexagon    # NPU 版（-DGGML_HEXAGON=ON），约 12 min
./build.sh cpu                        # 纯 CPU 对照（可选）
```

构建三坑（一次踩完永久有效）：

- **ggml 必须是 sd.cpp pin 的 commit**：随手拉上游 master 缺 `ggml-graph.h`，直接报错。
  `fetch_deps.sh` 会自己查 API 拿 sha。
- **hexagon 版不需要 Hexagon SDK**：`ggml/src/ggml-hexagon/sdk-shim/` 顶替
  （fastrpc/rpcmem/dspqueue 全是 `dlopen`+`dlsym`）。别为它去装 SDK。
- **JOBS=4~6**：`-j 24` 会 ninja+clang 死锁。

## 2. 部署（设备侧一次性准备）

```bash
"$ADB" shell "mkdir -p /data/local/tmp/qisplit/models /data/local/tmp/qisplit/lib"

# 权重（D 盘根目录 → 设备）
"$ADB" push "D:/qwen-image-2.1-turbo-Q4_K_M.gguf"      /data/local/tmp/qisplit/models/dit.gguf
"$ADB" push "D:/Qwen3VL-8B-Instruct-Q4_K_M.gguf"       /data/local/tmp/qisplit/models/te.gguf
"$ADB" push "D:/qwen-image/models/qwen_image_2.1_vae_bf16.safetensors" /data/local/tmp/qisplit/models/vae.safetensors

# 二进制（构建产物 strip 后推上）
STRIP=D:/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe
cp build-hex/bin/sd-cli /tmp/sd-cli-hex && "$STRIP" /tmp/sd-cli-hex
cp build-hex/bin/sd-server /tmp/sd-server-hex && "$STRIP" /tmp/sd-server-hex
"$ADB" push /tmp/sd-cli-hex    /data/local/tmp/qisplit/sd-cli-hex
"$ADB" push /tmp/sd-server-hex /data/local/tmp/qisplit/sd-server-hex

# DSP applet（llama-hex 编好的，收在 ref/hexagon/；与主机 ggml 版本无关，不用重编）
"$ADB" push ref/hexagon/libggml-htp-v73.so /data/local/tmp/qisplit/lib/

# 脚本
"$ADB" push tools/chain.sh tools/serve.sh /data/local/tmp/qisplit/

"$ADB" shell "chmod 755 /data/local/tmp/qisplit/sd-cli-hex /data/local/tmp/qisplit/sd-server-hex /data/local/tmp/qisplit/chain.sh /data/local/tmp/qisplit/serve.sh"
```

## 3. 一次性出图：`chain.sh`

三段链定版入口：末段（层 22..32，持采样器）spawn 前驱（11..22），后者再 spawn（0..11），
三进程全驻留（每段 ≈1450 MiB，一次 map 到底、权重零搬运），段间 socketpair 乒乓。
所有调优项（RoPE/FA/conv-direct/VAE tiling）**脚本里已带定版默认值**，不用手写环境变量。

```bash
"$ADB" shell "cd /data/local/tmp/qisplit && PROMPT='a lovely cat' W=512 H=512 sh chain.sh t1"
# 产物：out_t1.png
```

可覆盖的常用变量（全部有安全默认值）：`PROMPT / SEED / W / H / STEPS`；
回退开关见脚本头注释（如 `VAE_BACKEND=CPU`、`SD_ROPE_NATIVE=0`、`FLASH_ATTN=off`）。

**末段的后端是 `diffusion=HTP0,vae=HTP0,te=CPU`**：VAE 走 HTP（CPU 58~67 s → 1.6~7.8 s）、
TE 留在 CPU（v73 上 Adreno TE 慢一倍；V75 上数值还会错，见 `docs/npu-port.md`）。

## 4. 常驻服务：`serve.sh`

`sd-server` 当末段 + 两个常驻 worker。首次请求 87.8 s，**第二请求起 256² 45.7 s / 512² 56.9 s**，
换分辨率不用重启。

```bash
# PC 直连手机 IP（默认监听 0.0.0.0:18080）
"$ADB" shell "cd /data/local/tmp/qisplit && setsid nohup sh serve.sh 18080 >/dev/null 2>&1 </dev/null &"

# 或只走 adb forward（本机访问）
"$ADB" shell "cd /data/local/tmp/qisplit && LISTEN_IP=127.0.0.1 setsid nohup sh serve.sh 18080 >/dev/null 2>&1 </dev/null &"
"$ADB" forward tcp:18080 tcp:18080
```

请求走 sd.cpp 的 `/sdapi/v1/txt2img`；PC 侧也可以用 `python tools/webui.py` 当前端。
注意：**常驻模式下 TE 不释放**（后续请求还要编码 prompt），末段常驻约 4.4 GB —— 与 9B 混推
同跑会抢内存（见 §6）。

## 5. 看进度 / 取图

```bash
"$ADB" shell "cd /data/local/tmp/qisplit && tr '\r' '\n' < run.log | grep -aE 's/it|completed' | tail -5"
"$ADB" pull /data/local/tmp/qisplit/out_t1.png D:/out_t1.png
```

进度条用 `\r` 刷新，直接 `tail` 会糊成一行，先 `tr '\r' '\n'`。

## 6. 关闭与内存

```bash
"$ADB" shell "am force-stop dev.pocketmodel.studio"   # 连带停掉 App 起的服务
```

停自起的 `serve.sh`/`chain.sh`：按 PID kill（TERM 即可），**不要 `pkill -f`**（会杀掉 adb shell
自己）、**不要 `kill -9`**（fastrpc PD 泄漏 `pdId=2 err 1009`，只能重启设备）。

跑 Qwen-Image 前先停 9B 混推：TE 4.4 GB + DiT 3.9 GB + VAE 0.6 GB ≈ 9 GB，两者抢内存直接 OOM。

## 7. 规划分段（换模型 / 改切点时才用）

```bash
python tools/gguf_info.py   <dit.gguf>            # 结构 / 逐层字节
python tools/plan_stages.py <dit.gguf> --k 3      # 切点 + 边界代价
```

`plan_stages.py` 给的是「每段权重 + 每步跨段字节」。CPU 后端且内存够时它会给出 K=1，
这是正确答案——切分不加速。

---

## 附录 A：关键环境变量（脚本已内置，手动调参时用）

| 变量 | 定版值 | 作用 |
|---|---|---|
| `GGML_HEXAGON_OPQUEUE=1` | 开 | batch 串行（默认 8 会增大 IOVA 压力） |
| `GGML_HEXAGON_MBUF=128M,128M` | 开 | 权重块切小（默认 512M） |
| `SD_ROPE_NATIVE=1` | 开 | 原生 M-RoPE 内核：5.07 → 1.46 s/it |
| `SD_FLASH_ATTN_MAX_KV=0` | 开 | 放开 FA token 门槛（512² 也走 HMX；⚠️ KV 很长时数值漂） |
| `--conv-direct auto,vae=on` | 开 | 强制 VAE 用 HTP 原生 CONV_2D：17.43 → 1.55 s |
| `--vae-tiling --vae-tile-size 320` | 开 | 512² 最后一档 conv 上不了 DSP：VAE 99 → 7.8 s |
| `SD_DIT_RELEASE_TE=1` | chain 开 / serve 关 | 编码完释放 TE（只适合一次性跑） |
| `SD_DIT_DUMP_DIR=<dir>` | 空 | 落 `x_/joint_/vel_<step>.bin` 对账用 |

## 附录 B：已废弃做法（背景，勿照做）

- **单进程 + 权重流式（已整体移除）**：旧 runbook 的 `GGML_HEXAGON_MAPBUDGET` /
  `GGML_HEXAGON_RPCMEM_FREE` / `[stream]` 日志只属于那条路。它靠 LRU 换出把 3.9 GiB
  塞进 2304 MB 窗口，但 256² 每步搬 6.9 GB、8 步 54.9 GiB；分段之后权重零搬运，完全不需要。
  代价：**单进程跑整模型这条路没有了**——权重 > 2304 MB 会直接 map 失败，必须走分段。
- **OpenCL 出图**：后端能跑但**数值是错的**（与 CPU 逐像素 MAE 54），只曾用来做链条对照。
  v73 上 Adreno TE 数值对但慢一倍；V75 上 Adreno TE 直接算错（彩色条纹）。
- **旧的工作目录 `D:/qwen-image-split`**：是 12-11 之前的快照，ggml_extend/ggml_runner 已落后
  于本仓库，只在查历史日志时有价值。当前一切以 `D:/sd-split` 的 `overlay/` 为准。
