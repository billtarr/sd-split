# Runbook

全部命令在 `D:/qwen-image-split` 下执行，`ADB` 见文末。

## 1. 首次准备

```bash
./fetch_deps.sh          # sd.cpp 源码 + pin 住的 patched ggml + OpenCL 头
./build.sh               # CPU 版 → build-android/bin/sd-cli
./build.sh opencl        # OpenCL 版 → build-android-cl/bin/sd-cli（要 GPU 提速就用这个）
BUILD=build-hex ./build.sh hexagon   # NPU 版 → build-hex/bin/sd-cli（-DGGML_HEXAGON=ON）
```

三个坑：

- **ggml 必须是 sd.cpp pin 的那个 commit**。随手拉 `leejet/ggml` 的 master 会报
  `Patched ggml requires ggml-graph.h`——master 是上游版，缺这个头。`fetch_deps.sh` 会自己查 API 拿 sha。
- **OpenCL 的 cmake 变量是混合大小写** `OpenCL_INCLUDE_DIR` / `OpenCL_LIBRARY`。
  给全大写的 `OPENCL_INCLUDE_DIR` 不报错，但会被 `find_package` 忽略，最后报
  「Could NOT find OpenCL (missing: OpenCL_LIBRARY OpenCL_INCLUDE_DIR)」。
- **hexagon 版不需要 Hexagon SDK**：`ggml/src/ggml-hexagon/sdk-shim/` 顶替（fastrpc/rpcmem/dspqueue
  全是 `dlopen`+`dlsym`，只需要 ABI 类型和编译期常量）。别为了它去装 SDK。

## 2. 部署

```bash
ADB="C:/Users/normin/WorkBuddy/2026-10-27-09-59-46/tools/platform-tools/adb.exe"
export MSYS_NO_PATHCONV=1

"$ADB" shell "mkdir -p /data/local/tmp/qisplit/models"
"$ADB" push "D:/qwen-image-2.1-turbo-Q4_K_M.gguf"     /data/local/tmp/qisplit/models/dit.gguf
"$ADB" push "D:/Qwen3VL-8B-Instruct-Q4_K_M.gguf"      /data/local/tmp/qisplit/models/te.gguf
"$ADB" push "D:/qwen-image/models/qwen_image_2.1_vae_bf16.safetensors" /data/local/tmp/qisplit/models/vae.safetensors

STRIP=D:/ndk/android-ndk-r26d/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe
cp build-android-cl/bin/sd-cli /tmp/sd-cli-cl && "$STRIP" /tmp/sd-cli-cl
"$ADB" push /tmp/sd-cli-cl /data/local/tmp/qisplit/sd-cli-cl
"$ADB" shell "chmod 755 /data/local/tmp/qisplit/sd-cli-cl"

# NPU 版 + DSP applet（applet 是 llama-hex 编好的，已收进 ref/hexagon/，必须进设备 lib/）
cp build-hex/bin/sd-cli /tmp/sd-cli-hex && "$STRIP" /tmp/sd-cli-hex
"$ADB" push /tmp/sd-cli-hex /data/local/tmp/qisplit/sd-cli-hex
"$ADB" shell "mkdir -p /data/local/tmp/qisplit/lib && chmod 755 /data/local/tmp/qisplit/sd-cli-hex"
"$ADB" push ref/hexagon/libggml-htp-v73.so /data/local/tmp/qisplit/lib/libggml-htp-v73.so
# ↑ md5 应为 5037e15787c84bea0c6313b2448911d0（965384 字节）。
#   原件来自 D:/llama-hex/build-adb/ggml/src/ggml-hexagon/libggml-htp-v73.so，
#   与主机 ggml 版本无关，换 sd.cpp/ggml 不用重编 applet。
```

`adb.exe` 是 Windows 程序：**给的本地路径必须是 `D:/...`**，`/d/...` 会 `cannot stat`。

## 3. 出图（定版）

### 3.1 NPU（ggml-hexagon，推荐）

```bash
"$ADB" shell "cd /data/local/tmp/qisplit && \
  LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
  GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_MBUF=128M,128M \
  GGML_HEXAGON_MAPBUDGET=1400M GGML_HEXAGON_RPCMEM_FREE=1 \
  ./sd-cli-hex --diffusion-model models/dit.gguf --llm models/te.gguf --vae models/vae.safetensors \
  -p 'a lovely cat' --sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0 \
  -H 256 -W 256 --backend 'diffusion=HTP0,vae=CPU,te=CPU' -o out.png"
```

四个环境变量都用得上，**少一个就崩或变慢**：

| 变量 | 作用 | 少了会怎样 |
|---|---|---|
| `ADSP_LIBRARY_PATH=lib` | 让 DSP 找到 applet `libggml-htp-v73.so`（已推到 `lib/`） | 报 `0x80000406` skel 找不到 |
| `GGML_HEXAGON_MBUF=128M,128M` | 把权重块切小（默认 512M）→ 流式粒度细 | 每次换出要搬 492 MB，抖动大 |
| `GGML_HEXAGON_MAPBUDGET=1400M` | 权重常驻上限，超了按 LRU 换出 | 不设 = 全量 mmap → 2.3 GiB 窗口直接 `EFAULT` |
| `GGML_HEXAGON_RPCMEM_FREE=1` | 换出时真的销毁 dma_buf 归还 IOVA | 只 unmap 不归还 → 累计到 ~1.9 GB 后必失败 |
| `GGML_HEXAGON_OPQUEUE=1` | batch 串行 | 默认 8；流式下并发会增大 IOVA 压力 |

- **512² 要把预算降到 `1100M`**（1400M 会在第 6 步左右 mmap 失败）。
- `--sigmas qwen-image-2.1-turbo` 是 Turbo 的固定 8 步表，**步数由 sigma 个数决定**，
  所以 `--steps 8` 只是对齐语义。
- `--backend` 里 **`vae=CPU`、`te=CPU` 都不能省**：OpenCL 上 3D VAE 会输出粉色色块
  （`architecture.md` §1）；v73 上 TE 走 Adreno 数值对但慢一倍。
- 长任务用 `nohup ... > run.log 2>&1 &`，否则 adb 连接一断就带走子进程。

### 3.2 OpenCL（对照用）

```bash
"$ADB" shell "cd /data/local/tmp/qisplit && ./sd-cli-cl \
  --diffusion-model models/dit.gguf --llm models/te.gguf --vae models/vae.safetensors \
  -p 'a lovely cat' --sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0 \
  -H 512 -W 512 --backend 'diffusion=GPUOpenCL,vae=CPU,te=CPU' -o out.png"
```

⚠️ 这个后端**出图数值是错的**（与 CPU 逐像素 MAE 54），只用来做链条对照。

## 4. 看进度 / 取图

```bash
"$ADB" shell "cd /data/local/tmp/qisplit && tr '\r' '\n' < run.log | grep -aE 's/it|completed|\[stream\]' | tail -5"
"$ADB" pull /data/local/tmp/qisplit/out.png "D:/qwen-image-split/out.png"
```

进度条用 `\r` 刷新，直接 `tail` 会看到一整行糊在一起，先 `tr '\r' '\n'`。
`[stream]` 行是流式换出统计（`relinq/restore/copy/wait`），判断「慢在搬运还是慢在算力」看它。

## 5. 规划分段（只在装不下时用）

```bash
python tools/gguf_info.py    /d/qwen-image-2.1-turbo-Q4_K_M.gguf          # 结构 / 逐层字节
python tools/plan_stages.py  /d/qwen-image-2.1-turbo-Q4_K_M.gguf --k 3    # 切点 + 边界代价
```

`plan_stages.py` 给的是「每段权重 + 每步跨段字节」。**先看结论那一行**：
CPU 后端且内存够时，它会给 K=1，这是正确答案——切分不加速。

## 6. 关掉设备上的服务

```bash
"$ADB" shell "am force-stop dev.pocketmodel.studio"   # 会连带停掉 llama_split 子进程
```

跑 Qwen-Image 前先做这一步：TE 4.3 GB + DiT 4.0 GB + VAE 0.6 GB ≈ 9 GB，
和 9B 混推（5~8 GB）抢内存会直接 OOM 被杀。

---

`ADB="C:/Users/normin/WorkBuddy/2026-10-27-09-59-46/tools/platform-tools/adb.exe"`（项目内那份已随旧目录删除，见台账 `E-015`）
