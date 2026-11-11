# Qwen-Image 2.1 Turbo · 分段推理现状与设计

> 最后更新：2026-11-11
>
> 这份文档讲三件事：**现在跑多快、瓶颈在哪、以及「按层切段」这件事到底该怎么做**。
> 所有数字都是 8 Gen 2（SM8550 / v73 / 23.6 GB RAM / 6 核）上的实测，不是估算。

---

## 0. 一句话

Qwen-Image 2.1 Turbo（32 层单流 DiT，Q4_K_M 3.9 GiB）在手机上**能跑，且不用切段**：
DiT 放 Adreno OpenCL、TE 和 VAE 放 CPU，256²/8 步 **157 s**、512² 见 §3.2。
切段只在「内存装不下」时才有意义——它不会让单张图更快。

## 1. 实测：三种后端配置（256²，8 步 Turbo，cfg 1.0）

| 配置 | TE | DiT | VAE | 总计 | 出图 |
|---|---:|---:|---:|---:|---|
| 全 CPU | 20.25 s | **376.71 s**（48.5 s/it） | 39.99 s | 436.95 s | ✅ |
| 全 OpenCL | 28.62 s | 99.00 s（10.46 s/it） | 124.53 s | 252.68 s | ❌ 粉色块 + 条纹 |
| **DiT=OpenCL，TE/VAE=CPU** | ~20 s | **99.03 s**（10.46 s/it） | 36.75 s | **157.42 s** | ✅ |

三条结论，每条都是「换掉一个变量」测出来的：

1. **DiT 放 OpenCL 是 4.6× 提速**（48.5 → 10.46 s/it），且数值正确。
2. **VAE 必须留 CPU**：同一张图、同一个 latent，VAE 走 OpenCL 直接输出色块。
   代价是慢 3.4×（36.75 → 124.53 s），但至少图是对的。这和之前 Qwen-Image 1.0 链上
   「3D VAE 在 OpenCL 卡死/算错」是同一个坑，只是这次表现为「出图但全烂」。
3. **TE 也留 CPU**：OpenCL 反而慢（28.62 vs 20.25 s）。8B 的 prefill 是带宽活，
   Adreno 的收益抵不过权重搬运。

启动命令：

```bash
./sd-cli-cl --diffusion-model models/dit.gguf --llm models/te.gguf --vae models/vae.safetensors \
  -p 'a lovely cat' --sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0 \
  -H 256 -W 256 --backend 'diffusion=GPUOpenCL,vae=CPU,te=CPU' -o out.png
```

设备名来自 `sd-cli-cl --list-devices` → `GPUOpenCL` / `CPU`。

### 1.1 512² 的现实：内存先撞墙，速度反而次要

同样配置换 512²（同一 prompt）就崩了，三次尝试：

| 配置 | 速度 | 结果 |
|---|---|---|
| 同 256² 的命令，只改 `-H 512 -W 512` | 61~65 s/it | **第 4 步后卡死**：内存 22.9/23.1 GB 满、swap 用掉 2.5 GB、CPU 771% idle（纯抖动），第 5 步推进不去 |
| `--mmap --vae-tiling` | 81.5 s/it，采样 **651.87 s** | 采样跑完；VAE tiling 超过 10 分钟没有结束，手动终止 |
| `--params-backend 'te=disk'` | 90.7 s/it | 能跑，但比 256² 按 token 比例（4×）预期的慢得多 |

原因是**权重驻留方式**：TE 4.3 GB + DiT 4.0 GB + VAE 0.6 GB 常驻匿名内存，
512² 的激活与 prefix cache 再叠上去就顶到 23 GB。三种绕法都要付代价：
`--mmap` 让权重页可回收但读盘变慢，`te=disk` 同理，`--vae-tiling` 把 VAE 拆块算。

**结论：512² 在这台机器上要 13 分钟以上，现阶段不实用。**
要跑 512² 得先解决内存——候选顺序：TAEQI 2.1 替换 VAE（解码 40 s → 秒级，且省 0.6 GB）、
换更小的 TE（Q4_0 / 3B 级）、或出完图后显式卸载 TE 再解码。


## 2. 内存账（实测日志）

```
total params memory size = 8951.71MB (VRAM 0.00MB, RAM 8951.71MB):
  text_encoders 4302.33MB(RAM), diffusion_model 4005.00MB(RAM), vae 644.38MB(RAM)
```

- DiT compute buffer 只有 **52 MiB（CPU）/ 63 MiB（OpenCL）**，512² 也才百 MB 级。
  → 峰值内存 ≈ 权重之和，切段能省的只有权重。
- 23.6 GB 的机器上这 9 GB 装得下，所以**当前配置不需要切段**。
  要切的场景是：8 GB 机型、或与别的模型共存（比如同时驻留一个 9B LLM）。

## 3. DiT 的结构与切点（`tools/gguf_info.py` / `tools/plan_stages.py`）

```
265 tensors · 权重 3.911 GiB · header 仅 0.03 MiB（元数据里没有 block 数，全靠张量形状反推）
32 × transformer_blocks，每层 117~125 MiB（avg 119.2）
单流块：img_norm1 → attn → img_norm2 → img_mlp（文本经 txt_in 投影后并入同一序列）
```

| 归属 | 张量 | 大小 |
|---|---|---:|
| 段 0 专属 | `txt_in.in_layer/out_layer` + `img_in` + `time_text_embed` | 83 MiB |
| 每步共用 | `modulation.1` | 44 MiB |
| 末段专属 | `norm_out.linear` + `proj_out` | 64 MiB |

K=3 的切法（VA 2048 MiB 那档的答案）：

```
stage 0: 层[0,10) 10 层  1338.6 MiB  含 txt_in/img_in/time_embed
stage 1: 层[10,21) 11 层 1295.0 MiB
stage 2: 层[21,32) 11 层 1371.4 MiB  含 norm_out/proj_out
边界代价：每步 36.00 MiB（1536 token × 3072 × f32），8 步共 288 MiB ≈ 9 ms/张
```

**边界代价比 LLM 高一个量级**：LLM 一次搬 16 KiB，DiT 一次搬 18 MiB（512² 时），
而且**每个 timestep 都要搬一遍**。K=3、20 步就是 1 GB 级拷贝。

## 4. sd.cpp 已经有的三件东西（不用自己造）

翻源码才发现，sd.cpp 早就把「分段」做进产品里了，只是没叫这个名字：

| 机制 | 位置 | 它解决什么 | 我们能怎么用 |
|---|---|---|---|
| `ggml_graph_cut` | `src/core/ggml_graph_cut.cpp` | 把整图按依赖自动切成 Segment，每段独立算 compute buffer、顺序执行 | 已经默认在跑（日志 `build cached graph cut plan`）；是「省峰值内存」的现成实现 |
| `--split-mode layer\|row` | `examples/cli` | `layer` = 整块 transformer block 分给不同 device；`row` = matmul 行切（**仅 CUDA**） | `--backend "diffusion=GPUOpenCL&cpu"` + `--split-mode layer` 就是按层分设备 |
| `--rpc-servers` | ggml-rpc | 把某段权重放到另一个进程/机器，走 TCP | 这才是「多进程 split」的现成通道，不用改 sd.cpp 的图构建 |

**这对我们意味着什么**：想要「多进程按层切」，最短路径不是自己写 driver，
而是 `rpc-server` + `--backend diffusion=<rpc 名>&cpu --split-mode layer`。
自己写 driver 的唯一理由是要跨后端异构 ABI（比如把一段做成 QNN 子图），
如果只是「CPU 装不下、把后面几层放到另一个进程」，RPC 够用，且是上游维护的。

但前提仍然是：**它只解决装不下**。在 8 Gen 2 上 DiT 全上 OpenCL 只要 4 GB 显存，
没有任何理由把层拆开——每拆一段就多一次 18 MiB 的往返 × 步数。

## 5. 下一步（按投入产出比）

| 优先级 | 事项 | 预期 |
|---|---|---|
| P0 | 512² 混合后端实测（DiT OpenCL） | 拿到目标分辨率的真实数字 |
| P0 | TE 结果缓存（`--conditioning-cache-size`） | 同 prompt 二次出图省 20 s |
| P1 | VAE 换 TAEQI 2.1（`docs/taesd.md`） | VAE 40 s → 秒级，画质略降 |
| P1 | 试 `--split-mode layer` + `CPU&GPUOpenCL` | 验证「部分层 GPU + 部分层 CPU」是否比全 GPU 快（预期更慢，但要量） |
| P2 | RPC 分段 PoC | 只在 8 GB 机型上才有意义 |
| P2 | Q6_K / Q8_0 DiT 对比 | 目前 Q4_K_M；Turbo 8 步对量化更敏感，值得测画质 |
