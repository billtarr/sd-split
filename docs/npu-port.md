# 把 DiT 放到 NPU 上：可行性评估

> 背景：本项目当前 DiT 走 Adreno OpenCL（256² 8 步 157 s）。用户提出「用 ggml-hexagon 后端，同时吃 gguf 和 NPU」。
> 本文只记**结论与证据**，不含操作步骤（跑法见 `runbook.md`）。台账对应 `I-021`，历史禁令见 `H-004`。

## 0. 结论（2026-11-11 实测：已打通，图正确）

ggml-hexagon 后端已移植进 sd.cpp 并**跑出正确图像**，Q4_K_M 的 gguf **原样上 HTP**，不需要反量化。

**定版配置**（v73 / 8 Gen 2 / 单进程）：

```
GGML_HEXAGON_OPQUEUE=1 GGML_HEXAGON_MBUF=128M,128M \
GGML_HEXAGON_MAPBUDGET=1400M GGML_HEXAGON_RPCMEM_FREE=1
--backend 'diffusion=HTP0,vae=CPU,te=CPU'
```

| 尺寸 | s/it | generate | 对比 sd.cpp CPU | 对比 OpenCL 混跑 |
|---|---|---|---|---|
| 256² | **7.49** | **129.0 s** | 48.5 s/it / 437 s → **6.4×** | 157 s → 1.2× |
| 512² | **34.6** | **492.8 s** | 90.9 s/it → **2.6×** | 61~65 s/it → 1.8× |

**正确性判据（这是整件事的核心，不是速度）**：
256² 与 CPU 出图逐像素比 **MAE 1.23/255、最大差 88、14.7% 像素完全相同**。
（同一比法下 OpenCL 出图 MAE 54 —— 与台账里「OpenCL 算错」一致，可作为反向对照。）

速度结论要诚实：**比 CPU 快 2.6~6.4×，但并不可观**。瓶颈是 ggml-hexagon 的**逐 op 调度**
（llama 侧 pp 天花板 250~270 t/s，而 QNN 整图 AOT 能到 4000，见 `L-017`/`H-004`）。
这条路的价值是 **「直接吃社区 gguf + 单二进制可移植」**，不是「最快」。要最快仍是 QNN。

⚠️ 512² 的 IOVA 预算必须降到 **1100M**（1400M 会在第 6 步左右 `fastrpc_mmap` EFAULT）：
512² 的激活更大，留给权重的窗口相应更小。

## 1. `--rpc-servers` 路线：封闭

先试的是最省事的一条：sd.cpp 原生支持 `--rpc-servers`，设备上跑 `ggml-rpc-server -d HTP0`，
把 HTP 当成远端设备注册进来，sd.cpp 零改动。

**结果：编译期就被拦住。**

```
ggml/include/ggml-rpc.h:14:1: error: static assertion failed
  due to requirement 'GGML_OP_COUNT == 105'
  expression evaluates to '101 == 105'
```

根因不是版本号差两个 minor，而是 **op 枚举顺序不同**：

| 位置 | sd.cpp 的 ggml 0.27.0（leejet fork） | llama.cpp 的 ggml 0.29.0 |
|---|---|---|
| idx 91 | `GGML_OP_UNARY` | `GGML_OP_TOP_K_MASK` |

llama.cpp 在中间**插入**了 `GGML_OP_TOP_K_MASK`，后面全部 +1。改掉 `static_assert` 只是把
「编译失败」换成「rpc 两端编号错位」——**这条路关闭，没有配置层面的绕法。**

**但把差异量算清楚很重要**（一开始我写得过重了）：全部差异就是 1 个 op。

| | sd.cpp ggml 0.27 | llama.cpp ggml 0.29 |
|---|---|---|
| op 总数（含 COUNT） | 102 | 103 |
| 只在 llama 有 | — | `GGML_OP_TOP_K_MASK`（idx 91） |
| 只在 sd 有 | 无 | — |
| 编号漂移 | **只有 idx ≥ 91 的 11 个**，`UNARY` 91→92 起 | |

DiT 实际用到的 op 编号**全部相同**（`MUL_MAT` 29、`RMS_NORM` 25、`SOFT_MAX` 46、`ROPE` 48、
`FLASH_ATTN_EXT` 74、`ADD` 2、`MUL` 7、`CPY` 34、`CONT` 35、`VIEW` 37、`PERMUTE` 38、
`RESHAPE` 36、`SCALE` 32、`CONCAT` 22、`PAD` 64）。唯一踩线的是 `UNARY`（91 vs 92）。

这有两个含义：
- **rpc 依旧不行**：协议是整体版本化的，不是「用不到就不影响」。
- **移植源码几乎不受影响**：host 与 backend 编在同一份枚举上，天然一致。0.27 里没有
  `TOP_K_MASK`，所以在枚举尾部追加一个（保持前 101 个 op 的编号不变）就对齐了 —
  见 `ggml/include/ggml.h`。

## 2. HTP 侧本身是好的

设备实测（v73 / 8 Gen 2，`test-backend-ops`，后端 `HTP0` 正常起来，`domain=cdsp:3`）：

| 算子 | 结果 |
|---|---|
| `RMS_NORM` | 全 OK（f32 / 各种 `ne`） |
| `FLASH_ATTN_EXT`（f16 KV，hsk=64/128/256，kv=1024/2048） | **12/12 OK** |
| `FLASH_ATTN_EXT`（`type_K=q4_0`） | `not supported` —— DiT 用 f16 激活，不受影响 |
| `MUL_MAT`（Q4_K） | HTP 有原生 `HTP_TYPE_Q4_K` 内核 + repack 支持（源码确认） |

**Q4_K_M 可以原样上 HTP，不需要先反量化成 f16。** 这是这条路相对 QNN 的最大优势。

## 3. 移植方案（唯一可行，已落地）

不走 rpc，把 backend 源码搬进 sd.cpp 的 ggml。已完成的搬迁与适配：

| 部分 | 状态 |
|---|---|
| DSP 侧 applet `libggml-htp-v73.so` | **现成，与 ggml 版本无关**，直接用 `llama-hex` 编出来的那份（965 KB），运行时由 `ADSP_LIBRARY_PATH` 加载 |
| host 侧 `ggml-hexagon.cpp` + `htp-*.cpp` | 已搬入 `ggml/src/ggml-hexagon/`（13.6 k 行），适配 0.27 |
| IDL stub | `htp_iface.h` + `htp_iface_stub.c` 从 llama-hex 的 build 里取出**直接进源码树**，不再需要 `qaic` 工具 |
| SDK 头 | 无 SDK，用 `sdk-shim/` 的 7 个最小头（`AEEStdDef/AEEStdErr/remote/domain/rpcmem/dspqueue/HAP_farf`）。全部函数都走 `dlopen + dlsym`，只需要类型 |
| CMake | 自带 `CMakeLists.txt`，不建 skel、不装 SDK；`build.sh hexagon` 打开 `-DGGML_HEXAGON=ON` |

leejet 的 ggml 本身就带六成基础：`GGML_USE_HEXAGON` 注册钩子（`ggml-backend-reg.cpp:62/154`）、
`include/ggml-hexagon.h`、`ggml_add_backend(Hexagon)` 都在，只是 `ggml/src/ggml-hexagon/` 目录空着。

### 实际改动的 5 处（0.29 → 0.27）

| 问题 | 处理 |
|---|---|
| `ggml_backend_buffer_i` 少 3 个槽位 | 删掉 `.set_tensor_n/.get_tensor_n/.cpy_tensor_n` 的初始化，两个函数标 `[[maybe_unused]]` |
| 缺 `GGML_OP_TOP_K_MASK` | 在 `ggml.h` 枚举**尾部**追加（前 101 个编号不变），`ggml.c` 的两个名字表 + `static_assert` 从 101 改 102 |
| 缺 `ggml_rope_get_truncate()` | 直接读 `op_params[3]`。两边布局逐字节相同，0.27 那格是恒 0 的 `n_ctx`、0.29 是 `no_trunc`，语义恰好对齐 |
| `uint64` 类型 | shim 里必须是 `unsigned long long`（SDK 原样），用 `uint64_t` 会让 `htp_iface_hwinfo` 参数类型不匹配 |
| `HAP_debug_v2` | 本机没有任何库导出它（设备上在 `libcdsprpc`），shim 里给 `static inline` 实现 |

注意：`llama-hex` 是 llama.cpp 的树，sd.cpp 用的是 **leejet fork 的 ggml**（带 `ggml-graph.h`）。
两者不能互换，只能「把 backend 搬过去」。

## 4. 两个必须先说的风险

**① 历史禁令只覆盖 UNet，不覆盖 DiT，但也不能当作已翻案。**
`H-004`（禁止）的实测对象是 **SD1.5 的 UNet**（`conv` 密集），当时结论是
HTP 全开 corr 0.3024 / 把 IM2COL 判给 CPU 仅 0.3616，仍错。已排除 staging、cache sync、
stride、batching、图缓存、同步、融合、f32、flash attention（`90-archive/v73-NPU-结论-2026-10-29.md` §5）。

- **支持翻案**：DiT 是单流 transformer，**没有 conv / im2col / GROUP_NORM**，恰好是当年错误最集中的三处。
- **反对翻案**：IM2COL 回 CPU 后仍 0.36，说明**存在第二个未定位的错误源**。若它来自通用路径
  （多段子图边界、外部张量回写），DiT 一样会中。

**结论：已经过了这道关** —— 2026-11-11 实测 DiT 在 HTP 上出图与 CPU 参考 MAE 1.23/255，
说明「第二个未定位的错误源」在 DiT 路径上不存在。`H-004` 的禁令**只对 UNet 有效，不要外溢**。
（当年 UNet 的具体成因仍未定位，属另一条线。）

**② IOVA 窗口是「单进程」限制 —— 跨进程可以叠加，这是突破 2.3 GiB 的唯一正路。**

DiT 权重 3.9 GiB > fastrpc 单 PD 的 IOVA 窗口 2.3 GiB。这**只约束单个进程**：
一个进程一个 PD、一个 VA 窗口。两条路：

| 路线 | 做法 | 结果 |
|---|---|---|
| 单进程 + 权重流式（已落地，`I-022`） | `GGML_HEXAGON_MAPBUDGET` 限量常驻 + `RPCMEM_FREE` 真正归还 IOVA，按 LRU 换入换出 | 全权重都过 NPU，代价是每步搬 ~6.9 GB（见下）。**能跑，但不该是终点** |
| **多进程分段（下一步，`I-024`）** | 按层切 K 段，每段一个进程 / 一个 PD / 各自 2.3 GiB 窗口 | 每段 < 2.3 GB → **段内一次 map 到底、零搬运** |

单进程流式的实测代价（256²）：8 步里 `relinq=268 restore=240`，累计搬 **54.9 GiB**、
`wait` 30 s（占 129 s 的 23%）。**这就是分段要省掉的那部分。**

**「换地址代替搬运」的真实边界（`I-024`，别误用）**：DSP 侧 VA 按「唯一映射集合」记账 ——
同一 `(fd,offset,len)` 反复 map/unmap 不新增（`L-026`：小模型轮转 572 次、累计 39.7 GB 不耗尽），
但**每个不同映射各占一份**，所以大模型 18 个 buffer 合计 4.5 GB 会线性累积到 2.6 GB 就撞墙。
→ 要让 DSP 看到 3.9 GiB，无论切几次 map，唯一映射的总字节必然是 3.9 GiB > 2304 MB，
**单进程里「不搬」与「全权重可见」数学上不可兼得**。反倒是**分段之后“换地址”天然成立**：
每段装得下，一次 map 到底，`MAPBUDGET` / `RPCMEM_FREE` 全部可以关掉 —— 那才是真正的零搬运。

**分段不损失算力**：diffusion 每步是段间**串行依赖**，不是并发，单 CDSP 核在任一段都吃满。
代价只有段边界 IPC（`L-053`：每边界 ~15 ms + 激活 12.6 MB/边界 @256²）→ 8 步 × 2 边界 ≈ 0.24 s。
（与 `L-029`「两进程算力对折」不矛盾 —— 那测的是两进程**同时**在算。）

⚠️ **并发红线（`I-023` 教训）**：2026-11-11 用两个**同参数流式**的 `sd-cli-hex` 并发，
50 s 后整机 hard hang（黑屏/发热/电源键无响应，需放电）。而 `L-028` 的并发是「一个大模型已静止 +
一个小模型」—— 问题不在「两个 PD」，在**两个进程同时跑高频 `rpcmem_alloc/free` 抖动**
（本工程每进程每步 268 次）。并发测试必须：先起一个、确认静止，再起第二个；不要两个进程同一秒进入抖动。


## 5. 性能实测（诚实版，2026-11-11）

| 后端 | 256² 8 步 | 512² 8 步 |
|---|---|---|
| CPU（sd.cpp 原生） | 48.5 s/it · 437 s | 90.9 s/it |
| Adreno OpenCL（DiT 在 GPU，数值错） | 157 s 总计 | 61~65 s/it |
| **ggml-hexagon / HTP0（本次）** | **7.49 s/it · 129 s** | **34.6 s/it · 493 s** |
| QNN AOT 分段链（另一条线，`20-qwen-image-qnn`） | — | 32.1 s/it · 262 s |

结论：
- 比 CPU 快 **2.6~6.4×**，512² 上和 QNN 链**基本持平**（34.6 vs 32.1 s/it）；
- 但 512² 总时长 493 s vs QNN 262 s —— 差在**每步要搬 6.9 GB 权重**，
  去掉流式（多进程分段）后才有机会追平；
- 瓶颈是**逐 op 调度**，不是 HTP 算力（llama 侧 pp 天花板 250~270 t/s，QNN 整图 4000，
  见 `L-017`/`H-004`）。这条路的价值是**直接吃社区 gguf + 单二进制可移植**，不是「最快」。

## 6. 进度与下一步

已完成（原来计划的三步全部走完）：
1. ✅ 搬 backend → 编过 → `--list-devices` 里有 `HTP0`；
2. ✅ 256² 跑通，图与原 CPU 参考 MAE 1.23/255；
3. ✅ 8 步 s/it、512² 都拿到数（见 §5）。

下一步（2026-11-11 修正，用户指定方向）：
1. **多进程分段**：K 段 × 各自 PD，段内 `< 2.3 GB` → 一次 map 到底、**零搬运**。
   依据 `llamasplit/L-028`（两 PD 并存实测成立）+ `I-024`（分段后换地址才成立）。
   落地顺序：① 先确认 DiT 层的切分点（每段权重 = 层数 × ~117 MiB/层 @Q4_K_M）；
   ② sd.cpp 侧要新增「只加载 [first,last] 层 + 段边界激活透传」，与 `llamasplit` 的
   `LLAMA_LAYER_SPLIT` 同构（那边改的是 `llama-model.cpp` / `qwen35.cpp` + `done_getting_tensors(partial)`）；
   ③ 段间用 dma_buf fd（`SCM_RIGHTS`）或 shm 传 12.6 MB 激活。
2. **先做低风险的那半**（不必等分段）：把 LRU 换成「严格层序遍历」，每步搬运从 6.9 GB
   压到理论最小 ~2.5 GB（`I-024` ⑤）。
3. 单进程侧的调优空间已经很小：预算 1400M→1600M 反而 mmap 失败、s/it 几乎不变
   （7.45 vs 7.49）→ 确认是算力/调度瓶颈，不是 IO。**这条结论在流式前提下成立，不代表分段没收益。**


## 7. 移植踩到的 6 个坑（都靠反汇编或运行时自检定位）

| # | 症状 | 真因 | 修法 |
|---|---|---|---|
| 1 | `dspqueue_write failed: 0x48`，logcat 里 `fastrpc_buffer_ref failed (fd -880525312)` | `struct dspqueue_buffer` 字段序**不是** `{ptr,fd,offset,size,flags}`，而是 `{fd,size,offset,flags,ptr}`（库把 `ptr` 低 32 位当 fd） | 反汇编 `libggml-hexagon.so` 的 `push()` 拿到真实偏移；flags = `0x80|0x10` |
| 2 | `dspqueue_read failed: 0xc` 直接 abort | `AEE_EEXPIRED` 我在 shim 里猜成 `0x80000400+n`，真值是 **12**，等于「超时请重试」 | 按真实 `AEEStdErr.h` 重写错误码表 |
| 3 | `vmem 2097151998951424` 之类的天文数字 | `parse_size()` **默认单位是 MiB**，`GGML_HEXAGON_VMEM=2000000000` 被理解成 2000000000 MiB | 必须带单位：`1400M` / `2G` |
| 4 | `fastrpc_mmap ... EFAULT`，`mapped 958 + need 492 < budget` 却失败 | `fastrpc_munmap` **不把 IOVA 还给 DSP 分配器**，`reap` 只是 unmap | `GGML_HEXAGON_RPCMEM_FREE=1`，让 `relinquish()` 真的销毁 dma_buf |
| 5 | DSP 崩在 `op_matmul`，`Bad VA`，地址每次不同 | sd.cpp 的 staging 是 `std::swap(managed->buffer/data, staging->buffer/data)`，所以 `init_tensor` 登记的 tensor **不是**图上用的那个；`restore()` 换新块后没人重指 `data` → DSP 拿到已释放的旧地址 | 在 `mmap_tensor()`（每个真正上图的 tensor 必经）按地址区间补登记，`restore()` 时按 `saved_base` 区间重判并重指 |
| 6 | 收尾 `GGML_ASSERT(sbuf && sbuf->mem)` | `unclone_buffer()` 对已 relinquish（`mem==nullptr`）的 buffer 断言 | 加守卫：`mem` 为空直接返回 |

★ 方法论：**凡是不确定的 SDK 常量（请求号、scalar 编码、mmap flags、结构体字段序），
都去反汇编 `libcdsprpc.so` / 预编译的 `libggml-hexagon.so` 拿真值**，不要猜。
本文件里 1、2、4 三条都是这么定的。

