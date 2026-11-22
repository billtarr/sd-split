# DiT 层区间切分（多进程分段）

> 目标：让**全量 DiT 权重常驻 NPU**，且**不做权重搬运**。
> 状态：**第 1 层已实测通过** —— K=3（11/11/10）全驻留，8/8 步 velocity 与整模型**逐位相同**，
> 采样耗时 61.8 s → 40.3 s（−35%）。第 2 层（IPC/常驻在一台机器上跑成一条链）未做。

## 0. 为什么必须分段，而不是「换地址」

单进程一个 fastrpc PD，DSP 侧只有 **2304 MB** VA（`H-001`）。

关键机制（`llamasplit/docs/ledger.md` `L-026` 实测）：DSP 侧 VA 是**按「唯一映射集合」记账**的 ——
同一 `(fd, offset, len)` 反复 map/unmap **不新增**（小模型轮转 572 次、累计映射 39.7 GB 不耗尽），
但每个**不同**的映射各占一份，所以大模型 18 个 buffer 合计 4.5 GB 会线性累积到 2.6 GB 就 `0x1`。

推论：**要让 DSP 看到 3.9 GiB 权重，「不搬」与「全权重同时可见」在数学上不可兼得。**
无论怎么切 map、怎么换 offset，唯一映射的总字节必然还是 3.9 GiB > 2304 MB。

所以：
- **单进程**只能「搬」——现在的 `relinquish/restore`（`RPCMEM_FREE=1`，销毁 dma_buf 真还 VA + 拷回）
  已经是唯一解。代价 256² 8 步搬 **54.9 GiB**、`wait` 30 s（占 129 s 的 23%）。
- **多进程**才能扩容：每进程**独立 PD + 独立 SMMU domain + 独立 IOVA**（`L-028` 实测：
  进程 A 跑 4.5 GB 权重时启 B，B 正常出数、A 不受影响）。
- **切分之后「换地址」才成立**：每段权重 < 2.3 GB → 段内一次 map 到底，
  `MAPBUDGET` / `RPCMEM_FREE` 全部关掉，**零换入换出**。这才是省掉那 54.9 GiB 的地方。

⚠️ **并发红线**（2026-11-11 实测）：两个**同参数流式**的 `sd-cli-hex` 并发 → 50 s 后整机
hard hang（黑屏/发热/电源键无响应，需放电）。而 `L-028` 的并发是「一个大模型已静止 + 一个小模型」。
→ 问题不在「两个 PD」，在**两个进程同时跑高频 `rpcmem_alloc/free` 抖动**。
分段之后恰好消掉了这个抖动（不再流式），但要按「先起一个、确认静止、再起第二个」的节奏上。
（这是**另一条**雷：**fork 时父进程必须没碰过 backend**，见 §5「根因修正」与 llamasplit `L-067`。两条别混。）

## 1. 架构：采样器留在末段，前段只是函数服务

采样是 **ER-SDE 多阶段**格式（`src/runtime/denoiser.hpp` 主循环含 `stage_used`/`integration_points`
三阶修正），在外部 driver 里重写必然引入数值分叉。所以**不要让 driver 拥有采样器**：

```
段 0（layers 0..k）       段 1（layers k..32 + head + Sampler）
  (x, sigma) ──────────▶   收到 joint
  返回 joint  ◀──────────  跑自己的层 → velocity → 更新 x → 下一步
```

末段进程是**正常的 sd-cli 进程**，采样循环一行不改；它在每步开头向前一段要 joint。
前段进程是**无状态函数服务**：收 `(x, sigma)`、回 `joint`。

好处：① 采样器零改动，数值可复现；② 前段不需要知道 sigmas/步数；③ 加段只是加一环。

## 2. 已落地：层区间 + joint 落盘（第 1 层）

改的是 `src/model/diffusion/qwen_image_2_1.hpp`，全部由一个 env 开关控制，不动 pipeline：

| 变量 | 含义 |
|---|---|
| `SD_DIT_LAYER_FIRST` / `SD_DIT_LAYER_LAST` | 本段拥有的层区间 `[first, last)`，默认 `0..32` |
| `SD_DIT_JOINT_IN` | `first > 0` 时必填：从该目录读 `joint_<step>.bin` |
| `SD_DIT_X_IN` | 可选：从该目录读 `x_<step>.bin` 覆盖采样器给的 latent（重放用） |
| `SD_DIT_DUMP_DIR` | 可选：把 `x_/joint_/vel_<step>.bin` 写到该目录 |

实现要点：
- **只创建自己那几层**的 block（`blocks["transformer_blocks.N"]` 不创建就不会去 gguf 要张量），
  所以越界层的权重根本不会被分配 —— 这是「装得下」的前提，而不是先加载再丢。
- `forward()` 新增 `joint_in` 参数：非空则跳过 prelude（`img_in`/`txt_in`/refs）；
  `layer_last < num_layers` 时在 prefix slice / `norm_out` / `proj_out` 之前直接返回 `joint`。
- 段模式强制关掉 prefix cache（它的 K/V 跨全栈）。
- 中间段返回 zeros 当 velocity，让采样器继续跑 —— 这是**调试模式**，出图无意义。

文件格式：`int64 ndim | int64 dims[ndim] | float32[numel]`（`qwen_image_21_dump_tensor`）。

## 3. 怎么验证数值（不需要并发，不需要 IPC）

三个顺序跑，逐 step 对账：

```bash
# 基线：整模型，dump 每步 velocity
SD_DIT_DUMP_DIR=dump_full ./sd-cli-hex ... --seed 42 -o full.png

# A 段（layers 0..16）：dump 每步 joint。velocity 是 zeros，出图无效，忽略
SD_DIT_LAYER_FIRST=0  SD_DIT_LAYER_LAST=16 SD_DIT_DUMP_DIR=dump_a ./sd-cli-hex ... --seed 42 -o /dev/null

# B 段（layers 16..32）：吃 A 的 joint、复制基线的 x，dump 每步 velocity
SD_DIT_LAYER_FIRST=16 SD_DIT_LAYER_LAST=32 \
SD_DIT_JOINT_IN=dump_a SD_DIT_X_IN=dump_full SD_DIT_DUMP_DIR=dump_b ./sd-cli-hex ... --seed 42 -o split.png
```

判据：
- `dump_b/vel_<i>` vs `dump_full/vel_<i>` **逐 step 相对误差 < 1e-3**（同后端应更小）；
- 全 8 步都过，才算切分接对了 —— **单跑一步对上不算**（`L-035` 教训：曾因非确定性误判为对）。
- 视觉判据：`split.png` 与 `full.png` 的 MAE（复用现在这套比对脚本，参考值 256² 上是 1.23/255）。

注意：A 段自己的 x 从第 1 步起就是垃圾（velocity 是 zeros），所以 B 必须用 `SD_DIT_X_IN=dump_full`
重放基线的 x —— 这也正是 `SD_DIT_X_IN` 存在的理由。

## 4. 分割点怎么选

DiT 4005 MiB / 32 层 ≈ **125 MiB/层**（prelude 很小）。窗口 2304 MB = **2197 MiB** 可用：

| K | 每段层数 | 每段权重 | 判断 |
|---|---|---|---|
| 2 | 16 | ~2000 MiB | 加上 compute/pe/masks ≈ 2100 MiB，**太紧**（`L-058` 显示 1902 MiB 过、2023 MiB 崩） |
| **3** | **11 / 11 / 10** | **~1375 / 1375 / 1250 MiB** | **推荐**，留足余量（与 `L-046` 的 9B 三段同形） |
| 4 | 8 | ~1000 MiB | 段边界开销 ×3，收益递减 |

段边界开销：`L-053` 实测每边界 ~15 ms（graph build + submit 的固定成本，与段内层数无关）。
激活量 @256²：`32×32×64` latent 很小，`joint` 是 `32×32×4096` f32 ≈ **16 MB/边界**。
8 步 × 2 边界 × 16 MB ≈ 256 MB 总传输，相对 54.9 GiB 可忽略。

**分段不损失算力**：diffusion 每步是段间**串行依赖**，单 CDSP 核在任一段都吃满
（与 `L-029`「两进程同时算 → 算力对折」不冲突，那是并发场景）。

## 5. 第 2 层：IPC 与常驻（已落地，含一次整机 hang）

### 拓扑与协议

末段（22..32，**拥有采样器**）spawn 前驱（11..22），后者再 spawn（0..11）。每段都是同一个
`sd-cli` 二进制，靠 `SD_DIT_SPLIT_PLAN=11,22` + `SD_DIT_LAYER_FIRST/LAST` 决定自己是谁。

每步的乒乓（**全同步、无队列**，天然 lockstep，不需要步号对齐逻辑）：

```
末段  --send(x, step)-->  中间段  --send(x, step)-->  首段
      <--recv(joint)--           <--recv(joint)--
```

帧格式：`int32 step | int32 ndim | int64 dims[ndim] | f32 data[]`，走 `socketpair(AF_UNIX)`。
`x` 是 64 KB、`joint` 是 4.34 MB，一趟是内核态拷贝，实测整链没有可见开销
（三段合计 5.10 s/it ≈ 单独跑的 1.74+1.73+1.57 之和）。

采样器留在末段 → ER-SDE 多阶段循环一行没改，数值可复现（§6 的逐位一致就是这么来的）。

### 三个必须记住的实现细节

1. **子进程必须关掉继承的 fd**。`fork()` 复制整份 fd 表，ggml-hexagon 的 dspqueue / rpcmem
   句柄都没有 `CLOEXEC`；exec 后子进程还持有它们就会和父进程的 PD 撞车。
   做法：子进程里 `for (fd=3..1024) close(fd)`，只留 stdout/stderr 和 socketpair 那一端。
2. **子进程的 `-o` 要改掉**。它跑的是同一份 argv，会用占位 velocity 出一张垃圾图**覆盖真结果**
   （第一次实测 md5 不一致就是这个，不是算法错）。spawn 时把 `-o` 后面那个参数改成 `/dev/null`。
3. **子进程跑完最后一步就 `_exit(0)`**（`SD_DIT_SPLIT_STEPS=8`）。否则三个进程同时做 VAE 解码
   抢 CPU，端到端从 ~113 s 拖到 206 s。

### ⚠️ 整机 hard hang（第二次，2026-11-11）

把 spawn 从「父进程第一次 `compute()`」**提前到 `model.init()`**（想让子进程和父进程的 TE 并行），
结果三个进程几乎同时做 HTP init → 整机 hard hang（`adb` 先 `offline` 再彻底消失），
**只能断电池排线恢复**（`E-018`）。

对比：同样的链式代码，spawn 留在 `compute()`（懒启动）连续跑 5 次都没事 —— 因为那条路径下
**子进程是串行的**（父 spawn 子 1，子 1 到自己第一次 compute 才 spawn 子 2）。

**2026-11-11 12:40 补记（设备恢复后拿到设备侧日志，修正旧表述）**：第二次并不是「启动即挂」。
`log_early1.txt` 显示三段**全部起来了**（父 spawn pid 15981、B spawn 16033），三段都打印
`generate_image` / `sampling using Euler method`，父跑到 **1/8 @ 19.58 s/it**（懒启动是 5.10 s/it，
这里慢 3.8×），随后设备**自动重启**（`uptime` 归零，11:43 启动 → 约 11:50 重启，**不需要断电池**）。
另外三段日志混在同一个文件里 —— spawn 时只改了 `-o`，没重定向子进程 stdout/stderr
（llamasplit 每段都 `dup2` 到独立 `v2-X.log`，要照抄）。

已加的两道防护**实测无效**（2026-11-11 复现）：子进程 exec 前 `close(3..1024)` 关掉继承 fd、
`SD_DIT_SPLIT_STAGGER_MS=6000` 错开 HTP init —— 加上之后**仍然挂**，35 s 内 `adb` 就 `offline`。

→ **判定为禁止项**：`model.init()` 阶段 spawn 第二个 HTP 进程，两次尝试两次挂。
懒启动路径连跑 5 次（含 `I-027` 那批）都正常，**默认已改回懒启动**；
`SD_DIT_SPLIT_EARLY=1` 只留作复现用。

#### 根因修正（2026-11-11，对照 llamasplit 代码）

原先记的「同时存在多个正在初始化的 HTP 进程」**是错的**。翻 `D:/llamasplit` 的
`llama-split.cpp` 与台账 `L-067`：那里 `for (i...) fork()` 紧凑循环，K 个子进程的
`llama_backend_init()` 是**真并发**（`L-044`③ 两段 wall = `max(2044,2766)`，不是相加），
9B 三段常驻服务天天这么起。**并发 init 本身无害。**

真正的区别是**fork 的那一刻父进程是什么状态**：

| | llamasplit | 本项目 |
|---|---|---|
| fork 时父进程 | **从未碰过 backend**（`main()` 到 `run_v2()` 一行 llama 没调） | **HTP 会话已活**（权重已映射进 PD） |
| 子进程如何 init | 在自己进程里从零 init | 同样从零 init，但 fork 已复制父的 fastrpc/dma_buf VMA 与 fd 表 |
| 结果 | 稳定 | hard hang |

llamasplit 那条注释写得很直白（`llama-split.cpp:2340`）：
*the parent must never touch llama before fork(): a forked child inherits the fastrpc / backend
global state and segfaults on the first session allocation* —— 在 LLM 那边代价是 SIGSEGV，
在我们这儿代价是整机挂。**「先 fork 再 init」是硬前提，反过来不行。**

**那 58 s 怎么压**（路径 (b) 是这次新发现的，比 (a) 更彻底）：
1. **让 worker 不跑 TE**（改动小、安全）。worker 的 `forward` 走 `joint_in` 分支，`txt_in` 不执行，
   `context` 只用来定 token 数。父进程把 `context`（`(4096,9)`，约 147 KB）随第一个请求发下来，
   worker 直接用它、跳过 `get_learned_condition`。每个 worker 启动只剩「加载自己那 1.4 GB 分片」。
2. **改成 llamasplit 的形状：一个什么都不做的 launcher 先 fork 全部 worker，再各自 init**（首选方向）。
   这样既满足「fork 时父进程没碰过 backend」，又让三段的加载/init **并行**（`L-044`③ 证明 wall = max
   不是 sum）—— 端到端能真正回到 `max(段耗时)` 而不是 `sum`。代价：launcher 不能预加载权重，
   每段各读一份 gguf（页缓存共享，物理内存仍是一份）。
3. 如果还嫌慢，把 TE 挪到 worker 之外单独一个进程预热 —— TE 在 CPU 上，不碰 HTP，这条是安全的。

## 6. 实测结果（2026-11-11，v73 / 256² / 8 步 Turbo / seed 42）

| 段 | 层 | 配置 | `s/it` | 采样耗时 |
|---|---|---|---|---|
| 对照组 | 0..32 | `MAPBUDGET=1400M RPCMEM_FREE=1`（流式） | 7.73 | **61.8 s** |
| seg1 | 0..11 | `MAPBUDGET=1900M RPCMEM_FREE=0`（全驻留） | 1.74 | 13.9 s |
| seg2 | 11..22 | 同上 | 1.73 | 13.8 s |
| seg3 | 22..32（含 head） | 同上 | 1.57 | 12.6 s |
| **链式三段（懒启动）** | 同上三段，socketpair 串成一条 | 同上 | **5.10** | **40.8 s** |

- **数值：逐 step 比对 `vel_<i>`，8/8 步 `max|d| = 0`、`identical = 100%`** —— 完全逐位一致，
  比 llamasplit 的层切还干净（那边因图规模变化导致的 kernel 顺序差 ~0.3%）。
  这里每段是**连续层区间**，每层算术不变，所以没有舍入路径差异。
- 非平凡性已验证：`joint` 形状 `(4096, 265, 1, 1)`（265 = 256 图像 token + 9 文本 token），
  seg1→seg2 的 joint 相差 **27.6% relL2**，说明链路真的在推进；`vel` 与 `joint` 都不是零。
- **采样耗时 61.8 → 40.3 s（−35%）**，纯粹来自「不再流式换入换出」。
  注意 40.3 s 略高于「纯算力」估计（61.8 − 30 s wait ≈ 31.8 s），多出来的是 3 个进程
  各自的 graph build + submit 固定开销（`L-053`：每个 `llama_decode` 级别的边界 ~15 ms，
  但 diffusion 每步都要重建整图，成本比 LLM 高）。
- 端到端：懒启动链式 **171.76 s**，比整模型 133.83 s **还慢 38 s**。原因是子进程在
  「父进程第一次 `compute()`」时才 spawn，它的加载 + TE 全部串到第 0 步之前（实测 +58 s，
  而采样省下的只有 −21 s）。**这是当前唯一的性能缺口，下一步必须解决。**
  出图 `md5 = 39595a0728f7918443dcdee8269cdbff`，与整模型**完全相同**。

### ⚠️ 一个必须记住的组合坑

16 层 + **流式**（`MAPBUDGET=1400M` + `RPCMEM_FREE=1`）：第 0 步正常（`joint_000` 正常落盘），
**第 1 步 DSP 直接 abort，`dspqueue_read failed: 0x2e`**（不在 AEE 表里，等于「DSP 侧死了」）。
同样 16 层改成**全驻留**不会复现；8 层全驻留、11 层全驻留都正常跑完 8 步；
整模型 32 层用流式也正常。→ **流式 × 切分**这个组合有 bug，未定位。
但分段的定版配置**本来就不用流式**（每段装得下），所以不阻塞；哪天要回退到流式再查。

## 7. 判据

- 正确性：§3 的逐 step velocity 对账全过 + 出图与整模型 MAE 量级一致；
- 性能：**先看 `wait` 是否归零**（现在 30 s/129 s），再看 `s/it` 是否不低于单进程
  （分段不该让 DiT 计算变慢，只该省搬运）；
- 回归：`chainrun` 的 md5 判据不适用（那是 QNN 链），这里用 `out256htp.png` 的对照。

---

## 8. 死机的真正原因：OOM → 系统进程被杀 → OPPO 主动 panic（2026-11-11 定案）

> 结论：**不是 DSP/fastrpc 撞车，是内存耗尽。** 前面几节里「多进程/提前 spawn 导致挂机」的机制归因作废
> （现象保留），台账见 `I-029`。证据是设备自留的上次开机内核日志。

**取证**（`adb pull` 被 adbd 拦，`exec-out cat` 可读）：

```bash
F='/data/persist_log/DCS/de/minidump/SYSTEM_LAST_KMSG@…@2026_11_11_11_50_39.dat.gz'
adb exec-out "cat $F" > last.gz && gzip -dc last.gz > last.txt   # 196947 行
```

**时间线**（设备开机于 11:19:57，各时刻为 uptime 秒）：

| 时刻 | 事件 |
|---|---|
| 11:43:21 (1404.5) | OOM killer 杀 `rs.media.module` |
| 11:45:35 (1475.8) | 杀 **`ndroid.systemui`** ← 黑屏的直接原因 |
| 11:45:47 (1487.0) | 杀 `m.android.phone` |
| 11:47:03 (1613.9) | 杀 `exsystemservice` |
| 11:47:13 (1623.2) | 杀 `com.oplus.nas` |
| 11:48:36 (1719.3) | `DeathHealer: task watchdog:3084 blocked > 60 s in state 2`（D 态，Count:1） |
| 11:50:39 (1842.2) | `system_server:2693` / `2908` 同样 D 态 >60 s（Count 2/3）→ `kernel BUG at hung_task_enhance.c:299` → **Kernel panic** |

panic 现场：`CPU: 4 PID: 91 Comm: khungtaskd`、`pc: io_check_hung_detection+0x4d8 [hung_task_enhance]`；
同一时刻多核栈全在直接回收（`__alloc_pages_direct_reclaim` → `shrink_zones` → `shrink_lruvec` →
`shrink_inactive_list`），另有 `Comm: LowMemThread` → **内存耗尽的铁证**。全日志里没有一条
`adsprpc/fastrpc` 错误。

**因果链**（这就是「为什么会死机」）：

```
3 个 sd-cli 各跑完整 pipeline（TE 4.8 GB + DiT 分片 + VAE）≈ 13~20 GB
        ↓  23.6 GB 物理内存 + 9.9 GB swap 见底
内核 OOM killer 连杀 5 个系统进程（含 systemui / systemservice）
        ↓
SystemUI 死 → 黑屏；system_server 的依赖没了 → 多个线程卡在 D 态 >60 s
        ↓
OPPO 的 DeathHealer（hung_task_enhance）计满 3 次 → 主动 BUG() → kernel panic
        ↓
panic 不自动重启 → 黑屏 / 发热 / 电源键无响应（只能长按或断电池）
```

**为什么 llamasplit 3 进程没事**：它每段只 ~1.4 GB（还做了 PLE 切片），三段合计 ~4.5 GB。
**差别是内存量级，不是 fork 语义、也不是 PD 数量。** 由此也能解释为什么 `19.58 s/it`（懒启动是
5.10 s/it）：不是 DSP 抢算力，是整机在疯狂回收/换页。

**修法（已落地并实测通过，2026-11-11，见 `I-030`）**：worker 不加载 TE/VAE —— 它只跑层区间，
`context` 由末段写一个 `ctx.bin`（`(4096,9)` = 147480 B）交给它。实现三处：

| 位置 | 改动 |
|---|---|
| spawn | 注入 `SD_DIT_SPLIT_WORKER=1`，并把 `SD_DIT_CTX_FILE` 传下去 |
| `diffusion_engine.cpp` | worker 跳过 `llm`/`llm_vision`/`vae` 加载；`validate_registered_tensors()` 前把 `text_encoders.llm.`/`first_stage_model.`/`vae.` 加进 ignore 表（否则校验直接失败） |
| `image.cpp` | `cond_stage_model` 为空时跳过 `get_learned_condition` |
| `qwen_image_2_1.hpp` | 末段在发第一个 latent **之前**写 `ctx.bin`；worker 一律以该文件为准（不是「context 为空才读」——没有 conditioner 时上游会塞一个 `[3584,9]` 的占位值，而 DiT 要 `4096`） |

**实测**（`tools/memwatch.sh` 每 5 s 采样）：

| | 改之前 | 改之后 |
|---|---|---|
| 末段 RSS | ~4.4 GB | **4368 MB**（仍加载 TE+VAE） |
| worker RSS ×2 | ~4.4 GB 各 | **66 MB / 66 MB**（DiT 分片在 DSP dma_buf，不计入 RSS） |
| 三进程 sum_rss 峰值 | ~13.2 GB | **4556 MB** |
| `MemAvailable` 最低 | 打穿 → OOM → panic | **5340 MB** |
| 采样 | 5.10 s/it | **5.14 s/it** |
| 端到端 | 173.04 s | **125.43 s**（带采样器 131.10 s） |
| 出图 md5 | `39595a07…bff` | **`39595a07…bff`（相同）** |

结论：**worker 的 RSS 几乎全部来自 TE** —— 它根本没用到 TE，却各背了 4.8 GB。去掉之后三进程
与 llamasplit 同量级。剩下的大头是末段自己（TE 常驻 4.4 GB），要再压得让 TE 走内存映射或换更小量化。

**反面教训（别再犯）**：改成「launcher 先 fork 全部 worker 并行加载」看起来更优雅，
但它把三段加载叠在同一时间窗 → 峰值最高 → 正好撞死。**这里是内存受限，不是并发语义受限。**

---

---

## 9. 服务模式（`sd-server`，2026-11-11 打通）

**`tools/serve.sh` 现在可用**：末段就是 `sd-server`（持采样器 + HTTP），前两段是**常驻 worker**，
第一次请求把 worker 拉起来并加载完模型，之后每个请求只有编码 + 采样 + 解码。

| 形态 | 实测（256² / 8 步，v73） |
|---|---|
| 首次请求（含拉起两个 worker + 各段加载） | **87.8 s** |
| 第二个请求（换 prompt，worker 已在位） | **45.7 s** |
| 第三个请求（320×192，换分辨率） | **43.0 s** |
| 对照：`chain.sh` 一次性跑（每请求重付加载） | 77.3 s |

启动：`adb shell "cd /data/local/tmp/qisplit && nohup sh serve.sh 18080 > log_srv.txt 2>&1 &"`，
PC 侧直接 `POST http://<手机IP>:18080/sdapi/v1/txt2img`（默认 `LISTEN_IP=0.0.0.0`；
只想本机就 `LISTEN_IP=127.0.0.1` + `adb forward`）。

可用接口（见 `examples/server/api.md`）：`/sdapi/v1/txt2img`（A1111 风格，最省事）、
`/v1/images/generations`（OpenAI 风格）、`/sdcpp/v1/img_gen`（原生 + 异步 job）。

### 9.1 原先的坑（已修）：worker 继承 argv 自己也去 listen

`split_spawn_predecessor()` 拿 `/proc/self/cmdline` 重新 `execv("/proc/self/exe")`。
在 `sd-cli` 下没问题（子进程跑完就退）；但在 `sd-server` 下子进程**继承了 `--listen-port 18080`**，
也去 `svr.listen()`，永远走不到 `compute()` → 请求挂死。

修法（采用第 2 条）：`examples/server/main.cpp` 在 `new_sd_ctx` 之后、起 HTTP 之前判
`SD_DIT_SPLIT_WORKER`，走 `sd_serve_split_worker()`（新 C API）：

```
sd_ctx 建好 → worker? → yes: 不进 initialize_server_models / 不 listen，直接 serve_split_worker()
                       → no : 原样起 HTTP
```

`serve_split_worker` 是纯驱动循环：`收 latent → compute 本段 → 发 joint`，**没有采样器**，
日志里能看到 `split worker mode: layers [11, 22), no HTTP listener` 与 `split worker serving layers ...`，
`ps` 里三个 `sd-server-hex` 分别停在 `inet_accept`（末段）/ `unix_stream_data_wait`（两个 worker）。

### 9.2 关键前提：timestep 必须随 latent 一起下发

worker 没有采样器就推不出这一步的 `t`，而 `t` 会进 modulation。所以段间帧加了第三个头字段
（`step | ndim | timestep_bits`，NaN = 未知）。CLI 模式也改用它（以前各进程各算一遍同样的调度，
是隐含约定；现在统一以末段为准）。顺带的好处：worker 对**分辨率、步数、prompt 一无所知也不会错位**，
换分辨率不用重启——320×192 那次请求就是同一个常驻链直接出的。

### 9.3 每请求重发 context

`end_runners()` 里加了一条 `reset_split_state()`：末段每次生成结束把 `split_ctx_sent_` 清 0，
下一个请求重新沿链发 `(-1, ctx)`。日志证据：第二请求 `ctx sent [4096, 14]`（新 prompt 的 token 数变了），
两个 worker 各 `ctx recv [4096, 14]`。

### 9.4 TE 释放判据改 `atoi(...) != 0`

`image.cpp` 原先只看 `getenv("SD_DIT_RELEASE_TE") != nullptr`，`serve.sh` 里 `=0` 照样把 TE 放掉
（日志有 `conditioner released`），第二个请求就没有编码器了。现在按值判断，常驻服务下 TE 常驻
（末段 RSS 4.4 GB），一次性跑仍可 `=1` 省钱。


---

---

## 10. WebUI + 效率画像（2026-11-11）

### 10.1 用法

```bash
python tools/webui.py                    # 浏览器打开 http://127.0.0.1:7860
python tools/webui.py --port 8080 --server http://192.168.1.8:18080
```

一个文件、零第三方依赖（标准库 `http.server` + `urllib`），页面上可切两种后端：

| 模式 | 后端 | 每张耗时 |
|---|---|---|
| **分段链（默认）** | adb 跑设备上的 `tools/chain.sh`（三进程，末段持采样器） | **77 s**（4.5 s/it，含每请求重付加载） |
| 常驻分段服务 | 直连设备上的 `tools/serve.sh`（末段 `sd-server` + 两个常驻 worker） | **45.7 s**（第二请求起） |

`chain.sh` 现在吃 `PROMPT / STEPS / SEED / W / H / THREADS` 环境变量，所以页面上的参数真的生效。
出图在设备上落到 `out_<tag>.png`，webui pull 回来再 base64 内联。

两个实现坑：
- **不要用 `adb shell "... &"`**：Windows 上 adb 会一直等后台子进程的管道，直到 timeout 才返回。
  改成前台跑 + `Popen` 不等待 + 轮询 `out_<tag>.png`。
- webui 的 `kill_stale()` **只杀 `sd-cli-hex`，不碰 `sd-server-hex`** —— 否则跑一次 chain 就把常驻服务杀了。

### 10.2 效率画像：瓶颈不是采样，是 VAE 解码

同一张 256²/8 步（seed 42，md5 `39595a07…bff`）拆开看：

| 阶段 | 单进程（流式） | 分段链（K=3） | 说明 |
|---|---|---|---|
| 采样 | 61.8 s（7.73 s/it） | **40.4 s（5.05 s/it）** | 分段唯一省下来的：**−21 s（−34%）** |
| VAE 解码 | ~58 s | **60~67 s** | 两者相同，**占端到端 45%** |
| 模型加载 | 0（常驻） | ~20 s | 每次请求重付；`sd-server` 常驻则免掉 |
| 合计 | 110~117 s | 126~129 s | |

**结论 1：分段的收益是确定的，但被「每次重新加载」抵掉了。**
一次性跑时 chain 126 s 反而比常驻 server 117 s 慢 —— 因为 chain 每次要重新加载 4 GB 权重。
要让分段真正赢，得让 worker **常驻**（见 §9 的坑）或至少把加载并行掉。

**结论 2：VAE 的 60 s 才是最大单点。** 日志证据：

```
VAE weight type stat: bf16: 238
conv2d direct auto: CPU -> off (no implicit GEMM path)
conv3d direct auto: CPU -> off (no implicit GEMM path)
```

VAE 权重是 **bf16**（238 个张量），而 CPU backend 没有 implicit-GEMM 卷积路径，只能 im2col；
256² 的 VAE 解码在手机上正常应该 5~15 s，这里 60 s。**要再快，先动 VAE，不是动采样。**

三条可试的路（都还没做）：
1. **VAE 走 HTP**（`--backend vae=HTP0`）：conv 在 HTP 上有 implicit GEMM，但要先解决早期
   QNN 链里「VAE 算错」的老问题（`H-004` 只对 UNet 有效，VAE 要单独验）；
2. 试 `-t` 线程数（默认 `sd_get_num_physical_cores()`，8 核混大小核时未必最优，gemma 那边 4 线程 > 8）；
3. 把 VAE 权重转 fp16/fp32 试（bf16 在 x86 之外的 CPU 上往往没有 SIMD 加速路径）。

### 10.3 「提前 spawn」已不再是问题（但也没收益）

`SD_DIT_SPLIT_EARLY=1` 让 worker 在 `model.init()` 就 spawn、与末段 TE 并行加载。
11-11 重测（worker 已不加载 TE，内存安全）：

| | 懒启动 | 提前 spawn |
|---|---|---|
| 端到端 | **126.1 s** | 132.3 s |
| 采样 | 5.05 s/it | 5.05 s/it |
| 出图 md5 | `39595a07…bff` | **`39595a07…bff`（一致）** |
| 三进程 RSS | 4.4 GB + 66 MB ×2 | 同（MemAvailable 最低 5.56 GB，安全） |

**数值正确、内存安全，但没有更快** —— 末段的 TE 加载（21 s）本身就是瓶颈，并行只能省 worker 那点，
还被 VAE 解码的波动吃掉。所以默认仍走懒启动（`SD_DIT_SPLIT_EARLY` 只作实验开关）。

顺带修掉一个真 bug：worker 原本靠**文件**拿 context（末段写 `ctx.bin`）。提前 spawn 时 worker
会先于末段算完 TE 就绪，读到的必然是上一次请求的旧 ctx（实测 `[4096,16]`，模型要 9 个 token）
→ `invalid image token layout`。已改成**控制帧 `(-1, ctx)` 走 socket**，由 recv 天然定序，
文件只留作兜底。

---

---

## 11. VAE 搬到 HTP：58~67 s → 18.8 s（2026-11-11 定版）

> 这是目前最大的一次端到端提速。**采样优化（分段）省了 21 s，VAE 这一步省了 40~48 s。**

### 11.1 结论

| 形态 | VAE 后端 | VAE 解码 | 端到端 | 出图 |
|---|---|---|---|---|
| 单进程流式 | CPU | 58~67 s | 133.8 s | 基线 `39595a07…bff` |
| **单进程流式** | **HTP0** | **18.82 s** | **95.4 s** | MAE 0.030/255 · PSNR 63.2 dB |
| 分段链 K=3 | CPU | 60~67 s | 126~129 s | 同上基线 |
| **分段链 K=3** | **HTP0** | **19.19 s** | **85.1 s（WebUI 87 s）** | MAE 0.030/255 · PSNR 63.2 dB |

**数值**：与 CPU VAE 的图比，`MAE 0.030/255`、`max 5`、`92.01%` 像素完全相同、`PSNR 63.21 dB`。
两条路（单进程 / 分段链）的差异**一模一样**，说明误差全部来自 VAE 的 HTP fp16 路径，
跟分段无关。比 DiT 那段 HTP 的 1.23/255 还干净一个数量级。

### 11.2 为什么 CPU 那么慢

```
VAE weight type stat: bf16: 238
conv2d direct auto: CPU  -> off (no implicit GEMM path)
conv3d direct auto: CPU  -> off (no implicit GEMM path)
conv2d direct auto: HTP0 -> off (no implicit GEMM path)      ← HTP 也一样
```

`sd_backend_supports_conv_implicit_gemm()` 只对 **Vulkan / MUSA / CUDA / ROCm** 返回 true，
CPU 和 HTP 都走 im2col。所以不是「HTP 有 implicit GEMM 所以快」，而是 **DSP 的 SIMD（HVX）
把 im2col + matmul 跑得比 CPU 的 bf16 路径快 3 倍多**。bf16 在 ggml-cpu 上没有高效的卷积路径，
256² 这么小的图也要 60 s。

### 11.3 怎么开

```bash
# 分段链（定版，87 s）：VAE 默认已改成 HTP0
sh chain.sh mytag

# 回退到纯 CPU VAE（1900M 预算 + CPU 解码，126 s）：
VAE_BACKEND=CPU GGML_HEXAGON_MAPBUDGET=1900M sh chain.sh mytag
```

⚠️ **两个硬约束**（都是实测撞出来的）：
1. **`MAPBUDGET` 必须从 1900M 降到 1500M**。VAE 上 HTP 要常驻 644 MB IOVA，
   与 DiT 分片挤在同一个 2304 MB 窗口里：`1900 + 644 > 2304` 直接挂（`CHAIN exit=141`，SIGPIPE），
   `1500 + 644 ≈ 2144` 才稳。
2. `chain.sh` 的 `VAE_BACKEND` 只影响 `--backend vae=`，不动 TE（仍 CPU）。

`serve.sh`（分段常驻，末段 `sd-server`）同样 `vae=HTP0`，见 §9。

### 11.4 还没验的

- **512² 未测**：VAE 的激活和 IOVA 都会涨，`1500M + 644 MB` 可能不够，要单独调。
- VAE 在 HTP 上是 **fp16 累加**（`H-003` 那条低精度路径），这次 MAE 只有 0.03 是运气好
  —— 依赖它「永远这么好」需要多 prompt 抽查，不宜只信一张。
- `conv_direct` 对 HTP 是 off，意味着 conv 仍走 im2col；如果给 HTP 后端补 implicit GEMM，
  还能再快一截（工作量在 ggml-hexagon 侧）。

---

---

## 12. 512² 为什么这么慢 / 每段到底占多少内存（2026-11-11 实测）

### 12.1 512² 的账单：不是内存，是算力与 token 数

chain + VAE(HTP)、8 步、seed 42：

| 阶段 | 256² | 512² | 倍数 |
|---|---|---|---|
| 采样 | 8 × 5.05 = 40.4 s | 8 × **32.0** = 256 s | 6.3× |
| VAE 解码 | 19.2 s | **152.4 s** | 7.9× |
| 加载 + TE | ~25 s | ~30 s | — |
| **合计** | **85 s** | **437.7 s** | 5.1× |

**关键证据：512² 全程零搬运。** 日志里每一步都是

```
[stream] reap=0 map=17 (27.8 ms) unmap=0 (0.0 ms) relinq=0 (0.0 ms)
         restore=0 (0.0 ms) copy=0.0 GiB wait=0.0 ms mapped=483/1500 MiB cum=1933 MiB
```

`wait=0.0 ms`、`copy=0.0 GiB`、`relinq=0 restore=0`、权重 `mapped 1450/1500 MiB` 常驻。
所以**「512² 慢是因为内存不够」这个直觉是错的** —— 它连一次换入换出都没发生。

真正的原因是 **token 数**：joint 从 `(4096, 265)` 变成 `(4096, 4105)` ——
256² 是 `16×16 = 256` 个 image token + 9 个文本 token；512² 是 `64×64 = 4096` + 9
（VAE 下采样 16×，所以 256²→16²、512²→32²；再经 patch 2×2 → 16² 与 64²，与实测 265/4105 吻合）。

`265 → 4105` 是 **15.5×**，而 s/it 只涨 **6.3×**。反过来说明：
- FFN/GEMM 是线性于 token（15.5×），attention 是平方（240×），
- 实测 6.3× 说明 **256² 时的 5.05 s/it 里固定开销占比很大**（每步 graph build + submit + 权重 scan），
  512² 反而更「划算」：`256² = 0.157 s/token`，`512² = 0.062 s/token`（**2.5× 效率提升**）。

**结论**：512² 慢是必然的算力账，不是配置问题。要提速只能减少 token（更小的分辨率）或换更快的 kernel。

### 12.2 VAE 在 512² 上优势变小（19→152 s，7.9×）

- 256²：CPU 58 s → HTP **19 s**（3.1×）
- 512²：HTP **152 s**。CPU 侧没有同步实测，按 256² 的 58 s 线性推约 230 s（未验），
  即 HTP 仍快一些，但**远不如 256² 明显**。

原因大概率是 512² 的中间层激活超出 VTCM（8 MB），要落到 DDR 上往返；
也可能 conv 的 im2col 展开在 512² 时占比变高。**这条没深挖**，是 512² 现在的第一大单点（占 35%）。

### 12.3 每段到底占多少内存：两个口径别混

**口径 A：进程 RSS（`ps` 看到的）**

| 进程 | 256² | 512² |
|---|---|---|
| 末段（22..32 + head + 采样器 + TE + VAE） | **4368 MB** | **4394 MB** |
| worker（0..11） | 66 MB | 70 MB |
| worker（11..22） | 66 MB | 87 MB |
| 合计 | **4.5 GB** | **4.6 GB** |

**4.4 GB 几乎全是 TE**（Qwen3-VL-8B Q4_K_M ≈ 4.8 GB 的 gguf），不是 DiT。
DiT 的权重根本不在 RSS 里 —— 它们走 HTP 的 rpcmem/dma_buf。

**口径 B：DSP 侧 IOVA（真正的瓶颈）**

| 项 | 大小 |
|---|---|
| DiT 每段（11 层） | ≈ 1450 MiB |
| 末段额外 VAE | 644 MiB |
| **末段合计** | **≈ 2100 MiB** / 2304 MiB 窗口 |
| worker 每段 | ≈ 1450 MiB |

所以「每段 5 GB」是 RSS 口径且**只有末段**；按 DSP 侧算，每段是 **1.45 GB**，末段 2.1 GB。

### 12.4 和 llamasplit 对比：差的是「那个 8B 的 TE」

| | llamasplit（gemma4-e4b / qwen35） | qwen-image-split |
|---|---|---|
| 每段 DSP 驻留 | **1155~1496 MiB**（`L-033`，PLE 切片后） | **1450 MiB**（DiT 11 层） |
| 每段 RSS | 很小（权重走 mmap + DSP） | worker 66 MB；**末段 4.4 GB** |
| 额外常驻 | 无 | **TE = Qwen3-VL-8B Q4_K_M（4.8 GB）** |

**差别几乎全在那一个 TE**：llamasplit 是纯 LLM，模型本身就是那个 8B；
我们是「**8B 文本编码器 + 4 GB DiT + 644 MB VAE**」三件套，而 TE 只在第一步用一次，
却全程钉在末段进程里。DiT 段本身（1.45 GB）和 llamasplit 的每段是**同一量级**。

### 12.5 由此得到的优化项（未做）

1. **TE 下沉**（收益最大，3.4 GB）：TE 只在第一个 latent 之前用一次（21 s，CPU），
   算完就可以退出。做成「独立进程 / 按需加载 → 出 context → `_exit`」后，
   **末段 RSS 从 4.4 GB 降到 ~1 GB**，三进程总内存 4.6 GB → 1.2 GB，和 llamasplit 相当。
2. **TE 结果缓存**：同一个 prompt 反复调 seed/尺寸时，21 s 可以完全省掉（按 prompt 哈希缓存 context）。
3. **512² 的 VAE**（占 35%）：试 `--vae-tiling`，或与 CPU 做同条件对照，确认是不是 VTCM 溢出。

---

---

## 13. 生图过程的内存峰值（三口径实测，2026-11-11）

> 只问「峰值多少」会有三个不同答案，差 3 倍。**最容易漏掉的是 DSP 侧那 5.4 GB。**
> 采集工具 `tools/mempeak.sh`（每 2 s 采 `MemAvailable / MemFree / SwapFree / Cached / Shmem /
> SUnreclaim` + 三进程 RSS 之和，进程全退出后自动汇总）。

设备 MemTotal **23644 MB**，开机基线（无任何 sd 进程）：

| 字段 | 基线 |
|---|---|
| MemAvailable | 17.83 GiB |
| MemFree | 7.59 GiB |
| Cached | 2.66 GiB |
| Shmem | 13 MB |
| SwapFree | 7.79 GiB（已用 2.2 GB 是常驻的） |

### 13.1 256² 峰值（chain + VAE HTP，8 步）

| 指标 | 峰值 |
|---|---|
| **`MemAvailable`** | **5.27 GB** |
| `MemFree` | **91 MB**（几乎归零） |
| `Cached` | 5.97 GB（可回收） |
| `Shmem` | 12.7 MB（≈0） |
| `SUnreclaim` | 486 MB |
| **三进程 `sum_rss`** | **4.43~4.62 GB** |

### 13.2 三个口径的账

| 口径 | 大小 | 说明 |
|---|---|---|
| 进程 RSS | **4.6 GB** | 末段 4.4 GB ≈ 全是 TE；两个 worker 各 66~87 MB |
| **DSP 侧 dma_buf** | **≈ 5.4 GB** | 三段 DiT 各 1.45 + VAE 0.64 + op buffer `128M×2×3`。**不计入 RSS**，也不体现在 `Shmem`（走 ion/dma_heap） |
| 页缓存增长 | +3.2 GB | 读 `dit.gguf`/`te.gguf` 形成的 Cached（2.66 → 5.97） |
| **整机新增** | **≈ 13.2 GB** | `18.70 − 5.53 = 13.16 GB` |

**为什么 `MemFree` 归零却不 OOM**：Linux 的有效余量是 `MemAvailable`（含可回收的 Cached）。
峰值时还剩 5.97 GB Cached 能随时回收，所以没触发 OOM killer。

这也回头解释了 `I-029` 那次整机死机：当时**三个进程各带一份 TE**（RSS 13.2 GB），
加上 dma_buf 5.4 GB，总需求超过 23.6 GB，连 Cached 都被吃光 → OOM → SystemUI 被杀 → panic。

### 13.3 512² 更紧

| | 256² | 512² |
|---|---|---|
| `MemAvailable` 最低 | 5.27 GB | **4.21 GB** |
| 整机新增 | 13.2 GB | **14.5 GB** |
| 三进程 RSS | 4.43~4.62 GB | 4.52 GB |

**Swap 全程没动**（`SwapFree` 恒定 8.13 GB）→ 贴到红线但没开始换页。

### 13.4 实用结论

| 机型内存 | 256² | 512² |
|---|---|---|
| 24 GB（本机） | 从容（余 5.3 GB） | 可以（余 4.2 GB） |
| 16 GB | 可以（余 ~1 GB） | **危险**（余 <1 GB） |
| 12 GB | 勉强，需先清后台 | 不要试 |

**判断标准用 `MemAvailable ≥ 1 GB`，不要看 `MemFree`** —— 后者在正常情况下也常年接近 0。

---

---

## 14. TE 下沉：末段 RSS 4.48 GB → 24 MB（2026-11-11）

TE（Qwen3-VL-8B Q4_K_M，4.4 GB）整轮生成只在最前面用一次，却常驻整个进程。编码完释放即可。

### 13.1 做法

`image.cpp` 在所有文本编码完成之后（`get_learned_condition completed` 日志那行下面）：

```cpp
if (getenv("SD_DIT_RELEASE_TE") != nullptr && sd->cond_stage_model) {
    sd->release_conditioner();   // runner_end -> unregister(Conditioner) -> reset
}
```

`release_conditioner()` 是新增的：

```cpp
cond_stage_model->runner_end();
model_manager->unregister_param_tensors(ModelComponent::Conditioner, &released);
cond_stage_model.reset();     // ← 真正释放 backend buffer 的是这一步
```

⚠️ 别被 `unregistered 0.0 MiB` 误导：TE 的张量存储块不在 `Conditioner` 名下，
`reset()` 触发 runner 析构时 buffer 才一起还回去。

**安全性前提**（已确认）：`sd::Tensor<T>` 内部是 `std::vector<T>`，值语义；
`SDCondition.c_crossattn` 是深拷贝，释放 TE 不会把它一起清掉。

### 13.2 实测

| | 改之前 | 改之后 |
|---|---|---|
| 末段 RSS（峰值，TE 加载中） | 4.48 GB | 4.48 GB（不变，峰值一样） |
| 末段 RSS（稳态，采样/VAE 阶段） | **4.4 GB** | **19.9~28.8 MB** |
| 三进程 RSS 合计（稳态） | ≈ 4.6 GB | **≈ 75 MB** |
| `MemAvailable` | 14.3 GB | 18.7 GB |
| 端到端 | 85.1 s | **91.7 / 96.6 s** |
| 出图 md5 | `1ceb0cc0…` | **`1ceb0cc0…`（一致）** |

**注意峰值没变**：TE 加载时那 4.4 GB 仍然存在，只是用完就还。
所以这条省的是**稳态占用**，不是峰值 —— 真要降峰值只能把 TE 换成更小的量化。

### 13.3 什么时候不能开

`SD_DIT_RELEASE_TE` 只适合**一次性跑**（`chain.sh`）。常驻服务（`serve.sh`）**必须关掉**：
释放后再来的 prompt 没有编码器可用。默认值：`chain.sh` 里 `=1`，`serve.sh` 里显式 `=0`
（注意判据已改成「值非 0」，早先只看变量存在，`=0` 也会放掉，见 §9.4）。

---

---

## 15. 算子统计：66% 的时间花在一个退化的 RoPE 分解上（2026-11-11）

`GGML_HEXAGON_OPSTAT=1` 让后端把 DSP 回传的 per-op 计时聚合成表，退出时打印
（分段链的 worker 用 `_exit()` 跳过 atexit，再给 `GGML_HEXAGON_OPSTAT_CUM=1` 让它在每次
`synchronize()` 输出累计表，取最后一张即可）。

### 15.1 先确认口径

DSP 每个 batch 都会回一份 `htp_prof_desc`（`usecs` + 启停 cycle），但**必须先调用
`htp_iface_profiler(handle, mode=1)` 打开**，否则那块内存是脏的。第一次跑出来的
`sum(op) = 199886 s` 比 16 s 的窗口大 1.2 万倍，就是没开 profiling 直接读缓冲区。

打开后三个数字自洽：

| 量 | 含义 | 256²/8 步三段合计 |
|---|---|---|
| `dsp-window` | `Σ rsp.usecs`，DSP 真正在跑我们的 batch 的时间 | **39.38 s** |
| `sum(op)` | `Σ pd.usecs`，各 kernel 自报的时间 | 39.34 s |
| `unattributed` | 上面两个的差：提交、线程同步、VTCM 申请、权重扫描 | **0.04 s（0.1%）** |

**0.1% 就是「逐 op 转发」的全部代价**，可以忽略。之前担心的「每个 op 一次 IPC / 一次队列提交」
并不存在——22 408 个 op 平均 1.76 ms，时间全在 kernel 里面。

采样 wall 41 s，DSP 39.4 s → **host 侧只占 4%**。所以优化只能做在 op 内部。

### 15.2 分布：RoPE 66%，矩阵乘 23%

| 类 | 时间 | 占比 | 说明 |
|---|---|---|---|
| **RoPE（`REPEAT` + `MUL` @ `[2,64,265,32]`）** | **25.94 s** | **65.9%** | 每层 4 次 `REPEAT`（19.3 ms）+ 4 次 `MUL`（5.9 ms）= **101 ms/层/步** |
| 矩阵乘（`MUL_MAT` + `MUL_MAT_NX`） | 8.92 s | 22.6% | 见 15.3，效率其实是健康的 |
| `CONT` | 2.07 s | 5.3% | 8 300 次布局拷贝，0.25 ms/次 |
| `SOFT_MAX` / `SWIGLU` / `SCALE` / `ADD` / `CONCAT` … | ~2.4 s | 6% | 都是 265 token 的小算子 |

整层预算 153 ms/步，其中 RoPE 101 ms、矩阵乘 30 ms、注意力 17 ms、其它 5 ms。

### 15.3 矩阵乘不是瓶颈

按形状看单算子吞吐（`2×MAC/usec`）：

| 形状 | GFLOPS | 说明 |
|---|---|---|
| `MUL_MAT q4_K 24576x265` | **7 673** | MLP gate/up，最大的一块 |
| `MUL_MAT q4_K 4096x265` | **5 600~5 760** | proj / qkv |
| `MUL_MAT q4_K 4096x265`（fused `MUL_MAT_NX`） | 3 400 | |
| `MUL_MAT f32 128x256x32` | **35** | **注意力 QK^T**，15.7 ms/次 |
| `MUL_MAT f32 265x256x32` | 400 | 注意力 ·V |

量化权重那几块跑到 5.6~7.7 TFLOPS，比之前用 LLM prefill 反推的 4.3 TFLOPS 还高——
**matmul 这条路没有多少油水了**。真正差的是 f32 注意力：QK^T 只有 35 GFLOPS
（f32 走 HVX，且 N=265 太小，HMX 用不上），但它总共只占 8~11%。

### 15.4 为什么 RoPE 会慢成这样

`src/core/ggml_extend.cpp` 的 `ggml_ext_rope_apply()` 里，给 HTP 走的是 **interleaved 的手工分解**：

```cpp
x   = ggml_cont(ggml_permute(x, 0, 2, 1, 3));          // 整块转置
x   = ggml_reshape_4d(x, 2, d_head/2, L, n_head*N);    // [2, 64, 265, 32]
x   = ggml_cont(ggml_permute(x, 3, 0, 1, 2));          // 再转一次
x_0 = ggml_view_3d(...);  x_1 = ggml_view_3d(...);
x_0 = ggml_repeat(x_0, temp_x);                        // ← 把半向量复制成 2 宽
x_1 = ggml_repeat(x_1, temp_x);
x_out = add_inplace(mul(x_0, pe_0), mul(x_1, pe_1));
```

问题是 **`REPEAT` 的目标张量内层维 `ne0 = 2`**（因为要把 `[cos,-sin; sin,cos]` 的 2×2 旋转矩阵
按位置乘上去）。HVX 一次要处理 128 B，这里只有 8 B，等于完全不能向量化：
4.3 MB 的重复拷贝耗 19.3 ms ⇒ **225 MB/s**，比 DSP 上正常的 memcpy 慢一到两个数量级。
每层 q、k 各一次，四次 REPEAT 就 77 ms。

（`Rope::apply_rope` 还有一条 `sd_ggml_rope_op()` 自定义算子快路，但
`ggml_backend_supports_op(HTP)` 不认，所以必然落到手工分解。）

### 15.5 可做的优化（按收益排序，均未实施）

1. **RoPE 换成原生 rope kernel**（收益最大）。HTP 后端本来就有 `htp/rope-ops.h`，
   llama.cpp 的 `GGML_OP_ROPE` 在 HTP 上是直接支持的。把 interleaved 分支替换成一次
   `ggml_rope_ext(x, ..., GGML_ROPE_TYPE_NORMAL, rot_dim)`，内层维回到 128，
   同时干掉那两次 `cont` 转置。预期 65.9% → 5% 以内。
   **风险**：`pe` 是预计算好的 cos/sin 表，`ggml_rope_ext` 收的是 freqs，末位精度会变，
   md5 不会逐位一致，判据要改成 MAE/PSNR。
2. **注意力 f32 → f16（或开 flash attn）**：QK^T 35 → 数百 GFLOPS 量级，省 8~11%。
   注意 HTP fp16 是低精度路径（台账 `I-020`），要先量 MAE。
3. `CONT`（5.3%）大部分由 RoPE 的 permute 产生，跟着第 1 条一起消失。
4. 矩阵乘：**没有明显空间**，大块已经 5.6~7.7 TFLOPS。

外推（只算 RoPE）：采样 5.11 → **≈1.9 s/it**，端到端 95.6 → **≈70 s**；
RoPE + 注意力都修 → **≈1.6 s/it**。

### 15.6 一个没解释的副产物

末段 VAE 的 DSP 时间只有 **3.83 s**，而 VAE 解码 wall 是 **17.8 s**。
15.2 的表里 `IM2COL` 一共 3.07 s，也就是说 **VAE 有 14 s 根本不在 DSP 上**
（host 侧建图/重排，或 graph cut 掉回 CPU 的算子）。512² 的 VAE 要 152 s，
大概率同源，值得单独查一次。

---

---

## 16. 权重流式已整体删除（2026-11-11）

分段（每段 11 层 ≈ 1450 MiB）之后流式没有任何用处，但它一直以「窗口不足时的兜底」名义留着。
2026-11-11 决定彻底删掉，不留死代码。删掉的东西（`ggml-hexagon.cpp` 净 -242 行）：

| 位置 | 内容 |
|---|---|
| session 成员 | `stream_budget/mapped/total_map/op_idx`、10 个计数器、4 个 `unordered_*` 集合 |
| session 方法 | `stream_reserve()`（LRU 回收 + 必要时强 flush）、`stream_forget()` |
| `shared_buffer` | `relinquish()` / `restore()`、`backup` / `saved_size` / `saved_base` / `relinquished` / `tensors` 登记 |
| `mmap_tensor()` | 预算分支、映射失败重试、`stream_last_use` 记账 → 现在只剩「没映射就 map」 |
| 其他 | `synchronize()` 的 `[stream]` 统计、`GGML_HEXAGON_MAPBUDGET` / `GGML_HEXAGON_RPCMEM_FREE` 解析 |

**行为未变**：256² 出图 `md5 = 1ceb0cc0…`（与删除前逐位一致），采样 5.07~5.15 s/it，
端到端 95.6 s；DiT 1450 + VAE 644 = 2094 MiB 仍在 2304 MiB 窗口内，一次 map 到底。

**代价**：单进程加载整个 3.91 GiB DiT 会直接 `fastrpc_mmap` 失败，没有兜底了。
所以 `tools/serve1.sh`（单进程常驻服务）和 `tools/pd_probe.sh`（双 PD 探针）随之一并删除，
常驻服务只能走 `tools/serve.sh`（分段链）。

---

---

## 17. 原生 M-RoPE 内核：快 3×，但语义还没对齐（2026-11-11）

`ggml_ext_rope_apply()` 里加了一条快路（`SD_ROPE_NATIVE=1` 打开，**默认关闭**）。

### 17.1 做了什么

```cpp
// q/k: [d_head, n_head, L, N]，相邻对 (2j, 2j+1) 是旋转对
auto x4 = ggml_reshape_4d(ctx, x, 2, d_head/2, n_head, L);
x4      = ggml_cont(ctx, ggml_permute(ctx, x4, 1, 0, 2, 3));   // [pair, lane, n_head, L]
x4      = ggml_reshape_3d(ctx, x4, d_head, n_head, L);          // 头维 = [偶数位|奇数位]
x4      = ggml_rope_multi(ctx, x4, positions, freq_factors, d_head, sections,
                          GGML_ROPE_TYPE_MROPE, 0, 10000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
```

一次 `cont` + 一次原生 rope，替掉原来的「2×cont 转置 + 2×REPEAT + 2×MUL + ADD」。

三条从源码里敲定的事实：

| 事实 | 依据 |
|---|---|
| HTP 的 rope 内核里 **MROPE 必定是半切配对** | `htp/rope-ops.c:559` `is_neox = (mode & NEOX) \|\| (mode & MROPE)` |
| `sections` 比较的是**对索引**不是头维 | `sector = (i0/2) % sect_dims`，故传 `axes_dim/2 = {8,28,28}` |
| `theta_scale` 用**全局** n_dims，pe 用**逐轴** axes_dim | `theta_scale = powf(freq_base, -2/n_dims)` vs `rope_frequencies(dim=axes_dim[i])` |

第三条差一个逐对修正：`freq_factors[k] = 10000^(2*k_local/dim_axis - 2*k/head_dim)`，
内核按 `theta / ff` 应用（`rope_cache_hvx_32` 和标量路径都是除）。

### 17.2 实测

| | pe 路径（默认） | 原生 rope |
|---|---|---|
| 采样 | 5.07~5.15 s/it | **1.83~1.96 s/it** |
| 端到端（256²/8 步） | 95.6 s | **68.3 s** |
| 出图 | `md5 1ceb0cc0…` | **MAE 66.9/255、PSNR 10.6 dB（灰纹理）** |

速度是真的（`REPEAT` 那 66% 直接消失），但图完全不对——像是**位置信息根本没生效**，
即 rope 退化成了恒等（最可能：positions 没送到设备，全 0）。
`SD_ROPE_NATIVE=0` 时 md5 逐位回到 `1ceb0cc0…`，说明开关本身是干净的。

### 17.3 下一步

把 q 在 rope 前后落盘，两条路径逐元素比：

- native 输出 ≈ 输入 → positions 没上传（查 `make_input` 的 I32 路径）；
- 差异有结构（只是角度不同）→ 频率/配对。

---

## 18. VAE 的 17.8 s 里有 14 s 在 CPU（2026-11-11）

### 18.1 结论

VAE 解码 wall **17.8 s**，但 DSP 只忙 **3.83 s**。用 `GGML_SCHED_DEBUG=2` 看分配：
末段 VAE 图被切成 **31 段，HTP0 → CPU 交替**，其中 **15 段在 CPU**，
每段边界还要在 host 之间搬 36~72 MB 的激活。

### 18.2 谁掉到 CPU 了

形态固定为 **IM2COL + MUL_MAT 这一对**（conv2d 走 im2col），共同点是 im2col 产物很大：

| 节点 | im2col dst | 后端 |
|---|---|---|
| `node_925` | **324 MB** | CPU |
| `node_947` | **162 MB** | CPU |
| `node_961` | 36 MB | HTP0 |

所以是**尺寸门槛**（在 36~162 MB 之间的某个常量），不是类型或形状问题。

### 18.3 为什么 512² 更惨

激活随像素线性涨，落 CPU 的层数和每段拷贝都变大，所以 512² 的 VAE 从 17.8 s → **152 s**（8.6×）。

### 18.4 可试的修法（都未做）

1. **让大卷积走 HTP 原生 `CONV_2D`**：`conv_direct.cpp` 的 auto 判定要求后端有 implicit GEMM，
   HTP 被一刀切关掉。手动打开 `--conv2d-direct` 可能直接绕开 im2col。
2. **im2col 分块**：内核里已经有 `owb` 分块的搜索循环，看是哪条判定把大尺寸那一档拒绝了。
3. **换权重类型**：`GGML_HEXAGON_MM_SELECT` 之类，或把这几层的 bf16 转成 HTP 更擅长的量化格式。

---

---

## 19. VAE 掉 CPU 已修好：`--conv-direct auto,vae=on`（2026-11-11）

### 19.1 根因链

| 环节 | 事实 |
|---|---|
| sd.cpp 的 auto 探测 | `src/core/probe/conv_direct.cpp:88` 只在 `sd_backend_supports_conv_implicit_gemm()` 为真时才开 direct |
| HTP0 | 没有 implicit GEMM → 日志固定打 `conv2d direct auto: HTP0 -> off (no implicit GEMM path)` |
| 于是 conv2d 走 im2col | `IM2COL` + `MUL_MAT` 一对，im2col 产物 256² 就有 **324 MB**（162 MB 的也过不去） |
| HTP 的尺寸门槛 | `htp_kparams_solve_im2col()` 的 `owb` 循环到 1 仍不满足 VTCM 约束 → `return false` → 节点落 CPU |
| 图被切碎 | sched 把 VAE 图切成 **31 段 HTP0/CPU 交替**，每段边界搬 36~72 MB 激活 |

对账：VAE 的 DSP 时间只有 3.83 s，wall 17.8 s，差额 14 s 就是这些 CPU 段 + 边界拷贝。

### 19.2 修法

HTP 后端本来就有原生 `CONV_2D`（宿主 `htp_kparams_solve_conv2d`，DSP `htp/conv2d-ops.h`），
只是 auto 探测够不到。强制开：

```sh
--conv-direct auto,vae=on
```

`tools/chain.sh` 已设为默认（`CONV_DIRECT`，回退用 `CONV_DIRECT=auto`）。

### 19.3 实测

| | im2col（回退 CPU） | conv-direct（原生 CONV_2D） |
|---|---|---|
| 256² `decode_first_stage` | 17.43 s | **1.55 s** |
| 512² `decode_first_stage` | 152 s | **79.40 s** |
| 出图 vs 原路径 | — | **MAE 0.567/255 / PSNR 46.96 dB** |

512² 端到端 **323.65 s**（此前 437.7~492.8 s），采样 28.43 s/it。

统计佐证（末段）：`CONV_2D` 只占 DSP 的 5.6%、`PAD` 2.0%，**再没有 CPU 段**。

> 注意 `PAD`：VAE 里有 circular padding，走的是单独的 PAD 算子（2.0%），
> 不是 conv 的 p0/p1 参数。

---

## 20. 原生 M-RoPE 数值修好了（2026-11-11）

### 20.1 真正的 bug：悬垂指针，不是数学

`I-039` 那次「快 3× 但出图是灰纹理」跟 sections / freq_factors / pairing 都没关系。
`GGMLRunner::set_backend_tensor_data()` 只记录指针：

```cpp
void GGMLRunner::set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
    ggml_set_input(tensor);
    backend_tensor_data_map[tensor] = data;   // ← 只是存指针
}
```

真正的 `tensor_set` 发生在图分配之后。而 `make_input(sd::Tensor(...))` 传的是右值，
**那个 Tensor 在语句结束就析构了**，于是 positions / freq_factors 全是已释放内存。

后果正好是「rope 退化成恒等」（positions 全 0 → theta=0 → cos=1, sin=0），
和当时「灰纹理、位置信息没生效」的现象一致。

### 20.2 两处修改

```cpp
// qwen_image_2_1.hpp：保活到 compute 结束
sd::Tensor<int32_t> rope_pos_holder;
sd::Tensor<float>   rope_ff_holder;
...
rope_pos_holder = sd::Tensor<int32_t>({4 * n_tok}, std::move(pos));
rope_ff_holder  = sd::Tensor<float>({n_pair}, std::move(ff));
ctx.rope_positions    = make_input(rope_pos_holder);
ctx.rope_freq_factors = make_input(rope_ff_holder);
```

```cpp
// ggml_extend.cpp：rope 之后补上「交织还原」
x4 = ggml_reshape_4d(ctx, x4, d_head / 2, 2, n_head, L);   // [pair, lane, n_head, L]
x4 = ggml_cont(ctx, ggml_permute(ctx, x4, 1, 0, 2, 3));    // [lane, pair, n_head, L]
x4 = ggml_reshape_3d(ctx, x4, d_head, n_head, L);          // 回到交织
```

交织还原严格说不是必需的（q/k 用同一排列，注意力分数不变），
但保持和 pe 路径完全一致的布局更安全，代价只有一次 4.3 MB 的 cont。

### 20.3 实测（默认参数，256²/8 步 seed42）

| | pe 路径 | 原生 M-RoPE |
|---|---|---|
| 采样 | 5.07 s/it | **1.92 s/it** |
| 端到端 | 93.96 s | **53.19 s** |
| 出图 | 基准 | **MAE 0.447/255、PSNR 48.94 dB** |

回退开关：`SD_ROPE_NATIVE=0`。

### 20.4 修好后的算子分布（末段，DSP 5.17 s / 6854 op）

| 类 | 占比 | 备注 |
|---|---|---|
| `MUL_MAT`（含注意力） | 50.5% | 其中 **`f32 128x256x32`（QK^T）24.4%，只有 35.4 GFLOPS** |
| `CONT` | **18%** | 约 10% 是 rope 的去交织/还原（每次 ~2 ms / 4.3 MB） |
| `CONV_2D`（VAE） | 5.6% | 已经全在 HTP 上 |
| `MUL_MAT_NX` | 4.2% | q4_K 3312 GFLOPS |
| `SOFT_MAX` | 3.7% | |
| `SWIGLU` / `SCALE` / `PAD` | 3.0 / 2.9 / 2.0% | |
| **`ROPE`** | **0.8%** | 原来是 65.9% |

量化矩阵乘仍然健康：q4_K `24576x265` 7549 GFLOPS、`4096x265` 5680 GFLOPS。

**下一个瓶颈是 f32 注意力**（QK^T 35 GFLOPS + SOFT_MAX + PV ≈ 31%）。
HTP 有 `htp/flash-attn-ops.h`，但 sd.cpp 的 `sd_probe_flash_attention()` 对 HTP0 没有判过，
所以现在走的是 f32 的 mul_mat + soft_max 三段式。

---

---

## 21. Flash attention：256² 可用，512² 因 HMX 数值漂移被门槛挡住（2026-11-11）

### 21.1 先说 auto 探测为什么是假阴性

`flash attention auto: HTP0 -> off (insufficient probe memory)`。
`sd_run_backend_probe()` 用 `ggml_backend_dev_memory()` 做前置检查，而 HTP 的上报是

```cpp
static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = opt_mbuf_total;    // 就是 GGML_HEXAGON_MBUF 的总和（我们设 128M,128M）
    *total = opt_mbuf_total;
}
```

跟真正的预算（2304 MB IOVA 窗口）不是一回事，也不含"可用"语义。
注意 `capability.supported()`（探针图能不能建起来）在前、这个内存门槛在后，
也就是说 **HTP 其实支持 `GGML_OP_FLASH_ATTN_EXT`**，只是被一个不适用的内存检查拦住了。

所以用 `--flash-attn off,diffusion=on` 强制开（`tools/chain.sh` 已设为默认）。

### 21.2 256²：真的更快

| | 无 FA | FA（HMX） |
|---|---|---|
| 采样 | 1.92 s/it | **1.45~1.52 s/it** |
| 单段 DSP 窗口 | 5.172 s | **3.982 s** |
| 出图 | 基准 | **MAE 1.398/255、PSNR 39.15 dB** |

单段算子分布的变化（DSP 5.172 → 3.982 s）：

| | 无 FA | FA |
|---|---|---|
| `MUL_MAT` | 50.5%（其中 QK^T `f32 128x256x32` 24.4%、35.4 GFLOPS） | 33.8%（5776 GFLOPS） |
| `SOFT_MAX` | 3.7% | 0 |
| `FLASH_ATTN_EXT` | 0 | **5.7%**（0.227 s，替掉原来 ~1.5 s 的三段式） |

256² 的 `MUL_MAT f32 128x256x32`（QK^T）和 PV 那两块彻底消失了。

### 21.3 512²：HMX 数值不可用

| 512²（4105 token） | 采样 | vs 无 FA | vs 对方内核 |
|---|---|---|---|
| 无 FA（f32 三段式） | 17.80 s/it | — | — |
| FA · HMX（默认） | **5.56 s/it** | **MAE 24.60/255（PSNR 17.2）** | vs HVX 差 23.55 |
| FA · HVX（`GGML_HEXAGON_FA_SELECT=1`） | 21.85 s/it | MAE 4.21/255（PSNR 30.0） | — |

HMX 快 3.2×，但出图会换一只**毛色不同**的猫（构图、姿态、背景一致，只有颜色和细节变了）。
三方对账能定位到是 HMX 单个内核的问题：

- 256²（265 token）：HMX vs HVX 只差 **0.558/255**（47.6 dB），两者都对；
- 512²（4105 token）：HMX vs HVX 差 **23.55/255**，而 HVX 离 f32 参考只有 4.21。

→ 误差随 S 增长，最可能是 HMX 的 online-softmax 累加（4105 个 key 分 65 个 64-block）精度不够。
DSP 侧是预编译的 applet，改不动。
HVX 虽然准（4.21），但 **21.85 s/it 比不开 FA 还慢**，没有使用价值。

### 21.4 按 token 数设门槛

`ggml_extend.cpp` 的 `ggml_ext_attention_ext()` 里加了：

```cpp
if (can_use_flash_attn && sd_backend_is(backend, "HTP")) {
    static const int64_t max_kv = [] {
        const char* e = getenv("SD_FLASH_ATTN_MAX_KV");
        return e != nullptr ? (int64_t) atoll(e) : (int64_t) 1024;
    }();
    if (max_kv > 0 && L_k > max_kv) can_use_flash_attn = false;
}
```

256²（L_k = 265）走 FA，512²（L_k = 4105）自动退回 f32 三段式。
`SD_FLASH_ATTN_MAX_KV=0` 可解除限制（想要 512² 的 3.2× 就得接受换一只猫）。

⚠️ **1024 是启发值，只在 265 / 4105 两个点上验证过**，中间尺寸没测。

### 21.5 一个必须说清楚的结论：256² 的瓶颈已经不在采样

256²/8 步端到端 ~54 s，其中：

| 段 | 耗时 |
|---|---|
| TE（Qwen3-VL-8B，CPU） | **~21 s** |
| 采样（8 步 × 1.5 s） | ~12 s |
| VAE 解码 | 1.55 s |
| 首步建图 + 三段进程拉起等固定开销 | **~20 s** |

所以 FA 在 256² 上省的那 3.7 s，端到端几乎看不出来（53.19 → 54.35 就是噪声）。
**真正吃时间的是 TE 和固定开销。** 512² 才是采样占大头（142 s / 253 s）。

**而且 TE 的耗时极不稳定。** 同一份二进制、同一组参数连跑四次，端到端
**37.16 / 53.19 / 54.35 / 62.93 s**，差异几乎全在 `get_learned_condition completed`
（**6.49 / 21.20 / 33.44 s**）；采样（1.47~1.52 s/it）和 VAE（1.55~1.56 s）都很稳。

37.16 s 那次的分段：

| 段 | 耗时 |
|---|---|
| TE | 6.49 s |
| 采样（8 × 1.48 s） | 11.8 s |
| VAE 解码 | 1.56 s |
| 首步建图 + 三段进程拉起 | **~17 s** |

TE 是 Qwen3-VL-8B Q4_K_M 跑在 CPU 上，抖动来自设备后台负载和热状态，不是代码问题。
**要再压端到端，只能动 TE（更小的量化，或想办法上 HTP）或那 ~17 s 固定开销。**

---

---

## 22. 常驻分段服务（serve.sh）曾经不可用（2026-11-11，已修，见 §23）

### 22.1 现象

```sh
sh serve.sh 18080        # 正常起来，日志里 listening on: http://0.0.0.0:18080
POST /sdapi/v1/txt2img   # 永不返回（900 s 超时）
```

三个 `sd-server-hex` 全部停在 `inet_accept`，utime 不再增长；父进程日志停在

```
[W] qwen_image_2_1: split spawned predecessor pid 30123 for layers [11, 22)
[W] qwen_image_2_1: ctx sent [4096, 13] numel 53248 (via socket)
```

worker 那侧则打出 `listening on: http://0.0.0.0:18080` —— 它把自己当成了另一个 HTTP 服务。

### 22.2 根因

分段 worker 是 `execv("/proc/self/exe", 原 argv)`（`qwen_image_2_1.hpp:641`），子进程拿同一份
argv 重跑，靠"自己也会调用 generate"进入握手（握手机制挂在 `compute()` 里）：

- **sd-cli**：main 解析完参数立刻 `generate_image()` → 进 `compute()` → 识别自己是 worker → 握手。
- **sd-server**：main 解析完参数**去起 HTTP 监听** → 永远走不到 `compute()` → 父进程死等。

`SD_DIT_SPLIT_WORKER` 只在 `diffusion_engine.cpp:698` 被读来跳过 TE/VAE 加载，
**没有任何 main 用它做分发**，所以服务形态下这个变量等于没用。

### 22.3 顺带发现的一个 bug

`image.cpp` 的 TE 释放判据是

```cpp
if (getenv("SD_DIT_RELEASE_TE") != nullptr && sd->cond_stage_model) { ... }
```

**只看变量存在、不看值**。`serve.sh` 里写 `SD_DIT_RELEASE_TE=0` 照样会把 TE 放掉
（服务日志里能看到 `conditioner released`）。所以即使握手修好，第二个请求也会因为没有编码器而失败。
应该改成 `atoi(...) != 0`。

### 22.4 当前能用的形态

`python tools/webui.py`（PC 端，默认 `http://127.0.0.1:7860`）+ **chain 模式**：
每次生成重新拉起三进程。实测

| | 往返 | 采样 | generate |
|---|---|---|---|
| 256² | 57.6 s | 1.45 s/it | 50.0 s |
| 512² | 158.3 s | 5.38 s/it（HMX FA） | 153.4 s |

出图正常。代价就是每次多付一遍模型加载 + 建图（约 17 s）。

### 22.5 要修的话

1. `examples/server/main.cpp`：`SD_DIT_SPLIT_WORKER=1` 时不启 HTTP，直接跑一次生成以进入握手；
2. `image.cpp`：TE 释放判据改 `atoi(...) != 0`。

两处都已修，见 §23。

---

---

## 23. 常驻分段服务修好：worker 分流 + timestep 随 latent 下发（2026-11-11）

### 23.1 改了什么

| 位置 | 改动 |
|---|---|
| `examples/server/main.cpp` | `new_sd_ctx` 之后、起 HTTP 之前判 `SD_DIT_SPLIT_WORKER`（`atoi != 0`）→ 不进 `initialize_server_models`、不 `svr.listen`，直接 `sd_serve_split_worker()` |
| `include/stable-diffusion.h` + `src/stable-diffusion.cpp` | 新 C API `sd_serve_split_worker(sd_ctx_t*)` |
| `src/model/diffusion/model.hpp` | `DiffusionModelRunner` 加两个虚函数：`serve_split_worker(int)`（默认 false）、`reset_split_state()`（默认空） |
| `qwen_image_2_1.hpp` | 实现 worker 驱动循环；段间帧加第三个头字段 `timestep`；worker 用它代替自己的 `inputs.timesteps` |
| `diffusion_engine.cpp` | `end_runners()` 末尾调 `reset_split_state()`；`serve_split_worker()` 转发给 runner；`sd_split_worker_mode()` 判据改 `atoi(...) != 0` |
| `image.cpp` | `SD_DIT_RELEASE_TE` 判据改 `atoi(...) != 0` |

### 23.2 为什么必须让 timestep 跟着帧走

worker **没有采样器**（不跑 `sample()`），而 `t` 要进 DiT 的 modulation（`model.forward(..., timestep, ...)`）。
以前 CLI 模式下各进程跑同一份调度、各算各的 `t`，是隐含前提；worker 循环化以后这个前提没了，
所以帧格式从 `step | ndim | dims | data` 变成 `step | ndim | ts_bits | dims | data`
（`ts_bits` 是 float 的位模式；NaN = 不知道，退回调用方的）。

副作用（好东西）：worker 与**分辨率、步数、prompt 完全解耦**，同一个常驻链可以直接换尺寸/换步数。

### 23.3 每请求重发 ctx

`reset_split_state()` 在末段（`!split_worker_ && split_prev_fd_ >= 0`）把 `split_ctx_sent_` 清 0，
于是每个请求的第一步会重新发 `(-1, ctx)`。worker 收到就替换 `split_ctx_hold_`，不用重读文件。

### 23.4 实测（v73，256²/8 步，seed 42，HTTP `/sdapi/v1/txt2img`）

| 请求 | 耗时 | 备注 |
|---|---|---|
| 第 1 次 `a cat` | **87.8 s** | 含 spawn 两个 worker + 各段加载模型 |
| 第 2 次 `a red apple on a table` | **45.7 s** | worker 在位，ctx 重发 `[4096, 14]` |
| 第 3 次 `a lighthouse at sunset` 320×192 | **43.0 s** | 换分辨率，未重启 |
| 对照 `chain.sh` 一次性跑 | 77.3 s | 每请求重付加载 |

进程状态（`ps -A | grep sd-server`）：

```
17604  tail  RSS 4.4G  inet_accept              ← 末段 = HTTP 服务
17973  [11,22) RSS 66M unix_stream_data_wait    ← 常驻 worker
18030  [0,11)  RSS 66M unix_stream_data_wait    ← 常驻 worker
```

日志特征：`split worker mode: layers [11, 22), no HTTP listener`、`split worker serving layers ...`；
**没有** `conditioner released`（TE 保留），三张图内容都对（猫 / 红苹果 / 灯塔）。

---

## 24. TE 时间不稳定：根因不是抖动，是「首次」（2026-11-11 定案并修复）

### 24.1 现象

`I-044`：同一份二进制、同一组参数连跑四次，`get_learned_condition` 用掉 **6.49 / 21.20 / 33.44 s**，
端到端 37.16~62.93 s，差异几乎全在这里。

### 24.2 诊断：单调下降，说明是冷启动

同一个常驻进程连发三个请求（prompt 各不相同，排除缓存）：

```
get_learned_condition completed, taking 22.65s
get_learned_condition completed, taking  7.54s
get_learned_condition completed, taking  5.58s
```

**单调下降、不反弹** —— 不是随机抖动，是「第一次」的一次性开销。三条证据指向同一件事：

1. **CPU 频率根本没吃满**。编码时采样 `scaling_cur_freq`：cpu0（A510）1.67 GHz / cpu4（A715）1.92 GHz /
   cpu7（X3）1.59 GHz，上限分别是 2.02 / 2.80 / 3.19 GHz。大核只跑到 50~69%。
2. **8 线程被小核拖累**。TE 只编码 10~16 个 token，本质是「把 4.8 GB 权重扫一遍」的 GEMV：
   sd.cpp 按行切分给 n_threads=8，其中 3 份落在 2.0 GHz 的 A510 上，同步屏障等它们。
3. **内存有压力**。`SwapTotal 9.96 GB / SwapFree 8.24 GB`（已用 1.7 GB zram），`MemAvailable 9.6 GB`，
   首次大批量缺页/回填的代价明显高于稳态。

### 24.3 三个修复（都已落地）

| 修复 | 做法 | 效果 |
|---|---|---|
| **TE 预热** | 新增 `sd_warmup_conditioner(ctx, rounds)`；`sd-server` 在 `svr.listen()` **之前**跑 2 轮（`SD_TE_WARMUP`，默认 2，0 = 关） | 首次请求从 22.6 s 的 TE 变成 3.6 s；服务启动多付 ~13 s（那时还没人请求） |
| **cond 缓存** | sd.cpp 自带 `ConditioningCache`（LRU 4 条），相同 prompt 直接命中 | 同 prompt 第二次 TE = **0.00 s**，端到端 41 s |
| **绑大核 + 5 线程** | `serve.sh` 默认 `AFFINITY=f8`（cpu3~7）+ `THREADS=5`；toybox 的 `taskset` 只认十六进制掩码、不认 `-c` 和 `0x` 前缀 | 预热 27.8 → **12.6 s**、TE 4.6 → 4.2 s、**连 sampling 都从 39.4 → 37.6 s**（worker 继承亲和性，host 线程不再被小核拖） |

### 24.4 实测（v73，256²/8 步）

| 配置 | 预热 2 轮 | 首请求 TE | 稳态 TE | 首请求 sampling | 稳态 sampling |
|---|---|---|---|---|---|
| 默认（8 线程 / 不绑核） | 27.82 s | 3.61 s | 4.58~4.68 s | 53.15 s | 39.35~39.80 s |
| **`f8` + 5 线程（定版）** | **12.62 s** | 3.75 s | **4.22 s** | 48.81 s | **37.57 s** |
| `f0` + 4 线程 | 15.22 s | 4.56 s | 4.56~5.04 s | 50.68 s | 38.18 s |

端到端（常驻服务、全新 prompt）：**55 s**（改前 77.5~87.4 s）；相同 prompt 复用 cond：**41 s**。

### 24.5 还没解决的

稳态 4.2 s 对应 4.8 GB / 4.2 s ≈ **1.14 GB/s** 的有效带宽，只有 LPDDR5X（67 GB/s）的 1.7%。
瓶颈是 CPU 侧 Q4_K 的反量化 GEMV 吞吐，不是带宽。要再压得改 kernel（NEON 手写 / i8mm 不可用，
A510 不支持），或把 TE 挪出 CPU —— 但 4.8 GB 权重 > 2304 MB 的 fastrpc 窗口，HTP 侧放不下。
**结论：4.2 s 是当前形态的合理下限。**

---

## 25. 进 Pocket Model Studio（APK 集成，2026-11-12）

整条分段服务已经能作为**模型包**跑在 App 里，不再需要 adb / shell 脚本。

### 25.1 为什么入口必须是 jniLibs 里的 `libsd_server.so`

App 的 `targetSdk=35`，SELinux 只允许 exec APK 的 `nativeLibraryDir`（`apk_data_file`），
app 私有目录一律 EPERM。所以：

| 东西 | 放哪 | 原因 |
|---|---|---|
| `sd-server` 入口 | `app/src/main/jniLibs/arm64-v8a/libsd_server.so` | 唯一允许 exec 的位置；`RuntimeResolver.embedded()` 会优先用这份 |
| DSP applet `libggml-htp-v73.so` | 同一个 jniLibs | adsprpc 守护进程读不了 0700 的私有目录，`ADSP_LIBRARY_PATH` 只给 nativeLibraryDir |
| 权重 / manifest | `<modelDir>/weights`、`manifest.json` | 包内数据 |
| 分段 worker | **不用单独准备** | spawn 是 `execv("/proc/self/exe")`，天然指回 `libsd_server.so`，同一个文件被 exec 三次 |

`runtime/` 目录里只放一个 `VERSION` 就够：它的存在让 App 走「随包运行框架」分支，
入口再从 jniLibs 找（包里那份 `runtime/bin/sd-server` 只是给 `pack_model check` 用的备份）。

### 25.2 manifest 的关键几项

```json
"entry": { "bin": "sd-server", "portArg": "--listen-port",
           "healthPath": "/sdapi/v1/options", "stopMode": "sigterm" },
"env": { "SD_DIT_SPLIT_PLAN": "11,22", "SD_DIT_LAYER_FIRST": "22", "SD_DIT_LAYER_LAST": "32",
         "SD_DIT_SPLIT_STEPS": "{steps}", "SD_DIT_SPLIT_EXIT": "0",
         "SD_DIT_CTX_FILE": "{filesDir}/qisplit-ctx.bin", "SD_DIT_RELEASE_TE": "0",
         "SD_CPU_AFFINITY": "f8", "SD_TE_WARMUP": "2",
         "SD_ROPE_NATIVE": "1", "SD_FLASH_ATTN_MAX_KV": "0",
         "GGML_HEXAGON_OPQUEUE": "1", "GGML_HEXAGON_MBUF": "128M,128M" }
```

- `stopMode: sigterm` → App 走 `terminateChain(modelDir)`，按 `/proc/<pid>/cmdline` 含模型目录
  找整条链，先 TERM worker 再 TERM 父进程（**绝不能 kill -9**，会漏 fastrpc PD）。
- `healthPath` 用 `/sdapi/v1/options`：sd-server 没有 `/health`，这个 GET 返回 200。
- `SD_DIT_SPLIT_STEPS` 用 `{steps}` 占位，UI 改步数时 env 跟着变（App 的 ServiceInstance 支持 `{param}`）。
- **`SD_CPU_AFFINITY`**：APK 里没法在外面套 `taskset`（ProcessBuilder 直接拉起 exe），
  所以在 `main()` 开头调 `sched_setaffinity`，fork/exec 的子进程继承 —— 一个变量搞定整条链。

### 25.3 实测（v73，256²/8 步）

| | 结果 |
|---|---|
| App 启动 → `listening on 127.0.0.1:7800` | 含 TE 预热 44.4 s（App 里比 shell 的 12.6 s 慢，`prewarm` 先读了 9.7 GB 权重） |
| 首个请求（含拉起两个 worker） | **61.3 s** |
| 第二个请求（同 prompt，命中 cond 缓存） | **40.2 s** |
| 进程形态 | 三个 `libsd_server.so`：末段（父）+ [11,22) + [0,11)，worker 各 ~16 MB RSS |
| 出图 | 正确（猫） |
| UI | 卡片显示「2.1 Turbo (3 段 NPU) / 随包运行框架 / 运行中 :7800」，局域网 192.168.1.8:7860 可访问 |

### 25.4 打包与部署

```bash
# 1) 建包（权重用硬链接挂进来，不复制 9.7 GB）
python tools/pack_model.py init  ./build/qwen-image-split --kind image --embed-runtime
python tools/pack_model.py pack  ./build/qwen-image-split -o dist/qwen-image-2.1-turbo-split.mlm
# → 9.26 GB，store 模式，dit/te 按 2 GiB 分片（Android 的 ZipInputStream 不支持 zip64）

# 2) 装 APK（jniLibs 里的 libsd_server.so 必须是最新编译产物）
gradle -p D:/pocket-model-studio assembleDebug --offline && adb install -r app-debug.apk

# 3) 本机调试：直接用 run-as 布置目录（比走 9 GB 导入快）
adb shell "run-as dev.pocketmodel.studio sh -c 'M=files/models/qwen-image-2.1-turbo-split; \
  mkdir -p \$M/weights \$M/runtime; cp /data/local/tmp/qisplit/pkg_mf.json \$M/manifest.json; \
  ln -sf /data/local/tmp/qisplit/models/dit.gguf \$M/weights/dit.gguf; ...'"

# 4) 启动（App 需在前台跑过，ServiceHost 才注册了广播接收器）
adb shell am broadcast -a dev.pocketmodel.studio.CMD --es cmd start --es model qwen-image-2.1-turbo-split
adb forward tcp:7800 tcp:7800 && curl -s 127.0.0.1:7800/sdapi/v1/options
```

**注意**：上面第 3 步用符号链接指向 `/data/local/tmp/qisplit/models`，只是本机调试省 9.7 GB；
正式分发要走进 `dist/*.mlm`（App 内 SAF 导入，解出的是真实文件）。

---

## 26. 一次整机 thrash 的复盘：不是卡死，是旧进程没退干净（2026-11-12）

### 26.1 现象

用户在 App 里发起一个生图任务，随后点了一下「分辨率选择」，整机失去响应，4 分钟后 App 与服务全部消失。
表面看像「点一下 UI 就把服务搞挂了」。

### 26.2 真实因果链

| 时间 | 事件 |
|---|---|
| 06:48 | 第三个请求开始读 4.8 GB 的 TE 权重；日志进度条停在 `270/397`，速度从 1.26 GB/s 单调衰减到 711 MB/s |
| 06:48:20~25 | 用户点开参数对话框改分辨率，百度输入法连续按键（`input_oppo` 震动事件 × 8） |
| 06:48:30 | ColorOS 的 `o-kill(4012)` 开始批量清后台：quicksearchbox / sceneservice / camera |
| 06:48:31 | **输入法自己也被杀**（pid 359），系统连着重启它 3 次 |
| 06:52:54 | `am_proc_died: [0,18023,dev.pocketmodel.studio]` |

关键证据不是日志，是 **`dumpsys meminfo` 的 DMA-BUF 行**：

```
Before: DMA-BUF: 13,406,512K   MemAvailable 1.05 GB   SwapFree 3.42 GB
After : DMA-BUF:  5,395,004K   MemAvailable 13.71 GB  SwapFree 8.23 GB
```

### 26.3 根因：三个 5 小时 25 分的残留进程

`for d in /proc/[0-9]*; do grep -c dmabuf $d/maps; done` 直接点名：

```
41 /proc/13230  ./sd-server-hex --diffusion-model models/...
40 /proc/13209  ./sd-server-hex ...
27 /proc/13089  ./sd-server-hex ...
```

`ps -o ELAPSED` = **05:25:10** —— 是 TE 调优那一轮（01:38 左右）用 `serve.sh` 起的链。
当时只按一个 PID 停，父进程死后两个 worker 没退成，一直占着 13.4 GB dma-buf 活了 5 个多小时。

于是：设备 23.6 GB 总内存、DMA-BUF 吃掉 13.4 GB → 可用只剩 1 GB →
新服务照样加载 4 GB 权重 → 三方一起 swap thrash（`pswpout` 累计 51 GB、`pgscan_direct` 4.2 TB）
→ 整片被 o-kill。

**「点分辨率」只是最后一脚**：Compose 对话框 + 新建视图树 + 拉起输入法（~90 MB），
在可用内存只剩几百 MB 时就是压垮系统的最后一次分配。

### 26.4 ★ 排错教训：grep 名字不全 = 看不见

排查时我执行的是 `ps -A | grep -E 'libsd_server|pocketmodel'` —— **全部为空，于是先怀疑「内存泄漏」**。
真相是那三个进程的名字叫 `sd-server-hex`（shell 起的），命令行用的是相对路径 `./sd-server-hex`，
既不含 `libsd_server` 也不含 `qisplit`，两次 grep 都没命中。

**以后核对「服务是否真的停了」，必须按 `dmabuf` 映射数或 cmdline 匹配，不能只 grep 进程名。**

### 26.5 新增 `tools/stop.sh`

```sh
sh stop.sh            # 默认匹配 sd-server
```

- 按 cmdline 匹配整条链，用 `/proc/<pid>/environ` 区分 worker 与末段；
- **先 TERM worker 再 TERM 末段**（反了会让 worker 僵在 socket 等待）；
- 停完复查残留（仍在就再等 5 s 报出来）；
- 顺带打印 `MemAvailable` 和 `DMA-BUF`，一眼看出 dma-buf 有没有归还；
- **绝不 kill -9**（漏 fastrpc PD）。

### 26.6 App 侧两处防呆（不在 sd-split 里，改的是 pocket-model-studio）

**① 分辨率从参数对话框搬到卡片上**

原来 `分辨率` 是 `enum` 参数，得点「参数」图标弹对话框才能选 —— 而对话框会新建视图树、
点进输入框还会拉起输入法。现在直接渲染成卡片上的一行 FilterChip：

```
已停止                      [🔧] [📄]
分辨率  [256] [320] [384]
```

点一下即改，运行中改动提示「重启服务后生效」，**完全不弹窗**。

**② 启动前内存检查从 C++ 挪到 App**

一开始在 `examples/server/main.cpp` 里加了 `SD_MEM_GUARD_MB`（默认 2048），后来按要求撤回 ——
这类检查属于「产品决策」，应该由 App 做，运行框架不该内建策略。

实现（`ServiceInstance.doStart` 开头，`new_sd_ctx` 之前）：

```kotlin
val needMb = manifest.requires.minAvailableMb.takeIf { it > 0 } ?: 1024
val availMb = readMemAvailableMb()          // /proc/meminfo 的 MemAvailable
if (availMb > 0 && availMb < needMb) { fail("可用内存不足：${availMb} MB，需要 ${needMb} MB……"); return }
```

manifest 新增字段 `requires.minAvailableMb`（与 `ramMb` 区分：`ramMb` 是「设备总内存够不够」的宣传口径，
查了也拦不住 —— 出事时设备仍有 12 GB 总内存；真正要卡的是**空闲不够**）。qwen-image 包声明 **4096**。

### 26.7 实测（v73）

| 项 | 结果 |
|---|---|
| 残留进程清理 | TERM worker → TERM 末段，dma-buf 13.4 → 5.39 GB，可用 1.05 → 13.71 GB |
| 重启服务（256²） | 首个请求 83.7 s（含 spawn worker），第二个 48.1 s，出图正确 |
| 门槛调大验证 | `minAvailableMb=30000` → 卡片显示「可用内存不足：17824 MB，需要 30000 MB」，开关自动回弹，**没有启动进程** |
| 分辨率选 320 | 命令行 `-W 320 -H 320`，请求 76.0 s，出图正确 |
| 运行中参数图标 | 置灰，点击弹 Toast 提示先停服务 |
| 卡片 UI | `运行中 :7800` + 分辨率 chips + 打开 WebUI + 局域网地址 |

### 26.8 遗留

- **App 重启后部署列表为空**：`am force-stop` 后要重新点「部署」。服务列表没有持久化，属于体验问题。
- 512² 的 VAE 仍是大头（~79 s），与本次无关。

---

## 27. 分辨率：预设 + 自定义，以及那次 `ggml_can_repeat` 崩溃的排查（2026-11-12）

### 27.1 崩溃现场

WebUI 里点生成，后端 SIGABRT（`进程退出 code=134`）：

```
ggml.c:2306: GGML_ASSERT(ggml_can_repeat(b, a)) failed
[E] qwen_image_2_1: split worker recv latent failed (step 2)
```

调用点是 DiT 的 `modulate()`：

```cpp
auto rows  = ggml_ext_chunk(runner_ctx, params, 2, 1);   // 沿 batch 维切成 image / text 两组
auto apply = [&](ggml_tensor* part, ggml_tensor* row) {
    return ggml_mul(ctx, part, row);                      // ← can_repeat(b=row, a=part) 断言
};
```

要点：`modulation.1` 是 `Linear(hidden, 4*hidden)`，forward 里先 `chunk(mod, 4, 0)` 再在
`modulate` 里 `chunk(params, 2, 1)` —— **第二刀切的是 dim=1（batch 维）**，所以
`params->ne[1]` 必须是 2。而 `ggml_ext_chunk` 自身有 `GGML_ASSERT(x->ne[dim] % num == 0)`，
日志里没有触到那一行，说明崩的确实是后面的 `ggml_mul`，即某条分支下 `row` 与 `part`
的某一维不整除。

### 27.2 复现尝试：没复现

同一进程内连发 5 个请求，含两次分辨率切换：

| # | prompt | 尺寸 | 结果 |
|---|---|---|---|
| 1 | a cat | 320² | 200 / 76.6 s |
| 2 | a lighthouse at sunset | 320² | 200 / 63.7 s |
| 3 | a dog | **256²** | 200 / 43.0 s |
| 4 | a red apple on a table | **448²** | 200 / 140.0 s |
| 5 | a lighthouse on a cliff at sunset | **512²** | 200 / 240.4 s |

五张图内容都正确（512² 也没出现之前担心的 FA 数值漂移）。当时结论是「崩溃不可稳定复现，
所以不改逻辑」——**这个结论在 §27.6 被推翻了**：不是不可复现，是我一直走 curl（cfg=1.0），
没有走 WebUI 那条（cfg=6.0）路径。保留这段是为了记住「换条请求路径再试」。

### 27.3 埋了只在即将崩溃时触发的诊断

`modulate` 的 `apply` 里加了：

```cpp
if (!ggml_can_repeat(row, part)) {
    LOG_ERROR("qwen_image_2_1 modulate shape mismatch: row ne=[...] part ne=[...] prefix_length=%lld gate=%d", ...);
}
```

正常跑零开销、零噪声；一旦再崩，日志会直接给出是哪一维不整除（`part->ne[0] % row->ne[0]`
还是 `part->ne[1] % row->ne[1]`），届时能精确定位而不是猜。

### 27.4 分辨率 UI：预设 + 自由输入

- manifest：`size` 的 options 从 `['256','320','384']` 改为 **`['320','512']`**，默认 `320`。
- 卡片上渲染成 `[320²] [512²] [自定义]`：数字档位用 `²` 写法，一行放得下。
- 点「自定义」才弹输入框（`SizeInputDialog`），输入框校验 **16 的倍数、64~1024**：
  VAE 下采样 8×、DiT patch 2×，不是 16 的倍数会让 latent 切不整。
- 选中的自定义值直接变成 chip 标签（如 `448²`）并保持高亮，不会丢失。
- 改完都提示「重启服务后生效」——`-W/-H` 只在启动时进命令行。

### 27.5 512² 的实测代价

**240.4 s**（8 步，v73）。比 320² 的 63.7 s 慢 3.8 倍，主要是 VAE 解码（~79 s）与
attention 的二次项。数值这次是干净的，但 `SD_FLASH_ATTN_MAX_KV=0` 下 HMX 仍属「可能漂」，
出图前建议先用 20 步确认一次。

### 27.6 ★ 根因找到了：WebUI 把 `cfg_scale` 写死成 6.0

用户补充了关键信息「手动在 WebUI 点生成，**10 多秒后**崩溃」。19 秒这个量级正好是
TE 编码（~9 s）+ 首步建图，于是去翻 WebUI 的请求体：

```js
body: JSON.stringify({
  prompt, steps, width: size, height: size,
  cfg_scale: 6.0,          // ← 这里
  sampler_name: 'euler', seed: -1,
})
```

**Qwen-Image Turbo 是 8 步蒸馏模型，cfg 只能是 1.0。** `cfg_scale > 1` 会让 sd.cpp 启用
uncond 分支，把 cond/uncond 拼成 **batch=2** 一起跑；而分段链的 timestep 下发是单路的
（末段发一个 float，worker 用 `from_vector({ts})` 造出 `ne[1]=1` 的 tensor），
2 个 batch 的 latent 配 1 个 timestep → `modulation` 的 batch 维对不上 →
建图时 `GGML_ASSERT(ggml_can_repeat(b, a))` 直接 SIGABRT。

对照实验（同一进程、同一 prompt、只改 cfg）：

| cfg_scale | 结果 |
|---|---|
| 1.0 | **200 / 77.1 s**，出图正确 |
| 6.0 | **502 / 19.3 s，三个进程全死** |

这也解释了为什么之前用 curl 测了五个请求一次都没复现 —— 那些全是 `cfg_scale: 1.0`；
只有从 WebUI 点按钮才会带上 6.0。**「不可复现」其实是「没走同一条请求路径」。**

修复：

1. WebUI 的 `cfg_scale` 改为 `cfgScale` 变量，值由网关 `/health` 下发（读实例参数，
   不再写死）。网关 `health()` 新增 `cfg_scale` / `size` / `steps` 三个字段。
2. 尺寸下拉（含 640²/768²，768² 必 OOM）改成 **number input + datalist**，
   预设 320/512，可自由输入，前端校验 16 的倍数、64~1024。
3. 文案纠正：原文写「NPU 跑 DiT 的 QNN 分段链，**文本编码与解码在 CPU**」——
   解码（VAE）早就挪到 HTP 上了，现在只有文本编码在 CPU。

实测：WebUI 点生成 → **完成 79.6 s（8 步 320²）**，出图正确，进程全程存活。

### 27.7 遗留

- **分辨率有两处设置会打架**：卡片上的「分辨率」决定进程启动的 `-W/-H`；WebUI 页面里的
  「尺寸」下拉是请求级参数。两者不一致时以请求为准，用户容易困惑，后续应让 WebUI 读
  `/health` 里的 `sizes` 并默认跟随卡片。
- ~~崩溃根因未定位~~ → 已定位（§27.6），诊断代码保留：以后再出现 `!ggml_can_repeat` 会直接打印 shape。

---

## 28. WebUI：加种子、锁步数（2026-11-12）

### 28.1 步数锁死 8

Turbo 的步数与 sigmas、权重是配套训出来的，改步数只会出糊图或半成品，所以**从三处一起锁死**：

| 位置 | 改动 |
|---|---|
| `manifest.params` | 删掉 `steps`（1~20 可调）——App 的参数对话框里不再出现它 |
| `manifest.entry.args` / `env` | `--steps {steps}` → `--steps 8`；`SD_DIT_SPLIT_STEPS: "{steps}"` → `"8"` |
| `assets/webui/index.html` | 原来的 `<select>`（4/8/12/20）换成只读 `<span class="pill">8 · 锁定</span>` |

★ 为什么不「删参数但不改 args」：`ArgsBuilder.build()` 对未解析的占位符会替换成**空串**，
`--steps ""` 会直接把后端参数解析打崩。删参数必须同时把占位符改成字面量。

`/health` 也因此不再下发 `steps`（它读的是实例 params 表），WebUI 侧保留默认 `lockSteps=8`，
如果哪天包里的步数不是 8，health 一旦下发就会覆盖 pill 文本 —— 显示的永远是实际在用的那个数。

### 28.2 种子

- 新增 `种子` 输入框（默认 `-1`）+ `随机` 按钮：`-1` 每次随机，填 0~4294967295 的整数则复现。
- 前端校验：非整数或越界直接拦（后端 `resolve_seed` 只认 `>=0`，越界值会被静默处理，不如前端说清楚）。
- **随机时把实际种子回填到输入框**——否则随机出来的图永远无法复现。
- 状态行后缀 `· seed 1027286664`，输入框右侧另显示「本次种子 …」。

### 28.3 ★ 坑：`info` 是 JSON **字符串**，不是对象

第一版写成 `j.info && typeof j.info === 'object'`，结果跑完状态行没有 seed、输入框也没回填。
原因在 `examples/server/routes_sdapi.cpp:661`：

```cpp
out["info"] = jsoninfo.dump();     // ← dump() 出的是 string
```

这是 A1111 的历史惯例。改成先 `JSON.parse`：

```js
let info = j.info;
if (typeof info === 'string') { try { info = JSON.parse(info); } catch (e) { info = null; } }
if (info && typeof info === 'object' && Number.isFinite(Number(info.seed))) used = Number(info.seed);
```

（后端 `prepare_info_field()` 返回的确实是 json 对象，坑在最后一步 `dump()`。）

### 28.4 实测（v73，320²/8 步）

| 项 | 结果 |
|---|---|
| WebUI 点生成（seed=-1） | **79.5 s**，状态 `完成 · 79.5 s（8 步 320² · seed 1027286664）`，种子框自动回填 |
| 同 seed 连发两次（curl 直连网关） | 59.5 s / 59.8 s，**md5 完全相同** `8a22d90b0dd192721cedd52e3dc6e682` |
| `/health` | `cfg_scale: 1.0, size: 320`，不再有 `steps` |
| 步数 | 命令行为字面 `--steps 8`，页面上无任何入口可改 |

---

## 29. 算力口径重算：DiT + TE + VAE 全计入（2026-11-12 实测 / 数值 11-20 修正）

### 29.1 ★ token 数：16× 下采样，DiT 没有 patchify

> 2026-11-20 修正。早期版本按「VAE 下采样 8×」推成 1034/1610/4106，多算 4 倍；推导过程见 **§29.7**。

Qwen-Image 2.1 的 latent 是 **16× 下采样、64 通道**：
`vae.decoder.conv1.weight = [1152, 64, 1,3,3]`，encoder / decoder 各 **4 个 resample stage**。
DiT 侧 `img_in` 是 `Linear(64→4096)` 接 `patchify(x, 1, 1)` —— **1×1 patch，一个 latent 像素就是一个 token**。

| 尺寸 | latent | 图像 token | +文本 | T |
|---|---|---|---|---|
| 256² | 16×16 | 256 | 9 | **265** |
| 320² | 20×20 | 400 | 10 | **410** |
| 512² | 32×32 | 1024 | 9 | **1033** |

（文本 token 9~10 个，随 prompt 微变。）
「latent 32×32 = 1024 token」那个版本把 VAE 当成了 8× 下采样 —— 这是错的源头。

### 29.2 三个模型的 FLOPs

- DiT：总参数 **7.115 G**（265 张量累加），**每 token 参与乘加 6.9966 G**，d = 4096，L = **32**
- TE：N = **8.191 G**（Qwen3-VL-8B Q4_K_M，399 张量数出来）
- VAE：2.1 专用，**按层逐算**（不是按参数占比估，见 §29.7 ③）

每步 = `2·N_token·T`（线性）+ `4·T²·d·L`（attention）+ `2·N_once`（调制，每步一次）：

| 尺寸 | T | 线性 | attention | 每步 | 8 步 |
|---|---|---|---|---|---|
| 256² | 265 | 3.71 TF | 0.037 TF | **3.75 TF** | 30.0 TF |
| 320² | 410 | 5.74 TF | 0.088 TF | **5.83 TF** | 46.6 TF |
| 512² | 1033 | 14.46 TF | 0.56 TF | **15.02 TF** | 120.1 TF |

TE 只编码 9~16 token：`2 × 8.191 G × 13` = **0.21 TF**（attention 可忽略）。
VAE 按 `2·k²·Cin·Cout·H·W` 逐层累加：256² **0.81 TF**、320² **1.26 TF**、512² **3.22 TF**。
（早期按「decoder 各级参数占比 55/20/15/10」估成 26 / 40 / 103 TF，大了约 30 倍 —— 把 GFLOP 当 TFLOP 记了。）

### 29.3 实测有效算力（v73，2026-11-12 14:05 重测；数值 11-20 修正）

> 本节耗时在 §30（VAE tiling）与 §29.5（原生 M-RoPE 被误删、已补回）都修好之后重测。
> 三个尺寸各跑一次 `chain.sh`（同 prompt / seed42 / 8 步 / cfg 1.0 / `--vae-tiling --vae-tile-size 320`）。

| 段 | 尺寸 | 耗时 | FLOPs | 有效算力 |
|---|---|---|---|---|
| **DiT** | 256² | 1.46 s/it × 8 = 11.7 s | 30.0 TF | **2.6 TFLOPS** |
| **DiT** | 320² | 2.18 s/it × 8 = 17.4 s | 46.6 TF | **2.7 TFLOPS** |
| **DiT** | 512² | 5.32 s/it × 8 = 42.6 s | 120.1 TF | **2.8 TFLOPS** |
| **TE**（CPU） | 任意 | 4.2 s | 0.21 TF | **0.05 TFLOPS**（= 1.14 GB/s，纯带宽受限） |
| **VAE** | 256² | 1.65 s | 0.81 TF | **0.49 TFLOPS** |
| **VAE** | 320² | 2.10 s | 1.26 TF | **0.60 TFLOPS** |
| **VAE** | 512² | 7.82 s | 3.22 TF | **0.41 TFLOPS** |

**DiT 三档是 2.6 / 2.7 / 2.8，随 token 数缓慢上升**（token 涨 3.9 倍、时间只涨 3.6 倍）：
每步的固定开销（调制向量、图重建）被摊薄，而 attention 只占 1%~3.7%。这才是这块硬件的真实水平。

★ **VAE 只有 0.4~0.6 TFLOPS，不是「变差了」，是 conv 在 HTP 上本来就只有 matmul 的几分之一效率**：
`CONV_2D` 走的是完全不同的通路，层数多、每层尺寸小，访存占比高；再加上 tiling 会**重复计算 tile 重叠区**
（`target_overlap` 默认 0.5），实际计算量比理想值大 1.5~3 倍 —— 按实际量折算约 1.0~1.5 TFLOPS。
所以 VAE 的优化方向是**减重叠、减层**，而不是换后端。

#### 对外换算（别当性能承诺）

同引擎（stable-diffusion.cpp）、同模型的第三方 RTX 2060 6GB 实测（见台账 `L-078`）：512² 稳态 **1.06 s/it**、
折 **14.2 TFLOPS**（fp16 tensor 峰值的 28%）。拿它比：

| 口径 | 结论 |
|---|---|
| **DiT 单算** | 2.8 vs 14.2 → **差 5.1 倍**；两边都没吃满峰值（我们按 HTP fp16 估 22.5 算约 12%） |
| **端到端（512² / 8 步）** | 我们 **57 s**（VAE 在 NPU 7.8 s）vs 它 **≈97 s**（6 GB 装不下 VAE，只能丢 CPU 解码 72.3 s）→ **我们反而快** |
| **带宽** | 我们 LPDDR5X ≈ 68 GB/s，它 360 GB/s；TE 被按死在 1.14 GB/s 就是这条限制 |
| **能效** | 2.4 / 7.1 W = **0.34 TFLOPS/W** vs 它 14.2 / 160 W = 0.089 → **≈3.8 倍**（拿我们实测比它 TDP 上限） |

一句话：**纯算力是它的一半到五分之一，端到端因为 VAE 上了 NPU 反而更快，每瓦高约四倍。**
不要再说「算力等于 2060」——那是旧 token 口径下的 11~12 TFLOPS 推出来的，已作废（§29.7 ⑤）。

### 29.4 整链（DiT + VAE 全计入）

「纯计算」= 8 步采样 + VAE 解码，**不含**三进程拉起、TE 编码、首步建图；端到端是 `chain.sh` 的
`generate_image completed` 实测值（本次三次 52.6 / 41.8 / 71.3 s），App 常驻形态见下一小节。

| 尺寸 | DiT 8 步 | VAE | 纯计算 | **纯计算算力** | 端到端（chain） | 端到端算力 |
|---|---|---|---|---|---|---|
| 256² | 11.7 s | 1.65 s | 13.4 s | **2.3 TFLOPS**（30.8 TF） | 52.6 s | 0.6 |
| 320² | 17.4 s | 2.10 s | 19.5 s | **2.4 TFLOPS**（47.9 TF） | 41.8 s | 1.1 |
| 512² | 42.6 s | 7.82 s | 50.4 s | **2.4 TFLOPS**（123.3 TF） | **71.3 s** | 1.7 |

★ **320² 与 512² 的整链效率并列（2.45，两位小数都是 2.4）**：算力不是随分辨率单调上升 ——
DiT 那部分确实在涨（2.6→2.8），但 512² 的 VAE 占比从 11% 升到 16%，把整链拉回同一水平。
**所以「分辨率越高越划算」这句话只在 DiT 层成立，整链看不出来。**

★ 端到端与纯计算之间的差就是**固定开销 ≈ 20~40 s**（三进程拉起 + TE 4.2 s + 首步建图），
随页缓存状态大幅波动 —— 320² 端到端（41.8 s）比 256²（52.6 s）还短就是这个原因。
跨尺寸横比只看「纯计算算力」那列。

（TE 的 0.21 TF 没计入：相对 DiT 可以忽略。）

#### App 内实测（14:05，常驻服务形态）

`am broadcast -a dev.pocketmodel.studio.CMD --es cmd start --es model qwen-image-2.1-turbo-split`
起的服务（`libsd_server.so` 走 nativeLibraryDir，参数与 manifest 一致），同一 prompt / seed42：

| 尺寸 | DiT | VAE | generate | HTTP 往返 | 端到端算力 |
|---|---|---|---|---|---|
| 320² | 2.13 s/it | 2.67 s | 37.16 s | **37.4 s** | 1.3 |
| 512² | 5.43 s/it | 6.89 s | 56.50 s | **56.9 s** | **2.2** |

比 `chain.sh` 还快 15~25 s（512²：56.5 vs 71.3）—— 常驻 worker 不用重新拉起、TE 与页缓存全热。
**这两行才是用户实际感知的数字**，对外报 320² 42 s / 512² 57 s 也是这里来的。

### 29.5 ★★ 3.1~3.3 倍回退：根因是原生 M-RoPE 建图块被误删（11-12 14:00 定位并修好）

**现象**（上午 11:00 的复测）：

| 尺寸 | 11-11 定版 | 复测（坏） | 倍数 | 修好后 |
|---|---|---|---|---|
| 256² DiT | 1.45 s/it（2.6 TFLOPS） | 4.52 s/it（0.8） | 3.1× | **1.46 s/it（2.6）** |
| 320² DiT | — | 6.90 s/it（0.8） | — | **2.18 s/it（2.7）** |
| 512² DiT | 5.38 s/it（2.8 TFLOPS） | 17.5 s/it（0.9） | 3.3× | **5.32 s/it（2.8）** |

**根因**：`8070048`（11-12 00:30，「sd-server 当分段末段 / timestep 随 latent 下发」）在重构
`compute()` 里取 timestep 的那几行时，**把整块「原生 M-RoPE 建图」代码一起删掉了** ——
包括 `rope_pos_holder` / `rope_ff_holder` 两个生命周期容器、positions 装箱、逐对频率修正、
`ctx.rope_positions / rope_freq_factors / rope_sections` 的赋值。

后果是 `GGMLRunnerContext::rope_positions` **永远 nullptr**：

- `ggml_extend.cpp:1008` 的 `ggml_ext_rope_apply()` 快路判据是 `runner_ctx->rope_positions != nullptr`，
  于是 HTP 的 `htp/rope-ops.c` 原生内核成了**死代码**，q/k 的 rope 一律走通用的
  「转置 + repeat 成 2 宽 + 2 mul + add」分解；
- 最内层维只有 2，HVX 无法向量化 —— 单个 `REPEAT` 就是 19 ms；
- 和 13:20 那版 `OPSTAT` 对得上：320² 里 `REPEAT|f32|2x64x411 n=328` **占 53%（9.65 s）**，
  那个 `2` 就是 rope 分解的「repeat 成 2 宽」；
- `chain.sh` / `serve.sh` / manifest 里的 `SD_ROPE_NATIVE=1` 也一起变成了**空开关**
  （这个变量当时随那段代码一起没了，`grep getenv` 在 overlay/src 里已经搜不到它）。

**怎么找到的**（值得复用）：不要按「谁可能变慢」猜，直接查**机制是否还活着**。
`grep -rn "rope_positions" overlay/src/` 只有一处读、零处写 → 快路恒不成立；
`git log -S "rope_positions" -- overlay/` 立刻定位到 8070048 删掉了赋值。
比 `git checkout` 重编二分快一个数量级（重编一次 12 min）。

**修法**：把那段原样补回 `overlay/src/model/diffusion/qwen_image_2_1.hpp` 的 `run()` 里
（容器声明放在 `build` 之前 —— `make_input` 只登记 `backend_tensor_data_map[ptr->data]`，
真正拷贝在图分配之后，传临时 Tensor 会留悬垂指针，这正是 11-11 `41088d4` 修过的坑）。
`SD_ROPE_NATIVE=0` 的回退开关也一并恢复。

**修复后实测**（v73 / seed42 / 8 步）：

| 尺寸 | DiT | VAE | 端到端（chain） |
|---|---|---|---|
| 256² | 1.46 s/it | 1.65 s | **52.6 s**（坏时 85.1） |
| 320² | 2.18 s/it | 2.10 s | **41.8 s**（坏时 79.9） |
| 512² | 5.32 s/it | 7.82 s | **71.3 s**（坏时 198.8，今早 261） |

数值：256² 出图与修复前比 **MAE 0.629/255、max 30、54.75% 像素完全相同、PSNR 46.85 dB** ——
和 11-11 记的 0.45/255、48.9 dB 同一量级（fp16 rope vs f32 pe 的正常差别），不是算错。

**之前排除的六条全部正确但都不是原因**（serve 模式 / 绑核 / 降频 / 冷页缓存 / FA 开关 / 任务类型），
而当时最可疑的「删流式（net −242 行）、OPSTAT、modulate 诊断」**恰恰都是无辜的**。
教训：**「重构时顺手删掉一段看似无关的建图代码」不会报错、不会改变出图对错，只会慢 3 倍**，
所以这类回退只能靠「机制还在不在」来查，不能靠时序联想。

---

### 29.6 ★ 口径核实：Q4_K_M 到底改了什么（16:30，逐张量复核）

用 `python tools/gguf_info.py <dit.gguf> --flops` 直接解析 GGUF 头（265 个张量 + 2 条 KV）并
按张量形状算 FLOPs，不靠估算（脚本已入库，可复跑）：

**① 参数量：7.1151 G，与旧口径一致；但 4.2 GB ≠ 2.1 G 参数**

| 项 | 数值 |
|---|---|
| 总参数（逐张量累加） | **7.1151 G** |
| 文件大小 | 4 199 565 120 B |
| 折合位宽 | **4.72 bit/参数**（Q4_K_M 标称 4.6，含 f32 scale 与少量 Q5_K/Q8_0/BF16 混搭） |
| 量化分布 | Q4_K 6.459 G（90.8%）/ Q5_K 0.554 / Q8_0 0.035 / BF16 0.034 / F32 0.017 / Q6_K 0.017 |

常见的误算：拿 4.2 GB 除以 2 B 得到「2.1 G 参数」。**4 bit 只是每参数占 0.6 字节，参数量不变**。

**② 每 token 参与乘加的只有 6.9966 G，不是 7.115 G**

| 组成 | 参数量 | 是否随 token 放大 |
|---|---|---|
| attn q/k/v | 1.611 G | 是 |
| attn out | 0.537 G | 是 |
| img_mlp（gate_up 24576×4096 + out 4096×12288） | 4.832 G | 是 |
| 头（img_in / norm_out / proj_out） | 0.017 G | 是 |
| **modulation + time_text_embed** | **0.0849 G** | **否，每步只算一次** |
| norm / scale（RMSNorm 权重） | 0.034 G | 逐元素，可忽略 |

旧公式用 7.115 G 乘 token 数，把「每步一次」的调制也算成「每 token 一次」→ 高估 1.4~1.7%。
按 6.9966 G/T 算出的每步 FLOPs 是 **3.75 / 5.83 / 15.02 TF**（265 / 410 / 1033 token），
与 §29.2 的表一致。

**③ 量化不改变 MAC 次数，所以 FLOPs 口径成立**

HTP 上 Q4_K 的 matmul 是 `dequantize_tiled_weight_to_fp16_task_q4_*()` 先把权重解包成 fp16，
再走 HMX 的 fp16 累加（`hmx-mm-kernels-tiled.h`）。也就是说：

- **乘加次数仍然是 N·T**，与权重位宽无关 → `2·N·T` 是对的，和 llama.cpp 报 tok/s 时的算法同源；
- 变的是**字节数**（4 bit → 每参数 0.6 B）和**多出来的解包整数运算**（LUT + scale 乘法，HVX 上跑）；
- 解包开销**不在 FLOPs 里，但在时间里** → 所以 2.3~2.8 TFLOPS 是「**含解包开销的等效值**」，
  纯 MAC 速率比这更高。这也是它比理论峰值好看/难看都要打折的原因。

**④ Q4_K_M 的真正意义是「装得下」，不是「算得快」**

7.115 G 参数 fp16 要 **14.2 GB**；每进程 VA 窗口 2304 MB、每段 1.45 GB。**fp16 权重根本没法三段切分驻留**。
所以 Q4_K_M 是这条链能成立的前提；FLOPs 一点没省（该算的还是 66 TF/步）。

**⑤ 算术强度：这一步是纯算力受限，不是带宽受限**

512² 每步 **7.5 T 次 MAC**（15.02 TF ÷ 2）、权重 4.2 GB → **每字节权重被复用 ~1800 次**；256² 也有 ~450 次。
  HTP 侧要「算力受限」只需 ~165 MAC/字节（22.5 TF/2 ÷ 68 GB/s），两档都在线上。
所以在 Q4 下继续压带宽（MBUF、预读）对 DiT 没有意义，能动的只有算法和精度。

**⑥ 误差条**

DiT 的 7.1151 G / 15.02 TF 是**逐张量精确值**；TE 的 8.191 G 是数张量得到的；
**VAE 的 0.81/1.26/3.22 TF 也已是逐层精确值**（`2·k²·Cin·Cout·H·W` 累加，见 §29.7 ③）——
早期那版 26/40/103 TF 的估算法已作废。

---

### 29.7 ★★ 算力口径终版修正：320² 是 410 token，不是 1610（11-20）

一个外部复核指出「1600 多算了 4 倍」，方向对、机制不对。逐项查了权重和代码，结论如下。

**① token 数的真值来自 VAE，不是 DiT**

| 证据 | 内容 | 结论 |
|---|---|---|
| `dit.safetensors` | `img_in.weight = [4096, 64]`（torch 序，in=64） | 输入 64 维 |
| `QwenImage2_1` 代码 | `img_in->forward(ctx, DiT::patchify(ctx, x, 1, 1))`（`qwen_image_2_1.hpp:522`） | **patch 1×1，没有 patchify** |
| `vae.safetensors` | `decoder.conv1.weight = [1152, 64, 1,3,3]` | **latent 就是 64 通道**（64 不是「16×2×2」） |
| `vae.safetensors` | encoder / decoder 各 **4 个 resample stage** | **下采样 16×**，不是文档里写的 8× |
| 11-11 op profile | `joint` 形状 `(4096, 265)`、`MUL_MAT f32 128x256x32` | 256² = 16²+9 = **265**，与 16× 自洽 |

所以 `265` 当年是量出来的、对的；§29.1 把它「修正」成 1034 才是错的那一步。
**320² = 20²+10 = 410，512² = 32²+9 = 1033。**

**② 修正后的算力（同机同 seed，8 步）**

| 尺寸 | T | 每步线性 | 每步 attn | 每步 | 8 步 | VAE | 合计 | 纯计算耗时 | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| 256² | 265 | 3.71 TF | 0.037 | **3.75 TF** | 30.0 TF | 0.81 TF | 30.8 TF | 13.4 s | **2.3** |
| 320² | 410 | 5.74 TF | 0.088 | **5.83 TF** | 46.6 TF | 1.26 TF | 47.9 TF | 19.5 s | **2.4** |
| 512² | 1033 | 14.46 TF | 0.56 | **15.02 TF** | 120.1 TF | 3.22 TF | 123.4 TF | 50.4 s | **2.4** |

端到端：320² 42 s → **1.1 TFLOPS**；512² 57 s → **2.2**。

> ★ 320² 与 512² 其实并列：47.9/19.5 = **2.45**、123.3/50.4 = **2.45**，两位小数都是 **2.4**。
> 本节初版按「随尺寸上升」的预期写成 2.4/2.5，是错的 —— 会涨的是 **DiT 单算**（2.6 / 2.7 / 2.8），
> 整链被 VAE 的占比摊平（VAE 只 0.4~0.6 TFLOPS，在 512² 占 16%）。

**③ VAE 的 FLOPs 第一次是算出来的，不是估的**

按 `vae.safetensors` 逐层累加 `2·k²·Cin·Cout·H·W`（5D 卷积取 kh,kw；stage 间分辨率翻倍）：
256² **0.81 TF**、320² **1.26 TF**、512² **3.22 TF**。§29.2 原文的「26 / 40 / 103 TF」大了约 30 倍
——那是把 GFLOP 当 TFLOP 记了。

**④ 自洽性旁证**

本机 LLM prefill 是 **4.4 TFLOPS**（8B Q4_K_M，pp512 271 t/s；文本 token 数无歧义）。
修正后的 DiT 2.4 与它同一量级、略低（f32 attention、N=410 偏小、每步重建图），合理；
若按旧的 12.5，DiT 会是 LLM 的 2.8 倍，反而不好解释。

**⑤ 对外表述（作废的三条）**

| 旧说法 | 新说法 |
|---|---|
| 「11~12 TFLOPS ≈ RTX 2060」 | 2.4 TFLOPS（实测）≈ 2060 **峰值 12.9 的 1/5** |
| 「按 FP16 峰值折算，利用率约八成」 | 45 TOPS 是 INT8 标称、FP16 无公开峰值 → **不折算利用率** |
| 「每瓦高 3~6 倍」 | 2.4/7.1 = **0.34 TFLOPS/W** vs 2060 峰值 12.9/160 = 0.08 → **4 倍**（拿实测比它峰值） |

**⑥ 为什么这个错误能活这么久**

`gguf_info.py --flops` 的 `--tokens` 默认值是 `1034,1610,4106`，一旦 §29.1 写死就再没被质疑过；
而 `joint (4096, 265)` 这条硬证据一直躺在 §6 和 §15 里，和它矛盾。**教训：算 FLOPs 前先量一次 token 数
（打印 `joint.ne[1]`），别从「下采样倍数」推。**

## 30. ★ VAE 512² 慢 45 倍的根因：最后一档 conv 上不了 HTP（2026-11-12 修好）

### 30.1 现象

| 尺寸 | VAE 解码 | FLOPs（逐层累加） | 时间/FLOPs |
|---|---|---|---|
| 256² | 1.22 s | 0.81 TF | 0.66 TFLOPS |
| 320² | 1.93 s | 1.26 TF | 0.65 TFLOPS |
| **512²** | **78.5~99 s** | 3.22 TF | **0.04 TFLOPS** |

FLOPs 只涨 4 倍（0.81 → 3.22），时间却涨 **64~81 倍**。前两档效率完全一致（0.65），512² 掉约 16 倍。

### 30.2 根因：`OPSTAT` 的 conv 档位缺了 512 那一档

320² 与 512² 各跑一次 `GGML_HEXAGON_OPSTAT=1`，对比 `CONV_2D` 的 signature：

```
# 320²  CONV_2D n=46, 479 ms，档位 40×40 / 80×80 / 160×160 / 320×320   ← 四档齐全
# 512²  CONV_2D n=34, 1710 ms，档位 32×32 / 64×64 / 128×128 / 256×256   ← 缺 512×512！
```

512² 的**最后一档（输出 512×512 的那几层）完全没有在 DSP 上出现** —— 它们整档落在 CPU，
按 f16 3×3 conv 在 5 线程上跑，就是那 70 多秒。

（`dsp-window` 佐证：512² 全程 45.8 s，DiT 8 步自己就要 ~45 s，VAE 在 DSP 上只剩 1.7 s。）

**为什么上不去**：HTP 的 `conv2d` kparams 求解要按 VTCM（8 MB）切输出行块，512×512×144 的输出
在 `htp_kparams_solve` 阶段就失败，整个 op 被调度器退回 CPU。320×320×144 是能解的上限附近。

### 30.3 修法：`--vae-tiling --vae-tile-size 320`

sd.cpp 自带空间分块解码（默认 tile 256）。切成 320 的 tile 后**每一档都能落在 HTP 上**：

| 尺寸 | 不开 tiling | tile=320 | 变化 |
|---|---|---|---|
| 320² | 1.93 s | **2.16 s** | +0.23 s（+12%，单 tile 走同一路径） |
| **512²** | **78.5~99 s** | **6.76~8.21 s** | **-71 s（11~12×）** |

为什么选 320 而不是默认 256：320 对 320² 几乎是零代价（只有一块），256 会把 320² 切成
2×2 带 overlap、反而要 4.8 s（实测）。tile 再大就会退回到"512 档上不了 HTP"的问题。

512² 的 `CONV_2D` 次数 34 → **414**（每块都跑一遍），但总时间从 1.71 s 变 2.61 s，
因为**全部在 DSP 上**，而不再是"少数几层在 CPU 上磨 70 秒"。

### 30.4 落地

| 位置 | 改动 |
|---|---|
| `tools/chain.sh` | 新增 `VAE_TILING="${VAE_TILING---vae-tiling --vae-tile-size 320}"`，关掉用 `VAE_TILING=` |
| `tools/serve.sh` | 同上（11-12 补，见 §30.7；此前漏了，常驻服务路径一直是「conv-direct 开、tiling 关」） |
| `manifest.entry.args` | 追加 `--vae-tiling --vae-tile-size 320` |
| `dist/*.mlm` | 需重新打包才带上（11-12 已重打） |

不需要重编二进制（sd.cpp 早就支持这组参数）。

### 30.5 实测（App 内，v73）

| | 改前 | 改后 |
|---|---|---|
| 512² VAE 解码 | 99.0 s | **8.21 s** |
| 512² 端到端 | 261.3 s | **171.0 s**（-34%） |
| 320² VAE | 1.93 s | 2.16 s |
| 出图 | — | 无接缝、内容正确（同 seed 与改前同图） |

★ 顺带更新 §29.4 的整链口径：512² 从 **0.5 TFLOPS（261 s）→ 0.72（123.3 TF / 171 s）**。
注意这仍然远低于 DiT 单算的 2.8 —— 171 s 里 DiT 只占 45 s，剩下全是拉起/编码/建图的固定开销。

★★ **14:00 追加**：本节 171.0 s 那组数据是在「原生 M-RoPE 被误删」的二进制上测的（见 §29.5）。
补回 rope 后 512² 端到端 **71.3 s**（DiT 5.32 s/it + VAE 7.82 s），整链 **1.7 TFLOPS** ——
本节表格里的「改后 171.0 s」只用于对比 VAE tiling 本身的效果，**不要当当前性能引用**。

### 30.6 11-11 的数据核对

`I-041` 记的「conv-direct 后 512² VAE 79.4 s」与今天不开 tiling 的 78.5~99 s **一致**，
说明这不是本轮的回退，而是**从 conv-direct 落地那天起就存在的瓶颈**：
conv-direct 把 512² 从 152 s 降到 79.4 s（1.9 倍），而 256² 是 17.43 → 1.55 s（11 倍）——
当时被归成「尺寸大所以要慢」，实际是最后一档根本没进 DSP。

### 30.7 遗留

- 320 是"能解 kparams 的最大 tile"的经验值，没做 336/352/384 的扫描；真正的上限在
  `htp_kparams_solve` 的 VTCM 约束里，若把 `GGML_HEXAGON_MBUF` 或 VTCM 占用调小也许能推到更大。
- tile 之间 overlap 默认 0.5，512² 实际算的像素比 512² 多约 1.5 倍；把 overlap 降到 0.25
  也许还能再省 1~2 s，但接缝风险要单独验证。

### 30.8 11-12 复核：三条路径逐条落地

改完要确认**不是只改了 chain.sh**。逐条核对结果：

| 路径 | 核对方式 | 结果 |
|---|---|---|
| `chain.sh` | 设备实跑 512²，日志 `decode_first_stage` | **7.57 s** ✓（端到端 198.8 s，DiT 17.5 s/it） |
| `serve.sh`（常驻 / App 等价配置） | `ps` 看 cmdline + 实跑 512² | cmdline 含 `--vae-tiling --vae-tile-size 320`，解码 **7.84 s** ✓ |
| 320² 默认档 | `chain.sh` 实跑 | **2.33 s**（改前 1.93，+0.4 s）✓ 无回退 |
| App manifest | 设备 `files/models/…/manifest.json` md5 | 与 `build/qwen-image-split/manifest.json` 同为 `590eeb03…` ✓ |
| App 二进制 | 已安装 `libsd_server.so` md5 vs `jniLibs` | 同为 `5d92c225…`，`grep -a -- "--vae-tile-size"` 命中 2 次 ✓ |
| 参数透传 | `ArgsBuilder.build()` | 无占位符的字面量原样下发；sdapi 路由 `gen_params = *default_gen_params` 继承 CLI 默认 ✓ |
| `dist/*.mlm` | 解包看 `manifest.json` | **原包是 02:09 打的，还是 `--steps {steps}` 的旧版**（缺 tiling、占位符会解析成空串直接把参数解析打崩）→ 已重打 |

★ 教训：**参数类改动要同时数「有几条入口」**。`--vae-tiling` 这类纯 CLI 开关没有编译期保护，
chain / serve / manifest / .mlm 四个地方各写一遍，漏一个不会有任何报错，只是慢 12 倍。

★ 上表的时间是 **rope 修复前**测的（512² DiT 17.5 s/it）。§29.5 补回原生 M-RoPE 后同一路径变成
**VAE 7.82 s、端到端 71.3 s**；结论（tiling 生效、无回退）不变。
