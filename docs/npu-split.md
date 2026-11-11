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

## 9. 服务模式（`sd-server`，2026-11-11）

两个脚本，按稳定度排序：

| 脚本 | 形态 | 实测 |
|---|---|---|
| **`tools/serve1.sh`（推荐先跑）** | **不切分**，单进程 + IOVA 流式（`MAPBUDGET=1400M RPCMEM_FREE=1`） | **116.5 s / 110.4 s**（256²、8 步），出图 md5 与 `sd-cli` 基线**完全相同** |
| `tools/serve.sh` | 分段链（末段 = `sd-server`，前两段 worker） | **暂不可用**，见下面的坑 |

启动：`adb shell "cd /data/local/tmp/qisplit && nohup sh serve1.sh 18080 > log_srv1.txt 2>&1 &"`，
PC 侧 `adb forward tcp:18080 tcp:18080`，然后 `POST http://127.0.0.1:18080/sdapi/v1/txt2img`。

可用接口（见 `examples/server/api.md`）：`/sdapi/v1/txt2img`（A1111 风格，最省事）、
`/v1/images/generations`（OpenAI 风格）、`/sdcpp/v1/img_gen`（原生 + 异步 job）。

### ⚠️ 分段 + server 的已知坑：worker 会继承 argv 自己也去 listen

`split_spawn_predecessor()` 是拿 `/proc/self/cmdline` 重新 `execv("/proc/self/exe")`。
在 `sd-cli` 下这没问题（子进程跑完就退）；但在 `sd-server` 下，子进程**继承了 `--listen-port 18080`**，
于是它也去 `svr.listen()`，日志里能看到第二个 `listening on: http://127.0.0.1:18080`，
而它本该是一个被动的 worker → 请求发起后 3 分钟没有任何 `completed in`，链路实际卡死。

修法（未做）三选一：
1. spawn 时不要 exec 自己，改成 exec **`sd-cli-hex`**（env `SD_DIT_CLI_PATH`）—— worker 用 CLI 形态跑，
   天然没有 server loop；
2. 给 worker 加 `SD_DIT_WORKER_MODE=1`，在 `main()` 里跳过 `svr.listen()`，只保留一个
   「收 x → compute → 发 joint」的驱动循环；
3. 至少在 spawn 时把 argv 里的 `--listen-port` 改成 `0`（能跑起来，但 worker 仍会浪费一次 bind 逻辑）。

另外两处为常驻 worker 准备好的改动（已编译，**未在 server 下验证**）：
`SD_DIT_SPLIT_EXIT=0` 时 worker 不再早退；末段步号在 `split_steps` 后回绕到 0，
worker 见到 step 0 就判定「新请求」并重读 `ctx.bin`（`split_ctx_hold_valid_ = false`）。

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
| **分段链（默认）** | adb 跑设备上的 `tools/chain.sh`（三进程，末段持采样器） | **126~129 s**（5.05~5.25 s/it） |
| 单进程常驻 | 转发到设备上的 `sd-server`（`tools/serve1.sh`） | **110~117 s**（7.7 s/it） |

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

`serve1.sh`（单进程常驻）也已改成 `vae=HTP0`，预算保持 1400M（VAE 644 MB 另算，实测通过）。

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

## 13. TE 下沉：末段 RSS 4.48 GB → 24 MB（2026-11-11）

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

`SD_DIT_RELEASE_TE` 只适合**一次性跑**（`chain.sh`）。常驻服务（`serve1.sh`）**必须关掉**：
释放后再来的 prompt 没有编码器可用。默认值：`chain.sh` 里 `=1`，`serve1.sh` 里不设（即关）。

---

