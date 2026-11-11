#!/usr/bin/env python3
"""plan_stages.py -- 给 diffusion 的 DiT 分段：按真实权重字节切，顺便算清楚每一笔传输代价。

和 llamasplit 的 plan_split.py 是同一个思路，但 diffusion 的账不一样，所以单独一份：

  * LLM：每 token 过一遍全链，切分**只省 VA**，代价是 K-1 次 16 KiB 拷贝（可忽略）。
  * DiT：**每个 timestep 都要过一遍全链**，一次要搬 n_tok × hidden × 4B（512² 是 12 MB 级），
    所以 K-1 次边界 × 步数 是真的要算的，不是小数。

    python tools/plan_stages.py /d/qwen-image-2.1-turbo-Q4_K_M.gguf --k 3 --height 512 --steps 8
    python tools/plan_stages.py model.gguf --k 0          # 自动：装得下就不切
    python tools/plan_stages.py model.gguf --budget 1800  # 每段权重上限 MiB（VA / 内存）

输出直接给出：每段的层区间、权重 MiB、每步跨段字节、整张图的边界传输总量、以及
「这台机器上该不该切」的结论。切点一律按「最慢段最短」求（= 权重字节最均匀）。
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_info import RE_LAYER, Reader  # noqa: E402

MIB = 1048576.0

# 不随层走的张量，按归属钉到某一段。名字取自 Qwen-Image 2.1 的 GGUF（ComfyUI 约定）。
FIRST_ONLY = ("txt_in.", "time_text_embed.", "img_in.")
LAST_ONLY = ("norm_out.", "proj_out.")
# modulation 每步都要用，且很小（44 MiB）——每段各放一份不划算，只放段 0，
# 算完之后把调制向量随 slot 头一起下传（6 × hidden，几 KiB）。
MODULATION_SHARE = ("modulation.",)


def load(path):
    r = Reader(path)
    per_layer = collections.defaultdict(int)
    first = last = mod = 0
    rest = collections.defaultdict(int)
    n_layer = 0
    for t in r.tensors:
        m = RE_LAYER.search(t["name"])
        if m:
            i = int(m.group(1))
            per_layer[i] += t["bytes"]
            n_layer = max(n_layer, i + 1)
        elif any(k in t["name"] for k in FIRST_ONLY):
            first += t["bytes"]
        elif any(k in t["name"] for k in LAST_ONLY):
            last += t["bytes"]
        elif any(k in t["name"] for k in MODULATION_SHARE):
            mod += t["bytes"]
        else:
            rest[t["name"]] += t["bytes"]
    return n_layer, [per_layer[i] for i in range(n_layer)], first, last, mod, rest


def best_split(layers, k, first, last, mod):
    """把层切成 k 段，使最重的一段最轻。DP，层数只有几十，代价可忽略。"""
    n = len(layers)
    if k > n:
        sys.exit(f"K={k} 大于层数 {n}")
    pre = [0]
    for b in layers:
        pre.append(pre[-1] + b)

    def seg_bytes(i, j):  # 层 [i,j)
        b = pre[j] - pre[i]
        if i == 0:
            b += first + mod
        if j == n:
            b += last
        return b

    INF = float("inf")
    # dp[k][j] = 前 j 层切成 k 段时最重段的最小值
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            for i in range(kk - 1, j):
                v = max(dp[kk - 1][i], seg_bytes(i, j))
                if v < dp[kk][j]:
                    dp[kk][j] = v
                    cut[kk][j] = i
    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = cut[kk][j]
        bounds.append((i, j))
        j = i
    return list(reversed(bounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--k", type=int, default=3, help="段数；0 = 按 budget 自动选最小可行 K")
    ap.add_argument("--budget", type=float, default=0,
                    help="每段权重上限 MiB（HTP VA 2048 → 给 1800；CPU 不限给 0）")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--patch", type=int, default=2, help="DiT patch size（Qwen-Image 是 2）")
    ap.add_argument("--text-tokens", type=int, default=512, help="文本条件 token 数（含 image token）")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=3072)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--bw", type=float, default=30.0, help="跨段拷贝的有效带宽 GB/s")
    a = ap.parse_args()

    n, layers, first, last, mod, rest = load(a.gguf)
    total = sum(layers) + first + last + mod + sum(rest.values())
    print(f"DiT: {n} 层, 权重合计 {total/MIB:.0f} MiB "
          f"(层 {sum(layers)/MIB:.0f} + 首 {first/MIB:.0f} + 末 {last/MIB:.0f} + mod {mod/MIB:.0f}"
          f" + 其它 {sum(rest.values())/MIB:.0f})")
    if rest:
        print("  其它张量: " + ", ".join(f"{k} {v/MIB:.1f}" for k, v in
                                       sorted(rest.items(), key=lambda x: -x[1])[:6]))

    img_tok = (a.width // 8 // a.patch) * (a.height // 8 // a.patch)
    n_tok = img_tok + a.text_tokens
    elem = 4 if a.dtype == "f32" else 2
    hidden_mib = n_tok * a.hidden * elem / MIB
    print(f"序列: 图像 token {img_tok} + 文本 {a.text_tokens} = {n_tok}，"
          f"hidden 一次 {hidden_mib:.2f} MiB ({a.dtype})")

    k = a.k
    if k == 0:
        k = 1
        if a.budget:
            while k < n:
                b = best_split(layers, k, first, last, mod)
                if all(b2 <= a.budget * MIB for _, _ in b for b2 in
                       [sum(layers[i:j]) + (first + mod if i == 0 else 0) + (last if j == n else 0)]):
                    break
                k += 1
        print(f"自动选 K={k}" + (f"（每段 ≤ {a.budget:.0f} MiB）" if a.budget else "（未给 budget，默认不切）"))

    bounds = best_split(layers, k, first, last, mod)
    bytes_of = [sum(layers[i:j]) + (first + mod if i == 0 else 0) + (last if j == n else 0)
                for i, j in bounds]

    print(f"\nK={k} 切分：")
    for idx, ((i, j), b) in enumerate(zip(bounds, bytes_of)):
        tag = []
        if i == 0:
            tag.append("含 txt_in/img_in/time_embed")
        if j == n:
            tag.append("含 norm_out/proj_out")
        print(f"  stage {idx}: 层[{i},{j}) {j-i:2d} 层  {b/MIB:7.1f} MiB  {' '.join(tag)}")

    cross = (k - 1) * hidden_mib
    total_cross = cross * a.steps
    t_per_step = cross / 1024 / a.bw * 1000  # ms
    print(f"\n边界代价：每步 {cross:.2f} MiB → {a.steps} 步共 {total_cross:.1f} MiB，"
          f"按 {a.bw:.0f} GB/s 约 {t_per_step:.1f} ms/步 ({t_per_step*a.steps:.0f} ms/张)")

    fits = not a.budget or max(bytes_of) <= a.budget * MIB
    print()
    if k == 1:
        print("结论：K=1（单进程）。没有 VA 约束时这是唯一正确的选择——"
              "切分不会让 DiT 变快，只会把一次 12 MiB 的拷贝乘上步数。")
    elif fits:
        print(f"结论：能切，最重段 {max(bytes_of)/MIB:.0f} MiB。"
              "但请记住：这只解决「装不下」，不解决「不够快」。")
    else:
        print(f"✗ 最重段 {max(bytes_of)/MIB:.0f} MiB 仍超预算 {a.budget:.0f} MiB，加大 K。")


if __name__ == "__main__":
    main()
