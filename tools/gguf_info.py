#!/usr/bin/env python3
"""gguf_info.py -- 只读 GGUF 头部（metadata + tensor 表），不碰权重数据。

用途：还没下完的模型也能算出「逐层字节 / 总量 / 量化类型分布」，给分段规划用。
不依赖 gguf 包（那个包在 __init__ 里就会去读 tensor 数据，半截文件直接炸）。

    python tools/gguf_info.py <model.gguf>            # 摘要 + 逐层字节
    python tools/gguf_info.py <model.gguf> --tensors  # 再列每个 tensor
    python tools/gguf_info.py <model.gguf> --json     # 机器可读
"""

import argparse
import collections
import json
import re
import struct
import sys

# 层命名：llama 系用 blk.N.*，diffusion 系（ComfyUI 约定）用 transformer_blocks.N.*
RE_LAYER = re.compile(r"(?:^|\.)(?:blk|transformer_blocks|double_blocks|single_blocks)\.(\d+)\.")

TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

# 每 32 个权重占多少字节（K-quant 按 256 算，下面单独处理）
BPW32 = {0: 4, 1: 2, 2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 9: 36, 20: 18, 24: 1,
         25: 2, 26: 4, 27: 8, 28: 8, 30: 2, 39: 17}
BPW256 = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292, 16: 66, 17: 74,
          18: 98, 19: 50, 21: 110, 22: 82, 23: 136, 29: 56, 34: 54, 35: 66}

SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
          10: "Q", 11: "q", 12: "d"}


def tensor_bytes(ttype, nelem):
    if ttype in BPW32:
        return nelem // 32 * BPW32[ttype] if BPW32[ttype] > 4 else nelem * BPW32[ttype]
    if ttype in BPW256:
        return nelem // 256 * BPW256[ttype]
    raise ValueError(f"未知量化类型 {ttype}")


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            sys.exit("不是 GGUF 文件")
        self.version, = struct.unpack("<I", self.f.read(4))
        self.n_tensors, self.n_kv = struct.unpack("<QQ", self.f.read(16))
        self.fields = collections.OrderedDict()
        for _ in range(self.n_kv):
            k = self.s()
            self.fields[k] = self.value(struct.unpack("<I", self.f.read(4))[0])
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self.s()
            nd, = struct.unpack("<I", self.f.read(4))
            dims = list(struct.unpack("<" + "Q" * nd, self.f.read(8 * nd)))
            ttype, = struct.unpack("<I", self.f.read(4))
            off, = struct.unpack("<Q", self.f.read(8))
            nelem = 1
            for d in dims:
                nelem *= d
            self.tensors.append({"name": name, "dims": dims, "type": ttype,
                                 "offset": off, "bytes": tensor_bytes(ttype, nelem)})
        self.header_bytes = self.f.tell()

    def s(self):
        n, = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t in SCALAR:
            return struct.unpack("<" + SCALAR[t], self.f.read(struct.calcsize(SCALAR[t])))[0]
        if t == 8:
            return self.s()
        if t == 9:
            et, = struct.unpack("<I", self.f.read(4))
            n, = struct.unpack("<Q", self.f.read(8))
            if et == 8:
                return [self.s() for _ in range(n)]
            sz = struct.calcsize(SCALAR[et])
            return list(struct.unpack("<" + SCALAR[et] * n, self.f.read(sz * n)))
        raise ValueError(f"未知字段类型 {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tensors", action="store_true", help="列出每个 tensor")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    r = Reader(a.gguf)
    total = sum(t["bytes"] for t in r.tensors)
    per_layer = collections.defaultdict(int)
    n_layer = 0
    other = collections.defaultdict(int)
    for t in r.tensors:
        m = RE_LAYER.search(t["name"])
        if m:
            i = int(m.group(1))
            per_layer[i] += t["bytes"]
            n_layer = max(n_layer, i + 1)
        else:
            other[t["name"]] += t["bytes"]

    tdist = collections.Counter()
    for t in r.tensors:
        tdist[TYPE_NAMES.get(t["type"], t["type"])] += t["bytes"]

    if a.json:
        print(json.dumps({
            "version": r.version, "n_tensors": r.n_tensors, "n_layer": n_layer,
            "total_bytes": total, "header_bytes": r.header_bytes,
            "fields": {k: v for k, v in r.fields.items() if not isinstance(v, list) or len(v) < 8},
            "per_layer_bytes": {i: per_layer[i] for i in sorted(per_layer)},
            "other_bytes": dict(other),
            "type_bytes": {k: v for k, v in tdist.items()},
            "tensors": r.tensors if a.tensors else None,
        }, ensure_ascii=False, indent=2))
        return

    print(f"GGUF v{r.version}  tensors {r.n_tensors}  header {r.header_bytes/1048576:.2f} MiB")
    print(f"权重合计 {total/2**30:.3f} GiB")
    for k in ("general.architecture", "general.name", "general.size_label",
              "general.file_type", "general.quantization_version"):
        if k in r.fields:
            print(f"  {k} = {r.fields[k]}")
    arch = r.fields.get("general.architecture", "")
    for k, v in r.fields.items():
        if k.startswith(f"{arch}.") and not isinstance(v, list):
            print(f"  {k} = {v}")

    print(f"\n量化分布（按字节）: " + "  ".join(
        f"{k} {v/2**30:.2f}G" for k, v in tdist.most_common()))
    print(f"block 层数 {n_layer}" + ("" if n_layer else "（非 transformer？没有 blk.N.*）"))
    if n_layer:
        vals = [per_layer[i] for i in range(n_layer)]
        print(f"逐层 MiB: min {min(vals)/2**20:.1f} max {max(vals)/2**20:.1f} "
              f"avg {sum(vals)/len(vals)/2**20:.1f}")
        for i in range(n_layer):
            print(f"  blk.{i:<3d} {per_layer[i]/2**20:8.2f} MiB")

    print(f"\n非层张量（{len(other)} 个）:")
    for k, v in sorted(other.items(), key=lambda x: -x[1]):
        print(f"  {k:<48s} {v/2**20:9.2f} MiB")

    if a.tensors:
        print("\n全部 tensor:")
        for t in r.tensors:
            print(f"  {t['name']:<52s} {TYPE_NAMES.get(t['type'], t['type']):>8s} "
                  f"{str(t['dims']):<24s} {t['bytes']/2**20:8.2f} MiB")


if __name__ == "__main__":
    main()
