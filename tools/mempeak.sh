#!/system/bin/sh
# mempeak.sh <tag> -- 每 2 s 采一次整机内存 + 三进程 RSS，直到 sd-cli-hex 全部退出
# 输出：idx MemAvailable MemFree SwapFree Cached Shmem SUnreclaim sum_rss n_proc
tag="${1:-x}"
out="/data/local/tmp/qisplit/mem_${tag}.txt"
: > "$out"
i=0
seen=0
while [ $i -lt 400 ]; do
    n=$(ps -A | grep -c 'sd-cli-hex')
    [ "$n" -gt 0 ] && seen=1
    read -r A F S C < /dev/null 2>/dev/null
    A=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    F=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
    S=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
    C=$(awk '/^Cached:/{print $2}' /proc/meminfo)
    H=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
    L=$(awk '/^SUnreclaim:/{print $2}' /proc/meminfo)
    R=$(ps -A -o RSS,NAME | grep 'sd-cli-hex' | awk '{s+=$1} END{printf "%d", s+0}')
    echo "$i $A $F $S $C $H $L $R $n" >> "$out"
    if [ "$seen" = "1" ] && [ "$n" = "0" ]; then
        break
    fi
    sleep 2
    i=$((i + 1))
done
# 汇总：MemAvailable 最低点、RSS 最高点
awk 'NR==1{ma=$2; rs=$8}
     {if($2<ma){ma=$2; mai=$1}; if($8>rs)rs=$8; sh=$6; li=$7; last=$0}
     END{printf "PEAK idx=%d MemAvailable_min=%.2f GB sum_rss_max=%.2f GB shmem_max=%.2f GB slab_max=%.2f GB samples=%d\n", mai, ma/1048576, rs/1048576, sh/1048576, li/1048576, NR}' "$out" >> "$out.sum"
cat "$out.sum"
