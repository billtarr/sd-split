#!/system/bin/sh
# 采样一个链式分段跑的峰值内存：/proc/meminfo 的 MemAvailable 最低点 + 每个 sd-cli 的 VmRSS 峰值。
# 用法：sh memwatch.sh <tag>
TAG=${1:-mw}
cd /data/local/tmp/qisplit || exit 1
OUT="mem_$TAG.txt"
: > "$OUT"
peak_rss=0
min_avail=999999999
seen=0
while :; do
    n=$(ps -A 2>/dev/null | grep -c ' sd-cli')
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    total=0
    detail=""
    for pid in $(ps -A 2>/dev/null | grep ' sd-cli' | awk '{print $2}'); do
        rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
        [ -z "$rss" ] && rss=0
        total=$((total + rss))
        detail="$detail $pid:$((rss / 1024))M"
    done
    echo "$(date +%H:%M:%S) procs=$n avail=$((avail / 1024))M sum_rss=$((total / 1024))M$detail" >> "$OUT"
    [ "$avail" -lt "$min_avail" ] && min_avail=$avail
    [ "$total" -gt "$peak_rss" ] && peak_rss=$total
    [ "$n" -gt 0 ] && seen=1
    [ "$seen" = "1" ] && [ "$n" = "0" ] && break
    sleep 5
done
echo "PEAK sum_rss=$((peak_rss / 1024))M  MIN avail=$((min_avail / 1024))M" >> "$OUT"
