#!/system/bin/sh
# 干净停掉整条分段链（末段 + 两个 worker）。
#
# 用法：sh stop.sh [匹配串]     默认 "sd-server"
#
# 为什么需要它（2026-11-12 事故）：
#   旧 serve.sh 链只按一个 PID 停，父进程 TERM 后两个 worker 没退成，一直活着
#   5 小时 25 分，占着 13.4 GB dma-buf。整机 MemAvailable 只剩 1 GB，
#   新起的服务照样加载 4 GB 权重 → 三方一起 swap thrash → 被 OPPO o-kill 整片清掉。
#   症状看起来像「内存泄漏」，其实是一次没停干净的旧进程。
#
# 规矩：
#   1) 先 TERM worker，再 TERM 末段 —— 反了会让 worker 在等 socket 时僵住；
#   2) **绝不 kill -9** —— 会漏 fastrpc PD（pdId 泄漏，只能重启设备）；
#   3) 停完必须验证，残留要报出来。

PATTERN="${1:-sd-server}"
SELF=$$

list_pids() {
    for d in /proc/[0-9]*; do
        p="${d#/proc/}"
        [ "$p" = "$SELF" ] && continue
        cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
        case "$cl" in
            *"$PATTERN"*) echo "$p" ;;
        esac
    done
}

is_worker() {
    tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -q '^SD_DIT_SPLIT_WORKER=1'
}

PIDS=$(list_pids)
if [ -z "$PIDS" ]; then
    echo "没有匹配 \"$PATTERN\" 的进程"
    exit 0
fi

WORKERS=""
MAINS=""
for p in $PIDS; do
    if is_worker "$p"; then
        WORKERS="$WORKERS $p"
    else
        MAINS="$MAINS $p"
    fi
done

echo "worker:${WORKERS:- 无}"
echo "末段  :${MAINS:- 无}"

if [ -n "$WORKERS" ]; then
    # shellcheck disable=SC2086
    kill -TERM $WORKERS 2>/dev/null
    sleep 3
fi
if [ -n "$MAINS" ]; then
    # shellcheck disable=SC2086
    kill -TERM $MAINS 2>/dev/null
    sleep 3
fi

LEFT=$(list_pids)
if [ -n "$LEFT" ]; then
    echo "!! 仍有残留：$LEFT"
    echo "   再等 5 s 复查；仍在就手工 kill -TERM（不要 -9）"
    sleep 5
    LEFT=$(list_pids)
    [ -n "$LEFT" ] && echo "!! 依然残留：$LEFT"
fi

# 顺带报一下内存：整条链退出后 dma-buf 应该显著下降
echo "--- 现状 ---"
grep -E 'MemAvailable' /proc/meminfo 2>/dev/null
dumpsys meminfo 2>/dev/null | grep -E '^DMA-BUF:' || true
