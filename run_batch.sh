#!/usr/bin/env bash
# run_batch.sh — 轻量版六卡批调度
# ============================================================================
# 把一批 round_id 分配到指定 GPU 上并行跑(每个 round 走完整 run_round.sh 四步),
# 并发度 = GPU 数量; 某块卡上的 round 跑完, 下一个排队的 round 顶上来。
#
# 前置: 每个 round_id 都已用 make_round_config 写好 runs/<id>/round.json
#       (--gpu 可不传, 这里会用 TRAJ_GPU 覆盖)
#
# 用法:
#   ./run_batch.sh <id1> <id2> ... <idN>
#   GPUS="0 1 2 3 4 5" ./run_batch.sh r1 r2 r3 r4 r5 r6 r7 r8     # 8 个任务跑在 6 卡上
#
# 环境变量:
#   GPUS  默认 "0 1 2 3 4 5"  (可用的卡号, 空格分隔)
#
# 日志: 每个 round 的全量输出在 runs/<id>/round.out; 批级进度打在本脚本 stdout。
# 注意: 每块卡同一时刻只跑一个 round (训练吃显存, 默认不超卖)。
# ============================================================================

set -uo pipefail
cd "$(dirname "$0")"

GPUS="${GPUS:-0 1 2 3 4 5}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"

if [ "$#" -eq 0 ]; then
    echo "usage: ./run_batch.sh <round_id> [<round_id> ...]"
    echo "       GPUS=\"0 1 2 3 4 5\" ./run_batch.sh r1 r2 ..."
    exit 1
fi

ROUNDS=("$@")
echo "[batch] $NGPU GPUs ($GPUS), ${#ROUNDS[@]} rounds: ${ROUNDS[*]}"

# 预检: 每个 round 必须已有 round.json
for id in "${ROUNDS[@]}"; do
    if [ ! -f "runs/$id/round.json" ]; then
        echo "[batch] ABORT: runs/$id/round.json 不存在, 先跑 make_round_config --id $id ..."
        exit 1
    fi
done

# gpu_pid[k] = 占用第 k 块卡的后台进程 PID (空=空闲)
declare -A gpu_pid
declare -A gpu_round

launch () {  # $1=gpu_号  $2=round_id
    local gpu="$1" id="$2"
    echo "[batch] -> GPU $gpu : $id   ($(date '+%H:%M:%S'))" >&2
    TRAJ_GPU="$gpu" nohup ./run_round.sh "$id" > "runs/$id/round.out" 2>&1 &
    gpu_pid["$gpu"]=$!
    gpu_round["$gpu"]="$id"
}

# 找一块空闲卡(顺带回收已结束的); 找到 -> 仅把卡号打到 stdout 并 return 0; 否则 return 1。
# 关键: 所有人类可读日志一律走 stderr(>&2), 否则会被 g="$(free_gpu)" 吞进 $g。
free_gpu () {
    local g pid rc
    for g in "${GPU_ARR[@]}"; do
        pid="${gpu_pid[$g]:-}"
        if [ -z "$pid" ]; then
            echo "$g"; return 0                     # 从未用过, 空闲
        fi
        if ! kill -0 "$pid" 2>/dev/null; then        # 该卡上的进程已退出 -> 回收
            wait "$pid" 2>/dev/null; rc=$?
            echo "[batch] <- GPU $g : ${gpu_round[$g]} done (rc=$rc)   ($(date '+%H:%M:%S'))" >&2
            gpu_pid["$g"]=""; gpu_round["$g"]=""
            echo "$g"; return 0
        fi
    done
    return 1
}

idx=0
TOTAL="${#ROUNDS[@]}"
while [ "$idx" -lt "$TOTAL" ]; do
    if g="$(free_gpu)"; then
        launch "$g" "${ROUNDS[$idx]}"
        idx=$((idx+1))
    else
        sleep 5   # 六卡都满, 歇会再轮询
    fi
done

# 等所有在跑的收尾
echo "[batch] all dispatched, waiting for the last ones..."
for g in "${GPU_ARR[@]}"; do
    pid="${gpu_pid[$g]:-}"
    if [ -n "$pid" ]; then
        wait "$pid" 2>/dev/null
        echo "[batch] <- GPU $g : ${gpu_round[$g]} done (rc=$?)"
    fi
done

echo "[batch] DONE. 看各 runs/<id>/verdict.json 与 lesson.md; 跨轮汇总 ledger.csv / LESSONS.md"
