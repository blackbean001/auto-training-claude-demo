#!/usr/bin/env bash
# run_round.sh — 一轮完整流水线: finetune -> export+quant -> compare -> analyze
#
# 前置: 先用 make_round_config 写好 runs/<id>/round.json
#   python -m auto_finetune.make_round_config --id <id> --notes "..." [旋钮...]
#
# 跑法 (后台 + 轮询, 不要在 agent 的一个 tool call 里干等几小时):
#   nohup ./run_round.sh <id> > runs/<id>/round.out 2>&1 &
#   # 然后轮询: tail runs/<id>/train.log ; 完事看 runs/<id>/verdict.json
#
# 产物都落在 runs/<id>/ : ckpt/  train.log export.log compare.log results.csv verdict.json
# 跨轮趋势汇总在 auto_finetune/ledger.csv

set -euo pipefail

ROUND_ID="${1:?usage: run_round.sh <round_id>}"
cd "$(dirname "$0")"                      # 切到项目根 (本脚本所在处)

RUN_DIR="runs/$ROUND_ID"
ROUND_JSON="$RUN_DIR/round.json"
[ -f "$ROUND_JSON" ] || { echo "[run_round] 缺 $ROUND_JSON, 先跑 make_round_config"; exit 1; }

SAVE_DIR="$RUN_DIR/ckpt"
LOG_DIR="$RUN_DIR"
mkdir -p "$SAVE_DIR"

# ── 注入本轮旋钮到环境 (TRAJ_*); trainer/cnn_gru 经 env_overrides 读取 ──
eval "$(python -m auto_finetune.emit_env "$ROUND_ID")"

echo "[run_round] $ROUND_ID"
echo "[run_round]   train_json = $TRAJ_TRAIN_JSON"
echo "[run_round]   baseline_A = $TRAJ_BASELINE"
echo "[run_round]   eval_json  = $TRAJ_EVAL_JSON"
echo "[run_round]   comparer   = $TRAJ_COMPARER"

# ── 1/4 finetune (trainer 自带 early-stop) ──
echo "[run_round] 1/4 finetune -> $SAVE_DIR"
AUTO_SAVE_DIR="$SAVE_DIR" AUTO_LOG="$LOG_DIR" \
  python trainer_Trajectory_v78_220817_hm.py "$SAVE_DIR" "$TRAJ_TRAIN_JSON" "$LOG_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

# ── 2/4 export + quant + ort + header (export 现在读 argv = 本轮 ckpt 目录) ──
echo "[run_round] 2/4 export+quant"
python export_onnx_buffer.py "$SAVE_DIR" 2>&1 | tee "$RUN_DIR/export.log"

MODEL_B="$SAVE_DIR/model_hossom.quant.onnx"
[ -f "$MODEL_B" ] || { echo "[run_round] export 失败: 没有 $MODEL_B"; exit 2; }

# ── 3/4 compare vs 冻结基线 A (用绝对路径调 compare; cwd 仍在项目根) ──
echo "[run_round] 3/4 compare"
python "$TRAJ_COMPARER" \
    --model_a     "$TRAJ_BASELINE" \
    --model_b     "$MODEL_B" \
    --json        "$TRAJ_EVAL_JSON" \
    --data_base   "$TRAJ_DATA_BASE" \
    --past_length "$TRAJ_PAST_LENGTH" \
    --csv         "$RUN_DIR/results.csv" \
    2>&1 | tee "$RUN_DIR/compare.log"

# ── 4/4 verdict + ledger ──
echo "[run_round] 4/4 analyze"
python -m auto_finetune.analyze "$ROUND_ID"

echo "[run_round] DONE. verdict -> $RUN_DIR/verdict.json ; ledger -> auto_finetune/ledger.csv"


