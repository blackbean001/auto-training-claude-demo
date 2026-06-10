# -*- coding: utf-8 -*-
"""
auto_finetune/emit_env.py
=========================
把 runs/<id>/round.json 翻译成一串 `export TRAJ_...=...`,
供 run_round.sh 用 `eval "$(python -m auto_finetune.emit_env <id>)"` 注入。

trainer / cnn_gru 通过 env_overrides.py 读取这些 TRAJ_* 变量;
run_round.sh 还会用到 TRAJ_TRAIN_JSON / TRAJ_BASELINE / TRAJ_EVAL_JSON /
TRAJ_DATA_BASE / TRAJ_PAST_LENGTH / TRAJ_COMPARER 这几个。
"""

import json
import os
import shlex
import sys

from auto_finetune import config as C


def _q(v):
    return shlex.quote(str(v))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m auto_finetune.emit_env <round_id>")
    round_id = sys.argv[1]
    path = os.path.join(C.RUNS_DIR, round_id, "round.json")
    if not os.path.isfile(path):
        sys.exit(f"[emit_env] no {path}")
    with open(path) as f:
        cfg = json.load(f)

    w = cfg["weights"]
    fl = cfg["flags"]
    lines = []

    def ex(name, val):
        lines.append(f"export {name}={_q(val)}")

    # 权重 / 容差 / patience
    ex("TRAJ_DISTANCE_WEIGHT", w["distance_weight"])
    ex("TRAJ_ANGLE_WEIGHT",    w["angle_weight"])
    ex("TRAJ_TIME_WEIGHT",     w["time_weight"])
    ex("TRAJ_FIT_WEIGHT",      w["fit_weight"])
    ex("TRAJ_DIST_TOL_LOOSE",  w["distance_tolerance_loose"])
    ex("TRAJ_DIST_TOL_STRICT", w["distance_tolerance_strict"])
    ex("TRAJ_HARDNESS",        w["hardness"])
    ex("TRAJ_PATIENCE",        w["patience"])

    # 学习率
    for i, lr in enumerate(cfg["lr"]):
        ex(f"TRAJ_LR{i}", lr)

    # flag (bool -> 1/0)
    ex("TRAJ_TUNE_TIME_ONLY", 1 if fl["TUNE_TIME_ONLY"] else 0)
    ex("TRAJ_TUNE_POLY_ONLY", 1 if fl["TUNE_POLY_ONLY"] else 0)
    ex("TRAJ_USE_APL_LOSS",   1 if fl["USE_APL_LOSS"]   else 0)
    ex("TRAJ_USE_TIME_CEIL",  1 if fl["USE_TIME_CEIL"]  else 0)
    ex("TRAJ_APL_TARGET",     fl["APL_TARGET"])
    ex("TRAJ_TIME_CEIL",      fl["TIME_CEIL"])

    # run_round.sh 用到的固定路径
    train_json_abs = os.path.join(C.PROJECT_DIR, cfg["train_data_json"])
    ex("TRAJ_TRAIN_JSON", train_json_abs)
    ex("TRAJ_BASELINE",   C.BASELINE_ONNX)
    ex("TRAJ_EVAL_JSON",  C.EVAL_JSON)
    ex("TRAJ_DATA_BASE",  C.DATA_BASE)
    ex("TRAJ_PAST_LENGTH", C.PAST_LENGTH)
    ex("TRAJ_COMPARER",   C.COMPARER)   # ★ 新增: compare 脚本的绝对路径
    # GPU: round.json 里的 gpu 作默认, 但外部已设的 TRAJ_GPU(批调度) 优先
    ex("TRAJ_GPU", os.environ.get("TRAJ_GPU", str(cfg.get("gpu", "0"))))

    print("\n".join(lines))


if __name__ == "__main__":
    main()



