# -*- coding: utf-8 -*-
"""
auto_finetune/make_round_config.py
==================================
生成并校验某一轮的 round.json -> runs/<id>/round.json
这是 agent 唯一被允许的"改超参"入口: 它只写一个 JSON, 不碰任何源码。

典型用法 (agent):
  # 从零开一轮
  python -m auto_finetune.make_round_config --id 0608_aw15 \
      --notes "angle_weight 10->15, 想压 small_curves 飞线" \
      --angle-weight 15

  # 基于上一轮微调一个变量 (推荐: 单变量迭代)
  python -m auto_finetune.make_round_config --id 0608_aw15_apl --from 0608_aw15 \
      --notes "在 aw15 基础上开 APL floor 防长度退化" \
      --use-apl-loss true --apl-target 0.9
"""

import argparse
import json
import os
import sys
from datetime import datetime

from auto_finetune import config as C


def _str2bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _default_round():
    return {
        "round_id": None,
        "created": None,
        "notes": "",
        "train_data_json": C.ALL_DATA_JSON[0],
        "weights": dict(C.DEFAULT_WEIGHTS),
        "lr": list(C.DEFAULT_LR),
        "flags": dict(C.DEFAULT_FLAGS),
    }


def _load_round(round_id):
    path = os.path.join(C.RUNS_DIR, round_id, "round.json")
    if not os.path.isfile(path):
        sys.exit(f"[make_round_config] --from {round_id}: no {path}")
    with open(path) as f:
        return json.load(f)


def _validate(cfg):
    errs = []
    if cfg["train_data_json"] not in C.ALL_DATA_JSON:
        errs.append(
            f"train_data_json '{cfg['train_data_json']}' 不在 config.ALL_DATA_JSON 候选清单里"
        )
    for k, v in cfg["weights"].items():
        if k in C.WEIGHT_RANGES:
            lo, hi = C.WEIGHT_RANGES[k]
            if not (lo <= v <= hi):
                errs.append(f"weights.{k}={v} 越界 [{lo},{hi}]")
    if len(cfg["lr"]) != 4:
        errs.append(f"lr 必须 4 段, 现在 {len(cfg['lr'])}")
    f = cfg["flags"]
    if f.get("TUNE_TIME_ONLY") and f.get("TUNE_POLY_ONLY"):
        errs.append("TUNE_TIME_ONLY 与 TUNE_POLY_ONLY 不能同时为 true")

    w = cfg["weights"]
    if w["distance_tolerance_loose"] > w["distance_tolerance_strict"]:
        errs.append(
                f"distance_tolerance_loose({w['distance_tolerance_loose']}) "
                f"> strict({w['distance_tolerance_strict']}), 写反了"
        )

    if errs:
        sys.exit("[make_round_config] 校验失败:\n  - " + "\n  - ".join(errs))


def main():
    p = argparse.ArgumentParser(description="Author/validate a finetune round config.")
    p.add_argument("--id", required=True, help="round id, 例如 0608_aw15")
    p.add_argument("--from", dest="from_id", default=None,
                   help="基于已有轮的 round.json 克隆再改 (推荐做单变量迭代)")
    p.add_argument("--notes", default=None, help="本轮意图 (会写进 ledger, 务必写清楚)")
    p.add_argument("--train-json", default=None, help="训练描述符 (须在 ALL_DATA_JSON 内)")
    p.add_argument("--gpu", default=None, help="本轮绑定的 GPU 号 (0~5); 不传则由批调度覆盖")

    # 权重 / 容差 / patience
    p.add_argument("--distance-weight", type=float)
    p.add_argument("--angle-weight",    type=float)
    p.add_argument("--time-weight",     type=float)
    p.add_argument("--fit-weight",      type=float)
    p.add_argument("--dist-tol-loose",  type=float)
    p.add_argument("--dist-tol-strict", type=float)
    p.add_argument("--hardness",        type=float)
    p.add_argument("--patience",        type=int)
    p.add_argument("--lr", type=float, nargs=4, metavar=("LR0", "LR1", "LR2", "LR3"))

    # flag
    p.add_argument("--tune-time-only", type=_str2bool)
    p.add_argument("--tune-poly-only", type=_str2bool)
    p.add_argument("--use-apl-loss",   type=_str2bool)
    p.add_argument("--apl-target",     type=float)
    p.add_argument("--use-time-ceil",  type=_str2bool)
    p.add_argument("--time-ceil",      type=float)
    args = p.parse_args()

    cfg = _load_round(args.from_id) if args.from_id else _default_round()
    cfg["round_id"] = args.id
    cfg["created"] = datetime.now().isoformat(timespec="seconds")
    cfg["from"]     = args.from_id          # 记录 parent, 供经验回写做受控对比
    
    if args.gpu is not None:
        cfg["gpu"] = args.gpu
    cfg.setdefault("gpu", "0")     # 默认 0; run_batch 会用 TRAJ_GPU 覆盖

    if args.notes is not None:
        cfg["notes"] = args.notes
    if args.train_json is not None:
        cfg["train_data_json"] = args.train_json

    w = cfg["weights"]
    for cli, key in [
        ("distance_weight", "distance_weight"), ("angle_weight", "angle_weight"),
        ("time_weight", "time_weight"), ("fit_weight", "fit_weight"),
        ("dist_tol_loose", "distance_tolerance_loose"),
        ("dist_tol_strict", "distance_tolerance_strict"),
        ("hardness", "hardness"), ("patience", "patience"),
    ]:
        v = getattr(args, cli)
        if v is not None:
            w[key] = v
    if args.lr is not None:
        cfg["lr"] = list(args.lr)

    fl = cfg["flags"]
    for cli, key in [
        ("tune_time_only", "TUNE_TIME_ONLY"), ("tune_poly_only", "TUNE_POLY_ONLY"),
        ("use_apl_loss", "USE_APL_LOSS"), ("apl_target", "APL_TARGET"),
        ("use_time_ceil", "USE_TIME_CEIL"), ("time_ceil", "TIME_CEIL"),
    ]:
        v = getattr(args, cli)
        if v is not None:
            fl[key] = v

    _validate(cfg)

    out_dir = os.path.join(C.RUNS_DIR, args.id)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "round.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(out)
    print(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



