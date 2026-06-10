# -*- coding: utf-8 -*-
"""
auto_finetune/config.py
=======================
全项目唯一真相源。改路径/候选数据/阈值只改这里。
agent 不应在别处硬编码这些值。
"""

import os

# 项目根 = 本文件上一级 (auto_finetune/ 的父目录)
PROJECT_DIR = os.environ.get(
    "TRAJ_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

LESSONS_MD = os.path.join(PROJECT_DIR, "auto_finetune", "LESSONS.md")

# ── 冻结基线 A ──────────────────────────────────────────────────────────────
# 跨轮唯一参照, 永远不变。所有 results.csv 的 delta_* 都是相对它算的。
# 必须填成那个"原始 hossom"导出的 quant onnx。
BASELINE_ONNX = os.path.join(
    PROJECT_DIR, "model", "Trajectory_v78_220817_hm", "model_hossom.quant.onnx"
)

# ── 评测描述符 (valid_set 里每个文件 = 一个类别) ───────────────────────────
# 注意: 与训练 json 是两个不同文件, 不要混用。
EVAL_JSON = os.path.join(PROJECT_DIR, "data", "data_Trajectory_v78_260525_jason_tcl.json")

# compare_onnx_hossom.py 的 --data_base (mm 数据所在根, 由 process_data.py 产出)
DATA_BASE = os.environ.get(
    "TRAJ_DATA_BASE",
    "/home/jason/sunia_trajectoryprediction/torch_version/data",
)

PAST_LENGTH = 33   # feature = (past_length-1, 2) = (32, 2), 与 model_hossom 导出一致

# ── 脚本 (项目根下 / torch_version 下) ──────────────────────────────────────
TRAINER  = "trainer_Trajectory_v78_220817_hm.py"
EXPORTER = "export_onnx_buffer.py"
# ★ 修复: compare 和它依赖的 train_polyhead 都在 torch_version/ 下。
#   用绝对路径调用 → 即使 cwd 在项目根, compare 内部 _HERE 仍解析到 torch_version,
#   train_polyhead 照样能 import; --csv 也照样落在项目根的 runs/ 下。
COMPARER = os.path.join(PROJECT_DIR, "torch_version", "compare_onnx_hossom.py")

# ── 运行产物 ────────────────────────────────────────────────────────────────
RUNS_DIR   = os.path.join(PROJECT_DIR, "runs")
LEDGER_CSV = os.path.join(PROJECT_DIR, "auto_finetune", "ledger.csv")

# ── 允许的训练描述符 (agent 只能从中选一个) ─────────────────────────────────
ALL_DATA_JSON = [
    "data/data_Trajectory_v78_260525_jason_tcl_reduce_fast.json",
    "data/data_Trajectory_v78_260525_jason_tcl.json",
    "data_Trajectory_v78_260525_hossom.json",
    "data_Trajectory_v78_260525_hossom_reduce_fast_but_small_curve.json",
    "data_Trajectory_v78_260525_jason_tcl_plus_badcases.json",
    "data_Trajectory_v78_260525_jason_tcl_reduce_fast_normal.json",
    "data_Trajectory_v78_260525_jason_tcl_reduce_fast_plus_badcases.json",
    "data_Trajectory_v78_260525_jason_tcl_reduce_slow.json",
    "data_Trajectory_v78_260525_jason_tcl_reduce_slow_normal.json"
]

# ── 类别匹配 (对 results.csv 的 file 列做子串匹配) ──────────────────────────
# ★ 接好后第一轮务必核对: 这两个子串要能命中 EVAL_JSON 里 curve 类文件的 basename。
CURVE_PATTERNS    = ["small_curves", "big_curves"]
STRAIGHT_PATTERNS = ["straight"]

# ── 默认旋钮 (镜像 trainer.train_finetune 当前取值) ─────────────────────────
DEFAULT_WEIGHTS = {
    "distance_weight":          10.0,
    "angle_weight":             10.0,
    "time_weight":              3.0,
    "fit_weight":               10.0,
    "distance_tolerance_loose": 1.5,
    "distance_tolerance_strict":3.0,
    "hardness":                 99.5,
    "patience":                 50,    # ★ 修复: 对齐 trainer 当前的 50 (原先是 10)
}
DEFAULT_LR = [0.0001, 0.0002, 0.00004, 0.000008]
DEFAULT_FLAGS = {
    "TUNE_TIME_ONLY": False,
    "TUNE_POLY_ONLY": False,
    "USE_APL_LOSS":   False,
    "APL_TARGET":     0.8,
    "USE_TIME_CEIL":  False,
    "TIME_CEIL":      0.5,
}

# ── 取值合法区间 (防 agent 手抖) ────────────────────────────────────────────
WEIGHT_RANGES = {
    "distance_weight":          (0.0, 50.0),
    "angle_weight":             (0.0, 50.0),
    "time_weight":              (0.0, 50.0),
    "fit_weight":               (0.0, 30.0),
    "distance_tolerance_loose": (0.1, 6.0),
    "distance_tolerance_strict":(0.1, 8.0),
    "hardness":                 (80.0, 99.99),
    #"patience":                 (1, 1000),
}

# ── 裁决阈值 (双目标) ───────────────────────────────────────────────────────
# 目标1: 保住预测长度。B 的 APL 不得低于 A 的 (1 - APL_REGRESS_FRAC) 倍。
APL_REGRESS_FRAC = 0.05      # 5%; 按实际 APL 量级标定
# 目标2: 压拐角飞线。curve 类的 RMSD 不得变差超过 RMSD_REGRESS_TOL (mm),
#        且 good% 不得下降超过 GOOD_REGRESS_TOL (百分点)。
RMSD_REGRESS_TOL = 0.0       # mm; 0 = 必须不劣于基线
GOOD_REGRESS_TOL = 0.0       # 百分点
