# -*- coding: utf-8 -*-
"""
env_overrides.py
================
让 trainer 的 loss 权重 / 学习率 和 cnn_gru 的全局 flag 都可以由环境变量 (TRAJ_*)
覆盖，从而被 auto_finetune 的每轮 round.json 驱动。

设计原则:
  · 没有任何 TRAJ_* 时，行为与你现在手写的 train_finetune() 完全一致 ——
    所以手动跑 trainer / 直接 import cnn_gru 不受影响。
  · 不依赖 auto_finetune 包，可被根目录的 trainer / cnn_gru 直接 import。
  · 默认值刻意与 cnn_gru 顶部当前字面量保持一致 (APL_TARGET=0.8, TIME_CEIL=0.5 ...)。

被谁用:
  trainer_*.py   : options = apply_trainer_env_overrides(options)
  cnn_gru_*.py   : _F = flags_from_env() ; TUNE_TIME_ONLY = _F["TUNE_TIME_ONLY"] ...
  (具体接线见 WIRING.md)
"""

import os
import logging

# 与 cnn_gru 顶部当前字面量一致的默认 flag
_FLAG_DEFAULTS = {
    "TUNE_TIME_ONLY": False,
    "TUNE_POLY_ONLY": False,
    "USE_APL_LOSS":   False,
    "USE_TIME_CEIL":  False,
    "APL_TARGET":     0.8,
    "TIME_CEIL":      0.5,
}


def _get_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _get_int(name):
    v = os.environ.get(name)
    return int(float(v)) if v not in (None, "") else None


def _get_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def flags_from_env():
    """返回 cnn_gru 顶部那 6 个 flag 的当前取值 (env 覆盖默认)。"""
    return {
        "TUNE_TIME_ONLY": _get_bool("TRAJ_TUNE_TIME_ONLY", _FLAG_DEFAULTS["TUNE_TIME_ONLY"]),
        "TUNE_POLY_ONLY": _get_bool("TRAJ_TUNE_POLY_ONLY", _FLAG_DEFAULTS["TUNE_POLY_ONLY"]),
        "USE_APL_LOSS":   _get_bool("TRAJ_USE_APL_LOSS",   _FLAG_DEFAULTS["USE_APL_LOSS"]),
        "USE_TIME_CEIL":  _get_bool("TRAJ_USE_TIME_CEIL",  _FLAG_DEFAULTS["USE_TIME_CEIL"]),
        "APL_TARGET":     (_get_float("TRAJ_APL_TARGET")
                           if _get_float("TRAJ_APL_TARGET") is not None
                           else _FLAG_DEFAULTS["APL_TARGET"]),
        "TIME_CEIL":      (_get_float("TRAJ_TIME_CEIL")
                           if _get_float("TRAJ_TIME_CEIL") is not None
                           else _FLAG_DEFAULTS["TIME_CEIL"]),
    }


# trainer options dict 里可被覆盖的 key -> 环境变量名
_OPT_FLOAT_MAP = {
    "distance_weight":          "TRAJ_DISTANCE_WEIGHT",
    "angle_weight":             "TRAJ_ANGLE_WEIGHT",
    "time_weight":              "TRAJ_TIME_WEIGHT",
    "fit_weight":               "TRAJ_FIT_WEIGHT",
    "distance_tolerance_loose": "TRAJ_DIST_TOL_LOOSE",
    "distance_tolerance_strict":"TRAJ_DIST_TOL_STRICT",
    "hardness":                 "TRAJ_HARDNESS",
}
_OPT_INT_MAP = {
    "patience": "TRAJ_PATIENCE",
}


def apply_trainer_env_overrides(options):
    """就地覆盖 trainer 的 options dict;返回同一个 dict。

    · 数值权重/容差/patience: 见 _OPT_FLOAT_MAP / _OPT_INT_MAP
    · 学习率: 若 TRAJ_LR0..TRAJ_LR3 全部存在, 用它们重建 4 段 Adam
      (clipnorm=1, 与 train_finetune 的当前写法一致)
    没有对应 env 时保持原值不动。
    """
    changed = {}

    for key, env in _OPT_FLOAT_MAP.items():
        val = _get_float(env)
        if val is not None and key in options:
            options[key] = val
            changed[key] = val

    for key, env in _OPT_INT_MAP.items():
        val = _get_int(env)
        if val is not None and key in options:
            options[key] = val
            changed[key] = val

    lrs = [_get_float(f"TRAJ_LR{i}") for i in range(4)]
    if all(lr is not None for lr in lrs):
        import tensorflow as tf  # lazy: 只有真要改 LR 时才碰 tf
        options["optimizers"] = [
            tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1) for lr in lrs
        ]
        changed["lr"] = lrs

    if changed:
        logging.info("[env_overrides] trainer options overridden by env: %s", changed)
    else:
        logging.info("[env_overrides] no TRAJ_* trainer overrides; using literals")
    return options


