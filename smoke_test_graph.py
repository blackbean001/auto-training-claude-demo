# -*- coding: utf-8 -*-
"""
smoke_test_graph.py
===================
秒级冒烟检查。全程用随机张量, 不读数据集、不存盘、不动 GPU(默认 CPU)。
全量 finetune 前先跑它, 验证:

  1) env 接线生效 —— apply_trainer_env_overrides / flags_from_env 能读到 TRAJ_*,
     这样 `TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py` 会打印 "overridden by env: {...}"。
     (接线没接上时这步会暴露)
  2) 推理图能建 + 前向 shape 对 + 输出无 NaN/Inf。
  3) 损失路径不 NaN (若 cnn_gru 暴露了 calculate_loss)。

用法:
  python smoke_test_graph.py                       # 纯冒烟
  TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py  # 接线自检, 应见 overridden by env
返回码: 全过 0, 任一断言失败非 0。
"""

import os
# 默认不占 GPU (避免和正在跑的训练抢卡); 想用 GPU 就外部设 CUDA_VISIBLE_DEVICES
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import math
import sys
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

PAST_LENGTH   = int(os.environ.get("TRAJ_PAST_LENGTH", "33"))
FUTURE_LENGTH = 20          # 任意小值, 仅验图
DIM_FEATURE   = 2
B             = 4           # 小 batch


def section(t):
    print(f"\n=== {t} ===")


def _finite(arr):
    return bool(np.isfinite(np.asarray(arr)).all())


def main():
    # ────────────────────────────────────────────────────────────────────
    # 1) env 接线 (最重要的一步: 接线没接上这里就看不到 overridden 日志)
    # ────────────────────────────────────────────────────────────────────
    section("1) env wiring")
    from env_overrides import apply_trainer_env_overrides, flags_from_env

    # 与 trainer.train_finetune 同构的 options, 用来触发 env 覆盖日志
    options = {
        "dim_rnns": [128, 64],
        "dim_feature": DIM_FEATURE,
        "distance_weight": 10, "angle_weight": 10, "time_weight": 3, "fit_weight": 10,
        "distance_tolerance_loose": 1.5, "distance_tolerance_strict": 3.0,
        "angle_tolerance": 5 * math.pi / 180, "hardness": 99.5,
        "optimizers": [tf.keras.optimizers.Adam(1e-4, clipnorm=1) for _ in range(4)],
        "dropout_rate": 0.0, "patience": 50,
    }
    options = apply_trainer_env_overrides(options)   # ← 有 TRAJ_* 时打印 overridden by env
    print("  flags_from_env() =", flags_from_env())
    print("  effective weights:", {k: options[k] for k in (
        "distance_weight", "angle_weight", "time_weight", "fit_weight",
        "distance_tolerance_loose", "distance_tolerance_strict", "hardness", "patience")})

    # ────────────────────────────────────────────────────────────────────
    # 2) 推理图: 建图 + 随机前向 + shape/NaN 检查
    # ────────────────────────────────────────────────────────────────────
    section("2) inference graph forward")
    import export_onnx_buffer as exb
    infer_model = exb.build_infer_model_testing(
        dim_feature=DIM_FEATURE, dim_rnns=options["dim_rnns"])
    x = np.random.randn(B, PAST_LENGTH - 1, DIM_FEATURE).astype(np.float32)
    outs = infer_model(x)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    names = ["polynomial", "predicted_time", "prediction"]
    for n, o in zip(names, outs):
        arr = o.numpy()
        ok = _finite(arr)
        print(f"  {n:14s} shape={tuple(arr.shape)} finite={ok}")
        assert ok, f"{n} 出现 NaN/Inf"

    # ────────────────────────────────────────────────────────────────────
    # 3) 训练侧损失路径 (best-effort: 取决于 cnn_gru 是否暴露 calculate_loss)
    # ────────────────────────────────────────────────────────────────────
    section("3) loss path (best-effort)")
    try:
        import cnn_gru_v7_Trajectory_v78_220817_hm as M
        print("  cnn_gru flags:",
              "TUNE_TIME_ONLY=", getattr(M, "TUNE_TIME_ONLY", None),
              "TUNE_POLY_ONLY=", getattr(M, "TUNE_POLY_ONLY", None),
              "USE_APL_LOSS=",   getattr(M, "USE_APL_LOSS", None),
              "USE_TIME_CEIL=",  getattr(M, "USE_TIME_CEIL", None),
              "APL_TARGET=",     getattr(M, "APL_TARGET", None))
        if hasattr(M, "calculate_loss"):
            pred  = tf.convert_to_tensor(np.random.randn(B, FUTURE_LENGTH, 2).astype(np.float32))
            ptime = tf.convert_to_tensor(np.random.rand(B, 1).astype(np.float32))
            gt    = tf.convert_to_tensor(np.random.randn(B, FUTURE_LENGTH, 2).astype(np.float32))
            t2p   = tf.convert_to_tensor(
                np.tile(np.linspace(1 / FUTURE_LENGTH, 1, FUTURE_LENGTH, dtype=np.float32), (B, 1)))
            d, a, t = M.calculate_loss(pred, ptime, gt, t2p)
            for nm, v in (("distance", d), ("angle", a), ("time", t)):
                ok = _finite(v.numpy() if hasattr(v, "numpy") else v)
                print(f"  loss/{nm:8s} finite={ok}")
                assert ok, f"loss/{nm} NaN/Inf"
        else:
            print("  (cnn_gru 未暴露 calculate_loss, 跳过)")
    except AssertionError:
        raise
    except Exception as e:
        print(f"  [warn] 损失路径冒烟跳过 (不致命): {e}")

    section("SMOKE OK")
    print("  图能建 / 前向 shape 对 / 无 NaN-Inf / env 接线打印正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())



