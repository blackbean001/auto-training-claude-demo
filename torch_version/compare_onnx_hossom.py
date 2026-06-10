"""
compare_onnx_v7.py
==================
Compare two ONNX trajectory-prediction models on every file in the valid_set
of a dataset JSON. Outputs a per-file metric table plus an OVERALL row.

Changes from compare_onnx.py (train_v6 / export_onnx_v6)
----------------------------------------------------------
  [1] Imports from train_v7 (not train).
      · TrajectoryDataset produces feature (B, 33, 2) in mm units via
        SG-smooth + equi-time resampling (aligned with C++ inference).
      · DISTANCE_TOLERANCE_LOOSE / STRICT: 1.5 px / 3.0 px → 0.16 mm / 0.32 mm.
      · Both models are mm-trained; unit is consistent.

  [2] Both models use TrajectoryDatasetV6Compat — same preprocessing as model_hossom:
        · No SG smooth, no equi-time resample (raw consecutive diffs)
        · ndir = seg[P-1] − seg[P-1-5]  (v6 coordinate-diff)
        · feature (past_length-1, 2) = (32, 2) when past_length=33
        · caption in the same v6 rotation frame for both models
      Single DataLoader shared by both models → fair comparison on identical inputs.
      Controlled via --past_length (default 33 → feature (32, 2)).

  [3] Both models output (dx_mm, dy_mm) — polar→xy done inside ONNX wrapper.
      --output_mode removed; xy is the only mode.

  [4] valid_dir threshold: 0.1 px → 0.01 mm.

  [5] AAE denominator bug fixed: uses angle_valid_count not total.
      aggregate() updated consistently.

  [6] nan_mask now also covers time_err.

  [7] Prerequisite: data files must be in mm units (process_data.py).

Metrics (all in mm where applicable)
-------------------------------------
  err        composite: ANGLE_WEIGHT*AAE + LENGTH_WEIGHT*RMSD + TIME_WEIGHT*ATE
  RMSD       root-mean-square distance  (pred vs time-weighted expected, mm)
  RMSD_1st   RMSD vs first future point  caption[:,0,:]  (mm)
  MSE        mean squared distance  (= RMSD², mm²)
  ADE        average displacement error  (= RMSD for single-point)
  AAE        average angle error  (degrees in display, radians internally)
  ATE        average time error   mean(1 − pred_time)
  APL        average predicted length  ‖pred_xy‖  (mm)
  good%      fraction satisfying loose-OR-strict accuracy gate

Usage
-----
  python compare_onnx_v7.py \
      --model_a  path/to/model_a.onnx \
      --model_b  path/to/model_b_v7.onnx \
      --json     data/data_Trajectory_v78_260525_jason_tcl_processed.json \
      --data_base /home/jason/sunia_trajectoryprediction/torch_version/data \
      [--split valid] \
      [--batch_size 512] \
      [--workers 4] \
      [--csv results_v7.csv]
"""

import argparse
import csv
import json
import logging
import math
import os
import sys

import numpy as np

# ── locate train_v7.py so we can reuse TrajectoryDataset / constants ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    from train_polyhead import (
        # Dataset and constants
        DISTANCE_TOLERANCE_LOOSE,   # 0.16 mm  (was 1.5 px in train_v6)
        DISTANCE_TOLERANCE_STRICT,  # 0.32 mm  (was 3.0 px in train_v6)
        ANGLE_TOLERANCE,
        ANGLE_WEIGHT,
        LENGTH_WEIGHT,
        TIME_WEIGHT,
        EPS,
        # needed by TrajectoryDatasetV6Compat
        MIN_PAST_LENGTH,
        NORMALIZE_EVENT_COUNT,
        MIN_DIRECTION_NORM,
        MAX_DISPLACEMENT,
        # helper functions used by normalize_v6_compat
        _rotation_matrix,
        _base_dir_from_feature,
        xy_to_polar,
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Cannot import from train_v7.py – make sure it is on sys.path.\n{e}"
    )

import torch
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ════════════════════════════════════════════════════════════════════════════
# V6-compat normalize and Dataset
#   Same SG-smooth + equi-time resampling as train_v7, but:
#     · ndir = smooth[-1] − smooth[-6]  (v6 coordinate-diff, not sum of diffs)
#     · feature shape (32, 2)  = past_length_a diffs
#   caption uses the same rotation R derived from ndir_v6 → self-consistent.
# ════════════════════════════════════════════════════════════════════════════

def normalize_v6_compat(
    start:         int,
    trace:         np.ndarray,   # (N, ≥3)  x_mm, y_mm, t_ms
    past_length:   int,          # 32 for model_hossom
    future_length: int,
    add_noise:     bool = False,
):
    """
    Exact replica of train_v6 normalize(), adapted for mm-unit data.

    Differences from train_v7 normalize_v6_compat (previous attempt):
      · NO SG smooth            — model_hossom was not trained with smoothing
      · NO equi-time resample   — model_hossom was trained on raw consecutive diffs
      · ndir = seg[P-1] − seg[P-1-5]   (v6 coordinate-diff, not sum of diffs)
      · feature = dxy[:past_length-1]   (P-1 = 31 diffs)  ← wait, see below

    NOTE on feature shape:
      train_v6 produces feature (P-1, 2) = (32, 2) when past_length=33.
      But model_hossom was exported with past_length=33 → feature (32, 2).
      Here past_length is passed as 33 (default in args.past_length_a=33
      after this fix) and feature = dxy[:past_length-1] = dxy[:32] = (32, 2).

    Data is already in mm (converted offline by process_data.py).
    No DPI scaling needed here.

    Returns feature (past_length-1, 2), caption (future_length, 2)
    """
    seg  = trace[start : start + past_length + future_length]
    ndir = seg[past_length - 1, :2] - seg[past_length - 1 - NORMALIZE_EVENT_COUNT, :2]
    diff = seg[1:] - seg[:-1]
    dxy  = diff[:, :2].copy()

    if np.any(np.abs(dxy) > MAX_DISPLACEMENT):
        return None, None

    if add_noise:
        dxy = dxy * (1.0 + 0.10 * np.random.normal())
        if np.any(np.abs(dxy) > MAX_DISPLACEMENT):
            return None, None

    if np.linalg.norm(ndir) < EPS:
        return None, None

    R       = _rotation_matrix(ndir)
    dxy     = dxy @ R                                                # (P+F-1, 2)

    feature = dxy[: past_length - 1].astype(np.float32)             # (P-1, 2)
    cap_dxy = np.zeros((future_length, 2), dtype=np.float32)
    fut     = dxy[past_length - 1:]
    cap_dxy[: fut.shape[0]] = fut
    caption = np.cumsum(cap_dxy, axis=0).astype(np.float32)         # (F, 2)

    return feature, caption


class TrajectoryDatasetV6Compat(Dataset):
    """
    Dataset for model_hossom evaluation.
    Uses exact train_v6 normalize() logic on mm-unit data:
      · No SG smooth, no equi-time resampling
      · Raw consecutive diffs (unequal interval, matching training condition)
      · ndir = seg[P-1] − seg[P-1-5]  (v6 coordinate-diff)
      · feature (past_length-1, 2) = (32, 2) when past_length=33
      · caption (future_length, 2) — in same v6 rotation frame
    """

    def __init__(
        self,
        filepath:      str,
        past_length:   int  = 33,   # same as train_v6 default; feature = (P-1,2)=(32,2)
        future_length: int  = 8,
        max_history:   int  = 1_000_000,
        cache_feature: bool = False,
    ):
        self.past_length   = past_length
        self.future_length = future_length
        total = past_length + future_length

        strokes = []
        current = []

        def _flush(rows):
            if len(rows) >= MIN_PAST_LENGTH:
                arr   = np.array(rows, dtype=np.float32)
                pad_n = past_length - MIN_PAST_LENGTH
                if pad_n > 0:
                    arr = np.vstack([np.tile(arr[0], (pad_n, 1)), arr])
                strokes.append(arr[:max_history])

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    _flush(current); current = []
                else:
                    cols = line.split()
                    if len(cols) >= 3:
                        try:
                            current.append([float(cols[0]),
                                            float(cols[1]),
                                            float(cols[2])])
                        except ValueError:
                            pass
            _flush(current)

        self.samples = []
        for stroke in strokes:
            n = stroke.shape[0]
            if n < total:
                continue
            for i in range(n - total + 1):
                anchor = stroke[i + past_length - 1, :2]
                ref    = stroke[i + past_length - 1 - NORMALIZE_EVENT_COUNT, :2]
                if np.max(np.abs(anchor - ref)) <= MIN_DIRECTION_NORM:
                    continue

                t0    = stroke[i + past_length - 1, 2]
                t_end = stroke[i + past_length + future_length - 1, 2]
                denom = float(t_end - t0) if (t_end - t0) > 1e-6 else 1e-6
                t_fut = ((stroke[i + past_length : i + total, 2] - t0) / denom
                         ).astype(np.float32)

                if cache_feature:
                    feat, cap = normalize_v6_compat(
                        i, stroke, past_length, future_length)
                    if feat is None or np.isnan(feat).any():
                        continue
                    self.samples.append((feat, cap, t_fut, True))
                else:
                    self.samples.append((i, stroke, t_fut, False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if item[3]:
            feat, cap, t_fut, _ = item
        else:
            i, stroke, t_fut, _ = item
            feat, cap = normalize_v6_compat(
                i, stroke, self.past_length, self.future_length, add_noise=True)
            if feat is None:
                feat, cap = normalize_v6_compat(
                    i, stroke, self.past_length, self.future_length, add_noise=False)

        if (feat is None or cap is None
                or np.isnan(feat).any() or np.isnan(cap).any()):
            P    = self.past_length - 1    # v6: feature is (P-1, 2)
            F    = self.future_length
            feat = np.zeros((P, 2), dtype=np.float32)
            cap  = np.zeros((F, 2), dtype=np.float32)
            feat[:, 0] = 1e-3

        base_dir_np = _base_dir_from_feature(feat)        # (2,)
        base_t  = torch.from_numpy(base_dir_np).unsqueeze(0)
        cap_t   = torch.from_numpy(cap).unsqueeze(0)
        ang, ln = xy_to_polar(base_t, cap_t)
        polar_gt = torch.stack([ang.squeeze(0), ln.squeeze(0)], dim=-1)

        return (
            torch.from_numpy(feat),
            polar_gt,
            torch.from_numpy(cap),
            torch.from_numpy(t_fut),
            torch.from_numpy(base_dir_np),
        )



# ════════════════════════════════════════════════════════════════════════════
# ONNX backend
# ════════════════════════════════════════════════════════════════════════════

class OnnxModel:
    """Thin wrapper around onnxruntime.InferenceSession."""

    def __init__(self, path: str):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        out_names = [o.name for o in self.sess.get_outputs()]
        logging.info("  ONNX outputs: %s", out_names)

        def _find(*candidates):
            for name in candidates:
                if name in out_names:
                    return name
            raise ValueError(
                f"None of {candidates} found in ONNX outputs {out_names}"
            )

        self.time_name = _find("predicted_time")
        self.pred_name = _find("tf.compat.v1.squeeze", "prediction")
        self.path = path

    def run(self, feature_np: np.ndarray):
        """
        feature_np : (B, P, 2)  float32
                     P matches what this model was trained with:
                       32 for model_hossom (TrajectoryDatasetV6Compat)
                       33 for train_v7 model (TrajectoryDataset)
        Both models output (dx_mm, dy_mm) — polar→xy done in ONNX wrapper.

        Returns
        -------
        prediction : (B, 2)   dx, dy  in mm
        pred_time  : (B,)     sigmoid in (0, 1)
        """
        res = self.sess.run(
            [self.time_name, self.pred_name],
            {self.input_name: feature_np},
        )
        pred_time  = res[0][:, 0]   # (B,)
        prediction = res[1]         # (B, 2)  — (dx_mm, dy_mm)
        return prediction, pred_time


# ════════════════════════════════════════════════════════════════════════════
# Core evaluation (one pass over a DataLoader)
# ════════════════════════════════════════════════════════════════════════════

def eval_loader(model: OnnxModel, loader: DataLoader, future_len: int) -> dict:
    """
    Single-pass evaluation.  Returns scalar metrics + raw accumulators
    (keys prefixed with '_') for cross-file aggregation.

    Both model output and ground-truth caption must be in mm units.
    This is guaranteed when:
      · data files are converted by process_data.py (px → mm)
      · TrajectoryDataset from train_v7 is used (feature & caption in mm)
      · ONNX models wrap polar→xy inside the export graph

    Changes vs compare_onnx.py (v6):
      · feature passed to model: (B, 33, 2) mm  [was (B, 32, 2) px]
      · valid_dir threshold: 0.01 mm  [was 0.1 px]
      · AAE denominator = angle_valid_count  [was total]
      · nan_mask covers time_err  [was missing]
    """
    scales = np.linspace(1.0 / future_len, 1.0, future_len, dtype=np.float32)

    dist_sq            = 0.0
    dist_sq_first      = 0.0
    angle_sum          = 0.0
    angle_valid_count  = 0      # separate denominator — fixes AAE deflation
    time_sum           = 0.0
    apl_sum            = 0.0
    good               = 0
    total              = 0

    for batch in loader:
        feature, _, caption, _, _base_dir = batch
        feat_np    = feature.numpy()   # (B, 33, 2)  mm
        caption_np = caption.numpy()   # (B,  F,  2) cumulative xy in mm
        B = feat_np.shape[0]

        # ── model inference ─────────────────────────────────────────────
        # Both models output (dx_mm, dy_mm) — polar→xy done in ONNX wrapper.
        pred_xy, pred_time = model.run(feat_np)   # (B, 2), (B,)

        # ── time-weighted expected ground-truth point ────────────────────
        pt     = pred_time[:, None]                                     # (B, 1)
        weight = np.clip(
            1.0 / future_len - np.abs(pt - scales[None, :]), 0, None
        ) * future_len                                                  # (B, F)
        w_sum    = weight.sum(axis=1, keepdims=True).clip(min=EPS)     # (B, 1)
        expected = (weight[:, :, None] * caption_np).sum(axis=1) / w_sum  # (B, 2)

        # ── distance errors ──────────────────────────────────────────────
        dist_exp   = np.linalg.norm(pred_xy - expected,             axis=-1)
        dist_first = np.linalg.norm(pred_xy - caption_np[:, 0, :], axis=-1)

        # ── angle error ──────────────────────────────────────────────────
        def _cosim(a, b):
            na = np.linalg.norm(a, axis=-1, keepdims=True).clip(min=EPS)
            nb = np.linalg.norm(b, axis=-1, keepdims=True).clip(min=EPS)
            return (a / na * (b / nb)).sum(axis=-1)

        # exclude near-zero vectors (0.01 mm threshold, was 0.1 px)
        valid_dir = (
            (np.linalg.norm(pred_xy,  axis=-1) > 0.01) &
            (np.linalg.norm(expected, axis=-1) > 0.01)
        )
        cos_sim        = _cosim(pred_xy, expected)
        angle_err_full = np.arccos(np.clip(cos_sim, -0.9999, 0.9999))  # (B,)

        # valid-only — for correct AAE denominator
        angle_err_valid    = angle_err_full[valid_dir]                  # (V,)
        # zero-filled — for good% gate (consistent with train_v7 evaluate())
        angle_err_for_good = np.where(valid_dir, angle_err_full, 0.0)  # (B,)

        time_err = 1.0 - pred_time                                      # (B,)
        apl      = np.linalg.norm(pred_xy, axis=-1)                     # (B,)

        # ── drop NaN / Inf rows ──────────────────────────────────────────
        nan_mask = (
            np.isnan(dist_exp)   | np.isinf(dist_exp)   |
            np.isnan(dist_first) | np.isinf(dist_first) |
            np.isnan(time_err)   | np.isinf(time_err)   # v6 was missing this
        )
        if nan_mask.any():
            ok                 = ~nan_mask
            dist_exp           = dist_exp[ok]
            dist_first         = dist_first[ok]
            time_err           = time_err[ok]
            apl                = apl[ok]
            angle_err_valid    = angle_err_full[ok & valid_dir]
            angle_err_for_good = angle_err_for_good[ok]
            B = int(ok.sum())

        # ── accumulate ───────────────────────────────────────────────────
        dist_sq           += (dist_exp   ** 2).sum()
        dist_sq_first     += (dist_first ** 2).sum()
        angle_sum         += angle_err_valid.sum()
        angle_valid_count += len(angle_err_valid)
        time_sum          += time_err.sum()
        apl_sum           += apl.sum()
        good              += int(
            (
                (dist_exp < DISTANCE_TOLERANCE_LOOSE) |
                (
                    (dist_exp < DISTANCE_TOLERANCE_STRICT) &
                    (angle_err_for_good < ANGLE_TOLERANCE)
                )
            ).sum()
        )
        total += B

    # ── scalar metrics ───────────────────────────────────────────────────
    n     = max(total,              1)
    n_ang = max(angle_valid_count,  1)

    rmsd       = math.sqrt(dist_sq       / n)
    rmsd_first = math.sqrt(dist_sq_first / n)
    mse        = dist_sq / n
    ade        = rmsd
    aae        = angle_sum / n_ang
    ate        = time_sum  / n
    apl_mean   = apl_sum   / n

    return dict(
        err        = ANGLE_WEIGHT * aae + LENGTH_WEIGHT * rmsd + TIME_WEIGHT * ate,
        RMSD       = rmsd,
        RMSD_first = rmsd_first,
        MSE        = mse,
        ADE        = ade,
        AAE        = aae,
        ATE        = ate,
        APL        = apl_mean,
        good       = good / n,
        total      = total,
        _dist_sq        = dist_sq,
        _dist_sq_first  = dist_sq_first,
        _angle_sum      = angle_sum,
        _angle_valid    = angle_valid_count,
        _time_sum       = time_sum,
        _apl_sum        = apl_sum,
        _good           = good,
        _total          = total,
    )


def aggregate(results: list) -> dict:
    """Weighted aggregate over per-file result dicts."""
    acc = dict(
        _dist_sq=0.0, _dist_sq_first=0.0,
        _angle_sum=0.0, _angle_valid=0,
        _time_sum=0.0,  _apl_sum=0.0,
        _good=0,        _total=0,
    )
    for m in results:
        for k in acc:
            acc[k] += m[k]

    n     = max(acc["_total"],       1)
    n_ang = max(acc["_angle_valid"], 1)

    rmsd       = math.sqrt(acc["_dist_sq"]       / n)
    rmsd_first = math.sqrt(acc["_dist_sq_first"] / n)
    mse        = acc["_dist_sq"] / n
    aae        = acc["_angle_sum"] / n_ang
    ate        = acc["_time_sum"]  / n
    apl        = acc["_apl_sum"]   / n

    return dict(
        err        = ANGLE_WEIGHT * aae + LENGTH_WEIGHT * rmsd + TIME_WEIGHT * ate,
        RMSD       = rmsd,
        RMSD_first = rmsd_first,
        MSE        = mse,
        ADE        = rmsd,
        AAE        = aae,
        ATE        = ate,
        APL        = apl,
        good       = acc["_good"] / n,
        total      = acc["_total"],
    )


# ════════════════════════════════════════════════════════════════════════════
# Printing helpers
# ════════════════════════════════════════════════════════════════════════════

METRIC_KEYS   = ["err", "RMSD", "RMSD_first", "MSE", "ADE", "AAE", "ATE", "APL", "good"]
METRIC_LABELS = ["err",  "RMSD", "RMSD_1st",  "MSE",  "ADE", "AAE(°)", "ATE", "APL", "good%"]
FILE_COL_W = 42


def _fmt(key: str, val: float) -> str:
    if key == "AAE":
        return f"{math.degrees(val):7.2f}"
    if key == "good":
        return f"{val * 100:6.1f}"
    return f"{val:8.4f}"


def _metric_row(ma: dict, mb: dict) -> list:
    cells = []
    for k in METRIC_KEYS:
        a = _fmt(k, ma[k]).strip()
        b = _fmt(k, mb[k]).strip()
        cells.append(f"{a} / {b}")
    return cells


def print_table(file_rows, overall_a, overall_b, name_a, name_b):
    COL_W = 18
    header_cells = [f"{lbl:^{COL_W}}" for lbl in METRIC_LABELS]
    n_col = FILE_COL_W + 2 + COL_W * len(METRIC_KEYS) + 2 * (len(METRIC_KEYS)-1) + 10
    sep   = "=" * n_col

    print()
    print(sep)
    print(f"  Model A : {name_a}")
    print(f"  Model B : {name_b}")
    print(f"  Units   : mm  (v6-compat preprocessing, same as model_hossom)")
    print(f"  Format  : A / B   (↓ better for err/RMSD/MSE/AAE/ATE,  ↑ better for good%)")
    print()
    print(f"  {'File':<{FILE_COL_W}}  {'  '.join(header_cells)}  {'n':>8}")
    print("-" * n_col)
    for label, ma, mb in file_rows:
        cells = _metric_row(ma, mb)
        row   = "  ".join(f"{c:^{COL_W}}" for c in cells)
        print(f"  {label:<{FILE_COL_W}}  {row}  {ma['total']:>8d}")
    print("-" * n_col)
    cells = _metric_row(overall_a, overall_b)
    row   = "  ".join(f"{c:^{COL_W}}" for c in cells)
    print(f"  {'OVERALL':<{FILE_COL_W}}  {row}  {overall_a['total']:>8d}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# CSV export
# ════════════════════════════════════════════════════════════════════════════

def save_csv(file_rows, overall_a, overall_b, name_a, name_b, csv_path):
    fieldnames = ["file"]
    for lbl in METRIC_LABELS:
        fieldnames += [f"A_{lbl}", f"B_{lbl}", f"delta_{lbl}"]
    fieldnames.append("n")

    def _row_dict(label, ma, mb):
        d = {"file": label}
        for key, lbl in zip(METRIC_KEYS, METRIC_LABELS):
            va, vb = ma[key], mb[key]
            if key == "AAE":
                va, vb = math.degrees(va), math.degrees(vb)
            if key == "good":
                va, vb = va * 100, vb * 100
            d[f"A_{lbl}"]     = round(va, 6)
            d[f"B_{lbl}"]     = round(vb, 6)
            d[f"delta_{lbl}"] = round(vb - va, 6)
        d["n"] = ma["total"]
        return d

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, ma, mb in file_rows:
            writer.writerow(_row_dict(label, ma, mb))
        writer.writerow(_row_dict("OVERALL", overall_a, overall_b))

    logging.info("CSV saved: %s", csv_path)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=(
            "Per-file metric comparison of two ONNX trajectory models "
            "(v6-compat / model_hossom preprocessing, mm units)."
        )
    )
    p.add_argument("--model_a",    required=True,
                   help="Path to ONNX model A (e.g. TF baseline)")
    p.add_argument("--model_b",    required=True,
                   help="Path to ONNX model B (e.g. train_v7 export)")
    p.add_argument("--json",       required=True,
                   help="Dataset JSON. valid_set paths must point to "
                        "mm-converted data (process_data.py output).")
    p.add_argument("--data_base",
                   default="/home/jason/sunia_trajectoryprediction/"
                           "torch_version/data",
                   help="Base directory prepended to JSON file paths")
    p.add_argument("--split",      default="valid",
                   choices=["valid", "train", "test"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--csv",        default=None,
                   help="Optional CSV output path")
    p.add_argument("--past_length", type=int, default=33,
                   help="past_length for both models (TrajectoryDatasetV6Compat). "
                        "feature shape = (past_length-1, 2). "
                        "Default 33 → feature (32, 2), matching model_hossom export.")
    args = p.parse_args()

    logging.info("Loading model A: %s  [v6_compat, past_length=%d]",
                 args.model_a, args.past_length)
    model_a = OnnxModel(args.model_a)
    logging.info("Loading model B: %s  [v6_compat, past_length=%d]",
                 args.model_b, args.past_length)
    model_b = OnnxModel(args.model_b)

    with open(args.json) as f:
        cfg = json.load(f)

    past_len   = cfg.get("past_length",   33)
    future_len = cfg.get("future_length",  8)

    if args.split == "train":
        raw_paths = []
        for group in cfg.get("train_set", []):
            for entry in group.get("index", []):
                raw_paths.append(
                    entry if isinstance(entry, str) else entry["file"]
                )
    else:
        split_key = {"valid": "valid_set", "test": "test_set"}[args.split]
        raw_paths = cfg.get(split_key, [])

    if not raw_paths:
        sys.exit(f"[ERROR] No paths for split '{args.split}' in {args.json}")

    file_rows, results_a, results_b = [], [], []
    logging.info("Evaluating %d file(s) [%s] …", len(raw_paths), args.split)

    for rel_path in raw_paths:
        full_path = os.path.join(args.data_base, rel_path)
        label     = os.path.basename(rel_path)

        if not os.path.exists(full_path):
            logging.warning("File not found, skipping: %s", full_path)
            continue

        # ── Single Dataset for both models (v6-compat / model_hossom style) ──
        # Both models evaluated on identical inputs: same preprocessing,
        # same rotation frame, same caption ground truth.
        ds = TrajectoryDatasetV6Compat(
            full_path, args.past_length, future_len,
            max_history=1_000_000, cache_feature=True,
        )

        if len(ds) == 0:
            logging.warning("No valid samples in: %s", full_path)
            continue

        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )

        ma = eval_loader(model_a, loader, future_len)
        mb = eval_loader(model_b, loader, future_len)

        file_rows.append((label, ma, mb))
        results_a.append(ma)
        results_b.append(mb)

        logging.info(
            "  %-50s  n=%d  A_err=%.4f  B_err=%.4f  "
            "A_RMSD=%.4f  B_RMSD=%.4f",
            label, ma["total"],
            ma["err"],  mb["err"],
            ma["RMSD"], mb["RMSD"],
        )

    if not file_rows:
        sys.exit("[ERROR] No data files were found/loaded.")

    overall_a = aggregate(results_a)
    overall_b = aggregate(results_b)
    name_a = os.path.basename(args.model_a)
    name_b = os.path.basename(args.model_b)

    print_table(file_rows, overall_a, overall_b, name_a, name_b)

    if args.csv:
        save_csv(file_rows, overall_a, overall_b, name_a, name_b, args.csv)


if __name__ == "__main__":
    main()


