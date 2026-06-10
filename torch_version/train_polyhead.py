"""
Handwriting Trajectory Prediction — PyTorch  (polar output)  v7
================================================================

Changes from v6
---------------
  [1] normalize() pipeline aligned with C++ PredictAI inference:
        · SG 7-point smoothing on past window  (mirrors sgSmoothing_7points)
        · Equi-time resampling at REFRESH_MILL ms steps  (mirrors filterPoint)
        · 33 diffs produced  (was 32; C++ uses past_length=33 diffs)
        · ndir = sum of last-4 diffs  (matches C++ dx=sum(xs[-4:]))
        · Unit: pixel (data pre-converted to mm via process_data.py / DPI=237.787)
  [2] DPI rescaling done offline (process_data.py); training data in mm units.
  [3] feature shape: (P, 2) = (33, 2)  [was (P-1, 2) = (32, 2)]
  [4] export script updated to use past_length input size of 33 (not 32).

Output representation
---------------------
For each future time step k the model outputs (angle_k, length_k) where:
  - base_dir   = unit vector of the most recent diff  (feature[-1])
  - angle_k    = signed angle from base_dir to past[-1] → point_k,  in (-π, π)
  - length_k   = ‖point_k - past[-1]‖  (displacement magnitude, mm)

Data pipeline  (v7 — aligned with C++ filterPoint → onPredict)
---------------------------------------------------------------
  · Strokes separated by blank lines
  · Each line: x  y  timestamp_ms  [other fields ignored]
  · Data pre-converted to mm units by process_data.py (DPI = 237.787)
  · normalize():
      1. SG 7-point smooth past window
      2. Equi-time resample → 33 diffs  (last − cur, REFRESH_MILL=4.167ms)
      3. ndir = sum(diffs[-4:])
      4. rotate by atan2(ndir_x, ndir_y) + 45°
      feature = rotated diffs  (33, 2)
      caption = raw future diffs rotated + cumsum  (F, 2)
  · Polar labels computed on-the-fly in Dataset

Model topology  (unchanged from v6: Conv1D + GRU + TCH)
--------------------------------------------------------
  conv1(2→32, k=7) → conv2(32→64, k=3) → GRU(64→dim_rnn)
  → context (B, dim_rnn)
  context → Linear(dim_rnn→1) + sigmoid  = predicted_time ∈ (0,1)
  context + t  →  TCH MLP  →  (angle_k, length_k)

Loss  (unchanged from v6)
--------------------------
  fit_loss   = angle_weight * angle_proj + length_weight * length_proj
  point_loss = percentile-hardness margin loss on single prediction point
  time_loss  = mean(1 − predicted_time)  + variance term
  curv_loss  = curvature-aware overshoot penalty on length
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import os, re, json, math, logging, argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import Adam



if __name__ == "__main__":
    main()
