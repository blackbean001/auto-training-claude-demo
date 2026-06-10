# LESSONS — 累积已验证规律 (Claude 每轮决策前先读这里)

> 每轮自动追加。单变量条目的 rule 可信; 多变量条目仅作记录。

## `0608_tw4`  [FAIL]  (from `—`)  2026-06-09T01:47
- change: (首轮)
- effect: (无跨轮对比)
- rule: —
- next: APL 退化 (B 比基线短). 首选 time_weight += 1 (主旋钮); 其次放宽 --dist-tol-strict; 最后手段才 --use-apl-loss true / 抬 --apl-target。
## `0608_tw4`  [FAIL]  (from `0608_base`)  2026-06-09T17:03
- change: time_weight 3.0→4.0
- effect: (无跨轮对比)
- rule: 首轮 / 无 parent 对比。仅相对冻结基线 A 评估 (见 verdict)。
- next: APL 退化 (B 比基线短). 首选 time_weight += 1 (主旋钮); 其次放宽 --dist-tol-strict; 最后手段才 --use-apl-loss true / 抬 --apl-target。
## `t0609_1728`  [FAIL]  (from `—`)  2026-06-09T21:59
- change: (首轮)
- effect: (无跨轮对比)
- rule: —
- next: curve 类未改善/变差. 首选 angle_weight += 5; 或加大 cnn_gru slope/speed 辅助系数(0.1); 或收紧 --dist-tol-loose。
