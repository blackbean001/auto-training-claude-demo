# Lesson — round `t0609_1728`

- 时间: 2026-06-09T21:59:04
- 裁决: **FAIL**  (apl_ok=True, curves_ok=False)
- parent: `—(首轮)`
- 意图(notes): 首通诊断

## 改了什么 (vs parent)
- 首轮, 无 parent。

## 指标怎么动 (本轮 B vs 上一轮 B)
- (无 parent verdict, 跳过跨轮对比; 见下方相对基线 A)

## 相对冻结基线 A (本轮 delta = B − A)
- OVERALL: APL Δ0.5518, RMSD Δ0.0649, good% Δ22.2841
- fast_big_curves: RMSD Δ-0.1497, APL Δ1.5880, good% Δ1.1066
- fast_small_curves: RMSD Δ0.2762, APL Δ0.8979, good% Δ-2.4928
- normal_big_curves: RMSD Δ-0.2662, APL Δ0.8057, good% Δ7.1290
- normal_small_curves: RMSD Δ0.0416, APL Δ0.8095, good% Δ2.0277
- slow_big_curves: RMSD Δ-0.0790, APL Δ0.4385, good% Δ33.0933
- slow_small_curves: RMSD Δ-0.0883, APL Δ0.1362, good% Δ36.9513

## 推断的经验 (候选规律)
- 首轮 / 无 parent 对比。仅相对冻结基线 A 评估 (见 verdict)。

## 下一步建议 (来自 verdict + 优先级)
- curve 类未改善/变差. 首选 angle_weight += 5; 或加大 cnn_gru slope/speed 辅助系数(0.1); 或收紧 --dist-tol-loose。
