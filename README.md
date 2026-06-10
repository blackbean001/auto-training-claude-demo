# 轨迹预测 Finetune 编排框架

一套围绕**手写笔轨迹预测模型**的自动化 finetune 编排层。在不重写已有 trainer / 模型的前提下，把"改超参 → 训练 → 导出 → 评测 → 自动归因"串成一条可复现、可并行、经验自增长的流水线。

## 这套东西解决什么

模型 finetune 时常陷入两个**互相冲突**的目标：

- **保长度**：候选模型的平均预测长度（APL）不能低于基线。
- **压飞线**：拐角处的轨迹误差（RMSD / 角度误差）不能变差、达标率不能下降。

压飞线往往会缩短预测、保长度又容易把飞线放出来。人工逐个调权重既慢又难归因。本框架把每一轮实验约束成**单变量改动 + 受控对比 + 自动裁决**，让"改了哪个旋钮 → 指标怎么动 → 该往哪走"被自动记录并积累成可查询的经验库。

## 流水线（一轮 = 4 步）

```
round.json ──emit_env──> TRAJ_* 环境变量 (含指定 GPU)
     │
     ├─1─ trainer       训练，按评测描述符选 best ckpt
     ├─2─ export_onnx   导出量化 onnx
     ├─3─ comparer      候选 B vs 冻结基线 A → results.csv (逐类别 + OVERALL)
     └─4─ analyze       → verdict.json + ledger.csv + lesson.md + LESSONS.md
```

- 所有旋钮只经由 `round.json` 注入环境变量，**不改源码常量**——trainer / model 仅在固定的几处读 env 覆盖默认值，不接也能照常手动跑。
- 每轮记录 `from`（父轮），靠单变量受控对比做干净归因。
- 并发写共享文件（ledger / 经验库）已加文件锁。

## 评测指标

`results.csv` 每个数据类别一行 + 一行 OVERALL，列为 `A_x / B_x / delta_x`（`delta = B − A`）。跨轮只看与训练权重无关的原始量：

| 指标 | 含义 | 方向 |
|---|---|---|
| `RMSD` | 距离误差 (mm) | ↓ 好 |
| `AAE(°)` | 角度误差 (度) | ↓ 好 |
| `ATE` | 时间误差 | ↓ 好 |
| `APL` | 平均预测长度 (mm) | 不退 |
| `good%` | 满足精度门的比例 | ↑ 好 |

裁决双目标同时满足才算 PASS，由 `analyze.py` 自动判。

## 目录结构

```
.
├── CLAUDE.md  WIRING.md            # 操作手册 / 一次性接线说明
├── run_round.sh  run_batch.sh      # 单轮 / 多卡并行批跑
├── env_overrides.py                # round.json 的 TRAJ_* env → 覆盖 trainer/flag
├── trainer_*.py  cnn_gru_*.py  dataset_*.py   # 训练（接 env，不被重写）
├── export_onnx_buffer.py  quantize_onnx.py    # 导出量化 onnx
├── smoke_test_graph.py             # 改代码前的接线 / 图自检
├── auto_finetune/
│   ├── config.py                   # 路径 / 候选数据 / 阈值 / 取值区间 —— 唯一真相源
│   ├── make_round_config.py        # 写 round.json（含越界 / loose≤strict 校验、记 from）
│   ├── emit_env.py                 # round.json → TRAJ_* export
│   ├── analyze.py                  # results.csv → verdict + ledger + lesson + LESSONS
│   ├── ledger.csv                  # 跨轮汇总（生成）
│   └── LESSONS.md                  # 累积已验证规律（生成，自增长，决策前必读）
├── torch_version/
│   ├── compare_onnx_hossom.py      # 评测器（依赖同目录 train_polyhead）
│   └── train_polyhead.py
└── runs/<id>/                      # 每轮产物: round.json / ckpt / *.log / results.csv
                                    #            verdict.json / lesson.md
```

## 快速上手

### 一次性接线

新装这套编排层时，按 `WIRING.md` 做一遍：放文件、填 `auto_finetune/config.py`（基线路径、评测描述符、数据根、类别匹配等），在 trainer 与 model 各接一处 env 读取。接线自检：

```bash
# round.json 能写能校验、env 能 emit、接线生效
python -m auto_finetune.make_round_config --id _wiretest --notes test --angle-weight 12
python -m auto_finetune.emit_env _wiretest
TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py   # 应打印 overridden by env
rm -rf runs/_wiretest
```

### 跑单轮

```bash
TRAJ_GPU=0 nohup ./run_round.sh <id> > runs/<id>/round.out 2>&1 &
tail -f runs/<id>/train.log
```

### 跑一批（多卡并行）

```bash
# 1. 决策前先读经验与上一批趋势
cat auto_finetune/LESSONS.md
column -t -s, auto_finetune/ledger.csv

# 2. 派生这一批（≤6 个，每个只动一个旋钮，全部 --from 上一批已完成的最佳轮）
python -m auto_finetune.make_round_config --id <id_a> --from <parent> --notes "..." --time-weight 4
python -m auto_finetune.make_round_config --id <id_b> --from <parent> --notes "..." --angle-weight 15

# 3. 多卡并行（卡满排队、空了顶上、每卡不超卖）
GPUS="0 1 2 3 4 5" nohup ./run_batch.sh <id_a> <id_b> ... > batch.out 2>&1 &

# 4. 跑完读结论
cat runs/<id>/lesson.md
cat auto_finetune/LESSONS.md
```

## 核心约定

- **改超参唯一入口** = `make_round_config`（只写 `round.json`，不碰源码），自带越界校验、`loose≤strict` 校验、记录父轮。
- **一轮只动一个旋钮**（单变量）；多变量改动的归因会被自动标记为不可靠。
- **冻结基线 A 永不改动**——它是所有跨轮 delta 的唯一参照。
- **训练 json ≠ 评测 json**，不混用。
- 结构项（学习率、网络维度、patience）实验期冻结；权重 / 损失开关全走 `round.json → env`。
- 产物（ckpt / onnx / `runs/` / `ledger.csv` / `LESSONS.md` / 锁文件）不入库。

## 经验库自增长

每轮 `analyze.py` 会把"改了什么 → 指标怎么动 → 候选规律 → 下一步建议"写进该轮 `lesson.md`，并追加到全局 `auto_finetune/LESSONS.md`。下一轮决策前先读 `LESSONS.md`：已被多轮单变量实测一致验证的规律，优先级高于初始的方向性先验。如此每一批迭代都在前一批的实测结论上爬山。
