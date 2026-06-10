# CLAUDE.md — 轨迹预测 finetune 编排手册

> 给 Claude Code 的操作手册。**每批决策前依次读三样**：本文件 → `auto_finetune/LESSONS.md`(历史已验证规律, 自动增长) → 上一批各 `runs/<id>/verdict.json`。

## 0. 目标 (永远围绕这两条)

TCL-G100 stylus 轨迹预测。finetune TF 版 hossom 模型，在**不损失预测长度**的前提下**压住拐角飞线**。

- **目标1 保长度**：候选 B 的 `APL`(平均预测长度, mm) 不低于基线 A。
- **目标2 压飞线**：curve 类(`small_curves`/`big_curves`) 的 `RMSD`/`AAE` 不变差、`good%` 不下降。

两条天然冲突(压飞线常缩短预测)。每轮成败 = 两条**同时**满足，`verdict.json` 自动判。

> **经验库自动增长**：每轮 `analyze.py` 把"改了什么→指标怎么动→候选规律"写进 `runs/<id>/lesson.md`(每轮新 md) 并追加到 `auto_finetune/LESSONS.md`。**决策前先读 `LESSONS.md`**。§4 先验表是初始知识，`LESSONS.md` 是实测增补；冲突时以 `LESSONS.md` 里**单变量、已 ✅ 一致**的实测条目为准。

## 1. 数据流 (一轮 = 4 步, 由 `run_round.sh` 串好)

```
round.json ──emit_env──> TRAJ_* 环境变量 (含 TRAJ_GPU)
     │
     ├─1─ trainer_*.py SAVE_DIR TRAIN_JSON LOG   →  best ckpt (用 descriptor 的 valid_set 选优)
     ├─2─ export_onnx_buffer.py SAVE_DIR         →  SAVE_DIR/model_hossom.quant.onnx
     ├─3─ $TRAJ_COMPARER  A vs B                 →  runs/<id>/results.csv (逐类别 + OVERALL)
     └─4─ analyze.py                             →  verdict.json + ledger.csv + lesson.md + LESSONS.md
```

- trainer/export 经 `TRAJ_GPU` 绑定到本轮指定卡；trainer 经 `apply_trainer_env_overrides` 读 `TRAJ_*` 覆盖权重/容差/LR；cnn_gru 经 `flags_from_env` 读 flag。
- export 读 argv = 本轮 `SAVE_DIR`；compare 用绝对路径 `$TRAJ_COMPARER`(在 `torch_version/`)。
- analyze 经验回写靠 round.json 的 `from`(parent) 做受控对比 —— **单变量改动**才能干净归因；并发写 ledger/LESSONS 已加 `flock`。

## 2. 动词面

### 2.1 单轮 (调试 / 跑通用)
```bash
TRAJ_GPU=0 nohup ./run_round.sh <id> > runs/<id>/round.out 2>&1 &
tail -f runs/<id>/train.log
```

### 2.2 一批并行 (六卡, 日常迭代主力)
```bash
# 决策前: 读历史经验 + 上一批结论
cat auto_finetune/LESSONS.md                 # 历史已验证规律 (决策第一输入)
column -t -s, auto_finetune/ledger.csv       # 跨轮趋势

# 派生这一批 (≤6 个, 单变量, 全部 --from 上一批已完成的最佳轮)
python -m auto_finetune.make_round_config --id 0608_tw4 --from 0608_base --notes "tw 3->4" --time-weight 4
python -m auto_finetune.make_round_config --id 0608_aw15 --from 0608_base --notes "aw 10->15" --angle-weight 15
# ... 最多六个

# 六卡并行 (卡满排队、空了顶上、每卡不超卖)
GPUS="0 1 2 3 4 5" nohup ./run_batch.sh 0608_tw4 0608_aw15 ... > batch_0608.out 2>&1 &
tail -f batch_0608.out

# 跑完读经验
cat runs/<id>/lesson.md                      # 单轮: 改了什么→怎么动→规律
cat auto_finetune/LESSONS.md                 # 累积
```

- **改超参唯一入口 = `make_round_config`**(只写 round.json, 不碰源码)；它带 `--gpu`、越界校验、`loose≤strict` 校验、记 `from`。
- 改代码先 smoke：`python smoke_test_graph.py`；接线自检 `TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py` 应见 `overridden by env: {'angle_weight': 12.0}`。

## 3. 看什么指标 (跨轮只看权重无关的原始量)

`results.csv` 每数据文件一行(= 一类别) + 一行 OVERALL，列 `A_x / B_x / delta_x`，`delta = B − A`。

| 指标 | 含义 | 方向 |
|---|---|---|
| `RMSD` | 距离误差 (mm) | ↓ 好 |
| `AAE(°)` | 角度误差 (度) | ↓ 好 |
| `ATE` | 时间误差 mean(1−pred_time) | ↓ 好 |
| `APL` | 平均预测长度 (mm) ← **目标1** | 不退 |
| `good%` | 满足精度门的比例 | ↑ 好 |
| `err` | 合成量 = 加权和 | **跨轮禁用** ↓ |

核心看：curve 行 `delta_RMSD`/`delta_good%`，OVERALL `delta_APL`。

## 4. 旋钮清单 + 取值规则

### 4.1 可改 / 不可改
- **可改**: `time_weight`, `angle_weight`, `distance_weight`, `fit_weight`,
  `dist_tol_loose`, `dist_tol_strict`, `hardness`(谨慎)，
  flag: `use_apl_loss` / `apl_target` / `use_time_ceil` / `time_ceil` / `tune_time_only` / `tune_poly_only`
- **冻结**: `lr`, `dim_rnns`, `dim_feature`, **`patience`(固定 50)**

### 4.2 方向性经验 (初始先验; 有 LESSONS.md 实测后以实测为准)
| 旋钮 | 量级 | 步长 | 方向 |
|---|---|---|---|
| `time_weight` | O(1~3) | **±1** | **越大 → APL 越长**(单调, 最可靠的加长杠杆) |
| `angle_weight` | O(10) | **±5** | 越大 → 拐角越收敛/越不飞, 但可能缩短预测 |
| `distance_weight` | O(10) | **±5** | 越大 → 整体贴合越紧 |
| `fit_weight` | O(10) | **±5** | 越大 → 多项式拟合约束越强 |
| `dist_tol_loose`/`dist_tol_strict` | O(1) | **±0.5** | 收紧 → good% 门更严 |
| `hardness` | ~99.5 | ±0.5 | 百分位; 谨慎, 必须 <100 |

一句话: **O(1~3) change by 1, O(10) change by 5, 容差 ±0.5。**

### 4.3 取值优先级 (按序, 不跳级)
1. **APL 不够长 → `time_weight += 1`** (主旋钮, 最可靠的加长手段)
2. **拐角飞线 → `angle_weight += 5`**
3. **两个目标同时 FAIL → 先 `time_weight += 1`(保长度优先), 再 `angle_weight += 5`**
4. 直线被连累 → 数据配比(train json `size_per_batch`); 或 `angle_weight` 回退半档
5. `delta_ATE` 漂移 → `time_weight`
6. **最后手段(优先级最低)**: `--apl-target` / `--time-ceil` 带 relu 硬边界、易副作用, 仅当主旋钮(time/angle)推不动时才用。

### 4.4 通用原则
- **一轮只动一个旋钮**(单变量); 多变量 lesson 会标 `⚠️ 归因不可靠`。
- `--from <上一批最佳轮>` 克隆, 只覆盖要动的那一个 (parent 也是经验回写做对比的依据)。
- 按上表步长走; 若 `LESSONS.md` 已有实测斜率(如"每 +1→APL +X"), 用实测斜率估步长。
- 沿 `ledger.csv` / `LESSONS.md` 爬山: 同方向有效就继续; 过头(目标A好了但B退了)就回退半步二分。

## 5. 硬规则 (DO NOT)

- **不动冻结基线 A** (`config.BASELINE_ONNX`)：所有 delta 的唯一参照, 换了就没法跨轮比、经验也失真。
- **不用 `err` 跨轮比较**：含训练权重, 调权重后量纲变。只看 `RMSD/AAE/ATE/APL/good%`。
- **训练 json ≠ 评测 json**：训练 `..._reduce_fast.json`, 评测固定 `config.EVAL_JSON`(`..._tcl.json`)。不混用。
- **不手改模块级全局/源码常量**：权重、`USE_APL_LOSS/APL_TARGET...` 全走 round.json → env, 一律经 `make_round_config`。
- **不改 `patience`(50) / `lr` / `dim_*`**(结构项, 实验期冻结)。
- **新 parent 必须是已完成的轮**(有 verdict.json)：**不要从同批未完成的兄弟派生**(并行的兄弟之间无先后因果)。
- **一批 ≤6 个**(铺满卡不超卖, 每卡同时只跑一个 round)。
- **不 commit** ckpt/onnx/saved_model/*.h/runs//ledger.csv/LESSONS.md/*.lock。
- **不阻塞跑训练**: `nohup ... &` + 轮询 `train.log` / 最终 `verdict.json`。
- **`$TRAJ_COMPARER`(`torch_version/compare_onnx_hossom.py`) 依赖同目录 `train_polyhead`**：ImportError 先核对 `config.COMPARER`, 别改评测逻辑。
- 全量前先 `smoke_test_graph.py`。

## 6. 已知代码气味 (不要"顺手修", 除非任务明确要)

- `cnn_gru.calculate_jitter()` 里 `tf.config.experimental_run_functions_eagerly(True)` 是**全局副作用**, 只可离线分析, **绝不能进 train_step 链路**(会把整图翻 eager, 训练暴慢)。
- `cnn_gru.calculate_slope/speed` 里 `gather(..., [20,21,22,23])` 写死索引, 隐含 `past_length`; 动 `past_length` 会静默错。
- slope/speed 辅助损失系数(写死 `0.1`)、APL floor 惩罚(写死 `5.0`) 暂未做成 `TRAJ_*` 旋钮, agent 扫不到; 要调先接线(仿 `flags_from_env`)。
- trainer 用 descriptor 的 `valid_set` 选 best ckpt(不是 fast-valid); 想改 fast-valid 需显式加回 `load_fast_valid_set`。

## 7. 目录

```
.
├── CLAUDE.md  WIRING.md  run_round.sh  run_batch.sh  env_overrides.py
├── trainer_*.py  cnn_gru_*.py  dataset_*.py        # 训练 (接 env: trainer 1 处 + cnn_gru 2 处; GPU 从 TRAJ_GPU 读)
├── export_onnx_buffer.py  quantize_onnx.py         # 导出 (读 argv = 本轮 ckpt 目录; 绑 TRAJ_GPU)
├── smoke_test_graph.py
├── auto_finetune/
│   ├── config.py            # 路径/候选数据/阈值/ranges/LESSONS_MD — 唯一真相源
│   ├── make_round_config.py # 写 round.json (含 --gpu / from / 越界+loose≤strict 校验)
│   ├── emit_env.py          # round.json -> TRAJ_* export (含 TRAJ_GPU / TRAJ_COMPARER)
│   ├── analyze.py           # results.csv -> verdict + ledger + lesson + LESSONS (并发写加 flock)
│   ├── ledger.csv           # 跨轮汇总 (生成)
│   └── LESSONS.md           # ★ 累积已验证规律 (生成, 自增长, 决策前必读)
├── torch_version/
│   ├── compare_onnx_hossom.py   # 评测 ($TRAJ_COMPARER, 依赖同目录 train_polyhead)
│   └── train_polyhead.py
├── model/Trajectory_v78_220817_hm/model_hossom.quant.onnx   # 冻结基线 A
└── runs/<id>/               # 每轮: round.json ckpt/ *.log results.csv verdict.json lesson.md lesson.json
```

## 8. 并行迭代 loop (你自动跑这个)

一"批" = 一次最多 6 个单变量实验, 六卡并行。一轮迭代:

```
1. 读 LESSONS.md + 上一批各 verdict.json + ledger.csv
2. 挑“新 parent”: 上一批里 verdict=PASS 的; 没有则选净进步最大、且没把另一目标搞坏的那轮
   (若整批都把某目标搞坏 → 方向走过头, 新 parent 退回上一个好轮, 下批改小步长/换方向)
3. 按 §4.3 优先级, 基于新 parent 定这批要动的旋钮 (单变量, ≤6 个)
4. make_round_config --from <新parent> × N   (每个改一个旋钮)
5. GPUS="0 1 2 3 4 5" run_batch.sh <这批 ids>   (后台 + 轮询)
6. 跑完回到 1
```

**终止**: 出现 `verdict=PASS`(curve 改善且 APL 不退), 且后续几批在它基础上再改无净增益(LESSONS.md 出现"抖动级/再调无效")。产出 = 该轮 `runs/<id>/ckpt/model_hossom.quant.onnx`, 报告后停。

**纪律**(自动循环必须守): 新 parent 是已完成轮、不从同批兄弟派生; 每批 ≤6 且单变量; 拿不准 PASS 是否噪声时, 同配置多种子复跑估方差再下结论。




