# WIRING.md — 一次性接线 (装这套编排层时做一遍)

这套编排层**不重写**你的 trainer/model，只在两处接 env，让 round.json 能注入旋钮。
全部默认值与你现在的字面量一致 → 不接也能照常手动跑，接了之后 `run_round.sh` 才生效。

## 1. 放文件

\`\`\`
项目根/
  CLAUDE.md  WIRING.md  run_round.sh  env_overrides.py   ← 放根目录
  auto_finetune/  __init__.py config.py make_round_config.py emit_env.py analyze.py
chmod +x run_round.sh
\`\`\`

## 2. 填 `auto_finetune/config.py`

按训练机真实路径核对/修改：
- `BASELINE_ONNX` → 那个**原始 hossom** 导出的 `model_hossom.quant.onnx`（冻结基线 A）。
- `EVAL_JSON` → 评测描述符（`..._tcl.json`，valid_set 是逐类别文件）。
- `DATA_BASE` → 与 `compare_onnx_hossom.sh` 里 `--data_base` 一致。
- `ALL_DATA_JSON` → 列全允许的训练描述符。
- `CURVE_PATTERNS` → 必须能子串匹配 EVAL_JSON 里 curve 类文件的 basename（默认 `small_curves`/`big_curves`）。
- `APL_REGRESS_FRAC` → 按你实际 APL 量级标定（默认 5%）。

## 3. trainer 接 1 处 (`trainer_Trajectory_v78_220817_hm.py`)

在 `train_finetune()` 里，`options={...}` 那个 dict **构造完之后**、调 `cnn_gru...train(**options)` **之前**，插两行：

\`\`\`python
    logging.info('Options: %s',options)

    # ↓↓↓ 新增: 用 round.json 注入的 TRAJ_* 覆盖 options (没有 env 时不变) ↓↓↓
    from env_overrides import apply_trainer_env_overrides
    options = apply_trainer_env_overrides(options)
    # ↑↑↑ 新增 ↑↑↑

    cnn_gru_v7_Trajectory_v78_220817_hm.train(train_set=..., ...)
\`\`\`

(只接 `train_finetune`；`train1` 是手动用的，可不接。)

## 4. cnn_gru 接 1 处 (`cnn_gru_v7_Trajectory_v78_220817_hm.py`)

**4a.** 把文件顶部这块字面量赋值：

\`\`\`python
TUNE_TIME_ONLY=False
TUNE_POLY_ONLY = False
USE_APL_LOSS = False
USE_TIME_CEIL, TIME_CEIL = False, 0.5
\`\`\`

替换成从 env 读（默认值不变）：

\`\`\`python
from env_overrides import flags_from_env
_F = flags_from_env()
TUNE_TIME_ONLY = _F["TUNE_TIME_ONLY"]
TUNE_POLY_ONLY = _F["TUNE_POLY_ONLY"]
USE_APL_LOSS   = _F["USE_APL_LOSS"]
USE_TIME_CEIL  = _F["USE_TIME_CEIL"]
TIME_CEIL      = _F["TIME_CEIL"]
APL_TARGET     = _F["APL_TARGET"]          # 提到模块级, 让 train_step 用它
\`\`\`

**4b.** 在 `train_step` 里删掉那行写死的局部赋值，改用模块级 `APL_TARGET`：

\`\`\`python
            if USE_APL_LOSS:
                apl_pred   = tf.norm(prediction, axis=-1)
                # APL_TARGET = 0.8        ← 删掉这行局部字面量
                apl_floor  = tf.reduce_mean(tf.nn.relu(APL_TARGET - apl_pred))
                apl_loss  = 5.0 * apl_floor
\`\`\`

> 删掉局部 `APL_TARGET = 0.8` 后，函数会引用模块级 `APL_TARGET`（由 env 决定）。
> 这是接 env 必须的一步——否则 `--apl-target` 不会生效。

## 5. `.gitignore` 追加

\`\`\`
runs/
auto_finetune/ledger.csv
*.onnx
*.ort
saved_model/
*model_buffer*.h
\`\`\`

## 6. 自检 (接完跑一遍)

\`\`\`bash
# round.json 能写能校验
python -m auto_finetune.make_round_config --id _wiretest --notes test --angle-weight 12
# env 能 emit
python -m auto_finetune.emit_env _wiretest
# 接线生效: 不带 env, trainer 日志应打印 "no TRAJ_* trainer overrides";
# 带 TRAJ_ANGLE_WEIGHT=12 跑 smoke, 应打印 "overridden by env: {'angle_weight': 12.0}"
TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py
rm -rf runs/_wiretest
\`\`\`

看到 override 日志即接线成功，之后正常用 `make_round_config → run_round.sh → verdict.json` 即可。


