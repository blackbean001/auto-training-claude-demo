# CLAUDE.md — Trajectory Prediction Finetune Orchestration Handbook

> Operations handbook for Claude Code. **Before each batch decision, read three things in order**: this file → `auto_finetune/LESSONS.md` (historically verified rules, auto-growing) → each `runs/<id>/verdict.json` from the previous batch.

## 0. Goals (always revolve around these two)

TCL-G100 stylus trajectory prediction. Finetune the TF version of the hossom model to **suppress corner fly-off** **without losing prediction length**.

- **Goal 1 — keep length**: Candidate B's `APL` (average prediction length, mm) is not lower than baseline A.
- **Goal 2 — suppress fly-off**: For curve categories (`small_curves`/`big_curves`), `RMSD`/`AAE` does not get worse and `good%` does not drop.

The two conflict by nature (suppressing fly-off often shortens prediction). Each round succeeds = both satisfied **simultaneously**; `verdict.json` judges automatically.

> **The lessons library grows automatically**: Each round `analyze.py` writes "what changed → how metrics moved → candidate rule" into `runs/<id>/lesson.md` (a new md per round) and appends to `auto_finetune/LESSONS.md`. **Read `LESSONS.md` before deciding.** The prior table in §4 is initial knowledge; `LESSONS.md` is the measured supplement. On conflict, defer to the **single-variable, ✅-consistent** measured entries in `LESSONS.md`.

## 1. Data Flow (one round = 4 steps, wired by `run_round.sh`)

```
round.json ──emit_env──> TRAJ_* environment variables (incl. TRAJ_GPU)
     │
     ├─1─ trainer_*.py SAVE_DIR TRAIN_JSON LOG   →  best ckpt (selected via descriptor's valid_set)
     ├─2─ export_onnx_buffer.py SAVE_DIR         →  SAVE_DIR/model_hossom.quant.onnx
     ├─3─ $TRAJ_COMPARER  A vs B                 →  runs/<id>/results.csv (per-category + OVERALL)
     └─4─ analyze.py                             →  verdict.json + ledger.csv + lesson.md + LESSONS.md
```

- trainer/export are bound to this round's assigned GPU via `TRAJ_GPU`; trainer reads `TRAJ_*` via `apply_trainer_env_overrides` to override weights/tolerances/LR; cnn_gru reads flags via `flags_from_env`.
- export reads argv = this round's `SAVE_DIR`; compare uses the absolute path `$TRAJ_COMPARER` (in `torch_version/`).
- analyze writes back lessons using round.json's `from` (parent) for a controlled comparison — only **single-variable changes** allow clean attribution; concurrent writes to ledger/LESSONS already use `flock`.

## 2. Verb Surface

### 2.1 Single round (debug / smoke run)
```bash
TRAJ_GPU=0 nohup ./run_round.sh <id> > runs/<id>/round.out 2>&1 &
tail -f runs/<id>/train.log
```

### 2.2 One batch in parallel (six GPUs, the daily iteration workhorse)
```bash
# Before deciding: read historical lessons + previous batch conclusions
cat auto_finetune/LESSONS.md                 # historically verified rules (first input to decisions)
column -t -s, auto_finetune/ledger.csv       # cross-round trends

# Derive this batch (≤6, single-variable, all --from the best completed round of the previous batch)
python -m auto_finetune.make_round_config --id 0608_tw4 --from 0608_base --notes "tw 3->4" --time-weight 4
python -m auto_finetune.make_round_config --id 0608_aw15 --from 0608_base --notes "aw 10->15" --angle-weight 15
# ... up to six

# Six-GPU parallel (queue when full, fill in when free, no oversubscription per GPU)
GPUS="0 1 2 3 4 5" nohup ./run_batch.sh 0608_tw4 0608_aw15 ... > batch_0608.out 2>&1 &
tail -f batch_0608.out

# After it runs, read the lessons
cat runs/<id>/lesson.md                      # single round: what changed → how it moved → rule
cat auto_finetune/LESSONS.md                 # cumulative
```

- **The only entry point for changing hyperparameters = `make_round_config`** (writes round.json only, never touches source); it includes `--gpu`, out-of-range validation, `loose≤strict` validation, and records `from`.
- Smoke-test before changing code: `python smoke_test_graph.py`; wiring self-check `TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py` should show `overridden by env: {'angle_weight': 12.0}`.

## 3. Which Metrics to Watch (across rounds, only weight-independent raw quantities)

`results.csv` has one row per data file (= one category) + one OVERALL row, with columns `A_x / B_x / delta_x`, where `delta = B − A`.

| Metric | Meaning | Direction |
|---|---|---|
| `RMSD` | distance error (mm) | ↓ good |
| `AAE(°)` | angle error (degrees) | ↓ good |
| `ATE` | time error mean(1−pred_time) | ↓ good |
| `APL` | average prediction length (mm) ← **Goal 1** | no regression |
| `good%` | fraction meeting the accuracy gate | ↑ good |
| `err` | composite = weighted sum | **forbidden across rounds** ↓ |

Core focus: curve rows' `delta_RMSD`/`delta_good%`, and OVERALL `delta_APL`.

## 4. Knob List + Value Rules

### 4.1 Changeable / Frozen
- **Changeable**: `time_weight`, `angle_weight`, `distance_weight`, `fit_weight`,
  `dist_tol_loose`, `dist_tol_strict`, `hardness` (with caution),
  flags: `use_apl_loss` / `apl_target` / `use_time_ceil` / `time_ceil` / `tune_time_only` / `tune_poly_only`
- **Frozen**: `lr`, `dim_rnns`, `dim_feature`, **`patience` (fixed at 50)**

### 4.2 Directional priors (initial priors; once LESSONS.md has measurements, defer to those)
| Knob | Magnitude | Step | Direction |
|---|---|---|---|
| `time_weight` | O(1~3) | **±1** | **larger → longer APL** (monotonic, the most reliable lever for lengthening) |
| `angle_weight` | O(10) | **±5** | larger → corners more convergent / less fly-off, but may shorten prediction |
| `distance_weight` | O(10) | **±5** | larger → overall fit tighter |
| `fit_weight` | O(10) | **±5** | larger → stronger polynomial fitting constraint |
| `dist_tol_loose`/`dist_tol_strict` | O(1) | **±0.5** | tightening → stricter good% gate |
| `hardness` | ~99.5 | ±0.5 | percentile; cautious, must be <100 |

In one line: **O(1~3) change by 1, O(10) change by 5, tolerances ±0.5.**

### 4.3 Value priority (in order, no skipping)
1. **APL not long enough → `time_weight += 1`** (primary knob, most reliable lengthening means)
2. **Corner fly-off → `angle_weight += 5`**
3. **Both goals FAIL at once → first `time_weight += 1` (length-keeping takes priority), then `angle_weight += 5`**
4. Straight lines collateral-damaged → data ratio (train json `size_per_batch`); or back off `angle_weight` half a notch
5. `delta_ATE` drift → `time_weight`
6. **Last resort (lowest priority)**: `--apl-target` / `--time-ceil` carry relu hard boundaries, prone to side effects; use only when the primary knobs (time/angle) won't budge it.

### 4.4 General principles
- **Move only one knob per round** (single-variable); multi-variable lessons are tagged `⚠️ attribution unreliable`.
- `--from <best round of previous batch>` to clone, overriding only the one you intend to move (the parent is also the basis for lesson write-back comparison).
- Follow the step sizes in the table above; if `LESSONS.md` already has a measured slope (e.g. "each +1 → APL +X"), use the measured slope to estimate the step.
- Hill-climb along `ledger.csv` / `LESSONS.md`: continue if the same direction works; if overshot (Goal A improved but B regressed), back off half a step and bisect.

## 5. Hard Rules (DO NOT)

- **Do not touch the frozen baseline A** (`config.BASELINE_ONNX`): it is the sole reference for all deltas; changing it breaks cross-round comparison and distorts lessons.
- **Do not use `err` for cross-round comparison**: it contains training weights, so its dimension changes after tuning weights. Only look at `RMSD/AAE/ATE/APL/good%`.
- **Training json ≠ evaluation json**: training uses `..._reduce_fast.json`, evaluation is fixed to `config.EVAL_JSON` (`..._tcl.json`). Do not mix them.
- **Do not hand-edit module-level globals / source constants**: weights, `USE_APL_LOSS/APL_TARGET...` all go through round.json → env, always via `make_round_config`.
- **Do not change `patience` (50) / `lr` / `dim_*`** (structural items, frozen during the experiment period).
- **A new parent must be a completed round** (has verdict.json): **do not derive from an unfinished sibling in the same batch** (parallel siblings have no causal ordering between them).
- **A batch is ≤6** (fill the GPUs without oversubscribing; each GPU runs only one round at a time).
- **Do not commit** ckpt/onnx/saved_model/*.h/runs//ledger.csv/LESSONS.md/*.lock.
- **Do not block while training**: `nohup ... &` + poll `train.log` / final `verdict.json`.
- **`$TRAJ_COMPARER` (`torch_version/compare_onnx_hossom.py`) depends on `train_polyhead` in the same directory**: on ImportError, first check `config.COMPARER`, don't change the evaluation logic.
- Run `smoke_test_graph.py` before any full run.

## 6. Known Code Smells (do not "fix in passing" unless the task explicitly requires it)

- In `cnn_gru.calculate_jitter()`, `tf.config.experimental_run_functions_eagerly(True)` is a **global side effect**, usable only for offline analysis, and **must never enter the train_step path** (it flips the whole graph to eager, making training extremely slow).
- In `cnn_gru.calculate_slope/speed`, `gather(..., [20,21,22,23])` hard-codes indices that implicitly encode `past_length`; changing `past_length` will silently break it.
- The slope/speed auxiliary loss coefficients (hard-coded `0.1`) and the APL floor penalty (hard-coded `5.0`) are not yet exposed as `TRAJ_*` knobs and the agent can't see them; wire them up first (mirroring `flags_from_env`) before tuning.
- The trainer selects the best ckpt using the descriptor's `valid_set` (not fast-valid); to switch to fast-valid you must explicitly re-add `load_fast_valid_set`.

## 7. Directory

```
.
├── CLAUDE.md  WIRING.md  run_round.sh  run_batch.sh  env_overrides.py
├── trainer_*.py  cnn_gru_*.py  dataset_*.py        # training (reads env: trainer in 1 place + cnn_gru in 2; GPU from TRAJ_GPU)
├── export_onnx_buffer.py  quantize_onnx.py         # export (reads argv = this round's ckpt dir; binds TRAJ_GPU)
├── smoke_test_graph.py
├── auto_finetune/
│   ├── config.py            # paths/candidate data/thresholds/ranges/LESSONS_MD — single source of truth
│   ├── make_round_config.py # writes round.json (incl. --gpu / from / out-of-range + loose≤strict validation)
│   ├── emit_env.py          # round.json -> TRAJ_* export (incl. TRAJ_GPU / TRAJ_COMPARER)
│   ├── analyze.py           # results.csv -> verdict + ledger + lesson + LESSONS (concurrent writes use flock)
│   ├── ledger.csv           # cross-round summary (generated)
│   └── LESSONS.md           # ★ accumulated verified rules (generated, auto-growing, must-read before deciding)
├── torch_version/
│   ├── compare_onnx_hossom.py   # evaluation ($TRAJ_COMPARER, depends on train_polyhead in the same dir)
│   └── train_polyhead.py
├── model/Trajectory_v78_220817_hm/model_hossom.quant.onnx   # frozen baseline A
└── runs/<id>/               # each round: round.json ckpt/ *.log results.csv verdict.json lesson.md lesson.json
```

## 8. Parallel Iteration Loop (you run this automatically)

One "batch" = at most 6 single-variable experiments at once, six GPUs in parallel. One iteration:

```
1. Read LESSONS.md + each verdict.json from the previous batch + ledger.csv
2. Pick the "new parent": a round from the previous batch with verdict=PASS; if none, pick the one with the
   largest net improvement that did not wreck the other goal
   (if the whole batch wrecked some goal → direction overshot; roll the new parent back to the last good round,
    and next batch use a smaller step / change direction)
3. Per the §4.3 priority, based on the new parent, decide which knobs to move this batch (single-variable, ≤6)
4. make_round_config --from <new parent> × N   (each changes one knob)
5. GPUS="0 1 2 3 4 5" run_batch.sh <this batch's ids>   (background + poll)
6. When done, go back to 1
```

**Termination**: a `verdict=PASS` appears (curve improved and APL did not regress), and subsequent batches building on it yield no net gain (LESSONS.md shows "jitter-level / further tuning ineffective"). Deliverable = that round's `runs/<id>/ckpt/model_hossom.quant.onnx`; report and stop.

**Discipline** (the automatic loop must observe): the new parent is a completed round, never derived from a same-batch sibling; each batch ≤6 and single-variable; when unsure whether a PASS is noise, re-run the same config with multiple seeds to estimate variance before concluding.
