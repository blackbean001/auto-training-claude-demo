# Trajectory Prediction Finetune Orchestration Framework

An automation layer for finetuning a **stylus trajectory prediction model**. Without rewriting the existing trainer / model, it wires "tweak hyperparameters → train → export → evaluate → auto-attribute" into a single reproducible, parallelizable, self-growing pipeline.

## What this solves

Finetuning this model constantly runs into two **conflicting** goals:

- **Preserve length**: the candidate model's Average Prediction Length (APL) must not drop below the baseline.
- **Suppress fly-off**: trajectory error at corners (RMSD / angular error) must not get worse, and the pass rate must not decline.

Suppressing fly-off tends to shorten predictions, while preserving length tends to let fly-off back out. Tuning weights one by one by hand is both slow and hard to attribute. This framework constrains every round to a **single-variable change + controlled comparison + automatic verdict**, so that "which knob was turned → how the metrics moved → which way to go next" is recorded automatically and accumulated into a queryable lessons library.

## Pipeline (one round = 4 steps)

```
round.json ──emit_env──> TRAJ_* environment variables (incl. designated GPU)
     │
     ├─1─ trainer       train, select best ckpt per the eval descriptor
     ├─2─ export_onnx   export quantized onnx
     ├─3─ comparer      candidate B vs frozen baseline A → results.csv (per-category + OVERALL)
     └─4─ analyze       → verdict.json + ledger.csv + lesson.md + LESSONS.md
```

- All knobs are injected solely through `round.json` as environment variables — **no source constants are edited**. The trainer / model only read env overrides at a few fixed spots; without them they still run manually as usual.
- Each round records its `from` (parent round), enabling clean attribution via single-variable controlled comparison.
- Concurrent writes to shared files (ledger / lessons library) are guarded with file locks.

## Evaluation metrics

`results.csv` has one row per data category plus one OVERALL row, with columns `A_x / B_x / delta_x` (`delta = B − A`). Across rounds, only look at the raw quantities that are independent of training weights:

| Metric | Meaning | Direction |
|---|---|---|
| `RMSD` | Distance error (mm) | ↓ better |
| `AAE(°)` | Angular error (degrees) | ↓ better |
| `ATE` | Time error | ↓ better |
| `APL` | Average prediction length (mm) | must not regress |
| `good%` | Fraction meeting the accuracy gate | ↑ better |

A round PASSes only when both goals are satisfied simultaneously, judged automatically by `analyze.py`.

## Directory structure

```
.
├── CLAUDE.md  WIRING.md            # operations manual / one-time wiring guide
├── run_round.sh  run_batch.sh      # single round / multi-GPU parallel batch
├── env_overrides.py                # round.json's TRAJ_* env → override trainer/flags
├── trainer_*.py  cnn_gru_*.py  dataset_*.py   # training (reads env, not rewritten)
├── export_onnx_buffer.py  quantize_onnx.py    # export quantized onnx
├── smoke_test_graph.py             # wiring / graph self-check before editing code
├── auto_finetune/
│   ├── config.py                   # paths / candidate data / thresholds / value ranges — single source of truth
│   ├── make_round_config.py        # write round.json (with out-of-range / loose≤strict checks, records from)
│   ├── emit_env.py                 # round.json → TRAJ_* export
│   ├── analyze.py                  # results.csv → verdict + ledger + lesson + LESSONS
│   ├── ledger.csv                  # cross-round summary (generated)
│   └── LESSONS.md                  # accumulated verified rules (generated, self-growing, read before deciding)
├── torch_version/
│   ├── compare_onnx_hossom.py      # evaluator (depends on train_polyhead in the same dir)
│   └── train_polyhead.py
└── runs/<id>/                      # per-round artifacts: round.json / ckpt / *.log / results.csv
                                    #                      verdict.json / lesson.md
```

## Getting started

### One-time wiring

When installing this orchestration layer fresh, follow `WIRING.md` once: place the files, fill in `auto_finetune/config.py` (baseline path, eval descriptor, data root, category matching, etc.), and add one env-read spot each in the trainer and the model. Wiring self-check:

```bash
# round.json can be written and validated, env can be emitted, wiring takes effect
python -m auto_finetune.make_round_config --id _wiretest --notes test --angle-weight 12
python -m auto_finetune.emit_env _wiretest
TRAJ_ANGLE_WEIGHT=12 python smoke_test_graph.py   # should print "overridden by env"
rm -rf runs/_wiretest
```

### Run a single round

```bash
TRAJ_GPU=0 nohup ./run_round.sh <id> > runs/<id>/round.out 2>&1 &
tail -f runs/<id>/train.log
```

### Run a batch (multi-GPU parallel)

```bash
# 1. Before deciding, read the lessons and last batch's trend
cat auto_finetune/LESSONS.md
column -t -s, auto_finetune/ledger.csv

# 2. Derive this batch (≤6 rounds, each changing exactly one knob, all --from the best completed round of the last batch)
python -m auto_finetune.make_round_config --id <id_a> --from <parent> --notes "..." --time-weight 4
python -m auto_finetune.make_round_config --id <id_b> --from <parent> --notes "..." --angle-weight 15

# 3. Multi-GPU parallel (queues when full, fills in when free, no oversubscription per GPU)
GPUS="0 1 2 3 4 5" nohup ./run_batch.sh <id_a> <id_b> ... > batch.out 2>&1 &

# 4. Read the conclusions when done
cat runs/<id>/lesson.md
cat auto_finetune/LESSONS.md
```

## Core conventions

- **The only entry point for changing hyperparameters** = `make_round_config` (writes `round.json` only, never touches source), with built-in out-of-range checks, `loose≤strict` checks, and parent-round recording.
- **One knob per round** (single variable); attribution for multi-variable changes is automatically flagged as unreliable.
- **The frozen baseline A is never changed** — it is the sole reference for all cross-round deltas.
- **Training json ≠ eval json**, never mixed.
- Structural items (learning rate, network dimensions, patience) are frozen during experimentation; weights / loss switches all go through `round.json → env`.
- Artifacts (ckpt / onnx / `runs/` / `ledger.csv` / `LESSONS.md` / lock files) are not committed.

## Self-growing lessons library

Each round, `analyze.py` writes "what changed → how the metrics moved → candidate rule → next-step suggestion" into that round's `lesson.md`, and appends it to the global `auto_finetune/LESSONS.md`. Read `LESSONS.md` before deciding the next round: rules that have been consistently verified across multiple single-variable runs take priority over the initial directional priors. This way every batch climbs on the measured conclusions of the previous one.
