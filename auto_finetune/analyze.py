# -*- coding: utf-8 -*-
"""
auto_finetune/analyze.py
========================
一轮跑完后, 读 runs/<id>/results.csv (compare_onnx_hossom.py 产出), 做三件事:

  1) 裁决 (verdict): 双目标 pass/fail + 旋钮建议 -> runs/<id>/verdict.json + 打印
  2) 入账 (rollup):  把 OVERALL + curve 行的关键量 append 到 auto_finetune/ledger.csv
  3) ★ 经验回写 (lessons): 与上一轮 (round.json 的 "from" parent) 做受控对比,
     推断"改了什么 → 指标怎么动 → 候选规律", 写:
       · runs/<id>/lesson.md           本轮详细经验 (每轮一个新 md)
       · auto_finetune/LESSONS.md      累积先验库 (Claude 每轮必读, 自增长)

agent 决策时读: verdict.json(本轮诊断) + LESSONS.md(历史已验证规律) + ledger.csv(趋势)。

裁决依据 (跨轮可比, 只看权重无关的原始量, 不看合成 err):
  · 目标1 保长度: OVERALL 的 B_APL >= A_APL * (1 - APL_REGRESS_FRAC), 且 A_APL > 0
  · 目标2 压飞线: 每个 curve 类 delta_RMSD <= RMSD_REGRESS_TOL 且 delta_good% >= -GOOD_REGRESS_TOL
两者同时满足 -> PASS。
"""

import fcntl
from contextlib import contextmanager

@contextmanager
def _locked(path):
    """跨进程文件锁: 六卡并行时串行化对 ledger/LESSONS 的 append。"""
    lock = path + ".lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)     # 阻塞直到拿到锁
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

import csv
import json
import os
import sys
from datetime import datetime

from auto_finetune import config as C

# compare_onnx_hossom.save_csv 用 METRIC_LABELS 拼列名, 这里照抄列名
L_RMSD = "RMSD"
L_AAE  = "AAE(°)"
L_ATE  = "ATE"
L_APL  = "APL"
L_GOOD = "good%"

# 经验回写产物路径 (config 没定义时退回默认)
LESSONS_MD = getattr(C, "LESSONS_MD",
                     os.path.join(C.PROJECT_DIR, "auto_finetune", "LESSONS.md"))

# 跨轮可比较的旋钮 (用于 diff "改了什么")
_KNOB_KEYS = ["distance_weight", "angle_weight", "time_weight", "fit_weight",
              "distance_tolerance_loose", "distance_tolerance_strict", "hardness"]
_FLAG_KEYS = ["TUNE_TIME_ONLY", "TUNE_POLY_ONLY", "USE_APL_LOSS",
              "APL_TARGET", "USE_TIME_CEIL", "TIME_CEIL"]


def _f(row, col):
    """读一个浮点单元格; 缺列/空 -> None"""
    v = row.get(col, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_results(round_id):
    path = os.path.join(C.RUNS_DIR, round_id, "results.csv")
    if not os.path.isfile(path):
        sys.exit(f"[analyze] no {path} (compare 这步是不是失败了?)")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    overall = next((r for r in rows if r["file"].strip().upper() == "OVERALL"), None)
    if overall is None:
        sys.exit("[analyze] results.csv 里没有 OVERALL 行")
    file_rows = [r for r in rows if r["file"].strip().upper() != "OVERALL"]
    return overall, file_rows


def _match(name, patterns):
    return any(p in name for p in patterns)


# ─────────────────────────────────────────────────────────────────────────────
# 裁决
# ─────────────────────────────────────────────────────────────────────────────
def verdict(round_id):
    overall, file_rows = _read_results(round_id)

    a_apl = _f(overall, f"A_{L_APL}")
    b_apl = _f(overall, f"B_{L_APL}")
    apl_valid = (a_apl is not None and a_apl > 0)
    apl_floor = (a_apl * (1.0 - C.APL_REGRESS_FRAC)) if apl_valid else None
    apl_ok = (apl_valid and b_apl is not None and b_apl >= apl_floor)

    curve_rows = [r for r in file_rows if _match(r["file"], C.CURVE_PATTERNS)]
    curve_details = []
    curves_ok = True
    for r in curve_rows:
        d_rmsd = _f(r, f"delta_{L_RMSD}")
        d_good = _f(r, f"delta_{L_GOOD}")
        ok = ((d_rmsd is None or d_rmsd <= C.RMSD_REGRESS_TOL) and
              (d_good is None or d_good >= -C.GOOD_REGRESS_TOL))
        curves_ok = curves_ok and ok
        curve_details.append({
            "file": r["file"], "ok": ok,
            "B_RMSD": _f(r, f"B_{L_RMSD}"), "delta_RMSD": d_rmsd,
            "B_APL":  _f(r, f"B_{L_APL}"),  "delta_APL":  _f(r, f"delta_{L_APL}"),
            "delta_good%": d_good,
        })
    if not curve_rows:
        curves_ok = False

    straight_regressed = [
        {"file": r["file"], "delta_RMSD": _f(r, f"delta_{L_RMSD}")}
        for r in file_rows
        if _match(r["file"], C.STRAIGHT_PATTERNS)
        and (_f(r, f"delta_{L_RMSD}") or 0) > C.RMSD_REGRESS_TOL
    ]

    passed = apl_ok and curves_ok

    suggestions = []
    if not apl_valid:
        suggestions.append(
            "A_APL<=0 或缺失 — 基线 A 可能没正常载入/评测。先核对 BASELINE_ONNX 路径与 compare 日志, 再谈旋钮。")
    elif not apl_ok:
        suggestions.append(
            "APL 退化 (B 比基线短). 首选 time_weight += 1 (主旋钮); 其次放宽 --dist-tol-strict; "
            "最后手段才 --use-apl-loss true / 抬 --apl-target。")
    if curve_rows and not curves_ok:
        suggestions.append(
            "curve 类未改善/变差. 首选 angle_weight += 5; 或加大 cnn_gru slope/speed 辅助系数(0.1); "
            "或收紧 --dist-tol-loose。")
    if not curve_rows:
        suggestions.append(
            "results.csv 没匹配到 curve 类. 核对 config.CURVE_PATTERNS 是否与 EVAL_JSON 文件名一致。")
    if straight_regressed and curves_ok:
        suggestions.append(
            "curve 改善但直线被连累. 数据配比(train json size_per_batch); 或 angle_weight 回退半档。")
    if passed:
        suggestions.append("双目标达成. 可固化本轮 ckpt, 或在此基础上继续单变量微调。")

    result = {
        "round_id": round_id,
        "verdict": "PASS" if passed else "FAIL",
        "apl_ok": apl_ok, "curves_ok": curves_ok,
        "overall": {
            "A_APL": a_apl, "B_APL": b_apl, "apl_floor": apl_floor,
            "B_RMSD": _f(overall, f"B_{L_RMSD}"), "delta_RMSD": _f(overall, f"delta_{L_RMSD}"),
            "B_good%": _f(overall, f"B_{L_GOOD}"), "delta_good%": _f(overall, f"delta_{L_GOOD}"),
            "delta_APL": _f(overall, f"delta_{L_APL}"),
        },
        "curves": curve_details,
        "straight_regressed": straight_regressed,
        "suggestions": suggestions,
    }

    out = os.path.join(C.RUNS_DIR, round_id, "verdict.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _round_cfg(round_id):
    path = os.path.join(C.RUNS_DIR, round_id, "round.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"weights": {}, "flags": {}, "lr": [], "train_data_json": "", "notes": "", "from": None}


def _load_verdict(round_id):
    path = os.path.join(C.RUNS_DIR, round_id, "verdict.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 入账
# ─────────────────────────────────────────────────────────────────────────────
LEDGER_FIELDS = [
    "round_id", "ts", "verdict", "from", "notes", "train_json",
    "angle_weight", "time_weight", "distance_weight", "fit_weight",
    "use_apl_loss", "apl_target",
    "OVERALL_B_RMSD", "OVERALL_B_APL", "OVERALL_B_good%",
    "OVERALL_delta_RMSD", "OVERALL_delta_APL", "OVERALL_delta_good%",
    "small_curves_B_RMSD", "small_curves_delta_RMSD", "small_curves_B_APL", "small_curves_delta_APL",
    "big_curves_B_RMSD", "big_curves_delta_RMSD", "big_curves_B_APL", "big_curves_delta_APL",
]


def _curve_of(result, name, key):
    for c in result["curves"]:
        if name in c["file"]:
            return c.get(key)
    return None


def rollup(round_id, result):
    cfg = _round_cfg(round_id)
    w, fl = cfg.get("weights", {}), cfg.get("flags", {})
    ov = result["overall"]
    row = {
        "round_id": round_id, "ts": datetime.now().isoformat(timespec="seconds"),
        "verdict": result["verdict"], "from": cfg.get("from"),
        "notes": cfg.get("notes", ""), "train_json": cfg.get("train_data_json", ""),
        "angle_weight": w.get("angle_weight"), "time_weight": w.get("time_weight"),
        "distance_weight": w.get("distance_weight"), "fit_weight": w.get("fit_weight"),
        "use_apl_loss": fl.get("USE_APL_LOSS"), "apl_target": fl.get("APL_TARGET"),
        "OVERALL_B_RMSD": ov["B_RMSD"], "OVERALL_B_APL": ov["B_APL"], "OVERALL_B_good%": ov["B_good%"],
        "OVERALL_delta_RMSD": ov["delta_RMSD"], "OVERALL_delta_APL": ov["delta_APL"],
        "OVERALL_delta_good%": ov["delta_good%"],
        "small_curves_B_RMSD": _curve_of(result, "small_curves", "B_RMSD"),
        "small_curves_delta_RMSD": _curve_of(result, "small_curves", "delta_RMSD"),
        "small_curves_B_APL": _curve_of(result, "small_curves", "B_APL"),
        "small_curves_delta_APL": _curve_of(result, "small_curves", "delta_APL"),
        "big_curves_B_RMSD": _curve_of(result, "big_curves", "B_RMSD"),
        "big_curves_delta_RMSD": _curve_of(result, "big_curves", "delta_RMSD"),
        "big_curves_B_APL": _curve_of(result, "big_curves", "B_APL"),
        "big_curves_delta_APL": _curve_of(result, "big_curves", "delta_APL"),
    }
    #os.makedirs(os.path.dirname(C.LEDGER_CSV), exist_ok=True)
    #new_file = not os.path.isfile(C.LEDGER_CSV)
    #with open(C.LEDGER_CSV, "a", newline="") as f:
    #    wtr = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
    #    if new_file:
    #        wtr.writeheader()
    #    wtr.writerow(row)
    with _locked(C.LEDGER_CSV):
        #new_file = not os.path.isfile(C.LEDGER_CSV)
        new_file = (not os.path.isfile(C.LEDGER_CSV)) or os.path.getsize(C.LEDGER_CSV) == 0
        with open(C.LEDGER_CSV, "a", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            if new_file:
                wtr.writeheader()
            wtr.writerow(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# ★ 经验回写
# ─────────────────────────────────────────────────────────────────────────────
def _num(x):
    return x if isinstance(x, (int, float)) else None


def _changed_knobs(cur_cfg, par_cfg):
    """返回 [(name, old, new), ...]: 本轮相对 parent 改动的旋钮/flag/数据。"""
    changed = []
    cw, pw = cur_cfg.get("weights", {}), par_cfg.get("weights", {})
    for k in _KNOB_KEYS:
        cv, pv = cw.get(k), pw.get(k)
        if cv is not None and pv is not None and round(float(cv), 6) != round(float(pv), 6):
            changed.append((k, pv, cv))
    cf, pf = cur_cfg.get("flags", {}), par_cfg.get("flags", {})
    for k in _FLAG_KEYS:
        cv, pv = cf.get(k), pf.get(k)
        if cv is not None and pv is not None and cv != pv:
            changed.append((k, pv, cv))
    if cur_cfg.get("lr") != par_cfg.get("lr"):
        changed.append(("lr", par_cfg.get("lr"), cur_cfg.get("lr")))
    if cur_cfg.get("train_data_json") != par_cfg.get("train_data_json"):
        changed.append(("train_data_json", par_cfg.get("train_data_json"), cur_cfg.get("train_data_json")))
    return changed


def _arrow(delta):
    if delta is None:
        return "?"
    if delta > 1e-9:
        return "↑"
    if delta < -1e-9:
        return "↓"
    return "→"


def _move(cur, par):
    """cur_B - par_B; 两者任一缺失 -> None"""
    if _num(cur) is None or _num(par) is None:
        return None
    return round(cur - par, 6)


def _fmt(x):
    return "n/a" if _num(x) is None else f"{x:.4f}"


def build_lesson(round_id, result):
    """生成本轮经验 (dict + markdown 文本)。"""
    cfg = _round_cfg(round_id)
    par_id = cfg.get("from")
    par_cfg = _round_cfg(par_id) if par_id else None
    par_v = _load_verdict(par_id) if par_id else None

    ov = result["overall"]
    cur_apl, cur_rmsd, cur_good = ov["B_APL"], ov["B_RMSD"], ov["B_good%"]

    # 改了什么
    changed = _changed_knobs(cfg, par_cfg) if par_cfg else []

    # 指标怎么动 (本轮 B vs parent B)
    moves = {}
    if par_v:
        pov = par_v["overall"]
        moves["OVERALL_APL"]  = (pov["B_APL"],  cur_apl,  _move(cur_apl,  pov["B_APL"]))
        moves["OVERALL_RMSD"] = (pov["B_RMSD"], cur_rmsd, _move(cur_rmsd, pov["B_RMSD"]))
        moves["OVERALL_good%"]= (pov["B_good%"],cur_good, _move(cur_good, pov["B_good%"]))
        for cname in ("small_curves", "big_curves"):
            cur_c = _curve_of(result, cname, "B_RMSD")
            par_c = next((c.get("B_RMSD") for c in par_v.get("curves", []) if cname in c["file"]), None)
            moves[f"{cname}_RMSD"] = (par_c, cur_c, _move(cur_c, par_c))

    # 推断候选规律
    derived = []
    if par_v is None:
        derived.append(("info", "首轮 / 无 parent 对比。仅相对冻结基线 A 评估 (见 verdict)。"))
    elif len(changed) == 0:
        ad = moves.get("OVERALL_APL", (None, None, None))[2]
        rd = moves.get("OVERALL_RMSD", (None, None, None))[2]
        derived.append(("repeat",
            f"与 parent 同配置(重复实验): APL 抖动 {_fmt(ad)}, RMSD 抖动 {_fmt(rd)} —— "
            f"小于此幅度的 delta 视为噪声, 不应据其下结论。"))
    elif len(changed) == 1:
        k, old, new = changed[0]
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            kdir = "↑" if new > old else "↓"
            apl_m = moves.get("OVERALL_APL", (None, None, None))[2]
            sc_m  = moves.get("small_curves_RMSD", (None, None, None))[2]
            derived.append(("single",
                f"单变量验证: {k} {old}→{new} ({kdir}) → "
                f"OVERALL APL {_arrow(apl_m)}{_fmt(apl_m)}, small_curves RMSD {_arrow(sc_m)}{_fmt(sc_m)}。"))
            # 与已知先验一致性核对
            if k == "time_weight" and apl_m is not None:
                ok = (apl_m > 0) == (new > old)
                derived.append(("prior",
                    f"先验'time_weight↑→APL↑': {'✅ 一致' if ok else '⚠️ 不一致(异常, 检查)'}"
                    f" (本轮每 {abs(new-old):g} 单位 → APL {_fmt(apl_m)})。"))
            if k == "angle_weight" and sc_m is not None:
                ok = (sc_m < 0) if (new > old) else (sc_m > 0)
                derived.append(("prior",
                    f"先验'angle_weight↑→curve RMSD↓': {'✅ 一致' if ok else '⚠️ 不一致'} (small_curves RMSD {_fmt(sc_m)})。"))
        else:
            derived.append(("single", f"单变量改动(非数值): {k}: {old} → {new}。"))
    else:
        names = ", ".join(f"{k}({old}→{new})" for k, old, new in changed)
        derived.append(("multi",
            f"⚠️ 多变量改动({len(changed)}个): {names} —— 归因不可靠, 下轮请回到单变量。"))

    lesson = {
        "round_id": round_id, "from": par_id, "verdict": result["verdict"],
        "notes": cfg.get("notes", ""), "changed": [list(c) for c in changed],
        "moves": moves, "derived": [d[1] for d in derived],
        "next_suggestions": result["suggestions"],
    }
    md = _lesson_md(round_id, cfg, result, par_id, changed, moves, derived)
    return lesson, md


def _lesson_md(round_id, cfg, result, par_id, changed, moves, derived):
    ov = result["overall"]
    L = []
    L.append(f"# Lesson — round `{round_id}`")
    L.append("")
    L.append(f"- 时间: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- 裁决: **{result['verdict']}**  (apl_ok={result['apl_ok']}, curves_ok={result['curves_ok']})")
    L.append(f"- parent: `{par_id or '—(首轮)'}`")
    L.append(f"- 意图(notes): {cfg.get('notes','') or '—'}")
    L.append("")
    L.append("## 改了什么 (vs parent)")
    if not par_id:
        L.append("- 首轮, 无 parent。")
    elif not changed:
        L.append("- 无旋钮改动 (与 parent 同配置, 等于重复实验/估方差)。")
    else:
        for k, old, new in changed:
            tag = "  ⚠️多变量" if len(changed) > 1 else ""
            L.append(f"- `{k}`: {old} → {new}{tag}")
    L.append("")
    L.append("## 指标怎么动 (本轮 B vs 上一轮 B)")
    if not moves:
        L.append("- (无 parent verdict, 跳过跨轮对比; 见下方相对基线 A)")
    else:
        for key, (pv, cv, dl) in moves.items():
            L.append(f"- {key}: {_fmt(pv)} → {_fmt(cv)}  ({_arrow(dl)}{_fmt(dl)})")
    L.append("")
    L.append("## 相对冻结基线 A (本轮 delta = B − A)")
    L.append(f"- OVERALL: APL Δ{_fmt(ov['delta_APL'])}, RMSD Δ{_fmt(ov['delta_RMSD'])}, good% Δ{_fmt(ov['delta_good%'])}")
    for c in result["curves"]:
        L.append(f"- {c['file']}: RMSD Δ{_fmt(c['delta_RMSD'])}, APL Δ{_fmt(c['delta_APL'])}, good% Δ{_fmt(c['delta_good%'])}")
    L.append("")
    L.append("## 推断的经验 (候选规律)")
    for _, txt in derived:
        L.append(f"- {txt}")
    L.append("")
    L.append("## 下一步建议 (来自 verdict + 优先级)")
    for s in result["suggestions"]:
        L.append(f"- {s}")
    L.append("")
    return "\n".join(L)


def _lessons_digest(round_id, cfg, result, par_id, changed, moves, derived):
    """累积 LESSONS.md 的一条精简块。"""
    if not changed:
        chg = "(无改动)" if par_id else "(首轮)"
    else:
        chg = "; ".join(f"{k} {old}→{new}" for k, old, new in changed)
    eff = []
    for key in ("OVERALL_APL", "small_curves_RMSD", "big_curves_RMSD"):
        if key in moves:
            _, _, dl = moves[key]
            eff.append(f"{key} {_arrow(dl)}{_fmt(dl)}")
    eff = ", ".join(eff) if eff else "(无跨轮对比)"
    rule = next((txt for tag, txt in derived if tag in ("prior", "single", "multi", "repeat")), "")
    nxt = result["suggestions"][0] if result["suggestions"] else ""
    blk = [
        f"## `{round_id}`  [{result['verdict']}]  (from `{par_id or '—'}`)  {datetime.now().isoformat(timespec='minutes')}",
        f"- change: {chg}",
        f"- effect: {eff}",
        f"- rule: {rule}" if rule else "- rule: —",
        f"- next: {nxt}" if nxt else "- next: —",
        "",
    ]
    return "\n".join(blk)


def write_lessons(round_id, result):
    cfg = _round_cfg(round_id)
    par_id = cfg.get("from")
    par_cfg = _round_cfg(par_id) if par_id else None
    par_v = _load_verdict(par_id) if par_id else None
    changed = _changed_knobs(cfg, par_cfg) if par_cfg else []

    lesson, md = build_lesson(round_id, result)

    # 1) 每轮一个新 md
    per_round = os.path.join(C.RUNS_DIR, round_id, "lesson.md")
    with open(per_round, "w") as f:
        f.write(md)
    # 同时存一份结构化, 方便程序读
    with open(os.path.join(C.RUNS_DIR, round_id, "lesson.json"), "w") as f:
        json.dump(lesson, f, indent=2, ensure_ascii=False)

    # 2) 追加到累积先验库 (重建 derived 以复用)
    _, _ = build_lesson, write_lessons  # noqa
    # 复算 moves/derived 用于 digest (build_lesson 已算过, 这里直接重取)
    # 为避免重复实现, 直接从 lesson dict 还原 digest 所需信息:
    moves = lesson["moves"]
    derived = [("single" if lesson["changed"] and len(lesson["changed"]) == 1 else
                ("multi" if lesson["changed"] and len(lesson["changed"]) > 1 else
                 ("repeat" if (par_id and not lesson["changed"]) else "info")),
                t) for t in lesson["derived"]]
    digest = _lessons_digest(round_id, cfg, result, par_id, changed, moves, derived)
    #os.makedirs(os.path.dirname(LESSONS_MD), exist_ok=True)
    #if not os.path.isfile(LESSONS_MD):
    #    with open(LESSONS_MD, "w") as f:
    #        f.write("# LESSONS — 累积已验证规律 (Claude 每轮决策前先读这里)\n\n"
    #                "> 每轮自动追加。单变量条目的 rule 可信; 多变量条目仅作记录。\n\n")
    #with open(LESSONS_MD, "a") as f:
    #    f.write(digest)
    with _locked(LESSONS_MD):
        #if not os.path.isfile(LESSONS_MD):
        if (not os.path.isfile(LESSONS_MD)) or os.path.getsize(LESSONS_MD) == 0:
            with open(LESSONS_MD, "w") as f:
                f.write("# LESSONS — 累积已验证规律 (Claude 每轮决策前先读这里)\n\n"
                        "> 每轮自动追加。单变量条目的 rule 可信; 多变量条目仅作记录。\n\n")
        with open(LESSONS_MD, "a") as f:
            f.write(digest)
    return per_round


# ─────────────────────────────────────────────────────────────────────────────
def _print(result, lesson_path):
    ov = result["overall"]
    print("\n" + "=" * 64)
    print(f"  ROUND {result['round_id']}   VERDICT: {result['verdict']}")
    print(f"  APL ok={result['apl_ok']}   curves ok={result['curves_ok']}")
    print("-" * 64)
    print(f"  OVERALL  B_APL={ov['B_APL']}  (floor={ov['apl_floor']})  "
          f"B_RMSD={ov['B_RMSD']} (Δ{ov['delta_RMSD']})  good%={ov['B_good%']} (Δ{ov['delta_good%']})")
    for c in result["curves"]:
        flag = "ok " if c["ok"] else "BAD"
        print(f"  [{flag}] {c['file']}  RMSD Δ{c['delta_RMSD']}  APL={c['B_APL']} (Δ{c['delta_APL']})  good% Δ{c['delta_good%']}")
    if result["straight_regressed"]:
        print(f"  ! 直线被连累: {[s['file'] for s in result['straight_regressed']]}")
    print("-" * 64)
    for s in result["suggestions"]:
        print("  -> " + s)
    print(f"  lesson -> {lesson_path}")
    print(f"  lessons (累积) -> {LESSONS_MD}")
    print("=" * 64 + "\n")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m auto_finetune.analyze <round_id>")
    round_id = sys.argv[1]
    result = verdict(round_id)
    rollup(round_id, result)
    lesson_path = write_lessons(round_id, result)   # ★ 经验回写
    _print(result, lesson_path)


if __name__ == "__main__":
    main()



