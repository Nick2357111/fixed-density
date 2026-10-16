"""Fixed Density v1 · benchmark 运行 / 拟合 / 统计 / HTML 报告。

从固定参数表（core 的 ParamSpec）出发，对 benchmark/data.csv 中每个谱面计算 n，
用分层标定集拟合连续映射曲线 g，再反算 value′ 并统计 dvalue，最后生成单文件 HTML 报告。
本文件同时是命令行入口：``python -m fixed_density.benchmark``。

依赖：numpy / scipy / matplotlib（绘图）。
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .core import *  # noqa: F401,F403
from .core import _REGISTRY
from .reader import read_mania_beatmap


# ===========================================================================
# 路径设置
# ===========================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent  # 仓库根目录
BENCHMARK_DIR = ROOT_DIR / "benchmark"
EXTRACTED_DIR = ROOT_DIR / "benchmark_extracted"  # 预处理后的物件头时间戳数据集
DEFAULT_CSV_PATH = BENCHMARK_DIR / "data.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "report" / "benchmark_report.html"
FIT_FOLDER = "course"  # 原标定 folder；现作为“纯留出评测集”从标定池中排除
FIT_MIN_VALUE = 2.0    # 拟合参考的 value 下限：含低 value 段以标定映射曲线的低段四个转折点
FIT_EXCLUDE_FOLDERS = ("course", "ln")  # 分层抽样的标定池中排除的 folder
                                       # course=原始标定留出集；ln=LN 谱仅作迁移评测，不参与标定
FIT_STRATA_SIZE = 60   # 分层抽样标定集的目标样本数（按难度分位分层）
FIT_VALUE_BINS = 5     # 按 value 分位数的分层箱数
FIT_SEED = 0           # 抽样随机种子（固定，保证可复现）
FIT_USE_IRR = True     # 拟合模型是否包含 irr（局部间隔多样性）项
FIT_USE_CHORD = True   # 拟合模型是否包含 chord（和弦度 = 音符数/起音数）项
FIT_USE_DUR = False    # 拟合模型是否包含 ln_dur（谱面时长）项——“长谱/耐力”维度实验开关

# LN 的 value 变换 g_ln：与 RC 的 g 同一套 softplus 分段曲线（两端线性渐近、全程光滑、
# 严格单调，斜率有界），把 dan 标尺搬到 ln_value 标尺上。
# 标定集（含段位单曲）的 value 覆盖 1~17，故拟合范围取 [1, 17.5]；低 value 段（1~6）只有
# 段位单曲、且残差最大，按 RC 的做法在低段也放衔接点，让曲线能弯折去抵消该段的系统性偏差，
# 而不是用单一直线外推。两侧量纲相同（斜率量级 ~1），界取 [0.2, 2.5]：实测放宽到 [0.1, 4.0]
# 会让低段斜率顶到边界、反而略差（R² 0.9654→0.9638，整体原始 RMSE 0.913→0.920）。
LN_MAP_KNOTS = ((1.5, 1.0), (3.5, 1.2), (6.0, 1.5), (9.5, 1.2), (13.0, 1.0), (15.5, 1.0))
LN_MAP_FIT_RANGE = (1.0, 17.5)
LN_MAP_SLOPE_FLOOR = 0.2
LN_MAP_SLOPE_CEIL = 2.5
LN_MAP_ITERS = 4


def fit_ln_map(k_values, values, b_lo: float = 1e-4, b_hi: float = 10.0,
               grid: int = 400, pin_a: float | None = None, pin_b: float | None = None) -> dict:
    """拟合 LN 的 value 变换：ln_value = a·ln(b·k·n + 1) = g_ln(value)。

    仿照 RC 的映射逻辑（ln n = g(value) + …），此处 g_ln 取同一套 softplus 分段曲线
    （map_basis / map_forward / map_invert 直接复用），σ 以带边界最小二乘求解。
    a、b 与曲线交替拟合（LN_MAP_ITERS 轮）：先用原始 value 定 (a,b)，再以
    y = a·ln(1+b·k·n) 为目标解 g_ln；随后用 g_ln(value) 作新目标重定 (a,b)。
    给定 pin_a / pin_b 时跳过交替拟合、直接固定 (a,b) 只解一次曲线
    —— 用于让报告与已冻结的最终模型（tosu 插件内硬编码的 a、b）严格一致。

    返回 dict：a / b / map_coef / map_knots / r2_log（变换后口径 R²）/ r2_raw（原始口径 R²）。
    调用方约定：ln_value = a·ln(1+b·k·n) 即“变换后的值”，
    原始口径的预测 value = map_invert(fit, ln_value)。
    """
    n = np.asarray(k_values, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    knots = LN_MAP_KNOTS
    nseg = len(knots) + 1
    if n.size == 0:
        return {"a": 0.0, "b": 1.0, "map_coef": [0.0] + [1.0] * nseg, "map_knots": knots,
                "r2_log": 0.0, "r2_raw": 0.0}

    use = (v >= LN_MAP_FIT_RANGE[0]) & (v <= LN_MAP_FIT_RANGE[1]) & (n > 0.0)
    if int(use.sum()) < nseg + 3:
        use = n > 0.0
    A = np.array([map_basis(float(x), knots) for x in v[use]])
    lb = np.full(A.shape[1], -np.inf, dtype=np.float64)
    ub = np.full(A.shape[1], np.inf, dtype=np.float64)
    lb[1:1 + nseg] = LN_MAP_SLOPE_FLOOR
    ub[1:1 + nseg] = LN_MAP_SLOPE_CEIL

    a, b, _ = fit_ln_log(n, v, b_lo, b_hi, grid)
    pinned = pin_a is not None and pin_b is not None
    if pinned:
        a, b = float(pin_a), float(pin_b)
    coef = [0.0] + [1.0] * nseg
    if pinned:
        coef = [float(x) for x in lsq_linear(A, a * np.log1p(b * n[use]),
                                             bounds=(lb, ub)).x]
    else:
        for _ in range(LN_MAP_ITERS):
            coef = [float(x) for x in lsq_linear(A, a * np.log1p(b * n[use]),
                                                 bounds=(lb, ub)).x]
            fit = {"map_coef": coef, "map_knots": knots}
            a, b, _ = fit_ln_log(n, np.array([map_forward(fit, float(x)) for x in v]),
                                 b_lo, b_hi, grid)
    fit = {"map_coef": coef, "map_knots": knots}
    pred = a * np.log1p(b * n[use])
    y = np.array([map_forward(fit, float(x)) for x in v[use]])

    def _r2(p, t) -> float:
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        return 1.0 - float(np.sum((p - t) ** 2)) / ss_tot if ss_tot > 0 else 1.0

    fit.update({"a": a, "b": b, "r2_log": _r2(pred, y),
                "r2_raw": _r2(pred, v[use])})
    return fit

# ===========================================================================
# CSV 读取与谱面文件定位
# ===========================================================================
def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # 兼容列名 expected（当前 data.csv）/ value（旧版）：统一规整为 value
    for row in rows:
        if not row.get("value"):
            row["value"] = row.get("expected", "")
    return rows


_WIN_ILLEGAL_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_stem(name: str) -> str:
    """把 CSV 中的谱面名转换为可安全用作文件名的 stem。

    替换 Windows 非法字符为下划线，去掉首尾空白与尾部点/空格；
    空结果回退为 "_"。提取脚本与读取端共用，保证命名一致。
    """
    s = _WIN_ILLEGAL_RE.sub("_", name).strip()
    return s.rstrip(". ") or "_"


def extracted_json_path(folder: str, name: str) -> Path:
    """提取数据文件中该谱面对应的 JSON 路径。"""
    return EXTRACTED_DIR / folder / f"{sanitize_stem(name)}.json"


_COLUMNS_CACHE: dict = {}  # (folder, name) -> (轨道列表, OD)；调参循环中每轮省去重复 JSON 解析


def _load_extracted(folder: str, name: str):
    """读取并缓存 benchmark_extracted 中的 (轨道列表, OD)；文件不存在时返回 None。

    返回的轨道列表与 read_mania_beatmap 同构，兼容旧版仅存头时间戳的 JSON（长度按 0 处理，
    OD 缺失时回退 8.0）；调用方不得修改返回的列表结构。
    """
    ck = (folder, name)
    if ck in _COLUMNS_CACHE:
        return _COLUMNS_CACHE[ck]
    path = extracted_json_path(folder, name)
    if not path.is_file():
        _COLUMNS_CACHE[ck] = None
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cols = []
    for col in data["columns"]:
        if col and isinstance(col[0], (list, tuple)):
            cols.append([(int(t), int(l)) for t, l in col])
        else:
            cols.append([(int(t), 0) for t in col])
    got = (cols, float(data.get("od", 8.0)))
    _COLUMNS_CACHE[ck] = got
    return got


def load_extracted_columns(folder: str, name: str):
    """读取 benchmark_extracted 中预处理好的物件 (头时间戳, 长度)。文件不存在时返回 None。"""
    got = _load_extracted(folder, name)
    return None if got is None else got[0]


def load_extracted_od(folder: str, name: str) -> float:
    """读取该谱面的 OD（用于 LN 头/尾判定窗）；缺失时回退 8.0。"""
    got = _load_extracted(folder, name)
    return 8.0 if got is None else got[1]


_RATE_SUFFIX_RE = re.compile(r"\s+x\d[\d.]*\s*$")


def _is_rate_variant(stem: str, target: str) -> bool:
    """候选名去掉前缀后，剩余部分是否以 ' x<数字>' 变速后缀结尾。"""
    remainder = stem[len(target):].strip()
    return bool(_RATE_SUFFIX_RE.search(remainder))


def resolve_osu_path(folder: str, name: str):
    """定位谱面文件：精确 → 忽略空白/大小写精确 → 唯一前缀匹配（含子目录）。

    前缀匹配存在多个候选时，排除“变速变体”候选（剩余部分形如 x1.2/x0.95），
    若剩余唯一则采用，否则报匹配歧义。
    """
    fdir = BENCHMARK_DIR / folder
    direct = fdir / f"{name}.osu"
    if direct.is_file():
        return direct, "精确匹配"

    if not fdir.is_dir():
        return None, "文件夹不存在"

    files = list(fdir.rglob("*.osu"))
    target = name.strip().casefold()

    norm_matches = [p for p in files if p.stem.strip().casefold() == target]
    if len(norm_matches) == 1:
        return norm_matches[0], "规范化匹配"
    if len(norm_matches) > 1:
        return None, "匹配歧义"

    prefix_matches = [p for p in files if p.stem.casefold().startswith(target)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], "前缀匹配"
    if len(prefix_matches) > 1:
        base_matches = [p for p in prefix_matches
                        if not _is_rate_variant(p.stem.casefold(), target)]
        if len(base_matches) == 1:
            return base_matches[0], "前缀匹配（排除变速变体）"
        return None, "匹配歧义"
    return None, "文件未找到"

# ===========================================================================
# benchmark 执行
# ===========================================================================
def run_benchmark(
    algo: AlgorithmSpec,
    params: dict[str, float],
    rows: list[dict],
    source: str = "extracted",
) -> list[dict]:
    """对每一行 CSV 计算 n，返回逐谱面结果（同一数据结构驱动全部统计与报告）。

    source="extracted"：直接读取 benchmark_extracted 中预处理好的物件头时间戳；
    source="osu"：沿用 .osu 解析 + 文件定位（resolve_osu_path + read_mania_beatmap）。
    """
    results = []
    for idx, row in enumerate(rows, 1):
        name, folder = row["name"], row["folder"]
        value = float(row["value"])
        rec = {
            "idx": idx,
            "name": name,
            "folder": folder,
            "pattern": (row.get("pattern") or "").strip() or "未标注",
            "value": value,
            "value_mapped": None,  # 拟合出连续映射曲线后于 apply_fit_and_stats 中填充
            "n": None,
            "value_pred": None,
            "dvalue": None,
            "grade": None,
            "value_pred_mapped": None,
            "dvalue_mapped": None,
            "grade_mapped": None,
            "elapsed_ms": None,
            "irr": 0.0,
            "chord": 1.0,
            "ln_N": 0.0,
            "ln_dur": 0.0,
            "match": "",
            "error": "",
        }
        if source == "extracted":
            path = extracted_json_path(folder, name)
            match = "提取数据"
            missing = not path.is_file()
        else:
            path, match = resolve_osu_path(folder, name)
            missing = path is None
        rec["match"] = match
        if missing:
            rec["error"] = "未找到提取数据" if source == "extracted" else match
            results.append(rec)
            print(f"[{idx}/{len(rows)}] 跳过 {folder}/{name}：{rec['error']}")
            continue
        try:
            t0 = time.perf_counter()
            columns = (load_extracted_columns(folder, name) if source == "extracted"
                       else read_mania_beatmap(path))
            n_value = float(algo.func(columns, params))
            rec["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
            rec["n"] = n_value
            rec["irr"] = irregularity_cached(folder, name, columns)
            rec["chord"] = chord_ratio(columns)
            rec["ln_N"] = math.log(max(sum(len(c) for c in columns), 1))
            rec["ln_dur"] = math.log(max(chart_duration_s(columns), 1e-3))
        except Exception as exc:  # 单个谱面失败不影响整体报告
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{idx}/{len(rows)}] 失败 {folder}/{name}：{rec['error']}")
        results.append(rec)
    return results


def select_calibration(results: list[dict]) -> list[dict]:
    """挑选指数拟合的标定集：在「非 FIT_EXCLUDE_FOLDERS 且 value ≥ FIT_MIN_VALUE」的谱面中，
    按 value 分位箱分层、按各层样本数比例抽样 FIT_STRATA_SIZE 张。

    按难度分层保证全难度段都被覆盖；配额严格按比例（最大余数法），不额外保底，
    避免过度代表稀有难度层而扭曲标定分布。固定种子（FIT_SEED）保证可复现。
    """
    pool = [r for r in results
            if r["n"] is not None and r["value"] >= FIT_MIN_VALUE
            and r["folder"] not in FIT_EXCLUDE_FOLDERS]
    if len(pool) <= FIT_STRATA_SIZE:
        return pool

    vals = np.array([r["value"] for r in pool], dtype=np.float64)
    edges = np.quantile(vals, np.linspace(0.0, 1.0, FIT_VALUE_BINS + 1))
    edges[-1] += 1e-9
    strata: dict[int, list[dict]] = {}
    for r in pool:
        b = int(np.searchsorted(edges, r["value"], side="right")) - 1
        b = min(max(b, 0), FIT_VALUE_BINS - 1)
        strata.setdefault(b, []).append(r)

    keys = sorted(strata)
    total = float(len(pool))
    raw = {k: FIT_STRATA_SIZE * len(strata[k]) / total for k in keys}
    quota = {k: int(math.floor(raw[k])) for k in keys}
    left = FIT_STRATA_SIZE - sum(quota.values())
    for k in sorted(keys, key=lambda k: -(raw[k] - quota[k]))[:max(left, 0)]:
        quota[k] += 1

    rng = np.random.default_rng(FIT_SEED)
    picked: list[dict] = []
    for k in keys:
        grp = strata[k]
        take = min(quota[k], len(grp))
        idx = rng.choice(len(grp), size=take, replace=False)
        picked.extend(grp[i] for i in idx)
    return picked


def fit_exponential(course_records: list[dict]) -> dict:
    """对给定标定谱面拟合连续映射曲线模型：

        ln n = g(value) + C·irr + E·chord + D·ln(N)

    g 以分段斜率 σ_0…σ_K 为参数、两端线性渐近、全程光滑且严格单调（见 MAP_KNOTS 注释）。
    以带边界的最小二乘求解，约束 σ_j ≥ MAP_SLOPE_FLOOR（保证单调 + 抑制过平段放大方差）。
    拟合仅使用 value ∈ MAP_FIT_RANGE 的标定谱面（含低段以标定四个转折点）。
    FIT_USE_IRR / FIT_USE_CHORD 分别控制 irr、chord 两列是否参与（关闭时其系数恒为 0）。
    """
    knots = MAP_KNOTS
    nk = len(knots)
    nseg = nk + 1
    use = [r for r in course_records
           if MAP_FIT_RANGE[0] <= r["value"] <= MAP_FIT_RANGE[1] and r["n"] > 0.0]
    if len(use) < nseg + 3:
        use = [r for r in course_records if r["n"] > 0.0]
    if len(use) < nseg + 3:
        raise RuntimeError("可用于标定的谱面不足或存在非正 n 值")

    feats = (["irr"] if FIT_USE_IRR else []) + (["chord"] if FIT_USE_CHORD else [])
    A = np.array([map_basis(r["value"], knots)
                  + [r.get(nm, 1.0 if nm == "chord" else 0.0) for nm in feats]
                  + [r.get("ln_N", 0.0)]
                  + ([r.get("ln_dur", 0.0)] if FIT_USE_DUR else []) for r in use])
    y = np.log(np.array([r["n"] for r in use], dtype=np.float64))

    # 边界约束：c 与各特征/长度系数自由；段斜率 σ_j 限制在 [floor, ceil]
    lb = np.full(A.shape[1], -np.inf, dtype=np.float64)
    ub = np.full(A.shape[1], np.inf, dtype=np.float64)
    lb[1:1 + nseg] = MAP_SLOPE_FLOOR
    if MAP_SLOPE_CEIL is not None:
        ub[1:1 + nseg] = MAP_SLOPE_CEIL

    coef = [float(x) for x in lsq_linear(A, y, bounds=(lb, ub)).x]
    pred = A @ np.asarray(coef)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    sig = coef[1:1 + nseg]  # 各段斜率 σ_0 … σ_K；首/末段即低/高端线性渐近线斜率
    # 系数按实际参与的列顺序取（irr / chord / dur 可能被 FIT_USE_* 关闭，下标随之平移）
    j = 1 + nseg
    c_irr = coef[j] if FIT_USE_IRR else 0.0
    j += 1 if FIT_USE_IRR else 0
    c_chord = coef[j] if FIT_USE_CHORD else 0.0
    j += 1 if FIT_USE_CHORD else 0
    c_lnn = coef[j]
    j += 1
    c_lndur = coef[j] if FIT_USE_DUR else 0.0
    fit = {"map_coef": coef, "map_knots": knots,
           "use_irr": FIT_USE_IRR, "use_chord": FIT_USE_CHORD, "use_dur": FIT_USE_DUR,
           "C": c_irr, "E": c_chord, "D": c_lnn, "D2": c_lndur,
           "log_A": coef[0], "B": sig[0], "A": math.exp(min(coef[0], 700.0)),
           "slopes": sig, "r2_log": r2}
    # 真实渐近线：v→−∞ 时 g ≈ c + σ_0·v；v→+∞ 用大 v 数值定截距
    v_big = max(v0 for v0, _s in knots) + 80.0
    fit["asym_lo"] = (coef[0], sig[0])
    fit["asym_hi"] = (map_forward(fit, v_big) - sig[-1] * v_big, sig[-1])
    return fit


def predict_mapped(n_value: float, fit: dict, irr: float = 0.0, ln_n: float = 0.0,
                   chord: float = 1.0, ln_dur: float = 0.0) -> float:
    """映射空间（曲线值）下的预测：g(value′) = ln n − C·irr − E·chord − D·ln N − D2·ln dur。"""
    return (math.log(n_value) - fit.get("C", 0.0) * irr - fit.get("E", 0.0) * chord
            - fit.get("D", 0.0) * ln_n - fit.get("D2", 0.0) * ln_dur)


def grade_of(abs_d: float) -> str:
    if abs_d <= GRADE_BOUNDS[0]:
        return GRADE_LABELS[0]
    if abs_d <= GRADE_BOUNDS[1]:
        return GRADE_LABELS[1]
    if abs_d <= GRADE_BOUNDS[2]:
        return GRADE_LABELS[2]
    return GRADE_LABELS[3]


# 口径字段配置：同一套图表/报告渲染逻辑在两种 value 口径间切换
SPACE_FIELDS = {
    "original": {
        "value": "value", "pred": "value_pred",
        "dvalue": "dvalue", "grade": "grade",
        "label_value": "value", "label_pred": "value′",
        "label_dvalue": "dvalue",
        "subtitle": "原始 value 口径",
        "note_fit": "反算 g(value′) = ln n{terms}，再经逆曲线 g⁻¹ 得到 value′。",
    },
    "mapped": {
        "value": "value_mapped", "pred": "value_pred_mapped",
        "dvalue": "dvalue_mapped", "grade": "grade_mapped",
        "label_value": "map(value)", "label_pred": "map(value′)",
        "label_dvalue": "dvalue(map)",
        "subtitle": "映射 value 口径（拟合所在空间）",
        "note_fit": "在曲线空间直接比较：g(value′) = ln n{terms}。",
    },
}


def apply_fit_and_stats(
    results: list[dict],
    fit: dict,
) -> None:
    """根据拟合结果同时填充原始口径与映射口径的预测值 / 差值 / 类别（就地）。"""
    b_ref = fit["B"] if abs(fit.get("B", 0.0)) > 1e-9 else 1.0  # 曲线低端斜率（仅作口径归一化）
    for rec in results:
        if rec["n"] is None:
            continue
        g_pred = predict_mapped(rec["n"], fit, rec.get("irr", 0.0), rec.get("ln_N", 0.0),
                                rec.get("chord", 1.0), rec.get("ln_dur", 0.0))
        g_true = map_forward(fit, rec["value"])
        # 原始口径：把曲线空间预测逆映射回 value
        rec["value_pred"] = map_invert(fit, g_pred)
        rec["dvalue"] = rec["value_pred"] - rec["value"]
        rec["grade"] = grade_of(abs(rec["dvalue"]))
        # 映射口径：曲线空间残差，按低端斜率归一化到 value 量级以便共用评价区间
        rec["value_mapped"] = g_true / b_ref
        rec["value_pred_mapped"] = g_pred / b_ref
        rec["dvalue_mapped"] = (g_pred - g_true) / b_ref
        rec["grade_mapped"] = grade_of(abs(rec["dvalue_mapped"]))


def summarize(dvalues: np.ndarray) -> dict:
    d = np.asarray(dvalues, dtype=np.float64)
    grades = [grade_of(abs(x)) for x in d]
    counts = {label: grades.count(label) for label in GRADE_LABELS}
    return {
        "count": int(d.size),
        "mean": float(np.mean(d)),
        "var": float(np.var(d)),  # 总体方差
        "std": float(np.std(d)),
        "rmse": float(np.sqrt(np.mean(d * d))),  # 均方根误差
        "abs_mean": float(np.mean(np.abs(d))),
        "abs_median": float(np.median(np.abs(d))),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "grade_counts": counts,
    }


# ===========================================================================
# 绘图（统一从 results / fit 派生，保证图、表、文字口径一致）
# ===========================================================================
def _setup_chinese_font() -> None:
    for font_name in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei"):
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _scatter_panel(ax, recs, sp: dict, title: str, show_legend: bool = False) -> tuple[float, float] | None:
    if not recs:
        ax.set_visible(False)
        return None
    for folder in sorted({r["folder"] for r in recs}):
        sub = [r for r in recs if r["folder"] == folder]
        xs = [r[sp["value"]] for r in sub]
        ys = [r[sp["pred"]] for r in sub]
        ax.scatter(xs, ys, s=16, alpha=0.75,
                   color=FOLDER_COLORS.get(folder, "#7f8c8d"), label=folder,
                   edgecolors="none")
    xs_all = np.array([r[sp["value"]] for r in recs])
    ys_all = np.array([r[sp["pred"]] for r in recs])
    lo = min(xs_all.min(), ys_all.min()) - 0.5
    hi = max(xs_all.max(), ys_all.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555", linewidth=1.2, label="y = x")
    if len(recs) >= 2:
        k1, k0 = np.polyfit(xs_all, ys_all, 1)
        grid = np.linspace(lo, hi, 100)
        ax.plot(grid, k1 * grid + k0, color="#c0392b", linewidth=1.6,
                label=f"线性拟合 y={k1:.3f}x+{k0:.3f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(sp["label_value"])
    ax.set_ylabel(sp["label_pred"])
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left")
    return (float(k1), float(k0)) if len(recs) >= 2 else None


def plot_overall_scatter(valid: list[dict], sp: dict) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    _scatter_panel(ax, valid, sp,
                   f"全部谱面：{sp['label_pred']} vs {sp['label_value']}（虚线为 y=x）",
                   show_legend=True)
    return _fig_to_base64(fig)


def plot_group_scatters(valid: list[dict], per_group: dict, sp: dict,
                        group_key: str, ncol: int = 3) -> str:
    """按某个分组字段（folder / pattern 等）分面绘制散点 + 线性拟合 + y=x。

    子图内点仍按 folder 着色（folder 单分面时自然单色，跨 folder 的
    pattern 分面可区分各点所属 folder）。
    """
    groups = sorted(per_group.keys())
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 5.4 * nrow), squeeze=False)
    for i, group in enumerate(groups):
        recs = [r for r in valid if r[group_key] == group]
        _scatter_panel(axes[i // ncol][i % ncol], recs, sp,
                       f"{group_key} = {group}（n={len(recs)}）", show_legend=False)
    for j in range(len(groups), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_exponential_fit(course_records: list[dict], fit: dict) -> str:
    xs = np.array([r["value"] for r in course_records])
    ys = np.array([r["n"] for r in course_records])
    grid = np.linspace(float(xs.min()), float(xs.max()), 300)
    # 曲线为 ln n 的拟合值：g(value) + C·irr + E·chord + D·lnN；用标定集均值补上后三项才能贴合散点
    shift = (fit.get("C", 0.0) * float(np.mean([r.get("irr", 0.0) for r in course_records]))
             + fit.get("E", 0.0) * float(np.mean([r.get("chord", 1.0) for r in course_records]))
             + fit.get("D", 0.0) * float(np.mean([r.get("ln_N", 0.0) for r in course_records]))
             + fit.get("D2", 0.0) * float(np.mean([r.get("ln_dur", 0.0) for r in course_records])))
    curve = np.array([math.exp(map_forward(fit, v) + shift) for v in grid])

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(xs, ys, s=28, color=FOLDER_COLORS["course"], alpha=0.85,
               edgecolors="white", linewidths=0.5, label="标定谱面")
    ax.plot(grid, curve, color="#c0392b", linewidth=1.8,
            label=f"拟合曲线 n = exp(g(value) + C·mean(irr) + D·mean(lnN))，R²(log)={fit['r2_log']:.5f}")
    ax.set_yscale("log")
    ax.set_xlabel("value（原始难度）")
    ax.set_ylabel("n（对数轴）")
    ax.set_title("标定谱面：n 与 value 的连续曲线拟合")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    return _fig_to_base64(fig)


def plot_mapping_curve(fit: dict) -> str:
    """绘制连续映射曲线，并换算到 value 等效刻度（低端斜率归一为 1）以便与 y = x 对照。

    g 本身是 ln n 量纲（斜率 ~0.15），无法与 y = x 直接比较；这里做仿射归一
        g̃(v) = v_ref + (g(v) − g(v_ref)) / b，  b 为曲线低端斜率
    使低端斜率变为 1、并锚定在 v_ref 处，从而直观显示"value 轴被如何重分布
    （中段压缩）"，同时保留低端斜率 = 1 的参照。
    """
    b_ref = fit["B"] if abs(fit.get("B", 0.0)) > 1e-9 else 1.0
    v_ref = 0.5 * (MAP_FIT_RANGE[0] + MAP_FIT_RANGE[1])
    g_ref = map_forward(fit, v_ref)

    def eff(v: float) -> float:
        return v_ref + (map_forward(fit, v) - g_ref) / b_ref

    vs = np.linspace(0.0, 18.0, 1801)
    mapped = np.array([eff(v) for v in vs])
    # 真实渐近线（平移到各自实际位置，而非强制过锚点）
    a_lo, b_lo = fit["asym_lo"]
    a_hi, b_hi = fit["asym_hi"]
    lo_line = v_ref + (a_lo + b_lo * vs - g_ref) / b_ref
    hi_line = v_ref + (a_hi + b_hi * vs - g_ref) / b_ref

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.axvspan(MAP_FIT_RANGE[0], MAP_FIT_RANGE[1], color="#f1c42f", alpha=0.16,
               label=f"曲线拟合范围 [{MAP_FIT_RANGE[0]:g}, {MAP_FIT_RANGE[1]:g}]")
    for v0, _s in fit.get("map_knots", MAP_KNOTS):
        ax.axvline(v0, color="#bbbbbb", ls=":", lw=1)
    ax.plot(vs, mapped, color="#3498db", lw=2.2, label="映射曲线 g(value)（等效 value 刻度）")
    ax.plot(vs, lo_line, color="#7f8c8d", ls="--", lw=1.3,
            label=f"低端线性渐近线（斜率 {b_lo / b_ref:.2f}）")
    ax.plot(vs, hi_line, color="#e67e22", ls="--", lw=1.3,
            label=f"高端线性渐近线（斜率 {b_hi / b_ref:.2f}）")
    ax.plot([0, 18], [0, 18], color="#999999", ls=":", lw=1, label="y = x（不扭曲）")
    ax.set_xlim(0.0, 18.0)
    ax.set_ylim(0.0, 19.0)
    ax.set_xlabel("原始 value")
    ax.set_ylabel("映射后 g(value)（等效 value 刻度）")
    sl = fit["slopes"]
    ax.set_title("value → 映射曲线（连续、两端线性渐近、无分段）；"
                 f"{len(sl)} 段，归一化斜率 " + " / ".join(f"{x / b_ref:.2f}" for x in sl))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


# ===========================================================================
# HTML 报告
# ===========================================================================
def _fmt(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _grade_badge(label: str) -> str:
    idx = GRADE_LABELS.index(label)
    return f'<span class="badge" style="background:{GRADE_COLORS[idx]}">{label}</span>'


# 可排序表格的交互脚本（普通字符串，避免 f-string 内花括号转义）：
# 点击 <table class="sortable"> 的任意图表头，按该列升/降序重排行，
# 纯数字文本按数值比较（兼容 +0.12 / -1.3 / 1,234），其余按中文文本比较。
SORTABLE_SCRIPT = """<script>
document.querySelectorAll('table.sortable').forEach(function (table) {
  var headers = table.rows[0].cells;
  Array.prototype.forEach.call(headers, function (th, col) {
    th.classList.add('sortable-th');
    th.insertAdjacentHTML('beforeend', ' <span class="sort-arr"></span>');
    th.addEventListener('click', function () {
      var asc = th.getAttribute('data-dir') !== 'asc';
      Array.prototype.forEach.call(headers, function (h) {
        h.setAttribute('data-dir', '');
        h.querySelector('.sort-arr').textContent = '';
      });
      th.setAttribute('data-dir', asc ? 'asc' : 'desc');
      th.querySelector('.sort-arr').textContent = asc ? '\\u25B2' : '\\u25BC';
      var rows = Array.prototype.slice.call(table.rows, 1);
      rows.sort(function (r1, r2) {
        var a = r1.cells[col].textContent.trim();
        var b = r2.cells[col].textContent.trim();
        var na = parseFloat(a.replace(/,/g, ''));
        var nb = parseFloat(b.replace(/,/g, ''));
        var cmp = (!isNaN(na) && !isNaN(nb)) ? na - nb : a.localeCompare(b, 'zh');
        return asc ? cmp : -cmp;
      });
      rows.forEach(function (r) { table.appendChild(r); });
    });
  });
});
</script>"""


# 报告动态脚本：读取内嵌 JSON 数据，双滑块按 value 过滤后在前端实时重算
# 指标卡 / 区间表 / 饼图(conic-gradient) / 直方图(canvas) / 分组表 / 明细表
DYNAMIC_REPORT_SCRIPT = """<script>
(function () {
  var CFG = JSON.parse(document.getElementById('report-data').textContent);
  var L = CFG.labels, G = CFG.grades;
  var iMin = document.getElementById('rmin'), iMax = document.getElementById('rmax');
  var fill = document.getElementById('rfill'), rlab = document.getElementById('rlab');

  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function fmt4(x) { return x.toFixed(4); }
  function fmtS(x) { return (x >= 0 ? '+' : '') + x.toFixed(4); }
  function fmtG(x) { return String(Number(x.toPrecision(6))); }
  function gradeOf(d) {
    var a = Math.abs(d);
    return a <= 0.2 ? G.labels[0] : a <= 0.5 ? G.labels[1] : a <= 1.0 ? G.labels[2] : G.labels[3];
  }
  function gIdx(lab) { return G.labels.indexOf(lab); }

  function current() {
    var a = +iMin.value, b = +iMax.value;
    return a <= b ? [a, b] : [b, a];
  }
  function filtered(lo, hi) {
    return CFG.rows.filter(function (r) { return r.v >= lo - 1e-9 && r.v <= hi + 1e-9; });
  }
  function stats(ds) {
    if (!ds.length) return null;
    var n = ds.length, m = 0, i;
    for (i = 0; i < n; i++) m += ds[i];
    m /= n;
    var v = 0, am = 0, sq = 0;
    for (i = 0; i < n; i++) { v += (ds[i] - m) * (ds[i] - m); am += Math.abs(ds[i]); sq += ds[i] * ds[i]; }
    return { n: n, mean: m, variance: v / n, absMean: am / n, rmse: Math.sqrt(sq / n) };
  }
  function gradeCounts(recs) {
    var gc = {}; G.labels.forEach(function (l) { gc[l] = 0; });
    recs.forEach(function (r) { gc[gradeOf(r.d)]++; });
    return gc;
  }
  function set(id, text) { document.getElementById(id).textContent = text; }
  function setHtml(id, html) { document.getElementById(id).innerHTML = html; }

  function renderCards(st) {
    set('st-count', st ? String(st.n) : '—');
    set('st-mean', st ? fmtS(st.mean) : '—');
    set('st-var', st ? fmt4(st.variance) : '—');
    set('st-absmean', st ? fmt4(st.absMean) : '—');
    set('st-rmse', st ? fmt4(st.rmse) : '—');
  }

  function renderGrades(recs) {
    var gc = gradeCounts(recs), total = recs.length;
    setHtml('grade-body', G.labels.map(function (lab, i) {
      var c = gc[lab], pct = total ? (100 * c / total).toFixed(2) : '0.00';
      return '<tr><td><span class="badge" style="background:' + G.colors[i] + '">' + lab + '</span></td>'
        + '<td>' + G.bounds[i] + '</td><td>' + c + '</td><td>' + pct + '%</td></tr>';
    }).join(''));
    var acc = 0, stops = [];
    G.labels.forEach(function (lab, i) {
      var frac = total ? gc[lab] / total : 0;
      stops.push(G.colors[i] + ' ' + (acc * 360).toFixed(2) + 'deg ' + ((acc + frac) * 360).toFixed(2) + 'deg');
      acc += frac;
    });
    document.getElementById('grade-pie').style.background = total
      ? 'conic-gradient(' + stops.join(',') + ')' : 'none';
  }

  function renderHist(recs) {
    var cv = document.getElementById('dhist'), ctx = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '15px sans-serif'; ctx.fillStyle = '#222'; ctx.textAlign = 'left';
    var padL = 56, padR = 18, padT = 40, padB = 52;
    var ds = recs.map(function (r) { return r.d; });
    if (!ds.length) {
      ctx.fillStyle = '#888';
      ctx.fillText(L.d + ' 分布（当前范围内无谱面）', padL, 24);
      return;
    }
    var lo = Math.floor(Math.min.apply(null, ds) / 0.1) * 0.1;
    var hi = Math.ceil(Math.max.apply(null, ds) / 0.1) * 0.1;
    var nb = Math.round((hi - lo) / 0.1) + 1;
    var counts = new Array(nb);
    for (var i = 0; i < nb; i++) counts[i] = 0;
    ds.forEach(function (d) {
      var k = Math.min(nb - 1, Math.max(0, Math.floor((d - lo) / 0.1 + 1e-9)));
      counts[k]++;
    });
    var maxC = Math.max.apply(null, counts);
    var iw = (W - padL - padR) / nb, ih = H - padT - padB, base = padT + ih;
    ctx.strokeStyle = '#e6eaf0'; ctx.fillStyle = '#8899a6';
    for (var g = 1; g <= 4; g++) {
      var y = base - ih * g / 4;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(String(Math.round(maxC * g / 4)), padL - 8, y + 4);
    }
    for (i = 0; i < nb; i++) {
      if (!counts[i]) continue;
      ctx.fillStyle = G.colors[gIdx(gradeOf(lo + (i + 0.5) * 0.1))];
      var h = counts[i] / maxC * ih;
      ctx.fillRect(padL + i * iw + 1, base - h, Math.max(1, iw - 2), h);
    }
    ctx.setLineDash([4, 3]); ctx.strokeStyle = '#666';
    [0, -0.2, 0.2, -0.5, 0.5, -1.0, 1.0].forEach(function (b) {
      if (b < lo || b > hi) return;
      var x = padL + (b - lo) / 0.1 * iw;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, base); ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(padL, base + 0.5); ctx.lineTo(W - padR, base + 0.5); ctx.stroke();
    var step = Math.max(1, Math.ceil(nb / 16));
    ctx.fillStyle = '#55707f'; ctx.textAlign = 'center'; ctx.font = '11px sans-serif';
    for (i = 0; i < nb; i += step) {
      ctx.fillText((lo + i * 0.1).toFixed(1), padL + i * iw + iw / 2, base + 16);
    }
    ctx.font = '13px sans-serif';
    ctx.fillText(L.d + ' = ' + L.p + ' − ' + L.v + '（柱按区间着色，分度 0.1）', padL, base + 40);
    ctx.font = '15px sans-serif'; ctx.fillStyle = '#222'; ctx.textAlign = 'left';
    ctx.fillText(L.d + ' 分布图（分度区间 0.1，N=' + ds.length + '）', padL, 24);
    var lx = W - padR - 480;
    G.labels.forEach(function (lab, i) {
      ctx.fillStyle = G.colors[i]; ctx.fillRect(lx, 14, 12, 12);
      ctx.fillStyle = '#333'; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(lab + '（' + G.bounds[i] + '）', lx + 16, 24);
      lx += 122;
    });
  }

  function groupStats(recs, key) {
    var m = {};
    recs.forEach(function (r) {
      var k = r[key];
      if (!m[k]) m[k] = { key: k, c: 0, s: 0, s2: 0, sa: 0, gc: {} };
      var o = m[k];
      o.c++; o.s += r.d; o.s2 += r.d * r.d; o.sa += Math.abs(r.d);
      var g = gradeOf(r.d); o.gc[g] = (o.gc[g] || 0) + 1;
    });
    return Object.keys(m).map(function (k) {
      var o = m[k];
      o.mean = o.s / o.c;
      o.variance = o.s2 / o.c - o.mean * o.mean;
      o.absMean = o.sa / o.c;
      o.rmse = Math.sqrt(o.s2 / o.c);
      return o;
    });
  }
  function statRows(list, sortByCount) {
    list = list.slice();
    list.sort(sortByCount
      ? function (a, b) { return b.c - a.c || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0); }
      : function (a, b) { return a.key < b.key ? -1 : a.key > b.key ? 1 : 0; });
    return list.map(function (o) {
      var cells = G.labels.map(function (lab, i) {
        return '<td style="color:' + G.colors[i] + '">' + (o.gc[lab] || 0) + '</td>';
      }).join('');
      return '<tr><td><b>' + esc(o.key) + '</b></td><td>' + o.c + '</td>'
        + '<td>' + fmtS(o.mean) + '</td><td>' + fmt4(o.variance) + '</td><td>' + fmt4(o.absMean) + '</td>'
        + '<td>' + fmt4(o.rmse) + '</td>'
        + cells + '</tr>';
    }).join('');
  }

  function detailRows(recs) {
    return recs.map(function (r) {
      var gi = gIdx(gradeOf(r.d));
      return '<tr><td>' + esc(r.fo) + '</td><td>' + esc(r.pa) + '</td><td>' + esc(r.nm) + '</td>'
        + '<td>' + fmtG(r.v) + '</td><td>' + fmtG(r.n) + '</td>'
        + '<td>' + fmt4(r.pr) + '</td><td>' + fmtS(r.d) + '</td>'
        + '<td><span style="color:' + G.colors[gi] + '">' + gradeOf(r.d) + '</span></td></tr>';
    }).join('');
  }

  function syncSlider() {
    var lo = +iMin.value, hi = +iMax.value;
    var l = Math.min(lo, hi), h = Math.max(lo, hi);
    rlab.textContent = L.v + ' ∈ [' + l.toFixed(1) + ', ' + h.toFixed(1) + ']　谱面 ' + filtered(l, h).length + ' 张';
    var p1 = (l - CFG.vmin) / (CFG.vmax - CFG.vmin) * 100;
    var p2 = (h - CFG.vmin) / (CFG.vmax - CFG.vmin) * 100;
    fill.style.left = p1 + '%';
    fill.style.width = Math.max(0, p2 - p1) + '%';
  }

  function refresh() {
    var rng = current(), recs = filtered(rng[0], rng[1]);
    renderCards(stats(recs.map(function (r) { return r.d; })));
    renderGrades(recs);
    renderHist(recs);
    setHtml('folder-body', statRows(groupStats(recs, 'fo'), false));
    setHtml('pattern-body', statRows(groupStats(recs, 'pa'), true));
    setHtml('detail-body', detailRows(recs));
    set('det-count', String(recs.length));
    syncSlider();
  }

  iMin.min = iMax.min = CFG.vmin;
  iMax.max = iMin.max = CFG.vmax;
  iMin.step = iMax.step = 0.1;
  iMin.value = CFG.vmin;
  iMax.value = CFG.vmax;
  iMin.addEventListener('input', refresh);
  iMax.addEventListener('input', refresh);
  document.getElementById('rreset').addEventListener('click', function () {
    iMin.value = CFG.vmin; iMax.value = CFG.vmax; refresh();
  });
  refresh();
})();
</script>"""


def build_html_report(
    algo: AlgorithmSpec,
    params: dict[str, float],
    csv_path: Path,
    results: list[dict],
    fit: dict,
    overall: dict,
    per_folder: dict,
    per_pattern: dict,
    charts: dict[str, str],
    timings: dict,
    space: str = "original",
) -> str:
    sp = SPACE_FIELDS[space]
    valid = [r for r in results if r[sp["dvalue"]] is not None]
    failed = [r for r in results if r[sp["dvalue"]] is None]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 参数表（算法参数 + value 映射变形参数）----
    param_rows = "".join(
        f"<tr><td>{p.label}</td><td><code>{p.name}</code></td>"
        f"<td>{params[p.name]:g}</td><td>{p.help_text}</td></tr>"
        for p in algo.params
    )
    _sl = fit["slopes"]
    _kn = "、".join(f"({v0:g}, {s:g})" for v0, s in fit.get("map_knots", MAP_KNOTS))
    _bounds = (f"段斜率 σ ∈ [{MAP_SLOPE_FLOOR:g}, {MAP_SLOPE_CEIL:g}]" if MAP_SLOPE_CEIL is not None
               else f"段斜率 σ ≥ {MAP_SLOPE_FLOOR:g}")
    param_rows += (
        f"<tr><td>连续映射曲线 g(value)</td><td><code>MAP_KNOTS / MAP_FIT_RANGE</code></td>"
        f"<td>衔接点 {_kn}；拟合范围 [{MAP_FIT_RANGE[0]:g}, {MAP_FIT_RANGE[1]:g}]</td>"
        f"<td>由标定谱面带边界最小二乘拟合的连续曲线（两端线性渐近、光滑无分段、"
        f"严格单调，{_bounds}）；"
        f"{len(_sl)} 段斜率约 " + " / ".join(f"{x:.3f}" for x in _sl) + "；不再是分段映射</td></tr>"
    )

    lv, lp, ld = sp["label_value"], sp["label_pred"], sp["label_dvalue"]

    # ---- 总览卡片 ----
    cards = f"""
    <div class='cards'>
      <div class='card'><div class='num' id='st-count'>{len(valid)}</div><div class='lab'>成功计算谱面</div></div>
      <div class='card'><div class='num'>{len(failed)}</div><div class='lab'>失败/缺失谱面</div></div>
      <div class='card'><div class='num' id='st-mean'>{overall['mean']:+.4f}</div><div class='lab'>{ld} 均值</div></div>
      <div class='card'><div class='num' id='st-var'>{overall['var']:.4f}</div><div class='lab'>{ld} 方差</div></div>
      <div class='card'><div class='num' id='st-absmean'>{overall['abs_mean']:.4f}</div><div class='lab'>|{ld}| 均值</div></div>
      <div class='card'><div class='num' id='st-rmse'>{overall['rmse']:.4f}</div><div class='lab'>{ld} 均方根误差</div></div>
      <div class='card'><div class='num'>{timings['total_sec']:.1f}s</div><div class='lab'>n 值计算总耗时</div></div>
    </div>"""

    # ---- 区间统计表 ----
    total_grades = sum(overall["grade_counts"].values())
    bound_txt = GRADE_BOUND_TXT
    grade_rows = ""
    for label, txt in zip(GRADE_LABELS, bound_txt):
        c = overall["grade_counts"][label]
        pct = 100.0 * c / total_grades if total_grades else 0.0
        grade_rows += (
            f"<tr><td>{_grade_badge(label)}</td><td>{txt}</td>"
            f"<td>{c}</td><td>{pct:.2f}%</td></tr>"
        )

    # ---- 分 folder 表 ----
    folder_rows = ""
    for folder in sorted(per_folder):
        s = per_folder[folder]
        gc = s["grade_counts"]
        folder_rows += (
            f"<tr><td><b>{folder}</b></td><td>{s['count']}</td>"
            f"<td>{s['mean']:+.4f}</td><td>{s['var']:.4f}</td><td>{s['abs_mean']:.4f}</td>"
            f"<td>{s['rmse']:.4f}</td>"
            f"<td style='color:{GRADE_COLORS[0]}'>{gc['精确']}</td>"
            f"<td style='color:{GRADE_COLORS[1]}'>{gc['接近']}</td>"
            f"<td style='color:{GRADE_COLORS[2]}'>{gc['一般']}</td>"
            f"<td style='color:{GRADE_COLORS[3]}'>{gc['误判']}</td></tr>"
        )

    # ---- 分 pattern 表（按数量降序，便于查看主要类型）----
    pattern_rows = ""
    for pat in sorted(per_pattern, key=lambda p: (-per_pattern[p]["count"], p)):
        s = per_pattern[pat]
        gc = s["grade_counts"]
        pattern_rows += (
            f"<tr><td><b>{pat}</b></td><td>{s['count']}</td>"
            f"<td>{s['mean']:+.4f}</td><td>{s['var']:.4f}</td><td>{s['abs_mean']:.4f}</td>"
            f"<td>{s['rmse']:.4f}</td>"
            f"<td style='color:{GRADE_COLORS[0]}'>{gc['精确']}</td>"
            f"<td style='color:{GRADE_COLORS[1]}'>{gc['接近']}</td>"
            f"<td style='color:{GRADE_COLORS[2]}'>{gc['一般']}</td>"
            f"<td style='color:{GRADE_COLORS[3]}'>{gc['误判']}</td></tr>"
        )

    # ---- 失败表 ----
    failed_rows = "".join(
        f"<tr><td>{r['folder']}</td><td>{r['name']}</td><td>{r['match'] or r['error']}</td></tr>"
        for r in failed
    ) or "<tr><td colspan='3' style='text-align:center;color:#888'>无</td></tr>"

    # ---- 明细表（可折叠）----
    detail_rows = ""
    for r in sorted(valid, key=lambda x: (x["folder"], x["name"])):
        idx = GRADE_LABELS.index(r[sp["grade"]])
        detail_rows += (
            f"<tr><td>{r['folder']}</td><td>{r['pattern']}</td><td>{r['name']}</td>"
            f"<td>{r[sp['value']]:g}</td><td>{r['n']:.6g}</td>"
            f"<td>{r[sp['pred']]:.4f}</td><td>{r[sp['dvalue']]:+.4f}</td>"
            f"<td><span style='color:{GRADE_COLORS[idx]}'>{r[sp['grade']]}</span></td></tr>"
        )

    use_irr = fit.get("use_irr", FIT_USE_IRR)
    use_chord = fit.get("use_chord", FIT_USE_CHORD)
    terms, parts = "", []
    if use_irr:
        terms += f" − {fit['C']:.6f}·irr"
        parts.append(f"{fit['C']:.6f} · irr")
    if use_chord:
        terms += f" − {fit['E']:.6f}·chord"
        parts.append(f"{fit['E']:.6f} · chord")
    terms += f" − {fit['D']:.6f}·ln N"
    parts.append(f"{fit['D']:.6f} · ln N")
    fit_note = sp["note_fit"].format(terms=terms)
    fit_formula = "ln n = g(value) + " + " + ".join(parts)
    feats_desc = []
    if use_irr:
        feats_desc.append("irr = 局部间隔多样性（窗口归一化 KDE 熵，见 irregularity()）")
    if use_chord:
        feats_desc.append("chord = 和弦度（音符数/起音数，见 chord_ratio()）")
    feats_desc.append("N = 物件总数（长度/耐力维度）")
    disabled = []
    if not use_irr:
        disabled.append("irr")
    if not use_chord:
        disabled.append("chord")
    fit_feat_note = "、".join(feats_desc) + "，均只依赖物件时间戳；"
    if disabled:
        fit_feat_note += f"{'、'.join(disabled)} 项已从拟合模型中禁用（系数恒为 0）；"

    # ---- 前端动态数据（滑块过滤 + 实时重算统计）----
    v_all = [r[sp["value"]] for r in valid]
    payload_json = json.dumps({
        "labels": {"v": lv, "p": lp, "d": ld},
        "grades": {"labels": GRADE_LABELS, "bounds": GRADE_BOUND_TXT, "colors": GRADE_COLORS},
        "vmin": math.floor(min(v_all) * 10.0) / 10.0,
        "vmax": math.ceil(max(v_all) * 10.0) / 10.0,
        "rows": [
            {"fo": r["folder"], "pa": r["pattern"], "nm": r["name"],
             "v": r[sp["value"]], "n": r["n"], "pr": r[sp["pred"]], "d": r[sp["dvalue"]]}
            for r in sorted(valid, key=lambda x: (x["folder"], x["name"]))
        ],
    }, ensure_ascii=False, separators=(",", ":"))

    # ---- 映射关系曲线（仅映射口径报告附图）----
    mapcurve_block = ""
    if charts.get("mapcurve"):
        mapcurve_block = (
            "\n  <h2 style=\"margin-top:22px\">value → map(value) 映射曲线（横轴原始 value，0–18）</h2>\n"
            f"  <img src=\"{charts['mapcurve']}\" alt=\"mapcurve\">\n"
            "  <p class=\"note\">蓝实线：完整映射；灰虚线：仅压缩变换 z；黄底：正弦变形带（作用于压缩值 "
            "z ∈ [6−r, 6+r]）；竖点线：分段边界 8 / 12；点线 y=x 供参照。</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>osu!mania 4K n 值算法 benchmark 报告</title>
<style>
  body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; margin: 0; background:#f4f6f8; color:#222; }}
  header {{ background:#1f2d3d; color:#fff; padding:26px 36px; }}
  header h1 {{ margin:0 0 6px; font-size:24px; }}
  header .sub {{ color:#bcc7d3; font-size:13px; }}
  main {{ max-width:1180px; margin:0 auto; padding:24px 28px 60px; }}
  section {{ background:#fff; border-radius:10px; padding:22px 26px; margin:18px 0;
            box-shadow:0 1px 4px rgba(0,0,0,.07); }}
  h2 {{ font-size:18px; border-left:5px solid #3498db; padding-left:10px; margin:0 0 16px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ border:1px solid #e0e4e8; padding:7px 10px; text-align:center; }}
  th {{ background:#f0f3f6; }}
  td:nth-child(2) {{ text-align:left; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .card {{ flex:1 1 150px; background:#f8fafc; border:1px solid #e3e8ee; border-radius:8px;
          padding:14px; text-align:center; }}
  .card .num {{ font-size:22px; font-weight:700; color:#1f2d3d; }}
  .card .lab {{ font-size:12px; color:#778592; margin-top:4px; }}
  .badge {{ color:#fff; padding:2px 10px; border-radius:10px; font-size:12px; }}
  img {{ max-width:100%; display:block; margin:10px auto; }}
  .two-col {{ display:flex; gap:20px; flex-wrap:wrap; align-items:center; }}
  .two-col > div {{ flex:1 1 380px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }}
  table.sortable th.sortable-th {{ cursor:pointer; user-select:none; white-space:nowrap; }}
  table.sortable th.sortable-th:hover {{ background:#dde4ea; }}
  .sort-arr {{ color:#34678f; font-size:10px; margin-left:2px; }}
  details {{ margin-top:8px; }}
  summary {{ cursor:pointer; font-weight:600; color:#346; }}
  .scroll {{ max-height:520px; overflow:auto; margin-top:10px; }}
  .note {{ font-size:12px; color:#778592; }}
  .range-box {{ background:#f8fafc; border:1px solid #e3e8ee; border-radius:8px; padding:12px 16px; margin-bottom:16px; }}
  .range-head {{ font-size:13px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
  .range-val {{ font-weight:700; color:#1f2d3d; }}
  #rreset {{ border:1px solid #c8d2dc; background:#fff; border-radius:5px; padding:2px 12px; cursor:pointer; font-size:12px; }}
  #rreset:hover {{ background:#eef2f6; }}
  .dual {{ position:relative; height:32px; margin:4px 2px 0; }}
  .dual .track {{ position:absolute; left:8px; right:8px; top:13px; height:6px; background:#dde4ea; border-radius:3px; }}
  .dual .fill {{ position:absolute; top:0; height:100%; background:#3498db; border-radius:3px; }}
  .dual input[type=range] {{ -webkit-appearance:none; appearance:none; position:absolute; left:0; top:0; width:100%; height:32px; margin:0; background:transparent; pointer-events:none; }}
  .dual input[type=range]::-webkit-slider-runnable-track {{ height:32px; background:transparent; }}
  .dual input[type=range]::-webkit-slider-thumb {{ -webkit-appearance:none; pointer-events:auto; width:14px; height:28px; margin-top:2px; border-radius:4px; background:#3498db; border:1px solid #2a7cb0; cursor:ew-resize; }}
  .dual input[type=range]::-moz-range-track {{ height:32px; background:transparent; }}
  .dual input[type=range]::-moz-range-thumb {{ pointer-events:auto; width:12px; height:28px; border-radius:4px; background:#3498db; border:1px solid #2a7cb0; cursor:ew-resize; }}
  .pie {{ width:230px; height:230px; border-radius:50%; margin:8px auto; border:1px solid #e0e4e8; }}
  #dhist {{ max-width:100%; display:block; margin:10px auto; }}
</style>
</head>
<body>
<header>
  <h1>osu!mania 4K 难度 n 值算法 · Benchmark 测试报告（{sp['subtitle']}）</h1>
  <div class="sub">
    算法：{algo.title}（<code>{algo.key}</code>）　|　数据：{csv_path.name}（{len(results)} 行）
    　|　生成时间：{now}
  </div>
</header>
<main>

<section>
  <h2>1. 参数设置与拟合结果</h2>
  <p class="note">{algo.description}</p>
  <table>
    <tr><th>参数</th><th>键名</th><th>当前值</th><th>说明</th></tr>
    {param_rows}
  </table>
  <p style="margin-top:14px">
    value 映射为<b>连续单调曲线</b>（由标定谱面带边界最小二乘拟合，两端线性渐近、全程 C∞、无分段）：<br>
    g(v) = c + Σ<sub>j</sub> σ<sub>j</sub>·ψ<sub>j</sub>(v)，ψ 由 softplus 转折基
    φ<sub>i</sub>(v) = s<sub>i</sub>·softplus((v − v<sub>0,i</sub>)/s<sub>i</sub>) 构成，
    softplus(z)=ln(1+e<sup>z</sup>)；衔接点 {_kn}，拟合范围 [{MAP_FIT_RANGE[0]:g}, {MAP_FIT_RANGE[1]:g}]（低 value 段含四个转折点）；
    {_bounds} 保证严格单调，并抑制过平段放大反解方差。<br>
    拟合（标定集：非 {FIT_EXCLUDE_FOLDERS} 且 value ≥ {FIT_MIN_VALUE:g} 的谱面，按难度分位分层抽样 {fit.get('n_calib', 0)} 张）：
    <b>{fit_formula}</b>，
    对数空间 R² = <b>{fit['r2_log']:.6f}</b>；
    {fit_feat_note}
    {fit_note}
  </p>
  <p class="note">n 值计算总耗时 {timings['total_sec']:.2f}s（含读文件），
     单谱平均 {timings['avg_ms']:.1f} ms，最慢 {timings['max_ms']:.1f} ms。</p>
  {mapcurve_block}
</section>

<section>
  <h2>2. 总体指标</h2>
  <div class="range-box">
    <div class="range-head">
      <b>统计范围</b>
      <span class="note">拖动两端滑块按 <b>{lv}</b> 过滤：第 2、3、4、6、7、9 节实时重算；
        第 5 节及分组点阵图为全量静态图，不随滑块变化。</span>
      <span id="rlab" class="range-val"></span>
      <button id="rreset" type="button">重置</button>
    </div>
    <div class="dual">
      <div class="track"><div class="fill" id="rfill"></div></div>
      <input type="range" id="rmin" aria-label="最小 {lv}">
      <input type="range" id="rmax" aria-label="最大 {lv}">
    </div>
  </div>
  {cards}
  <p class="note" style="margin-top:12px">
    {ld} = {lp} − {lv}；方差为总体方差（ddof=0）。
    区间：|{ld}|≤0.2 精确，0.2&lt;|{ld}|≤0.5 接近，0.5&lt;|{ld}|≤1 一般，|{ld}|&gt;1 误判。
  </p>
</section>

<section>
  <h2>3. 区间统计</h2>
  <div class="two-col">
    <div>
      <table>
        <tr><th>类别</th><th>区间</th><th>数量</th><th>占比</th></tr>
        <tbody id="grade-body">
        {grade_rows}
        </tbody>
      </table>
    </div>
    <div>
      <div id="grade-pie" class="pie"></div>
      <p class="note" style="text-align:center">{ld} 区间占比（随滑块实时更新）</p>
    </div>
  </div>
</section>

<section>
  <h2>4. {ld} 分布（分度 0.1）</h2>
  <canvas id="dhist" width="1080" height="300"></canvas>
</section>

<section>
  <h2>5. {lp} – {lv} 散点 / 线性拟合 / y=x 参照</h2>
  <div class="two-col">
    <div><img src="{charts['scatter']}" alt="scatter"></div>
    <div><img src="{charts['expfit']}" alt="expfit"></div>
  </div>
  <p class="note">左：全部谱面 {lv} 与拟合 {lp} 的点阵，红色实线为 {lp} 对 {lv} 的线性拟合趋势，
     灰色虚线为 y=x；右：course 谱面 n 与 map(value) 的指数拟合（纵轴对数）。</p>
</section>

<section>
  <h2>6. 分 folder 详细统计</h2>
  <table class="sortable">
    <tr><th>folder</th><th>数量</th><th>{ld} 均值</th><th>{ld} 方差</th><th>|{ld}| 均值</th><th>{ld} 均方根误差</th>
        <th>精确</th><th>接近</th><th>一般</th><th>误判</th></tr>
    <tbody id="folder-body">
    {folder_rows}
    </tbody>
  </table>
  <h2 style="margin-top:24px">分 folder 点阵图（含线性拟合与 y=x 虚线）</h2>
  <img src="{charts['folders']}" alt="folders">
</section>

<section>
  <h2>7. 分 pattern 详细统计</h2>
  <table class="sortable">
    <tr><th>pattern</th><th>数量</th><th>{ld} 均值</th><th>{ld} 方差</th><th>|{ld}| 均值</th><th>{ld} 均方根误差</th>
        <th>精确</th><th>接近</th><th>一般</th><th>误判</th></tr>
    <tbody id="pattern-body">
    {pattern_rows}
    </tbody>
  </table>
  <p class="note" style="margin-top:8px">pattern 取自 CSV 的 pattern 列，按数量降序排列。</p>
  <h2 style="margin-top:24px">分 pattern 点阵图（含线性拟合与 y=x 虚线，点色按 folder 区分）</h2>
  <img src="{charts['patterns']}" alt="patterns">
</section>

<section>
  <h2>8. 失败 / 缺失谱面（{len(failed)}）</h2>
  <table>
    <tr><th>folder</th><th>name</th><th>原因</th></tr>
    {failed_rows}
  </table>
</section>

<section>
  <details>
    <summary>9. 全谱面明细（<span id="det-count">{len(valid)}</span> 条，点击展开；点击表头排序）</summary>
    <div class="scroll">
    <table class="sortable">
      <tr><th>folder</th><th>pattern</th><th>name</th><th>{lv}</th><th>n</th><th>{lp}</th><th>{ld}</th><th>类别</th></tr>
      <tbody id="detail-body">
      {detail_rows}
      </tbody>
    </table>
    </div>
  </details>
</section>

</main>
<script id="report-data" type="application/json">{payload_json}</script>
{DYNAMIC_REPORT_SCRIPT}
{SORTABLE_SCRIPT}
</body>
</html>"""
    return html


# ===========================================================================
# 主流程
# ===========================================================================
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="osu!mania 4K n 值算法 benchmark")
    parser.add_argument("--algo", default="object_sequence",
                        help=f"算法 key（默认 object_sequence，可选：{', '.join(_REGISTRY)}）")
    parser.add_argument("--list-algos", action="store_true", help="列出所有已注册算法后退出")
    parser.add_argument("--half-life", type=float, help="object_sequence: 衰减半衰期 H(ms)，默认 1000")
    parser.add_argument("--quantile", type=float, help="object_sequence: n 值分位数/目标均值，默认 0.96")
    parser.add_argument("--extra-rate", type=float,
                        help="object_sequence: 配对轨邻域额外 contribution 倍率，默认 0.1（0 关闭）")
    parser.add_argument("--extra-b", type=float,
                        help="object_sequence: 邻域耦合项偏移 b，默认 2")
    parser.add_argument("--extra-log-base", type=float,
                        help="object_sequence: 邻域耦合项对数底数，默认 2")
    parser.add_argument("--merge-power", type=float,
                        help="object_sequence: 双轨合并幂次 i，(x^i+y^i)^(1/i)，默认 2（hypot）")
    parser.add_argument("--merge2-power", type=float,
                        help="object_sequence: 两合并序列二次合并幂次 i2，默认 2")
    parser.add_argument("--contrib-power", type=float,
                        help="object_sequence: 初始 contribution 合并范数幂次 p，(x^p+y^p)^(1/p)，默认 2")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="NAME=VALUE", help="通用参数覆盖，可多次使用（适配任意已注册算法）")
    parser.add_argument("--params-json", default=None,
                        help="从 JSON 文件的 params 字段读取算法参数覆盖"
                             "（如 fixed_density/params_final.json，用于复现最终模型）")
    parser.add_argument("--no-irr", action="store_true",
                        help="拟合模型去掉 irr 项（即 C=0），改用逐键局部熵层替代")
    parser.add_argument("--no-chord", action="store_true",
                        help="拟合模型去掉 chord（和弦度）项（即 E=0）")
    parser.add_argument("--dur", action="store_true",
                        help="拟合模型加入 ln_dur（谱面时长）项——“长谱/耐力”维度实验开关")
    parser.add_argument("--csv", dest="csv_path", default=str(DEFAULT_CSV_PATH), help="benchmark CSV 路径")
    parser.add_argument("--exclude-folders", default="ln",
                        help="统计口径中排除的 folder（逗号分隔，默认 ln）：ln 目录是纯 LN 谱，"
                             "其难度由 ln_value 承载（见 LN 报告），不应计入 RC 精度；"
                             "传空字符串表示不排除")
    parser.add_argument("--source", choices=("extracted", "osu"), default="extracted",
                        help="谱面数据来源：extracted=读取 benchmark_extracted 预处理数据（默认），osu=解析 .osu 文件")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="原始 value 口径 HTML 报告输出路径")
    parser.add_argument("--output-mapped", default=str(ROOT_DIR / "report" / "benchmark_report_mapped.html"),
                        help="映射 value 口径 HTML 报告输出路径")
    return parser.parse_args(argv)


def resolve_params(algo: AlgorithmSpec, args: argparse.Namespace) -> dict[str, float]:
    params = algo.default_params()
    if args.half_life is not None:
        params["half_life_ms"] = args.half_life
    if args.quantile is not None:
        params["quantile"] = args.quantile
    if args.extra_rate is not None:
        params["extra_rate"] = args.extra_rate
    if args.extra_b is not None:
        params["extra_b"] = args.extra_b
    if args.extra_log_base is not None:
        params["extra_log_base"] = args.extra_log_base
    if args.merge_power is not None:
        params["merge_power"] = args.merge_power
    if args.merge2_power is not None:
        params["merge2_power"] = args.merge2_power
    if args.contrib_power is not None:
        params["contrib_power"] = args.contrib_power
    valid_names = {p.name for p in algo.params}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set 参数格式应为 NAME=VALUE：{item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in valid_names:
            raise SystemExit(f"算法 {algo.key} 没有参数 {key}；可用参数：{', '.join(sorted(valid_names))}")
        params[key] = float(raw)
    if args.params_json:
        data = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        for key, raw in data.get("params", {}).items():
            if key not in valid_names:
                raise SystemExit(
                    f"参数文件 {args.params_json} 含未知参数 {key}；可用参数："
                    f"{', '.join(sorted(valid_names))}")
            params[key] = float(raw)
    return params


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.list_algos:
        for key, spec in _REGISTRY.items():
            print(f"{key}: {spec.title}")
            for p in spec.params:
                print(f"    {p.name} = {p.default}  # {p.label}")
        return

    if args.algo not in _REGISTRY:
        raise SystemExit(f"未知算法 {args.algo}；已注册：{', '.join(_REGISTRY)}")
    algo = _REGISTRY[args.algo]
    params = resolve_params(algo, args)
    # 拟合模型的特征开关（--no-irr 即 C=0；--no-chord 即 E=0；--dur 即加入 D2·ln dur）
    global FIT_USE_IRR, FIT_USE_CHORD, FIT_USE_DUR
    if args.no_irr:
        FIT_USE_IRR = False
    if args.no_chord:
        FIT_USE_CHORD = False
    if args.dur:
        FIT_USE_DUR = True
    csv_path = Path(args.csv_path)
    output_path = Path(args.output)
    output_mapped_path = Path(args.output_mapped)

    _setup_chinese_font()

    print(f"算法：{algo.key} | 参数：{params} | 数据源：{args.source}")
    rows = load_csv(csv_path)
    print(f"读取 CSV：{csv_path}（{len(rows)} 行）")

    t_start = time.perf_counter()
    results = run_benchmark(algo, params, rows, source=args.source)
    total_sec = time.perf_counter() - t_start
    exclude_folders = {s.strip() for s in str(args.exclude_folders).split(",") if s.strip()}
    if exclude_folders:
        kept = [r for r in results if r["folder"] not in exclude_folders]
        print(f"统计口径：排除 folder {sorted(exclude_folders)}"
              f"（{len(results) - len(kept)} 张），保留 {len(kept)} 张")
        results = kept
    elapsed = [r["elapsed_ms"] for r in results if r["elapsed_ms"] is not None]
    timings = {
        "total_sec": total_sec,
        "avg_ms": float(np.mean(elapsed)) if elapsed else 0.0,
        "max_ms": float(np.max(elapsed)) if elapsed else 0.0,
    }

    calib_records = select_calibration(results)
    if len(calib_records) < 3:
        raise SystemExit(f"可用于标定拟合的谱面不足 3 个（需 value >= {FIT_MIN_VALUE} 的非 "
                         f"{FIT_EXCLUDE_FOLDERS} 谱面）")
    fit = fit_exponential(calib_records)
    fit["n_calib"] = len(calib_records)
    _sl = fit["slopes"]
    print(f"连续映射曲线：{len(_sl)} 段，斜率 " + " / ".join(f"{x:.4f}" for x in _sl) +
          f"，衔接点 {fit.get('map_knots')}，拟合范围 {MAP_FIT_RANGE}，"
          f"C={fit.get('C', 0.0):.6f}, E={fit.get('E', 0.0):.6f}, "
          f"D={fit.get('D', 0.0):.6f}, R2(log)={fit['r2_log']:.6f}")

    # 同时填充原始口径与映射口径的预测/差值/类别
    apply_fit_and_stats(results, fit)

    # 两种口径各自的统计与图表（共享同一份 results 与 fit，保证口径可追溯）
    def build_for_space(space: str) -> tuple[dict, dict, dict, dict[str, str]]:
        sp = SPACE_FIELDS[space]
        valid = [r for r in results if r[sp["dvalue"]] is not None]
        overall = summarize([r[sp["dvalue"]] for r in valid])
        per_folder = {
            folder: summarize([r[sp["dvalue"]] for r in valid if r["folder"] == folder])
            for folder in sorted({r["folder"] for r in valid})
        }
        per_pattern = {
            pat: summarize([r[sp["dvalue"]] for r in valid if r["pattern"] == pat])
            for pat in sorted({r["pattern"] for r in valid})
        }
        print(f"  [{sp['subtitle']}] 总体：",
              {k: overall[k] for k in ("count", "mean", "var", "abs_mean")}, overall["grade_counts"])
        charts = {
            "scatter": plot_overall_scatter(valid, sp),
            "folders": plot_group_scatters(valid, per_folder, sp, "folder"),
            "patterns": plot_group_scatters(valid, per_pattern, sp, "pattern"),
            "expfit": plot_exponential_fit(calib_records, fit),
        }
        if space == "mapped":  # 映射口径报告附映射关系曲线
            charts["mapcurve"] = plot_mapping_curve(fit)
        return overall, per_folder, per_pattern, charts

    print("生成图表（原始口径 + 映射口径）...")
    overall, per_folder, per_pattern, charts = build_for_space("original")
    overall_m, per_folder_m, per_pattern_m, charts_m = build_for_space("mapped")

    html = build_html_report(algo, params, csv_path, results, fit,
                             overall, per_folder, per_pattern, charts, timings,
                             space="original")
    output_path.write_text(html, encoding="utf-8")
    print(f"原始 value 口径报告已生成：{output_path}")

    html_m = build_html_report(algo, params, csv_path, results, fit,
                               overall_m, per_folder_m, per_pattern_m, charts_m, timings,
                               space="mapped")
    output_mapped_path.write_text(html_m, encoding="utf-8")
    print(f"映射 value 口径报告已生成：{output_mapped_path}")


if __name__ == "__main__":
    main()
