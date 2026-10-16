"""参数轮询（黑盒优化）脚本
=============================

在给定范围内搜索使“均方根误差（RMSE）”最小的算法参数组合。

优化目标
--------
对齐 fixed_density.benchmark 的完整流程：用候选参数计算全部谱面 n → 在「非 course 且
value >= FIT_MIN_VALUE」的谱面中按 (folder × 难度分位) 分层抽样作标定集做指数拟合
→ 反算 value′ → 统计 dvalue →
取 RMSE 作为目标（口径由 --space 指定，默认原始 value 口径）。
可选 --filter-min/--filter-max 把统计范围限制在某个 value 区间（与报告滑块一致）。

轮询参数与范围
--------------
  half_life_ms    100 - 2000
  quantile        0.80 - 0.99
  extra_rate      0.1 - 3
  extra_b         ln(a) - 2*ln(a)（a = extra_log_base）→ 以乘子 t∈[1,2] 间接搜索
  extra_log_base  1.5 - 5
  merge_power     log10 0.5 - 3（即原始幂次 10^0.5 ~ 10^3）
  merge2_power    log10 0.5 - 3
  contrib_power   log10 0.5 - 3

降低调用次数的做法
------------------
单次目标评估要跑完整 643 个谱面（约 12 s），因此调用次数是主要成本，脚本：
  1. 在单位超立方体上用 Sobol' 低差异序列做初始设计（空间填充优于纯随机），
     该批点用多进程并行评估，摊薄固定开销；
  2. 之后用高斯过程（sklearn, Matern-ARD + 白噪声）拟合目标曲面，
     以 Expected Improvement 采集函数提议下一个点——确定性平滑目标下
     8 维通常 60 次以内收敛，明显优于网格/随机/坐标下降；
  3. 每次评估结果写入 JSONL 日志并进入内存缓存，重复点零成本跳过；
     中断后 --resume 可直接续跑；
  4. 默认禁用“无改善提前停止”（--patience 0），跑满 --max-evals / --time-budget-sec
     预算以优先保证结果准确；如需早停可显式设置 --patience N。

按参数定位的分块搜索
--------------------
依据各参数在 algo_object_sequence 中的作用分为三层，逐层搜索（--stage auto，默认）：
  1. 结构层 structural：half_life_ms（全局衰减尺度）与三个 Lp 范数幂次
     contrib_power / merge_power / merge2_power（4 维），额外层倍率压到最低；
  2. 耦合成层 coupling：额外 contribution 三参数 extra_rate / extra_b_t /
     extra_log_base（3 维），结构层固定为上阶段最优；
  3. 联合精修 joint：8 维、从前面阶段最优热启动，捕捉跨层交互（如 H×extra_rate）。
每阶段自动把最优热启动给下一阶段，最终取各阶段中 RMSE 最小者。低维子问题
收敛更快、更省调用；也可用 --stage structural|coupling|joint 只跑某层，
用 --warm-start 指定起点。

为避免“分块弱于直接 8 维搜索”，默认再套多轮循环（--rounds，默认 2）并逐轮
收缩信赖域（--radius-decay，默认 0.5）：第 1 轮半径 1（全空间）做全局粗搜，
之后每轮都在以“当前最优”为中心、半径递减的盒内精搜。各块之间反复迭代、
由粗到细，逐步逼近跨层交互最优；预算倾向联合精修（structural/coupling/joint
≈ 3:2:5）。--rounds 1 即退化为单轮分块搜索。

用法
----
  python -m fixed_density.tune --dry-run                 # 只打印配置与预算，不评估
  python -m fixed_density.tune                           # 默认：16 初始点 + 最多 200 次评估
  python -m fixed_density.tune --resume                  # 续跑上次日志
  python -m fixed_density.tune --space mapped --max-evals 120
  python -m fixed_density.tune --workers 1               # 串行（便于调试）

说明：本脚本单次运行可能持续数十分钟到数小时，建议先 --dry-run 确认预算，
并用 --time-budget-sec 设置上限。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from . import benchmark as _bt
from .benchmark import (
    DEFAULT_CSV_PATH,
    FIT_USE_CHORD,
    FIT_USE_IRR,
    SPACE_FIELDS,
    _REGISTRY,
    apply_fit_and_stats,
    fit_exponential,
    load_csv,
    run_benchmark,
    select_calibration,
    summarize,
)

INF = float("inf")

# 轮询参数空间：(名称, 下界, 上界, 缩放)；缩放 linear=线性、log10=以 10 为底的对数。
# log10 项的单位立方体坐标 u 映射为 10^(lo+(hi-lo)u)，即“对数调节”。
# 前 6 个为算法主体参数；后 3 个为逐键局部熵增益（γ_lane 单轨@贡献层、
# γ_pair 轨道对@配对合并层、γ_global 全谱@最终合并层），配合 --no-irr(C=0) 使用。
# 末 6 个为 LN 通道参数（ln_gain=0 时整层关闭，其余 5 维退化为无效维度）。
# 三个范数幂次与 cross_lane_mix(alpha) 见 PINNED，固定不搜。
PARAM_SPACES: tuple[tuple[str, float, float, str], ...] = (
    ("half_life_ms", 100.0, 2000.0, "linear"),
    ("quantile", 0.80, 0.99, "linear"),
    ("extra_rate", 0.1, 3.0, "linear"),
    ("extra_b_t", 1.0, 2.0, "linear"),   # 乘子 t：extra_b = t * ln(extra_log_base)，满足 ln(a)~2ln(a)
    ("extra_log_base", 1.5, 5.0, "linear"),
    ("soft_beta", 0.0, 3.0, "linear"),   # 合并算子 softmax 系数（0=使用 Lp 范数）
    ("irr_gain_lane", -0.4, 0.4, "linear"),
    ("irr_gain_pair", -0.4, 0.4, "linear"),
    ("irr_gain_global", -0.4, 0.4, "linear"),
    ("ln_gain", 0.0, 2.0, "linear"),         # κ：LN 通道整体缩放（0=关闭 LN 通道）
    ("ln_merge_power", 1.0, 6.0, "linear"),  # p：n_rc 与 κ·n_ln 的 Lp 合并幂次
    ("ln_rel_power", 1.0, 6.0, "linear"),    # q：相邻松手间距倒数的 Lp 合并幂次
    ("ln_jack_gain", 0.0, 8.0, "linear"),    # 松手→同轨下一次按下（控手）
    ("ln_hold_gain", -0.4, 0.6, "linear"),   # 同手按住占用乘子 exp(γ·occ)
    ("ln_cross_gain", 0.0, 64.0, "linear"),  # 按住期间同手另一轨的超额动作比（无量纲，均匀谱≈1）
)

# 不参与搜索、固定注入算法参数的项（沿用 best_params_len 的量级；alpha 恒为 0）
PINNED: dict[str, float] = {
    "merge_power": 1000.0,
    "merge2_power": 100.0,
    "contrib_power": 300.0,
    "cross_lane_mix": 0.0,
    "irr_local_gain": 0.0,
}
DIM = len(PARAM_SPACES)


# 进程内上下文（主进程与 worker 各自构建）
_CTX: dict = {}
# 内存缓存：舍入后的单位立方体坐标 -> RMSE（INF 表示该参数不可用）
_CACHE: dict = {}


# ===========================================================================
# 参数编解码
# ===========================================================================
def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def unit_to_named(x) -> dict[str, float]:
    """单位立方体坐标 -> 带物理量纲的参数（log10 项在返回前还原为原始数值）。"""
    out: dict[str, float] = {}
    for (name, lo, hi, scale), xi in zip(PARAM_SPACES, x):
        v = lo + (hi - lo) * clamp01(float(xi))
        out[name] = (10.0 ** v) if scale == "log10" else v
    return out


def named_to_algo(named: dict[str, float]) -> dict[str, float]:
    """物理参数 -> Fixed Density 算法参数字典（extra_b 由乘子与底数导出；PINNED 项固定注入）。"""
    base = math.log(named["extra_log_base"])
    out = {
        "half_life_ms": named["half_life_ms"],
        "quantile": named["quantile"],
        "extra_rate": named["extra_rate"],
        "extra_b": named["extra_b_t"] * base,   # ∈ [ln a, 2 ln a]
        "extra_log_base": named["extra_log_base"],
        "soft_beta": named.get("soft_beta", 0.0),
        "irr_gain_lane": named.get("irr_gain_lane", 0.0),
        "irr_gain_pair": named.get("irr_gain_pair", 0.0),
        "irr_gain_global": named.get("irr_gain_global", 0.0),
        "ln_gain": named.get("ln_gain", 0.0),
        "ln_merge_power": named.get("ln_merge_power", 2.0),
        "ln_rel_power": named.get("ln_rel_power", 2.0),
        "ln_jack_gain": named.get("ln_jack_gain", 1.0),
        "ln_hold_gain": named.get("ln_hold_gain", 0.0),
        "ln_cross_gain": named.get("ln_cross_gain", 0.0),
    }
    out.update(PINNED)
    return out


def _key(x) -> tuple:
    return tuple(round(clamp01(float(v)), 6) for v in x)


# ===========================================================================
# 按参数在算法中的定位分组（针对性优化）
#
# 依据 algo_object_sequence 的结构，6 个可搜参数分属两层：
#   A 结构层（决定赋值序列形状，强交互）：half_life_ms（全局时间衰减尺度）、soft_beta（合并算子）；
#   B 耦合成层（额外 contribution）：extra_rate（=0 时整层关闭，extra_b / extra_log_base
#     退化为无效维度）、extra_b_t、extra_log_base；
#   C 归一化层：quantile —— 仅出现在末端 find_n_quantile，其共性缩放会被
#     指数拟合的截距 A 吸收，杠杆最低。
#   D LN 层（ln_gain=0 时整层关闭）：ln_gain(κ 整体缩放)、ln_jack_gain / ln_cross_gain
#     （控手）、ln_hold_gain（占用）、ln_merge_power / ln_rel_power（两个幂次）。
#     注意 κ 与 ln_jack_gain / ln_cross_gain 存在乘积近似退化方向（κ·gain 才是有效量），
#     低预算时可将两个幂次与 ln_hold_gain 钉住，只搜 κ 与两个 gain。
# 范数幂次与 cross_lane_mix(alpha) 已钉住（PINNED），不再搜索。
# 6 维一次联合搜索即可，无需分块（避免“阶段固定坐标使热启动种子降级”的问题）。
# ===========================================================================
NAMES: list[str] = [name for name, *_ in PARAM_SPACES]
NAME_INDEX: dict[str, int] = {name: i for i, name in enumerate(NAMES)}

STAGES: dict[str, dict] = {
    "joint": {"free": tuple(NAMES), "pin": {}},
}
STAGE_ORDER = ("joint",)
STAGE_FRACTION = {"joint": 1.0}


def default_named() -> dict[str, float]:
    """无热启动时的初始物理参数（取范围内较温和的默认值）。"""
    return {
        "half_life_ms": 1000.0, "quantile": 0.96, "extra_rate": 0.1,
        "extra_b_t": 1.5, "extra_log_base": 2.0,
        "soft_beta": 0.0,
        "irr_gain_lane": 0.0, "irr_gain_pair": 0.0, "irr_gain_global": 0.0,
        "ln_gain": 0.0, "ln_merge_power": 2.0, "ln_rel_power": 2.0,
        "ln_jack_gain": 1.0, "ln_hold_gain": 0.0, "ln_cross_gain": 0.0,
    }


def named_to_unit(named: dict[str, float]) -> list[float]:
    """物理参数 -> 单位立方体坐标（unit_to_named 的逆；log10 项先取对数）。"""
    out: list[float] = []
    for name, lo, hi, scale in PARAM_SPACES:
        z = math.log10(float(named[name])) if scale == "log10" else float(named[name])
        out.append(clamp01((z - lo) / (hi - lo)))
    return out


def load_warm_start(path) -> dict[str, float] | None:
    """从 best_params.json 读取热启动物理参数（缺失字段用默认值补齐）。"""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        named = data.get("named_params") or {}
        merged = default_named()
        for k in NAMES:
            if k in named:
                merged[k] = float(named[k])
        return merged
    except Exception:
        return None



# ===========================================================================
# 目标函数
# ===========================================================================
def _setup_ctx(csv_path: str, source: str, space: str, vmin: float, vmax: float,
               exclude_folders=("course",), use_irr: bool = True) -> None:
    """构建进程内上下文（每个进程加载一次 CSV 行）。

    exclude_folders 支持字符串（可按逗号分隔多个 folder）或可迭代的 folder 名序列；
    注意不可直接 tuple(<字符串>)，否则会被拆成单个字符。
    use_irr=False 时关闭拟合模型的 irr 项（C=0），由逐键局部熵层替代。
    """
    if isinstance(exclude_folders, str):
        exclude_folders = tuple(s.strip() for s in exclude_folders.split(",") if s.strip())
    _bt.FIT_USE_IRR = bool(use_irr)
    _CTX.clear()
    _CTX.update({
        "rows": load_csv(Path(csv_path)),
        "source": source,
        "space": space,
        "vmin": vmin,
        "vmax": vmax,
        "exclude_folders": tuple(exclude_folders),
        "use_irr": bool(use_irr),
    })


def eval_detailed(x) -> tuple[float, dict]:
    """评估一组参数，返回 (RMSE, 详情)。RMSE 为 INF 表示该参数不可用。"""
    algo_params = named_to_algo(unit_to_named(x))
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):  # 屏蔽 run_benchmark 的逐行跳过/失败输出
            results = run_benchmark(
                _REGISTRY["object_sequence"], algo_params, _CTX["rows"],
                source=_CTX["source"],
            )
            calib = select_calibration(results)
            if len(calib) < 3:
                return INF, {"error": "可用于标定拟合的谱面不足 3 个"}
            fit = fit_exponential(calib)
            apply_fit_and_stats(results, fit)
        rmse_by_space = {}
        for sp_key, sp in SPACE_FIELDS.items():
            field = sp["dvalue"]
            vals = [r[field] for r in results
                    if r[field] is not None and _CTX["vmin"] <= r["value"] <= _CTX["vmax"]
                    and r["folder"] not in _CTX["exclude_folders"]]
            if len(vals) >= 2:
                rmse_by_space[sp_key] = summarize(np.asarray(vals))
        if _CTX["space"] not in rmse_by_space:
            return INF, {"error": "统计样本不足"}
        detail = {
            "rmse_by_space": {k: v["rmse"] for k, v in rmse_by_space.items()},
            "grades_by_space": {k: v["grade_counts"] for k, v in rmse_by_space.items()},
            "fit": fit,
            "algo_params": algo_params,
            "named": unit_to_named(x),
            "count": rmse_by_space[_CTX["space"]]["count"],
        }
        return float(rmse_by_space[_CTX["space"]]["rmse"]), detail
    except Exception as exc:  # 任意异常都视为不可用，避免优化被单个坏点打断
        return INF, {"error": f"{type(exc).__name__}: {exc}"}


def eval_unit(x) -> float:
    """缓存感知的目标评估（不含 I/O，可在 worker 中调用）。"""
    key = _key(x)
    if key in _CACHE:
        return _CACHE[key]
    rmse, _ = eval_detailed(x)
    _CACHE[key] = rmse
    return rmse


def _worker_eval(x):
    """多进程 worker 入口：返回 (key, rmse)。"""
    return list(_key(x)), eval_unit(x)


def _init_worker(csv_path, source, space, vmin, vmax, exclude_folders, use_irr=True):
    _setup_ctx(csv_path, source, space, vmin, vmax, exclude_folders, use_irr)


# ===========================================================================
# 高斯过程 + EI 采集
# ===========================================================================
def _penalize(y: np.ndarray) -> np.ndarray:
    """把不可用点（INF）替换为大惩罚值，供 GP 拟合。"""
    finite = y[np.isfinite(y)]
    pen = float(finite.max()) * 1.5 + 1.0 if finite.size else 1e6
    return np.where(np.isfinite(y), y, pen)


def _fit_gp(x: np.ndarray, y: np.ndarray, seed: int) -> GaussianProcessRegressor:
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(x.shape[1], 0.3),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-8, (1e-12, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                 n_restarts_optimizer=2, random_state=seed)
    gp.fit(x, _penalize(y))
    return gp


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    """最小化问题的 EI（返回越大越值得试探）。"""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-12)
    z = (best - mu - xi) / sigma
    return (best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def _propose(gp, x_known: np.ndarray, best: float, xi: float, seed: int,
             n_cand: int = 20000, n_refine: int = 5) -> list[np.ndarray]:
    """用随机候选池 + 局部精修最大化 EI，返回按 EI 降序的候选点列表。

    维度取自 x_known，故可复用于任意活跃子空间（分块搜索）。
    """
    x_known = np.atleast_2d(np.asarray(x_known, dtype=np.float64))
    dim = int(x_known.shape[1])
    rng = np.random.default_rng(seed)
    pool = rng.random((n_cand, dim))
    mu, sigma = gp.predict(pool, return_std=True)
    ei_all = _expected_improvement(mu, sigma, best, xi)
    order = np.argsort(-ei_all)[:n_refine]

    refined: list[tuple[float, np.ndarray]] = []
    bounds = [(0.0, 1.0)] * dim

    def neg_ei(z):
        m, s = gp.predict(np.asarray(z).reshape(1, -1), return_std=True)
        return -float(_expected_improvement(m, s, best, xi)[0])

    known = {_key(r) for r in x_known}
    for i in order:
        start = pool[i]
        try:
            res = minimize(neg_ei, start, method="L-BFGS-B", bounds=bounds)
            cand = np.clip(res.x, 0.0, 1.0)
            val = -float(res.fun)
        except Exception:
            cand, val = start, float(ei_all[i])
        if _key(cand) not in known:
            refined.append((val, cand))
    refined.sort(key=lambda t: -t[0])
    return [c for _, c in refined]


# ===========================================================================
# 优化主循环
# ===========================================================================
def _load_log(log_path: Path) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    if not log_path.is_file():
        return xs, ys
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                xs.append(list(rec["x"]))
                ys.append(float(rec["rmse"]))
            except Exception:
                continue
    return xs, ys


def _append_log(log_path: Path, key: tuple, rmse: float) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"x": list(key), "rmse": rmse}) + "\n")


def _fmt_sec(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        return "?"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def optimize(args, free_names=None, base_named=None, log_path=None, best_out=None,
             max_evals=None, fresh_log=True, seed_history=False, radius=1.0) -> dict:
    """在给定“活跃子空间”上做 Sobol' + GP-EI 搜索，返回结果摘要。

    分块搜索要点：只在 free_names 指定的维度上搜索，其余维度固定为
    base_named（热启动）与阶段 pin 覆盖值。GP/提议始终在低维子空间进行，
    完整 8 维向量仅用于缓存键与最终评估。

    radius 为信赖域半径（单位立方体坐标，1 = 全空间）：搜索被限制在以
    base_named 为中心的盒内，多轮调用时逐轮收缩，实现由粗到细的精细搜索。
    """
    log_path = Path(log_path) if log_path else Path(args.log)
    best_out = Path(best_out) if best_out else Path(args.best_out)
    max_evals = int(max_evals) if max_evals is not None else int(args.max_evals)

    _setup_ctx(args.csv, args.source, args.space,
               float(args.filter_min), float(args.filter_max),
               args.exclude_folder, not args.no_irr)

    # 固定坐标 = 热启动值 + 当前阶段的 pin 覆盖（如结构层关闭额外层）
    base = dict(base_named or default_named())
    base.update({k: float(v) for k, v in STAGES.get(args.stage, {}).get("pin", {}).items()})
    free_names = list(free_names or NAMES)
    free_idx = [NAME_INDEX[n] for n in free_names]
    dim_a = len(free_idx)
    base_vec = np.asarray(named_to_unit(base), dtype=np.float64)

    # 信赖域：以 base_vec 为中心、半径 radius 的盒（单位立方体坐标；radius=1 即全空间）。
    # 多轮分块搜索时逐轮收缩半径，使每轮在“上一轮最优点”附近精细搜索。
    box_lo = np.clip(base_vec[free_idx] - radius, 0.0, 1.0)
    box_hi = np.clip(base_vec[free_idx] + radius, 0.0, 1.0)
    box_w = np.maximum(box_hi - box_lo, 1e-9)

    def embed(u) -> np.ndarray:
        """相对坐标（[0,1]）-> 完整 8 维（映射到当前信赖域盒内）。"""
        v = base_vec.copy()
        v[free_idx] = box_lo + box_w * np.asarray(u, dtype=np.float64)
        return v

    def to_rel(full) -> list[float]:
        """完整 8 维 -> 相对信赖域盒的 [0,1] 坐标（GP 训练/提议统一用相对坐标）。"""
        return [clamp01((float(full[i]) - box_lo[j]) / box_w[j])
                for j, i in enumerate(free_idx)]

    X: list[list[float]] = []   # 完整 8 维（缓存键 / 收尾）
    U: list[list[float]] = []   # 活跃子空间相对坐标（GP 训练）
    Y: list[float] = []
    if (args.resume or seed_history) and log_path.is_file():
        fixed_idx = [i for i in range(DIM) if i not in set(free_idx)]
        n_all = 0
        for x, y in zip(*_load_log(log_path)):
            n_all += 1
            _CACHE[_key(x)] = y  # 全部进缓存：跨阶段/跨次运行都不重复评估
            # 可用于 GP 的条件：固定坐标与当前基向量一致，且活跃坐标落在当前信赖域盒内
            if not all(abs(x[i] - base_vec[i]) < 1e-5 for i in fixed_idx):
                continue
            if all(0.0 <= (x[i] - box_lo[j]) / box_w[j] <= 1.0
                   for j, i in enumerate(free_idx)):
                X.append(list(_key(x)))
                U.append(to_rel(x))
                Y.append(y)
        print(f"  [{args.stage}] 载入历史 {n_all} 条（缓存生效），其中 {len(X)} 条进入 GP 训练"
              f"（半径 {radius:.3g}）")
    elif fresh_log and log_path.exists():
        log_path.unlink()

    t_start = time.perf_counter()

    def budget_left() -> bool:
        if len(X) >= max_evals:
            return False
        if args.time_budget_sec > 0 and (time.perf_counter() - t_start) >= args.time_budget_sec:
            return False
        return True

    def record(full, rmse: float) -> None:
        key = _key(full)
        _CACHE[key] = rmse
        X.append(list(key))
        U.append(to_rel(full))
        Y.append(rmse)
        _append_log(log_path, key, rmse)

    def report_progress(tag: str) -> None:
        finite = [y for y in Y if math.isfinite(y)]
        best = min(finite) if finite else INF
        used = time.perf_counter() - t_start
        rate = used / max(len(X), 1)
        remain = max(max_evals - len(X), 0)
        print(f"  [{args.stage}/{tag}] 评估 {len(X)}/{max_evals}，当前最优 RMSE = {best:.6f}，"
              f"已用 {_fmt_sec(used)}，预计剩余 {_fmt_sec(rate * remain)}")

    print(f"  阶段 [{args.stage}] 活跃维度 {dim_a}/{DIM}：{', '.join(free_names)}")

    # --- 阶段 1：Sobol' 初始设计（含热启动种子点，并行评估未评估点）---
    n_init = max(int(args.init_points), 2 * dim_a + 1)
    sobol = Sobol(d=dim_a, scramble=True, seed=args.seed)
    init_full = [embed(row) for row in sobol.random(n_init)]
    todo = [base_vec] + [p for p in init_full if _key(p) not in _CACHE]
    todo = [p for p in todo if _key(p) not in _CACHE]
    print(f"  阶段 1：Sobol' 初始设计 {n_init} 点 + 热启动种子（其中 {len(todo)} 点需评估），"
          f"workers = {args.workers}")

    if todo and args.workers > 1 and dim_a >= 2:
        workers = min(int(args.workers), len(todo))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(args.csv, args.source, args.space,
                                           float(args.filter_min), float(args.filter_max),
                                           args.exclude_folder, not args.no_irr)) as ex:
            for k, rmse in ex.map(_worker_eval, todo):
                record(k, rmse)
                report_progress("init")
    else:
        for p in todo:
            record(p, eval_unit(p))
            report_progress("init")

    # --- 阶段 2：GP-EI 贝叶斯优化（串行，最省调用）---
    print(f"  阶段 2：GP-EI 贝叶斯优化（剩余预算 {max(max_evals - len(X), 0)} 次评估）")
    finite = [y for y in Y if math.isfinite(y)]
    best = float(min(finite)) if finite else INF
    stale = 0

    while budget_left() and math.isfinite(best):
        # 每轮用“截至当前”的全部评估重建设计矩阵，确保 GP 能看到新样本
        u_arr = np.asarray(U, dtype=np.float64)
        y_arr = np.asarray(Y, dtype=np.float64)
        gp = _fit_gp(u_arr, y_arr, args.seed + len(X))
        picked_full = None
        for c in _propose(gp, u_arr, best, float(args.xi), args.seed + len(X)):
            full = embed(c)
            if _key(full) not in _CACHE:
                picked_full = full
                break
        if picked_full is None:  # EI 候选已全部评估过：随机补点，避免停滞
            rng = np.random.default_rng(args.seed + 1000 + len(X))
            for row in rng.random((2000, dim_a)):
                full = embed(row)
                if _key(full) not in _CACHE:
                    picked_full = full
                    break
        if picked_full is None:
            print("  活跃子空间内已无未评估点，提前结束")
            break

        rmse = eval_unit(picked_full)
        record(picked_full, rmse)
        if math.isfinite(rmse) and rmse < best - float(args.tol):
            best = rmse
            stale = 0
        else:
            stale += 1
        report_progress("bo")
        if args.patience > 0 and stale >= args.patience:
            print(f"  连续 {stale} 次无改善（tol={args.tol}），提前停止")
            break

    # --- 收尾：对最优点做一次带详情的评估并落盘 ---
    if not math.isfinite(best):
        return {"ok": False, "reason": "未找到任何可用参数组合", "evals": len(X),
                "stage": args.stage}

    y_arr = np.asarray(Y, dtype=np.float64)
    best_full = np.asarray(X[int(np.nanargmin(np.where(np.isfinite(y_arr), y_arr, np.nan)))],
                           dtype=np.float64)
    rmse, detail = eval_detailed(best_full)
    algo_params = named_to_algo(unit_to_named(best_full))

    payload = {
        "rmse": rmse,
        "rmse_by_space": detail.get("rmse_by_space", {}),
        "grades_by_space": detail.get("grades_by_space", {}),
        "fit": detail.get("fit", {}),
        "algo_params": algo_params,
                "named_params": unit_to_named(best_full),
        "stage": args.stage,
        "free_params": free_names,
        "evaluations": len(X),
        "elapsed_sec": time.perf_counter() - t_start,
        "space": args.space,
        "filter": [float(args.filter_min), float(args.filter_max)],
        "use_irr": not bool(args.no_irr),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    best_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    set_args = " ".join(f"--set {k}={v:.6g}" for k, v in algo_params.items())
    payload["cli"] = (f"python -m fixed_density.benchmark {set_args}"
                      + (" --no-irr" if args.no_irr else ""))
    return {"ok": True, "best": payload, "evals": len(X), "log": str(log_path)}


def run(args) -> dict:
    """入口：单阶段直接搜索；auto 多轮分块搜索（结构层 → 耦合成层 → 联合精修）。

    多轮（--rounds）逐轮收缩信赖域（--radius-decay）：第 1 轮半径 1（全空间）
    做全局粗搜，之后每轮在以“当前最优”为中心、半径按 decay 递减的盒内精搜，
    使各块之间反复迭代、逐步逼近跨层交互最优，避免单轮分块弱于直接 8 维搜索。
    """
    if args.stage != "auto":
        return optimize(args, free_names=STAGES[args.stage]["free"],
                        base_named=load_warm_start(args.warm_start))

    base = load_warm_start(args.warm_start)
    if base:
        print(f"热启动：{args.warm_start}")
    total = int(args.max_evals)
    rounds = max(1, int(args.rounds))
    decay = float(args.radius_decay)
    log_path = Path(args.log)
    results: dict[str, dict] = {}
    best_rmse = INF
    for r in range(rounds):
        radius = min(1.0, decay ** r)   # 1.0, decay, decay²…
        print(f"\n===== 第 {r + 1}/{rounds} 轮（信赖域半径 {radius:.3g}）=====")
        for st in STAGE_ORDER:
            free = STAGES[st]["free"]
            stage_evals = max(2 * len(free) + 1,
                              int(round(total / rounds * STAGE_FRACTION[st])))
            args.stage = st
            best_out = Path(args.best_out) if st == "joint" else \
                Path(args.best_out).with_name(f"best_params_{st}.json")
            first_call = (r == 0 and st == STAGE_ORDER[0])
            res = optimize(args, free_names=free, base_named=base, log_path=log_path,
                           best_out=best_out, max_evals=stage_evals,
                           fresh_log=first_call, seed_history=not first_call,
                           radius=radius)
            if not res.get("ok"):
                print(f"  阶段 [{st}] 失败：{res.get('reason')}")
                continue
            key = f"r{r + 1}-{st}"
            results[key] = res
            print(f"  阶段 [{key}] 完成：最优 RMSE = {res['best']['rmse']:.6f}，"
                  f"评估 {res['evals']} 次")
            # 只有更优的结果才推进热启动/信赖域中心，保证跨轮不倒退
            if res["best"]["rmse"] < best_rmse:
                best_rmse = res["best"]["rmse"]
                base = res["best"]["named_params"]

    if not results:
        return {"ok": False, "reason": "所有阶段均失败"}

    # 取所有轮次/所有阶段中 RMSE 最小者作为最终结果
    final_key = min(results, key=lambda s: results[s]["best"]["rmse"])
    payload = results[final_key]["best"]
    payload["best_stage"] = final_key
    payload["rounds"] = rounds
    payload["stages"] = {st: {"rmse": results[st]["best"]["rmse"],
                              "evaluations": results[st]["evals"],
                              "named_params": results[st]["best"]["named_params"]}
                         for st in results}
    Path(args.best_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return results[final_key]


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="搜索使 RMSE 最小的算法参数（GP-EI 贝叶斯优化）")
    p.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="benchmark CSV 路径")
    p.add_argument("--source", choices=("extracted", "osu"), default="extracted",
                   help="谱面数据来源，默认 extracted（读取 benchmark_extracted）")
    p.add_argument("--space", choices=("original", "mapped"), default="original",
                   help="RMSE 目标口径：original=原始 value（默认），mapped=映射 value")
    p.add_argument("--filter-min", default=-math.inf, help="统计范围 value 下限（默认不限）")
    p.add_argument("--filter-max", default=math.inf, help="统计范围 value 上限（默认不限）")
    p.add_argument("--exclude-folder", default="course",
                   help="目标统计中排除的 folder（默认 course，即只统计非 course 谱面）")
    p.add_argument("--stage", choices=("auto",) + STAGE_ORDER, default="auto",
                   help="搜索阶段：auto=结构层→耦合成层→联合精修（默认）；也可只跑单一阶段")
    p.add_argument("--warm-start", default=None,
                   help="热启动 JSON（如上一轮 best_params.json），提供各阶段固定坐标初值")
    p.add_argument("--init-points", type=int, default=16, help="Sobol' 初始设计点数（默认 16）")
    p.add_argument("--max-evals", type=int, default=200, help="总评估次数上限（默认 200）")
    p.add_argument("--rounds", type=int, default=2,
                   help="auto 分块搜索的轮数；≥2 时逐轮收缩信赖域精搜（默认 2）")
    p.add_argument("--radius-decay", type=float, default=0.5,
                   help="每轮信赖域半径的衰减系数，第 r 轮半径 = decay^(r-1)（默认 0.5）")
    p.add_argument("--patience", type=int, default=0,
                   help="连续无改善多少次后停止；0 表示禁用（默认 0，跑满预算以保准确）")
    p.add_argument("--tol", type=float, default=1e-4, help="判定“改善”的 RMSE 阈值（默认 1e-4）")
    p.add_argument("--xi", type=float, default=0.01, help="EI 采集函数的探索系数（默认 0.01）")
    p.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 1)),
                   help="初始设计阶段的并行进程数（默认 CPU-1，上限 12）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    p.add_argument("--time-budget-sec", type=float, default=0.0,
                   help="墙钟时间预算（秒），0 表示不限（默认 0）")
    p.add_argument("--log", default="tune_log.jsonl", help="评估记录 JSONL（断点续跑用）")
    p.add_argument("--resume", action="store_true", help="从已有 --log 续跑")
    p.add_argument("--best-out", default="best_params.json", help="最优参数输出 JSON")
    p.add_argument("--no-irr", action="store_true",
                   help="拟合模型去掉 irr 项（C=0），改由逐键局部熵增益层替代（配合 γ 三项）")
    p.add_argument("--dry-run", action="store_true", help="只打印配置与预算，不评估")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    print("轮询参数空间：")
    for name, lo, hi, scale in PARAM_SPACES:
        if scale == "log10":
            rng = f"log10∈[{lo}, {hi}]（原始 {10 ** lo:.4g} ~ {10 ** hi:.4g}）"
        else:
            rng = f"[{lo}, {hi}]"
        label = "（extra_b = t·ln a）" if name == "extra_b_t" else ""
        print(f"  {name:<16} {rng}{label}")
    print(f"目标口径：{args.space} | 统计范围 value ∈ [{args.filter_min}, {args.filter_max}]"
          f" | 排除 folder：{args.exclude_folder}")
    print(f"钉住参数（不搜索）：{PINNED}")
    _use_irr = not bool(args.no_irr)
    _fit_feats = "、".join([n for n, on in (("irr", _use_irr), ("chord", FIT_USE_CHORD))
                            if on]) or "无"
    print(f"拟合模型特征：{_fit_feats} + lnN"
          + ("（--no-irr：C=0，irr 信息改由 γ 三层承担）" if args.no_irr else ""))
    if args.stage == "auto":
        rounds = max(1, int(args.rounds))
        per_round = float(args.max_evals) / rounds
        plan = "  ".join(
            f"{st}({len(STAGES[st]['free'])}D/{max(2 * len(STAGES[st]['free']) + 1, int(round(per_round * STAGE_FRACTION[st])))}次)"
            for st in STAGE_ORDER)
        print(f"搜索模式：auto 分块搜索 × {rounds} 轮（信赖域半径逐轮 ×{args.radius_decay}）"
              f"—— 每轮 {plan}；总预算 {args.max_evals} 次")
    else:
        free = STAGES[args.stage]["free"]
        print(f"搜索模式：单阶段 {args.stage}（活跃 {len(free)}D：{', '.join(free)}），"
              f"预算 {args.max_evals} 次")

    if args.dry_run:
        print("--dry-run：未执行任何评估")
        return

    result = run(args)
    if not result["ok"]:
        print(f"优化失败：{result['reason']}（评估 {result['evals']} 次）")
        raise SystemExit(1)

    b = result["best"]
    print("\n===== 最优结果 =====")
    print(f"RMSE（{b['space']} 口径）= {b['rmse']:.6f}"
          f" | 各口径：{ {k: round(v, 6) for k, v in b['rmse_by_space'].items()} }")
    print(f"拟合：A={b['fit'].get('A'):.6g}, B={b['fit'].get('B'):.6f}, "
          f"R²(log)={b['fit'].get('r2_log'):.6f}")
    print("算法参数：")
    for k, v in b["algo_params"].items():
        print(f"  {k:<16} = {v:.6f}")
    print(f"评估次数：{b['evaluations']} | 耗时：{_fmt_sec(b['elapsed_sec'])}")
    print(f"最优参数已写入：{args.best_out}；评估记录：{result['log']}")
    print(f"\n复现命令：\n  {b['cli']}")


if __name__ == "__main__":
    main()
