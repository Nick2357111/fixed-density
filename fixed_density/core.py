"""Fixed Density v1 —— osu!mania 4K 难度计算核心。

本模块只依赖 numpy，包含：
  * 可调参数的统一注册表（ParamSpec / AlgorithmSpec / register_algorithm）；
  * 物件赋值序列（object_sequence）算法及其全部子过程（RC 通道 + LN 通道）；
  * value 的连续映射曲线 g（正解 map_forward / 反解 map_invert）；
  * 不规则度（irregularity）与逐键局部熵等辅助特征。

算法概述（详细推导见仓库 README）：
  1. 逐轨按相同时间戳分组，计算每组的注入 contribution
     （同轨前/后间距倒数经合并算子合并，另叠加配对轨邻域额外贡献）；
  2. 组内多次按键按“超限替代”用 OVERLAP_PENALTY 累加；
  3. 赋值序列按相邻物件 dt 递推衰减 value = value·0.5^(dt/H) + contribution；
  4. 同手（轨道对）两条序列 softmax 合并，再对各配对序列做二次合并；
  5. 对合并后的赋值序列求分位数 n（find_n_quantile）；
  6. n 经映射曲线反解得到 rc_value；LN 通道独立给出 ln_value。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ===========================================================================
# 固定常量（沿用原算法，不列入“程序可调参数”）
# ===========================================================================
OVERLAP_PENALTY = 20000.0   # 同轨同时间戳重叠物件的额外惩罚（每个多余物件）
INV_DIST_SCALE = 1000.0     # 相邻物件间距倒数的缩放：1000 / dt
ABSORB_DIST_SCALE = 0.5     # 松手吸收判定：距离 < 该倍数×本地节拍即视为被邻近起音吸收
LN_TAIL_WINDOW_RATIO = 1.5  # LN 尾判窗宽 / 头判窗宽（社区惯例：尾判为 1.5 倍判定）
LN_HEAD_WINDOW_A = 64.0     # 头判窗总宽 = A - 3·OD（ms），OD 越高判定越紧
LN_HEAD_WINDOW_B = 3.0
LN_MIN_HOLD_MS = 20.0       # 有效性门·最短点击补足 a（ms）：玩家一次点击的物理最短时长
LN_VALID_OD_GAIN = 0.4      # 有效性门·OD 项系数 b：门位置 lo' = a + b·(W_head + W_tail)
LN_VALID_RAMP = 1.3         # 有效性门过渡宽度：门宽 hi = LN_VALID_RAMP · lo'
TIME_EPS = 1e-6             # 浮点时间戳比较容差
N_PRECISION = 1e-12         # n 值二分精度

# dvalue 评价区间（含下沿、含上沿的方式在分类函数中实现）
GRADE_BOUNDS = (0.2, 0.5, 1.0)
GRADE_LABELS = ("精确", "接近", "一般", "误判")
GRADE_BOUND_TXT = ("|d|≤0.2", "0.2<|d|≤0.5", "0.5<|d|≤1", "|d|>1")
GRADE_COLORS = ("#2ecc71", "#f1c42f", "#e67e22", "#e74c3c")

FOLDER_COLORS = {
    "course": "#8e44ad",
    "jack": "#e74c3c",
    "speed": "#3498db",
    "stamina": "#2ecc71",
    "tech": "#e67e22",
}


# ===========================================================================
# 算法注册机制：新增算法只需实现函数并用 @register_algorithm 注册
# ===========================================================================
@dataclass(frozen=True)
class ParamSpec:
    """单个可调参数的描述。"""

    name: str
    default: float
    label: str
    help_text: str


@dataclass(frozen=True)
class AlgorithmSpec:
    """一个可被 benchmark 调用的难度算法。"""

    key: str
    title: str
    params: tuple[ParamSpec, ...]
    func: object
    description: str

    def default_params(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


_REGISTRY: dict[str, AlgorithmSpec] = {}


def register_algorithm(key: str, title: str, params: tuple[ParamSpec, ...], description: str = ""):
    """注册一个算法：func(columns, params: dict) -> float(n)。"""

    def _decorator(func):
        _REGISTRY[key] = AlgorithmSpec(key, title, tuple(params), func, description)
        return func

    return _decorator


# ===========================================================================
# n 值求解：排序 + 前缀和二分，分位数作为上界初值
# ===========================================================================
def find_n_quantile(a, target_avg: float = 0.96, precision: float = N_PRECISION) -> float:
    """求 n 使 mean(min(1, n/a_i)) == target_avg。

    与 find_n.find_n_fast 数学等价（排序 + 1/a 前缀和，每步 O(log N)），
    区别是二分上界初值取 a 的 target_avg 分位数：
    在该分位数处至少占比 target_avg 的元素 a_i <= n（贡献 1），
    故 f(n) >= target_avg，构成合法上界，区间比 [0, max] 更窄。
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        raise ValueError("赋值列表为空，无法计算 n")
    if target_avg <= 0.0:
        return 0.0
    if target_avg >= 1.0:
        return float(a.max())

    a_sorted = np.sort(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_a = np.where(a_sorted > 0.0, 1.0 / a_sorted, 0.0)
    prefix_inv = np.empty(a_sorted.size + 1, dtype=np.float64)
    prefix_inv[0] = 0.0
    np.cumsum(inv_a, out=prefix_inv[1:])
    total_inv = prefix_inv[-1]

    n_count = a_sorted.size
    target_sum = target_avg * n_count

    def partial_sum(value: float) -> float:
        k = int(np.searchsorted(a_sorted, value, side="right"))
        return k + value * (total_inv - prefix_inv[k])

    lo = 0.0
    hi = float(np.quantile(a_sorted, target_avg))  # 第 target_avg 分位数作为初值上界
    if hi <= 0.0 or partial_sum(hi) < target_sum:  # 极端分布下的安全兜底
        hi = float(a_sorted[-1])

    while hi - lo > precision:
        mid = (lo + hi) * 0.5
        if partial_sum(mid) < target_sum:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


# ===========================================================================
# 简化算法：物件赋值序列（逐物件 0.5^(dt/H) 衰减）
# ===========================================================================
def _group_lane(col: list[tuple[int, int]]) -> list[tuple[float, int, float | None, float | None]]:
    """把单轨物件按相同时间戳分组。

    返回 [(time, group_size, prev_dt, next_dt), ...]，按时间升序；
    prev_dt / next_dt 为与同轨前/后一组物件的时间差(ms)，首/末组为 None。
    """
    times = [float(t) for t, _ in col]  # read_mania_beatmap 已按时间排序
    groups: list[tuple[float, int]] = []
    i, m = 0, len(times)
    while i < m:
        j = i
        while j + 1 < m and abs(times[j + 1] - times[i]) < TIME_EPS:
            j += 1
        groups.append((times[i], j - i + 1))
        i = j + 1

    result = []
    for idx, (t, size) in enumerate(groups):
        prev_dt = t - groups[idx - 1][0] if idx > 0 else None
        next_dt = groups[idx + 1][0] - t if idx + 1 < len(groups) else None
        result.append((t, size, prev_dt, next_dt))
    return result


_EMPTY_GAPS: dict = {}  # cross_lane_mix=0 时代替 global_gaps（.get 恒返回 None）


def _lp_norm(a: float, b: float, power: float) -> float:
    """数值稳定的 Lp 范数 (|a|^p + |b|^p)^(1/p)。

    等价于把较大者提出：hi * (1 + (lo/hi)^p)^(1/p)，其中 hi = max(|a|, |b|)、
    lo = min(|a|, |b|)。p → ∞ 时收敛到 max(|a|, |b|)，与直接计算数学等价，
    但 (lo/hi) ≤ 1 保证幂运算不会溢出（大幂次下原式会 OverflowError）。
    """
    hi = max(abs(a), abs(b))
    if hi == 0.0:
        return 0.0
    lo = min(abs(a), abs(b))
    return hi * (1.0 + (lo / hi) ** power) ** (1.0 / power)


def _merge_op(a: float, b: float, power: float, soft_beta: float = 0.0) -> float:
    """把两个分量合并为一个值（所有合并点统一入口）。

    soft_beta > 0：softmax 加权平均（光滑的 max）
        w = ½·(1 + tanh(½·β·(a−b)))，  M = w·a + (1−w)·b
    tanh 处处光滑且对任意实数稳定（不溢出），故该算子全程连续、无分段：
    β → ∞ 收敛到 max(a, b)，β → 0 收敛到算术平均。
    soft_beta = 0：退回数值稳定的 Lp 范数（与旧行为完全一致）。
    """
    if soft_beta > 0.0:
        w = 0.5 * (1.0 + math.tanh(0.5 * soft_beta * (a - b)))
        return w * a + (1.0 - w) * b
    return _lp_norm(a, b, power)


def _merge_vec(a, b, power: float, soft_beta: float = 0.0):
    """_merge_op 的逐元素向量化版本（数值语义与标量版一致，供按轨批处理）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if soft_beta > 0.0:
        w = 0.5 * (1.0 + np.tanh(0.5 * soft_beta * (a - b)))
        return w * a + (1.0 - w) * b
    hi = np.maximum(np.abs(a), np.abs(b))
    lo = np.minimum(np.abs(a), np.abs(b))
    out = np.zeros_like(hi)
    nz = hi > 0.0
    out[nz] = hi[nz] * (1.0 + (lo[nz] / hi[nz]) ** power) ** (1.0 / power)
    return out


def _lane_contribs_vec(groups, mate_groups, contrib_power, cross_lane_mix, soft_beta,
                       extra_rate, extra_b, extra_log_base, gaps_get,
                       loc_onsets=None, loc_entropy=None, irr_gain=0.0):
    """按轨一次性向量化计算每组物件的注入贡献（等价于逐组标量计算，但消除数百万次调用）。

    原 contribution（每组在自身时刻注入的贡献，连续、无分段）：
      单件：c = (prev_inv^p + next_inv^p)^(1/p)，prev/next_inv = 1000 / dt（p = contrib_power，
            另可由 soft_beta 切换为 softmax 加权平均）；
      dt 为“同轨前/后间距”与“全局（全轨去重起音）前/后间距”的连续凸组合
            dt = (1 − cross_lane_mix)·dt_lane + cross_lane_mix·dt_global（取 0 时即纯同轨）；
      同刻堆叠（size > 1）逐件按“超限替代”用 OVERLAP_PENALTY 累加，避免除零。

    额外 contribution（额外贡献层位于原 contribution 与赋值序列生成之间）：
      令 a = 1000 / c，在配对轨上取时间差 x ∈ [−4a, 4a] 的键，逐键按
      exp(−((log_base(x/a) + extra_b)^2)/2) 加权求和；最终
            contribution = c · (1 + extra_rate · extra)

    逐键局部不规则度（irr_gain ≠ 0 时启用，局部层位于额外层之后）：
      以该起音的局部邻域熵 irr_i 作乘子并入赋值序列（负 γ 削弱高不规则处的贡献）
            contribution *= exp(irr_gain · irr_i)
      loc_onsets / loc_entropy 为全轨去重起音序列与其逐键熵（等长、升序）。
    """
    n = len(groups)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    tt = np.empty(n); sz = np.empty(n); pd = np.empty(n); nd = np.empty(n)
    for k, g in enumerate(groups):
        tt[k] = g[0]; sz[k] = g[1]
        pd[k] = -1.0 if g[2] is None else g[2]
        nd[k] = -1.0 if g[3] is None else g[3]
    use_loc = irr_gain != 0.0 and loc_onsets is not None and loc_onsets.size > 0
    if cross_lane_mix != 0.0:
        gpd = np.empty(n); gnd = np.empty(n)
        for k, g in enumerate(groups):
            gp, gn = gaps_get(g[0]) or (None, None)
            gpd[k] = -1.0 if gp is None else gp
            gnd[k] = -1.0 if gn is None else gn
    else:
        gpd = gnd = None

    def _inv(dtl, dtg):
        if dtg is None:
            eff = dtl
        else:
            g = np.where(dtg > TIME_EPS, dtg, dtl)
            eff = (1.0 - cross_lane_mix) * dtl + cross_lane_mix * g
        with np.errstate(divide="ignore", invalid="ignore"):
            val = INV_DIST_SCALE / eff
        return np.where(dtl > TIME_EPS, val, 0.0)

    prev_inv = _inv(pd, gpd)
    next_inv = _inv(nd, gnd)

    # 原 contribution：单件为前后倒数 Lp 合并；多件（同刻堆叠）按“超限替代”逐件累加
    base = _merge_vec(prev_inv, next_inv, contrib_power, soft_beta)
    multi = sz > 1.0
    if multi.any():
        pen = np.full(int(multi.sum()), OVERLAP_PENALTY)
        pen_pen = _merge_op(OVERLAP_PENALTY, OVERLAP_PENALTY, contrib_power, soft_beta)
        base = base.copy()
        base[multi] = ((sz[multi] - 2.0) * pen_pen
                       + _merge_vec(prev_inv[multi], pen, contrib_power, soft_beta)
                       + _merge_vec(pen, next_inv[multi], contrib_power, soft_beta))

    # 额外 contribution：配对轨邻域高斯耦合（按 base 决定时间窗 [t−4a, t+4a]，a = 1000/base）
    if extra_rate != 0.0 and mate_groups:
        mt = np.array([g[0] for g in mate_groups], dtype=np.float64)
        ms = np.array([g[1] for g in mate_groups], dtype=np.float64)
        pos = base > 0.0
        a = np.where(pos, 1000.0 / np.where(pos, base, 1.0), 0.0)
        lo = np.searchsorted(mt, tt - 4.0 * a, side="left")
        hi = np.searchsorted(mt, tt + 4.0 * a, side="right")
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total:
            idx = np.repeat(np.arange(n), counts)
            within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            flat_j = lo[idx] + within
            x = np.abs(mt[flat_j] - tt[idx])
            log_denom = math.log(extra_log_base)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log(x / a[idx]) / log_denom + extra_b
                w = ms[flat_j] * np.exp(-(z * z) * 0.5)
            w = np.where(np.isfinite(w), w, 0.0)
            extra = np.bincount(idx, weights=w, minlength=n)
        else:
            extra = np.zeros(n)
    else:
        extra = np.zeros(n)
    contrib = base * (1.0 + extra_rate * extra)
    # 逐键局部不规则度：以该起音的局部熵作乘子并入赋值序列
    if use_loc:
        k0 = np.searchsorted(loc_onsets, tt)
        kk = np.minimum(k0, loc_onsets.size - 1)
        hit = (k0 < loc_onsets.size) & (np.abs(loc_onsets[kk] - tt) < TIME_EPS)
        contrib = contrib * np.exp(irr_gain * np.where(hit, loc_entropy[kk], 0.0))
    return contrib


def _counts_at(ts, cs, times):
    """取 cs 在 times 各时刻的值（仅精确命中处非零，其余为 0）。向量化，容忍 TIME_EPS。"""
    out = np.zeros(times.size, dtype=np.int64)
    if ts.size:
        k = np.searchsorted(ts, times)
        kk = np.minimum(k, ts.size - 1)
        hit = (k < ts.size) & (np.abs(ts[kk] - times) < TIME_EPS)
        out[hit] = cs[kk[hit]]
    return out


def _lane_curve_vec(gt, contribs, ut, half_life_ms: float):
    """单轨衰减序列在 ut 各时刻的值（向量化，等价于 _lane_curve 的加法递推）。

    value_i = Σ_{k≤i} c_k · 0.5^((t_i − t_k)/H)。用分块 cumsum 计算（块内以
    2^(rel/H) 加权再乘回 0.5^(rel/H)），既保持等价又避免 2^(t/H) 上溢。
    """
    cc = np.zeros(ut.size, dtype=np.float64)
    if gt.size:
        k = np.searchsorted(gt, ut)
        kk = np.minimum(k, gt.size - 1)
        hit = (k < gt.size) & (np.abs(gt[kk] - ut) < TIME_EPS)
        cc[hit] = contribs[kk[hit]]
    v = np.empty(ut.size, dtype=np.float64)
    H = half_life_ms
    span = 400.0 * H  # 块内 2^(rel/H) ≤ 2^400，远低于 float64 上溢阈值
    carry = 0.0
    s, n = 0, ut.size
    while s < n:
        e = int(np.searchsorted(ut, ut[s] + span, side="right"))
        if e <= s:
            e = s + 1
        rel = ut[s:e] - ut[s]
        v[s:e] = np.exp2(-rel / H) * (carry + np.cumsum(cc[s:e] * np.exp2(rel / H)))
        if e < n:
            carry = float(v[e - 1]) * math.pow(0.5, (ut[e] - ut[e - 1]) / H)
        s = e
    return v


def _sample_decay_vec(st, sv, times, half_life_ms: float):
    """把“赋值突变衰减”序列在 times 上采样（替换语义，向量化）。

    函数在每个 st[k] 跳变为 sv[k] 后按 0.5^((t−st[k])/H) 衰减，之前为 0。
    """
    out = np.zeros(times.size, dtype=np.float64)
    if st.size:
        k = np.searchsorted(st, times, side="right") - 1
        ok = k >= 0
        kk = k[ok]
        out[ok] = sv[kk] * np.exp2(-(times[ok] - st[kk]) / half_life_ms)
    return out


def _merge_decay_sequences_vec(ta, va, ca, tb, vb, cb, power, half_life_ms, soft_beta):
    """合并两条“赋值突变衰减”序列（向量化版 _merge_decay_sequences）。

    输入/输出均为 (times, values, counts) 三个升序数组；在时间戳并集上采样后用
    _merge_vec 逐时刻合并，counts 取两条序列在该时刻的物件数之和（保持相同值个数一致）。
    """
    if ta.size == 0:
        return tb, vb, cb
    if tb.size == 0:
        return ta, va, ca
    ut = np.union1d(ta, tb)
    v = _merge_vec(_sample_decay_vec(ta, va, ut, half_life_ms),
                   _sample_decay_vec(tb, vb, ut, half_life_ms), power, soft_beta)
    c = _counts_at(ta, ca, ut) + _counts_at(tb, cb, ut)
    keep = c > 0
    return ut[keep], v[keep], c[keep]


# ===========================================================================
# LN 通道（松手流 + 松手→重按 + 按住占用）
#
# LN 谱面的困难有两类可测结构，都藏在“物件头时间戳”之外：
#   抓准：松手时刻 r = t + l 是与按下同权的独立判定点，同轨相邻松手的间距构成
#         第二套起音序列，可直接复用与 RC 相同的 contribution→合并→求 n 机制；
#   控手：同轨“松手 → 下一次按下”的间隔 δ_rp 越小，换指调度压力越大；
#         同手（轨道对）在某次松手时仍被其他 LN 占用，构成持续按住张力；
#         本轨长条按住期间同手另一轨的动作数超过其本地基线的程度（跨轨调度压力）。
# 单轨松手事件的注入贡献：
#   c = Lp(1000/δ_rr⁻, 1000/δ_rr⁺; q)          # 抓准：相邻松手间距倒数
#     + jack_gain · (1000/δ_rp)                # 控手：松手→同轨下一次按下
#     + cross_gain · (n / (本地基线率 · ℓ))     # 控手：按住期间同手另一轨的超额动作比
#     + OVERLAP_PENALTY · (size − 1)           # 同刻多件（超限替代同 RC 语义）
#     然后（启用占用层时）乘 exp(hold_gain · occ)，occ 为该松手时刻同手仍按住的键数。
# 该通道得到的 n_ln 与 RC 的 n_rc 以 Lp 合并：
#   n_total = (n_rc^p + (ln_gain·n_ln)^p)^(1/p)
# ln_gain = 0（默认）时完全不计算该通道，行为与旧版逐位一致。
# ===========================================================================
def _hand_occupancy(cols_pair, query_times) -> np.ndarray:
    """同一手（轨道对）在 query_times 各时刻“仍被 LN 按住”的键数。

    按住区间取半开 [t, t+l)，故某 LN 自身松手时刻不把自身计入，只反映
    “松手时同手还有其他键被按住”这一控手成分。
    """
    starts: list[float] = []
    ends: list[float] = []
    for col in cols_pair:
        for t, l in col:
            if l > 0:
                starts.append(float(t))
                ends.append(float(t + l))
    q = np.asarray(query_times, dtype=np.float64)
    if not starts:
        return np.zeros(q.size, dtype=np.float64)
    s = np.sort(np.asarray(starts, dtype=np.float64))
    e = np.sort(np.asarray(ends, dtype=np.float64))
    return (np.searchsorted(s, q, side="right")
            - np.searchsorted(e, q, side="right")).astype(np.float64)


def _ln_lane_contribs(col, hand_cols, partner_col, rel_power, jack_gain, hold_gain, cross_gain):
    """单轨 LN 松手事件的 (分组, 注入贡献)；该轨无 LN 时返回 (None, None)。

    groups 与 _group_lane 同构 [(t, size, prev_dt, next_dt), ...]，但 t 为松手
    时刻（t + 长度），prev/next_dt 为同轨相邻松手的间距。
    partner_col 为同手另一轨（用于跨轨按压计数）；hand_cols 为整手（用于占用）。
    """
    rels = sorted(float(t) + float(l) for t, l in col if l > 0)
    if not rels:
        return None, None
    groups = _group_lane([(r, 0) for r in rels])
    tt = np.array([g[0] for g in groups], dtype=np.float64)
    sz = np.array([g[1] for g in groups], dtype=np.float64)
    pd = np.array([-1.0 if g[2] is None else g[2] for g in groups], dtype=np.float64)
    nd = np.array([-1.0 if g[3] is None else g[3] for g in groups], dtype=np.float64)

    def _inv(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = INV_DIST_SCALE / d
        return np.where(d > TIME_EPS, v, 0.0)

    # 抓准：相邻松手间距倒数（与 RC 同一套前后间距 Lp 合并）
    contrib = _merge_vec(_inv(pd), _inv(nd), rel_power)

    # 控手：松手 → 同轨下一次按下的间隔
    if jack_gain != 0.0:
        presses = np.array(sorted(float(t) for t, _ in col), dtype=np.float64)
        if presses.size:
            k = np.searchsorted(presses, tt, side="right")  # 首个晚于松手时刻的按下
            d = np.full(tt.size, -1.0, dtype=np.float64)
            ok = k < presses.size
            d[ok] = presses[k[ok]] - tt[ok]
            contrib = contrib + jack_gain * _inv(d)

    # 控手：本轨长条按住期间，同手另一轨动作相对其本地基线的超额交互
    #   基线率由按住段两侧最近动作的间隔估计（不含段内动作，避免自我循环），
    #   期望动作数 = 基线率 × 保持时长；取实际/期望的无量纲比值（均匀谱 ≈ 1），
    #   从而剥掉“保持越短→谱越密”的密度代理成分，只留“我按住时另一轨是否异常繁忙”。
    if cross_gain != 0.0 and partner_col:
        pacts = sorted([float(t) for t, _ in partner_col]
                       + [float(t) + float(l) for t, l in partner_col if l > 0])
        if pacts:
            pt = np.asarray(pacts, dtype=np.float64)
            M = pt.size
            hs = np.array([float(t) for t, l in col if l > 0], dtype=np.float64)
            he = np.array([float(t) + float(l) for t, l in col if l > 0], dtype=np.float64)
            # 实际动作数（严格落在按住区间内）；kb/ka 为段前/段后最近动作的下标
            kb = np.searchsorted(pt, hs, side="right") - 1
            ka = np.searchsorted(pt, he, side="left")
            n_ev = np.maximum(ka - kb - 1, 0).astype(np.float64)
            has_b = kb >= 0
            has_a = ka < M
            gap_b = np.zeros(hs.size, dtype=np.float64)
            gap_a = np.zeros(hs.size, dtype=np.float64)
            if has_b.any():
                gap_b[has_b] = hs[has_b] - pt[kb[has_b]]
            if has_a.any():
                gap_a[has_a] = pt[ka[has_a]] - he[has_a]
            good_b = has_b & (gap_b > TIME_EPS)
            good_a = has_a & (gap_a > TIME_EPS)
            rate = np.zeros(hs.size, dtype=np.float64)   # 本地基线动作率（动作/ms）
            both = good_b & good_a
            rate[both] = 2.0 / (gap_b[both] + gap_a[both])
            only_b = good_b & ~good_a
            rate[only_b] = 1.0 / gap_b[only_b]
            only_a = good_a & ~good_b
            rate[only_a] = 1.0 / gap_a[only_a]
            expected = rate * (he - hs)
            inter = np.zeros(hs.size, dtype=np.float64)
            ok = expected > TIME_EPS
            inter[ok] = n_ev[ok] / expected[ok]
            acc = np.zeros(tt.size, dtype=np.float64)
            np.add.at(acc, np.searchsorted(tt, he), inter)  # 同刻多根长条按比值叠加
            contrib = contrib + cross_gain * acc

    # 控手：同手仍被按住的其他键数
    if hold_gain != 0.0:
        contrib = contrib * np.exp(hold_gain * _hand_occupancy(hand_cols, tt))

    multi = sz > 1.0
    if multi.any():
        contrib = contrib + OVERLAP_PENALTY * (sz - 1.0)
    return groups, contrib


def _ln_channel_n(columns, params: dict[str, float]) -> float:
    """LN 通道的 n（以松手事件为注入点，两级合并后求 n）；无 LN 时返回 0.0。"""
    rel_power = float(params.get("ln_rel_power", 2.0))
    jack_gain = float(params.get("ln_jack_gain", 1.0))
    hold_gain = float(params.get("ln_hold_gain", 0.0))
    cross_gain = float(params.get("ln_cross_gain", 0.0))
    if rel_power <= 0.0:
        raise ValueError("ln_rel_power 必须 > 0")
    lane_groups: list = []
    lane_contribs: list = []
    has_ln = False
    for i, col in enumerate(columns):
        start = i - (i % 2)
        hand = columns[start:start + 2]
        if len(hand) < 2:
            partner = None          # 奇数键时末轨无同手伙伴，跨轨按压项不参与
        else:
            partner = hand[1] if i == start else hand[0]
        groups, contrib = _ln_lane_contribs(col, hand, partner, rel_power,
                                            jack_gain, hold_gain, cross_gain)
        if groups is None:
            lane_groups.append([])
            lane_contribs.append(np.zeros(0, dtype=np.float64))
        else:
            has_ln = True
            lane_groups.append(groups)
            lane_contribs.append(contrib)
    if not has_ln:
        return 0.0
    return _merge_sequence_n(
        lane_groups, lane_contribs,
        float(params["merge_power"]), float(params["merge2_power"]),
        float(params.get("soft_beta", 0.0)), float(params["half_life_ms"]),
        float(params["quantile"]))


def _merge_pair_sequences(lane_groups, lane_contribs, merge_power, soft_beta,
                          half_life_ms, pair_irr=None, irr_gain_pair=0.0):
    """第一级：配对双轨 (1,2)/(3,4) 合并为 (times, values, counts) 序列列表。"""
    k_total = len(lane_groups)
    out = []
    for start in range(0, k_total, 2):
        mate = start + 1 if start + 1 < k_total else start
        groups_a = lane_groups[start]
        groups_b = lane_groups[mate]
        if not groups_a and not groups_b:
            continue
        gt_a = np.array([g[0] for g in groups_a], dtype=np.float64)
        sa = np.array([g[1] for g in groups_a], dtype=np.int64)
        ut = gt_a if mate == start else np.union1d(
            gt_a, np.array([g[0] for g in groups_b], dtype=np.float64))
        va = _lane_curve_vec(gt_a, lane_contribs[start], ut, half_life_ms)
        if mate == start:
            v, c = va, _counts_at(gt_a, sa, ut)
        else:
            gt_b = np.array([g[0] for g in groups_b], dtype=np.float64)
            sb = np.array([g[1] for g in groups_b], dtype=np.int64)
            v = _merge_vec(va, _lane_curve_vec(gt_b, lane_contribs[mate], ut, half_life_ms),
                           merge_power, soft_beta)
            c = _counts_at(gt_a, sa, ut) + _counts_at(gt_b, sb, ut)
        # 轨道对熵 → 配对合并层
        if pair_irr is not None:
            v = v * np.exp(irr_gain_pair * entropy_at(*pair_irr[start // 2], ut))
        keep = c > 0
        out.append((ut[keep], v[keep], c[keep]))
    return out


def _merge_sequence_n(lane_groups, lane_contribs, merge_power, merge2_power, soft_beta,
                      half_life_ms, quantile, pair_irr=None, irr_gain_pair=0.0,
                      glob_irr=None, irr_gain_global=0.0) -> float:
    """两级合并（配对双轨 → 衰减序列二次合并）后求 n；无有效序列时返回 0.0。"""
    pair_sequences = _merge_pair_sequences(lane_groups, lane_contribs, merge_power, soft_beta,
                                           half_life_ms, pair_irr, irr_gain_pair)
    mt = mv = mc = np.zeros(0, dtype=np.float64)
    for ta, va, ca in pair_sequences:
        mt, mv, mc = _merge_decay_sequences_vec(mt, mv, mc, ta, va, ca,
                                                merge2_power, half_life_ms, soft_beta)
    if mv.size == 0:
        return 0.0
    # 全谱熵 → 最终合并层
    if glob_irr is not None:
        mv = mv * np.exp(irr_gain_global * entropy_at(glob_irr[0], glob_irr[1], mt))
    merged_values = np.repeat(mv, mc.astype(np.int64))
    return find_n_quantile(merged_values, quantile)


# ===========================================================================
# LN 参考难度：ln 倍率 + 含下界对数标定
#
# 与上面的“Lp 合并通道”并列的另一条 LN 口径（不复用其赋值序列，只共用同一套
# LN 结构量的定义）：
#   1) ln 倍率 k = F(ρ) · (γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross + γ_hold·S_hold)
#      四项 S 是 LN 事件上“每根 LN 有多强”的均值（前三项单位 1000/间隔(ms)，
#      S_hold 为“保持时长/该轨物件平均间隔”的无量纲比值），
#      “有多少 LN”由长条占比 ρ 经 Hill 型饱和函数承担：
#          F(ρ) = (1+c^h)·ρ^h/(ρ^h+c^h)，F(0)=0、F(1)=1、单调不减
#      高 ρ 区趋平（LN 谱几乎不受参数影响）、低 ρ 区陡降（压制“只带少量长条”的谱面）。
#      **不含常数项**：无 LN 的谱面 ρ = 0 且各项皆 0 ⇒ k = 0。
#   2) 新 n 值 n_LN = k · n_rc，即把 RC 求得的 n 直接按倍率缩放。
#   3) 含下界对数标定 ln_value = a·ln(b·n_LN + 1)：n_LN = 0 时 ln_value = 0，
#      于是“几乎没有 ln 的谱面显示 0”这一语义天然成立，无需额外截断。
#      a、b 由 ln benchmark 做最小二乘拟合（给定 b 时 a 有闭式解）。
# ===========================================================================
def is_ln_course(name: str) -> bool:
    """该 LN 谱是否属于“段位单曲”（dan course 马拉松）。

    它们混在 ln/ 目录里（文件名含 "Dan Courses" 与 "(Marathon)"），共 64/102 张，
    是低 value 段（1~6）唯一的样本来源，故 LN 拟合**默认包含**它们；
    该函数用于分组统计与 --exclude-course 对照实验。
    """
    return "Dan Course" in name


def ln_od_windows(od: float) -> tuple[float, float]:
    """该 OD 下 LN 头/尾判定的半宽 (W_head, W_tail)，单位 ms。

    头判窗总宽 = 64 - 3·OD（osu!mania 的 300 判定），故半宽 W_head = (64-3·OD)/2；
    社区惯例 LN 尾判为头判的 1.5 倍，故 W_tail = 1.5·W_head。
    """
    w_head = (LN_HEAD_WINDOW_A - LN_HEAD_WINDOW_B * od) / 2.0
    return max(w_head, 0.0), max(w_head, 0.0) * LN_TAIL_WINDOW_RATIO


def ln_valid_band(od: float) -> tuple[float, float]:
    """长度有效性门的临界区间 (lo, hi)，单位 ms。

    lo = LN_MIN_HOLD_MS + LN_VALID_OD_GAIN·(W_head + W_tail)
    其中 W_head + W_tail 是头/尾判定窗刚好不重叠的临界长度（头判半宽 + 1.5×头判半宽），
    LN_MIN_HOLD_MS 是玩家一次点击的物理最短时长补足，LN_VALID_OD_GAIN 为 OD 项系数。
    hi = LN_VALID_RAMP · lo。ℓ ≤ lo 视为单键，ℓ ≥ hi 视为完整长条，中间连续过渡。

    OD 越高判定越紧 → lo 越小 → 同样长度的长条权重越大；OD 越低则相反。
    OD8 时 lo = 45ms / hi = 49.5ms，OD9 时 lo = 43.1 / hi = 47.4，OD0 时 lo = 60 / hi = 66。
    """
    w_head, w_tail = ln_od_windows(od)
    lo = LN_MIN_HOLD_MS + LN_VALID_OD_GAIN * (w_head + w_tail)
    return lo, lo * LN_VALID_RAMP


def ln_length_weight(length_ms: float, od: float) -> float:
    """单根长条的“长度权重” u ∈ [0, 1]：过短视为单键，足够长视为完整长条。

    u = smoothstep(ℓ; lo, hi)，(lo, hi) 见 ln_valid_band。ℓ ≤ lo 记 0（当米），
    ℓ ≥ hi 记 1，中间按 0.5 附近连续赋权——即在开头就把长度过短的 LN 当米处理。

    默认不启用（由 ln_len_weight_gain=0 关闭）：逐根赋权会把 benchmark 里 value 13~16 的
    真 LN 谱（长条时长中位 55~108ms，p10 低至 28ms，谱内混有大量短条）整体压掉，
    LN RMSE 0.8396→1.02 量级。默认改由 ln_hold_validity 的“中位数门”承担同一语义。
    """
    lo, hi = ln_valid_band(od)
    if hi - lo <= TIME_EPS:
        return 1.0 if float(length_ms) >= hi else 0.0
    t = (float(length_ms) - lo) / (hi - lo)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def ln_hold_validity(columns, od: float) -> float:
    """全谱 LN 时长有效性 v ∈ [0, 1]（无长条时为 0）——默认的短条判据。

    v = smoothstep(ℓ_med; lo, hi)，ℓ_med 为该谱长条时长的中位数，(lo, hi) 见 ln_valid_band。
    v 作用在全部四个 S 上（长条短到判定上等同单键时整层结构量清零），等价于
    “这张谱总体上是不是长条谱”。

    用中位数而非逐根赋权：benchmark 多数 LN 谱都混有一小撮短条，逐根赋权会被误伤
    （LN RMSE 0.8396→1.02 量级），而中位数能反映整谱性质。

    标定依据（benchmark-ln 102 张 vs 实测 vibro 反键图）：
      实测量          ℓ_med = 42ms（p90 也是 42），全谱起音间隔 21ms，n_rc = 107
      benchmark LN 谱 ℓ_med ≥ 55ms（中位 108ms），间隔 ≥ 44ms
    而 OD8 时 hi = 49.5ms、OD0 时 hi = 66ms，故 benchmark LN 谱全部落在门之上（零代价），
    实测图（OD0 时 hi=66 > 42）被清零。必须真正归零：其 n_rc 远超 LN 标定集的 n 上限
    （46.6），残余 k 会被 a·ln(1+b·k·n) 放大（v=0.056 时仍有 ln_value≈8.7）。
    """
    lens = [float(l) for c in columns for _t, l in c if l > 0]
    if not lens:
        return 0.0
    lo, hi = ln_valid_band(od)
    if hi - lo <= TIME_EPS:
        return 1.0 if float(np.median(lens)) >= hi else 0.0
    t = float(np.clip((np.median(lens) - lo) / (hi - lo), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def ln_release_absorb(columns, gain: float) -> float:
    """全谱松手吸收折扣因子 w = 1 - gain·ā ∈ [0, 1]（gain ≤ 0 时恒为 1）。

    ā 为全谱所有 LN 松手事件的“吸收度”均值，单个事件的吸收度
    a = clip(1 - d/(0.5·g), 0, 1)：d 为该松手时刻到最近“其它起音”的距离，g 为本地节拍
    （全谱正间隔中位数）。a = 1 表示松手正好落在另一个起音上——roll 反键里 LN 头挨得
    很近，OD9 判定窗下松手可当作两头中点、整段直接按高密度米打，不需要独立的控手/抓准。

    实测（冻结 RC 参数）：按全谱粒度打折比逐事件打折更好，也比“短 LN 且尾重合”的联合
    判据更好（λ=0.25 时 LN RMSE 0.8386 / 0.8455 / 0.8464），说明该现象是整谱性质而非
    局部性质；λ≥0.5 后整谱打折反而变差（0.8951）。
    """
    if gain <= 0.0:
        return 1.0
    onsets = _onsets_of(columns)
    diffs = np.diff(onsets)
    diffs = diffs[diffs > TIME_EPS]
    if onsets.size == 0 or diffs.size == 0:
        return 1.0
    gap = float(np.median(diffs))
    if gap <= TIME_EPS:
        return 1.0
    rels: list[np.ndarray] = []
    for col in columns:
        r = np.array([float(t) + float(l) for t, l in col if l > 0], dtype=np.float64)
        if r.size:
            rels.append(r)
    if not rels:
        return 1.0
    rel_times = np.concatenate(rels)
    k = np.searchsorted(onsets, rel_times)
    d = np.full(rel_times.size, np.inf, dtype=np.float64)
    for sh in (-1, 0):                      # 最近起音只可能在插入点或其左邻
        kk = np.clip(k + sh, 0, onsets.size - 1)
        d = np.minimum(d, np.abs(onsets[kk] - rel_times))
    a = float(np.mean(np.clip(1.0 - d / (ABSORB_DIST_SCALE * gap), 0.0, 1.0)))
    return 1.0 - min(max(gain, 0.0), 1.0) * a


def _ln_len_weight(length_ms: float, od: float, gain: float) -> float:
    """长度权重 w = 1 - gain·(1 - u(ℓ, od)) ；gain=0 时恒为 1（默认关闭）。"""
    if gain <= 0.0:
        return 1.0
    return 1.0 - min(max(gain, 0.0), 1.0) * (1.0 - ln_length_weight(length_ms, od))


def _ln_release_stats(col, od: float, gain: float, ref_ms: float):
    """该轨松手时刻 → (长度权重 w, 长度因子 h)，同一时刻多根长条先取长度均值。

    w = _ln_len_weight(ℓ, od, gain)（过短长条视为单键）
    h = ℓ / (ℓ + ref_ms)          （在“有效长条”内部恢复“越长越难”的单调性；
                                   ref_ms ≤ 0 或未启用时恒为 1）
    """
    acc: dict[float, list[float]] = {}
    for t, l in col:
        if l > 0:
            acc.setdefault(float(t) + float(l), []).append(float(l))
    out: dict[float, tuple[float, float]] = {}
    for tau, vs in acc.items():
        m = sum(vs) / len(vs)
        h = 1.0 if ref_ms <= 0.0 else m / (m + ref_ms)
        out[tau] = (_ln_len_weight(m, od, gain), h)
    return out


def _ln_event_scores(col, partner_col, rel_power, od: float, gain: float, ref_ms: float):
    """单轨 LN 松手事件的四项结构量 (S_rel, S_jack, S_cross, S_hold)；该轨无 LN 时返回 None。

    与 _ln_lane_contribs 共用同一套定义，但这里取“每事件原值”而非注入贡献：
      S_rel   = Lp(1000/δ_rr⁻, 1000/δ_rr⁺; q)   抓准：同轨相邻松手的间距倒数
      S_jack  = 1000/δ_rp                       控手：松手→同轨下一次按下
      S_cross = 1000·max(n−exp, 0)/ℓ            跨轨：超出本地基线期望的动作速率
      S_hold  = ℓ / 该轨物件平均间隔             保持时长（无量纲，跨越多少个本地节拍）
    注意 S_hold 按“每根 LN”给出（尺寸为长条数），其余三项按“每个松手分组”给出；
    三者在下游只做逐分量求均值，故尺寸无需一致。

    逐根赋权：w = _ln_len_weight(ℓ, od, gain)（长度过短的长条判定上等同单键，权重 0，
    等于“在开头就当米处理”）；S_rel / S_jack 再乘长度因子 h = ℓ/(ℓ+ref_ms)，
    以恢复“有效长条中越长越难”的单调性（纯长度门会把这条信息整段削掉）。
    """
    rels = sorted(float(t) + float(l) for t, l in col if l > 0)
    if not rels:
        return None
    groups = _group_lane([(r, 0) for r in rels])
    tt = np.array([g[0] for g in groups], dtype=np.float64)
    pd = np.array([-1.0 if g[2] is None else g[2] for g in groups], dtype=np.float64)
    nd = np.array([-1.0 if g[3] is None else g[3] for g in groups], dtype=np.float64)
    st = _ln_release_stats(col, od, gain, ref_ms)
    w = np.array([st.get(float(x), (0.0, 1.0))[0] for x in tt], dtype=np.float64)
    h = np.array([st.get(float(x), (0.0, 1.0))[1] for x in tt], dtype=np.float64)

    def _inv(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = INV_DIST_SCALE / d
        return np.where(d > TIME_EPS, v, 0.0)

    s_rel = _merge_vec(_inv(pd), _inv(nd), rel_power) * w * h

    s_jack = np.zeros(tt.size, dtype=np.float64)
    presses = np.array(sorted(float(t) for t, _ in col), dtype=np.float64)
    if presses.size:
        k = np.searchsorted(presses, tt, side="right")
        d = np.full(tt.size, -1.0, dtype=np.float64)
        ok = k < presses.size
        d[ok] = presses[k[ok]] - tt[ok]
        s_jack = _inv(d) * w * h

    ln_w = np.array([_ln_len_weight(l, od, gain) for _t, l in col if l > 0], dtype=np.float64)
    s_cross = np.zeros(tt.size, dtype=np.float64)
    if partner_col:
        pacts = sorted([float(t) for t, _ in partner_col]
                       + [float(t) + float(l) for t, l in partner_col if l > 0])
        if pacts:
            pt = np.asarray(pacts, dtype=np.float64)
            M = pt.size
            hs = np.array([float(t) for t, l in col if l > 0], dtype=np.float64)
            he = np.array([float(t) + float(l) for t, l in col if l > 0], dtype=np.float64)
            kb = np.searchsorted(pt, hs, side="right") - 1
            ka = np.searchsorted(pt, he, side="left")
            n_ev = np.maximum(ka - kb - 1, 0).astype(np.float64)
            has_b = kb >= 0
            has_a = ka < M
            gap_b = np.zeros(hs.size, dtype=np.float64)
            gap_a = np.zeros(hs.size, dtype=np.float64)
            if has_b.any():
                gap_b[has_b] = hs[has_b] - pt[kb[has_b]]
            if has_a.any():
                gap_a[has_a] = pt[ka[has_a]] - he[has_a]
            good_b = has_b & (gap_b > TIME_EPS)
            good_a = has_a & (gap_a > TIME_EPS)
            rate = np.zeros(hs.size, dtype=np.float64)   # 本地基线动作率（动作/ms）
            both = good_b & good_a
            rate[both] = 2.0 / (gap_b[both] + gap_a[both])
            only_b = good_b & ~good_a
            rate[only_b] = 1.0 / gap_b[only_b]
            only_a = good_a & ~good_b
            rate[only_a] = 1.0 / gap_a[only_a]
            length = he - hs
            ok = (rate > 0.0) & (length > TIME_EPS)
            # 超出基线期望的动作数，换算成与 S_rel/S_jack 同量纲的速率
            s_cross = np.zeros(hs.size, dtype=np.float64)
            s_cross[ok] = (INV_DIST_SCALE * np.maximum(n_ev[ok] - rate[ok] * length[ok], 0.0)
                           / length[ok]) * ln_w[ok]

    # S_hold：保持时长 / 该轨物件平均间隔（无量纲，衡量一根长条跨越多少个本地节拍）。
    # 保持“相对本地节奏”的长度而非绝对毫秒，避免又一次引入密度代理。
    lens = np.array([float(l) for _t, l in col if l > 0], dtype=np.float64)
    base_gap = float(np.mean(np.diff(presses))) if presses.size > 1 else 0.0
    if base_gap > TIME_EPS and lens.size:
        s_hold = lens / base_gap * ln_w
    else:
        s_hold = np.zeros(lens.size, dtype=np.float64)
    return s_rel, s_jack, s_cross, s_hold


def ln_hold_ratio(columns, od: float, gain: float = 0.0) -> float:
    """长条占比 ρ = Σ 长度权重 / 物件总数（无物件时为 0）。

    gain > 0 时“长度过短的长条”权重为 0（等同单键）、“中等”按 0~1 赋权，
    见 ln_length_weight / _ln_len_weight；默认 gain = 0 即普通计数占比。
    """
    total = sum(len(c) for c in columns)
    if total == 0:
        return 0.0
    holds = sum(_ln_len_weight(l, od, gain)
                for c in columns for _t, l in c if l > 0)
    return holds / total


def ln_structure_scores(columns, params: dict[str, float], od: float):
    """谱面的 LN 结构得分 (S_rel, S_jack, S_cross, S_hold)：全谱 LN 事件上的均值。

    只描述“每根 LN 有多强”，不含“有多少 LN”——后者由 ln_multiplier 里的 ρ 因子
    单独承担（ρ = 长条占比）。无 LN 时返回 (0.0, 0.0, 0.0, 0.0)。
    """
    rel_power = float(params.get("ln_rel_power", 2.0))
    if rel_power <= 0.0:
        raise ValueError("ln_rel_power 必须 > 0")
    gain = float(params.get("ln_len_weight_gain", 0.0))
    ref_ms = float(params.get("ln_hold_ref_ms", 80.0))
    parts: list[tuple] = []
    for i, col in enumerate(columns):
        start = i - (i % 2)
        hand = columns[start:start + 2]
        partner = None if len(hand) < 2 else (hand[1] if i == start else hand[0])
        got = _ln_event_scores(col, partner, rel_power, od, gain, ref_ms)
        if got is not None:
            parts.append(got)
    if not parts:
        return 0.0, 0.0, 0.0, 0.0
    s_rel, s_jack, s_cross, s_hold = (
        float(np.mean(np.concatenate([p[j] for p in parts]))) for j in range(4))
    # 松手吸收折扣：整谱粒度地降低“抓准 / 控手”两项（S_cross 描述保持期间配对轨的动作，
    # S_hold 描述长度，都与松手是否被邻近起音吸收无关）。
    w = ln_release_absorb(columns, float(params.get("ln_absorb_gain", 0.25)))
    # 时长有效性门：长条短到判定上等同单键时，整层结构量清零（见 ln_hold_validity）。
    v = ln_hold_validity(columns, od)
    return s_rel * w * v, s_jack * w * v, s_cross * v, s_hold * v


def ln_rho_saturation(rho: float, h: float, c: float) -> float:
    """纯 Hill 型饱和函数 F(ρ) = (1+c^h)·ρ^h/(ρ^h+c^h)（lin=0 时的 ln_rho_curve）。

    满足 F(0)=0、F(1)=1、单调不减；c 很大时退化为幂律 ρ^h，c 很小则退化为阈值开关。
    """
    return ln_rho_curve(rho, h, c, 0.0)


def ln_rho_curve(rho: float, h: float, c: float, lin: float = 0.0) -> float:
    """ρ 饱和曲线 F(ρ) = lin·ρ + (1-lin)·(1+c^h)·ρ^h/(ρ^h+c^h)。

    纯 Hill（lin=0）在 ρ≪c 段按 ρ^h 塌向 0，实战中“长条占比很小但确有 LN 难度”的
    谱面会被直接归零。叠一条线性 toe（lin>0）后低 ρ 段近似线性（斜率 lin），
    既保留 ρ 落在 (非 LN 谱 max≈0.14, LN 谱 min≈0.25) 之间时的陡峭过渡，
    又不会把少量长条的谱面压到 0。仍满足 F(0)=0、F(1)=1、单调不减。

    实测（冻结 RC 参数，h=6、c=0.33）：lin=0.10 时 LN RMSE 与纯 Hill 持平
    （0.8549 vs 0.8547），非 LN 谱尾部 max 0.156→0.735；lin=0.20 才略升到 0.8576。
    """
    if rho <= 0.0:
        return 0.0
    k = min(max(lin, 0.0), 1.0)
    rh = rho ** h
    ch = c ** h
    return k * rho + (1.0 - k) * (1.0 + ch) * rh / (rh + ch)


def ln_multiplier(columns, params: dict[str, float], od: float, rate: float = 1.0) -> float:
    """ln 倍率 k = F(ρ) · (γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross + γ_hold·S_hold)。

    无常数项；F(ρ) 为 ρ 饱和曲线（见 ln_rho_curve）：
    h = ln_mult_exp（陡度，越大越接近阈值开关），c = ln_mult_rho_c（半饱和位置），
    lin = ln_mult_rho_lin（低 ρ 段线性 toe 权重，>0 时少量长条的谱面不再被归零）。

    rate 为变速倍率：变速时 S_rel / S_jack 的“间距倒数”随 1/rate 缩小，
    故这两项乘 rate 抵消（与 tosu 插件一致）；S_cross / S_hold 与时间缩放无关，不调整。

    选形态的依据（见 ln_benchmark 的形态对比实验）：幂律 ρ^m 在高 ρ 区仍在
    快速变化，而 LN 谱的残差与 ρ 正相关（说明模型对 ρ 的响应偏强）；Hill 型在高 ρ
    区趋平、在低 ρ 区陡降，因而**同时**改善 LN 谱拟合并压低“只带少量长条”的谱面。
    实测（h=4, c=0.4）：LN RMSE 1.002→0.884，非 LN 谱 ln_value 最大 7.95→0.63。

    S_hold（保持时长 / 该轨物件平均间隔）是第四项：诊断显示“长保持”在现有三项下
    被系统性低估。实验对比过三种归一化，只有“相对本地节拍”这一种有效
    （LN RMSE 0.884→0.864，三个特征桶同时改善），而 1000/ℓ 会被搜索归零。
    加权和须 ≥ 0（各 S 与权重均为非负时天然成立，此处兜住 --set 传入负权重的情形）。
    """
    g_rel = float(params.get("ln_mult_rel", 0.0))
    g_jack = float(params.get("ln_mult_jack", 0.0))
    g_cross = float(params.get("ln_mult_cross", 0.0))
    g_hold = float(params.get("ln_mult_hold", 0.0))
    if g_rel == 0.0 and g_jack == 0.0 and g_cross == 0.0 and g_hold == 0.0:
        return 0.0
    rho = ln_hold_ratio(columns, od, float(params.get("ln_len_weight_gain", 0.0)))
    if rho <= 0.0:
        return 0.0
    h = float(params.get("ln_mult_exp", 6.0))
    c = float(params.get("ln_mult_rho_c", 0.33))
    lin = float(params.get("ln_mult_rho_lin", 0.1))
    if h <= 0.0 or c <= 0.0:
        raise ValueError("ln_mult_exp 与 ln_mult_rho_c 必须 > 0")
    s_rel, s_jack, s_cross, s_hold = ln_structure_scores(columns, params, od)
    total = (g_rel * (s_rel * rate) + g_jack * (s_jack * rate)
             + g_cross * s_cross + g_hold * s_hold)
    if total < 0.0:
        raise ValueError("ln 倍率的加权和必须 ≥ 0（各 S 与权重均应为非负）")
    return ln_rho_curve(rho, h, c, lin) * total


def ln_log_value(n_rc: float, k: float, a: float, b: float) -> float:
    """含下界对数标定：ln_value = a·ln(b·k·n_rc + 1)（n_LN = k·n_rc）。"""
    return a * math.log1p(b * k * n_rc)


def fit_ln_log(n_values, values, b_lo: float = 1e-4, b_hi: float = 10.0,
               grid: int = 400) -> tuple[float, float, float]:
    """拟合 ln_value = a·ln(b·n + 1)，返回 (a, b, R²)。

    给定 b 时 a 有闭式最小二乘解 a = Σy·L / ΣL²（L = ln(b·n+1)），
    故只需对 b 做一维网格搜索。n = 0 的点 L = 0，对拟合无贡献，
    其预测值天然为 0（含下界语义）。
    """
    n = np.asarray(n_values, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if n.size == 0:
        return 0.0, 1.0, 0.0

    def resolve(b: float) -> tuple[float, float]:
        L = np.log1p(b * n)
        den = float(L @ L)
        if den <= 0.0:
            return 0.0, float("inf")
        a = float(L @ y) / den
        return a, float(np.sum((a * L - y) ** 2))

    bs = np.logspace(math.log10(b_lo), math.log10(b_hi), int(grid))
    costs = [resolve(b)[1] for b in bs]
    b_best = float(bs[int(np.argmin(costs))])
    a_best, cost = resolve(b_best)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - cost / ss_tot if ss_tot > 0.0 else 1.0
    return a_best, b_best, r2

@register_algorithm(
    key="object_sequence",
    title="物件赋值序列（逐物件 0.5^(dt/H) 衰减）",
    params=(
        ParamSpec(
            "half_life_ms",
            1000.0,
            "衰减半衰期 H (ms)",
            "相邻物件间隔 dt(ms) 时衰减系数为 0.5^(dt/H)，对应原时间序列算法中的 1000",
        ),
        ParamSpec(
            "quantile",
            0.96,
            "n 值目标均值 / 分位数初值",
            "求 n 使 mean(min(1, n/a)) 等于该值，并以该分位数作为二分初值上界",
        ),
        ParamSpec(
            "extra_rate",
            0.1,
            "额外 contribution 倍率",
            "配对轨 4a 邻域耦合项求和后以该倍率加在原 contribution 上（0 关闭）",
        ),
        ParamSpec(
            "extra_b",
            2.0,
            "邻域耦合项偏移 b",
            "额外 contribution 逐键为 exp(-((log(x/a)+b)^2)/2)，x 为与配对轨键的时间差，a = 1000/原contribution",
        ),
        ParamSpec(
            "extra_log_base",
            2.0,
            "邻域耦合项对数底数",
            "邻域耦合项中 log 的底数（默认 2），须 > 0 且 ≠ 1",
        ),
        ParamSpec(
            "merge_power",
            2.0,
            "双轨合并幂次 i",
            "配对双轨 (1,2)/(3,4) 合并为 (x^i + y^i)^(1/i)：i=2 即 hypot 平方开方，i=1 为直接相加，须 > 0",
        ),
        ParamSpec(
            "merge2_power",
            2.0,
            "两合并序列合并幂次 i2",
            "两条配对合并序列视为赋值突变衰减函数，在彼此时间戳插值采样后按 (x^i2+y^i2)^(1/i2) 二次合并，须 > 0",
        ),
        ParamSpec(
            "contrib_power",
            2.0,
            "初始贡献合并范数幂次 p",
            "初始 contribution 的前后间距倒数按 (x^p+y^p)^(1/p) 合并（Lp 范数），须 > 0",
        ),
        ParamSpec(
            "cross_lane_mix",
            0.0,
            "跨轨混合系数 α",
            "贡献间隔 = (1-α)·同轨间距 + α·全局起音间距（∈[0,1]，连续无分段）；"
            "0 表示仅用同轨间距（旧行为），增大可提升跨轨交替流、抑制同轨重复",
        ),
        ParamSpec(
            "soft_beta",
            0.0,
            "合并算子 softmax 系数 β",
            ">0 时合并算子改为 softmax 加权平均 w·a+(1-w)·b（w=½(1+tanh(½β(a-b)))），"
            "即光滑的 max，连续无分段；β 越大越接近 max，=0 时使用 Lp 范数合并",
        ),
        ParamSpec(
            "irr_local_gain",
            0.0,
            "逐键局部不规则度增益 γ（全谱熵 @ 贡献层，旧单通道）",
            "≠0 时把每个起音的局部邻域熵 irr_i 作为乘子并入注入贡献："
            "contribution *= exp(γ·irr_i)（在额外 contribution 层之后），"
            "熵的统计范围是全谱合并去重后的起音序列。γ=0 完全关闭该层、零开销。",
        ),
        ParamSpec(
            "irr_gain_lane",
            0.0,
            "单轨熵增益 γ_l（贡献层）",
            "以「当前轨自身起音序列」的逐键局部熵作乘子：contribution *= exp(γ_l·irr_lane(t))。"
            "γ=0 零开销。",
        ),
        ParamSpec(
            "irr_gain_pair",
            0.0,
            "轨道对熵增益 γ_p（配对合并层）",
            "以「当前轨道对 (0,1)/(2,3) 合并起音序列」的逐键局部熵作乘子，"
            "作用于第一级配对合并后的曲线：v(t) *= exp(γ_p·irr_pair(t))。γ=0 零开销。",
        ),
        ParamSpec(
            "irr_gain_global",
            0.0,
            "全谱熵增益 γ_g（最终合并层）",
            "以「全谱合并去重起音序列」的逐键局部熵作乘子，作用于第二级最终合并后的序列："
            "v(t) *= exp(γ_g·irr_global(t))。γ=0 零开销。",
        ),
        ParamSpec(
            "ln_gain",
            0.0,
            "LN 倍率增益 γ",
            "LN 倍率 k = 1 + γ·(n_ch / n_rc)，把 RC 求得的 n 乘性放大为 LN 的 n："
            "n_LN = k·n_rc = n_rc + γ·n_ch（n_ch 为 LN 通道强度，其内部已由"
            "松手间距 / 松手→重按 / 跨轨交互 / 按住占用各特征加权）。"
            "γ=0（默认）时 k=1、n_LN=n_rc，行为与旧版逐位一致（零回归）。",
        ),
        ParamSpec(
            "ln_rel_power",
            2.0,
            "LN 松手间距幂次 q",
            "LN 通道内相邻松手间距倒数按 Lp 范数 (x^q + y^q)^(1/q) 合并，须 > 0。",
        ),
        ParamSpec(
            "ln_jack_gain",
            1.0,
            "LN 松手→重按增益",
            "松手后同轨下一次按下的间隔 δ_rp 的倒数 (1000/δ_rp) 以该增益加进松手事件贡献，"
            "刻画“松手后须立刻再按”的控手压力；0 表示只用松手间距（纯抓准）。",
        ),
        ParamSpec(
            "ln_hold_gain",
            0.0,
            "LN 按住占用增益 γ",
            "松手时刻同手仍被其他 LN 按住的键数 occ 以 exp(γ·occ) 乘进贡献，"
            "刻画持续按住张力；γ=0（默认）关闭该层、零开销。",
        ),
        ParamSpec(
            "ln_cross_gain",
            0.0,
            "LN 按住下跨轨交互增益",
            "本轨长条按住期间、同手另一轨的“实际动作数 / 本地基线期望”比值以该增益加进该长条"
            "松手事件的贡献。比值为无量纲量（均匀谱 ≈ 1，越大表示按住时另一轨越异常繁忙）；"
            "基线率由按住段两侧最近动作的间隔估计（不含段内动作），期望 = 基线率 × 保持时长，"
            "动作数计入按压与松开；同刻多根长条按比值叠加；0（默认）关闭该层、零开销。",
        ),
        ParamSpec(
            "ln_mult_rel",
            0.0,
            "LN 倍率·抓准权重 γ_rel",
            "ln 倍率中“相邻松手间距倒数”这一项的权重（S_rel 为全谱 LN 松手事件上的均值，"
            "单位 1000/ms）；k = ρ^m·(γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross)，"
            "**不含常数项**，无 LN 的谱面 ρ = 0 ⇒ k = 0 ⇒ n_LN = 0 ⇒ ln_value = 0。"
            "0（默认）时该项不参与。",
        ),
        ParamSpec(
            "ln_mult_jack",
            0.0,
            "LN 倍率·控手权重 γ_jack",
            "ln 倍率中“松手→同轨下一次按下”这一项的权重"
            "（S_jack 为全谱 LN 松手事件上的均值，单位 1000/ms）。",
        ),
        ParamSpec(
            "ln_mult_cross",
            0.0,
            "LN 倍率·跨轨权重 γ_cross",
            "ln 倍率中“跨轨超额动作速率”这一项的权重"
            "（S_cross 为全谱 LN 松手事件上的均值，单位 1000/ms）。",
        ),
        ParamSpec(
            "ln_mult_hold",
            0.0,
            "LN 倍率·保持时长权重 γ_hold",
            "ln 倍率中“保持时长 / 该轨物件平均间隔”这一项的权重"
            "（S_hold 为全谱 LN 事件上的均值，无量纲：一根长条跨越多少个本地节拍）。"
            "诊断显示长保持被系统性低估，需正权重；0（默认）时该项不参与。",
        ),
        ParamSpec(
            "ln_mult_exp",
            6.0,
            "LN 倍率·ρ 饱和陡度 h",
            "长条占比的 Hill 型饱和函数 F(ρ) = (1+c^h)·ρ^h/(ρ^h+c^h) 的陡度（须 > 0）："
            "h 越大越接近阈值开关；c（ln_mult_rho_c）很大时退化为幂律 ρ^h。"
            "默认 6.0 由 (h,c) 二维扫描确定：h 在 4.5~7 之间 LN RMSE 几乎不变（0.849~0.858），"
            "但非 LN 谱尾部随 h 单调收窄，故取偏大值。",
        ),
        ParamSpec(
            "ln_mult_rho_c",
            0.33,
            "LN 倍率·ρ 半饱和位置 c",
            "Hill 函数 F(ρ) 的半饱和位置（须 > 0，默认 0.33，由二维扫描确定）："
            "ρ ≳ c 时 F 趋于饱和（典型 LN 谱几乎不受参数影响），"
            "ρ ≪ c 时 F 迅速趋 0（强烈压低“只带少量长条”的谱面）。",
        ),
        ParamSpec(
            "ln_mult_rho_lin",
            0.1,
            "LN 倍率·ρ 线性 toe 权重 lin",
            "F(ρ) = lin·ρ + (1-lin)·Hill(ρ)（∈[0,1]，须 ∈[0,1]）。"
            "纯 Hill（lin=0）在低 ρ 段按 ρ^h 塌向 0，会把“长条占比很小但确有 LN 难度”的"
            "谱面直接归零；lin>0 时低 ρ 段近似线性（斜率 lin），少量长条也能拿到非零 ln 难度。"
            "实测 lin=0.10 时 LN RMSE 与 lin=0 持平（0.8549 vs 0.8547），"
            "lin=0.20 才略升到 0.8576（非 LN 谱尾部 max 0.156→1.392）。",
        ),
        ParamSpec(
            "ln_absorb_gain",
            0.25,
            "LN 松手吸收折扣 λ",
            "∈[0,1]。roll 反键中 LN 头挨得很近，松手时刻距邻近起音不足半个本地节拍时，"
            "在 OD9 判定窗下可直接按两头中点打（当成高密度米），不需要独立控制；"
            "此时按 w = 1 - λ·a 逐事件降低该次松手对 S_rel / S_jack 的贡献（a∈[0,1] 为吸收度）。"
            "实测 λ=0 时 LN RMSE 0.8545、控手桶 1.104；λ=0.25 时 0.8386、控手桶 1.096；"
            "λ≥0.5 反而变差。",
        ),
        ParamSpec(
            "ln_len_weight_gain",
            1.0,
            "LN 长度权重增益（逐根赋权）",
            "∈[0,1]。按 ln_length_weight 给每根长条连续赋权 w = 1 - gain·(1 - u(ℓ, od))，"
            "u = smoothstep(ℓ; lo, hi)，lo = LN_MIN_HOLD_MS + LN_VALID_OD_GAIN·(W_head+W_tail)、"
            "hi = LN_VALID_RAMP·lo——长度过短的 LN 直接当米、中等长度按 0~1 赋权。"
            "该权重同时作用于 ρ 与四个 S。实测（冻结 RC 参数）gain=1 时 LN RMSE 0.9049、"
            "实测 vibro 反键图 ln_value 22.6→0.000；gain=0 退化为同区间的“中位数门”"
            "（ln_hold_validity），benchmark LN RMSE 0.8396（零代价）但只按整谱中位数判定。",
        ),
        ParamSpec(
            "ln_hold_ref_ms",
            40.0,
            "LN 长度因子参考时长 ℓ₀",
            "S_rel / S_jack 逐根乘 h = ℓ/(ℓ+ℓ₀)（≤0 时关闭）。纯长度门会把“越长越难”"
            "这条信息整段削掉（残差诊断里 lenw_mean 是最大相关项 +0.225），h 在“有效长条”"
            "内部恢复该单调性：ℓ≫ℓ₀ 时 h→1，ℓ≪ℓ₀ 时 h→0。"
            "实测（冻结 RC 参数，含合成谱拟合）：ℓ₀=0/40/80/150 时 LN RMSE 0.8684/0.8690/"
            "0.8750/0.8863，非合成谱 0.567/0.559/0.564/0.575，控手桶 1.100/1.094/1.062/1.043"
            "——ℓ₀ 越大控手桶越好但密度/混合桶与整体越差，取折中最省点 40ms。",
        ),
    ),
    description=(
        "将原 1ms 步长时间序列 IIR 简化为按物件递推：value = value*0.5^(dt/H) + contribution；"
        "双轨 (1,2)/(3,4) 按 (x^i+y^i)^(1/i) 合并（默认 i=2 即 hypot）得到两条合并赋值序列；"
        "两序列视为赋值处有突变的时间衰减函数，在彼此时间戳插值采样后按 (x^i2+y^i2)^(1/i2) "
        "二次合并（默认 i2=2），同一时间戳的相同值个数等于该时刻物件个数，最后求 n。"
        "初始 contribution 的前后间距倒数与重叠惩罚按 Lp 范数 (x^p+y^p+o^p)^(1/p) 合并（默认 p=2）；"
        "额外 contribution 乘以原始 contribution 后以倍率 extra_rate 叠加进 contribution"
        "（偏移 extra_b，对数底数 extra_log_base）；"
        "逐键局部不规则度可按统计范围分三层介入：单轨熵@贡献层（irr_gain_lane）、"
        "轨道对熵@配对合并层（irr_gain_pair）、全谱熵@最终合并层（irr_gain_global），"
        "另有旧单通道 irr_local_gain（全谱熵@贡献层）；三者皆 0 时零开销。"
        "LN 通道（ln_gain ≠ 0 时启用）：以 LN 松手时刻（t+长度）为注入点，贡献取相邻松手间距"
        "倒数（抓准）、松手→同轨下一次按下间距倒数（控手）、按住期间同手另一轨动作数相对"
        "本地基线的超额比（跨轨交互）、可选再乘同手按住占用因子，经同一套两级合并求 n_ln，"
        "最后与 RC 的 n_rc 以 Lp 合并；ln_gain=0 时零开销。"
    ),
)
def algo_object_sequence(columns, params: dict[str, float]) -> float:
    half_life_ms = float(params["half_life_ms"])
    quantile = float(params["quantile"])
    extra_rate = float(params["extra_rate"])
    extra_b = float(params["extra_b"])
    extra_log_base = float(params["extra_log_base"])
    merge_power = float(params["merge_power"])
    merge2_power = float(params["merge2_power"])
    contrib_power = float(params["contrib_power"])
    cross_lane_mix = float(params.get("cross_lane_mix", 0.0))
    soft_beta = float(params.get("soft_beta", 0.0))
    ln_gain = float(params.get("ln_gain", 0.0))
    ln_merge_power = float(params.get("ln_merge_power", 2.0))
    irr_local_gain = float(params.get("irr_local_gain", 0.0))
    irr_gain_lane = float(params.get("irr_gain_lane", 0.0))
    irr_gain_pair = float(params.get("irr_gain_pair", 0.0))
    irr_gain_global = float(params.get("irr_gain_global", 0.0))
    if extra_log_base <= 0.0 or extra_log_base == 1.0:
        raise ValueError("extra_log_base 必须 > 0 且 ≠ 1")
    if merge_power <= 0.0:
        raise ValueError("merge_power(i) 必须 > 0")
    if merge2_power <= 0.0:
        raise ValueError("merge2_power(i2) 必须 > 0")
    if contrib_power <= 0.0:
        raise ValueError("contrib_power(p) 必须 > 0")
    if not (0.0 <= cross_lane_mix <= 1.0):
        raise ValueError("cross_lane_mix 必须 ∈ [0, 1]")
    if soft_beta < 0.0:
        raise ValueError("soft_beta 必须 ≥ 0（0 表示使用 Lp 范数合并）")
    if ln_merge_power <= 0.0:
        raise ValueError("ln_merge_power 必须 > 0")

    lane_groups = [_group_lane(col) for col in columns]
    k_total = len(columns)

    # 逐键局部不规则度：各增益 ≠0 时才计算对应统计范围（带内容缓存；全部为 0 时零开销）
    #   loc_*   : 全谱合并去重（旧单通道 irr_local_gain，作用于贡献层）
    #   lane_irr: 单轨自身起音序列（贡献层）
    #   pair_irr: 轨道对合并起音序列（配对合并层）
    #   glob_irr: 全谱合并去重（最终合并层）
    loc_onsets = loc_entropy = None
    if irr_local_gain != 0.0:
        loc_onsets, loc_entropy = local_irregularity_cached(columns)
    lane_irr = ([local_irregularity_cached([columns[i]]) for i in range(k_total)]
                if irr_gain_lane != 0.0 else None)
    pair_irr = ([local_irregularity_cached(columns[s:s + 2]) for s in range(0, k_total, 2)]
                if irr_gain_pair != 0.0 else None)
    glob_irr = local_irregularity_cached(columns) if irr_gain_global != 0.0 else None

    # 全局（全轨去重）起音序列的前后间距，供贡献层的跨轨混合使用。
    # cross_lane_mix = 0 时混合项不参与计算，跳过构建（省去逐组字典查询与大量元组分配）
    gaps_get = _EMPTY_GAPS.get
    if cross_lane_mix != 0.0:
        all_onsets = sorted({float(t) for col in columns for t, _ in col})
        global_gaps: dict[float, tuple[float | None, float | None]] = {}
        for oi, ot in enumerate(all_onsets):
            pg = all_onsets[oi] - all_onsets[oi - 1] if oi > 0 else None
            ng = all_onsets[oi + 1] - all_onsets[oi] if oi + 1 < len(all_onsets) else None
            global_gaps[ot] = (pg, ng)
        gaps_get = global_gaps.get

    # 预计算每轨每组物件的注入贡献（额外 contribution 层位于原 contribution
    # 与赋值序列生成之间）：
    #   contribution = 原 contribution + extra_rate * 原 contribution * 配对轨邻域额外 contribution
    lane_contribs = []
    for i, groups_i in enumerate(lane_groups):
        mate = i + 1 if i % 2 == 0 else i - 1  # 4K：(0,1)、(2,3) 配对
        if not (0 <= mate < k_total):
            mate = i
        lane_contribs.append(_lane_contribs_vec(
            groups_i, lane_groups[mate], contrib_power, cross_lane_mix, soft_beta,
            extra_rate, extra_b, extra_log_base, gaps_get,
            loc_onsets, loc_entropy, irr_local_gain))

    # 单轨熵 → 贡献层
    if lane_irr is not None:
        for i in range(k_total):
            tt_i = np.array([g[0] for g in lane_groups[i]], dtype=np.float64)
            lane_contribs[i] = lane_contribs[i] * np.exp(
                irr_gain_lane * entropy_at(lane_irr[i][0], lane_irr[i][1], tt_i))

    # 第一/二级合并 + 求 n（RC 通道）
    # 第一级：配对双轨 (1,2)/(3,4) 各自合并为一条赋值序列（value 为两轨曲线在该
    #         时刻的合并值，count 为该时刻两轨物件总数，故同一时间戳的相同值个数
    #         = 物件个数）；第二级：各配对序列视为“赋值处有突变的时间衰减函数”，
    #         在彼此时间戳上插值采样后以 merge2_power 两两合并。
    n_rc = _merge_sequence_n(lane_groups, lane_contribs, merge_power, merge2_power, soft_beta,
                             half_life_ms, quantile, pair_irr, irr_gain_pair,
                             glob_irr, irr_gain_global)

    # LN 通道：LN 松手流（抓准）+ 松手→重按 + 同手按住占用（控手），
    # 与 RC 通道以 Lp 范数合并：n_total = (n_rc^p + (κ·n_ln)^p)^(1/p)
    # κ = 0 时直接返回 n_rc，行为与旧版逐位一致；纯 rice 谱的 n_ln = 0，同样不受影响。
    if ln_gain == 0.0:
        return n_rc
    n_ln = _ln_channel_n(columns, params)
    if n_ln <= 0.0:
        return n_rc
    return _lp_norm(n_rc, ln_gain * n_ln, ln_merge_power)


# ===========================================================================
# 连续映射曲线（数据拟合；两端线性渐近、全程光滑、无分段、严格单调）
#
# 以「分段斜率」σ_0 … σ_K 为参数（K 个衔接点 → K+1 段）：
#   g(v) = c + Σ_j σ_j · ψ_j(v)
#   ψ_0(v) = v − φ_0(v)；ψ_j(v) = φ_{j-1}(v) − φ_j(v)（1 ≤ j ≤ K−1）；ψ_K(v) = φ_{K-1}(v)
#   φ_i(v) = s_i · softplus((v − v0_i)/s_i)，softplus(z)=ln(1+e^z)
# v → −∞ 时 g ≈ c + σ_0·v；v → +∞ 时 g ≈ (…) + σ_K·v，两端均为线性渐近。
# 单调递增等价于全部 σ_j > 0；对 σ_j 施加下界（MAP_SLOPE_FLOOR）既保证单调，
# 又避免某段斜率过平导致该区间反解方差被放大。
# 低 value 段额外设置四个转折点（MAP_LOW_KNOTS），使低段不再是单一直线外推，
# 而是一条可弯折的渐进曲线，用于抵消低 value 区间（dan 整数化标定）的残余。
# 衔接点位置/宽度由坐标下降对 RMSE 搜索得到（非等分）；宽度下限 0.9 以保持光滑。
# ===========================================================================
MAP_LOW_KNOTS = ((0.3, 1.6), (2.45, 1.6), (3.85, 0.9), (4.5, 2.0))  # 仅低 value 段的四个转折点
MAP_HIGH_KNOTS = ((7.1, 1.6), (12.05, 0.9), (12.65, 0.9))
MAP_KNOTS = MAP_LOW_KNOTS + MAP_HIGH_KNOTS   # 衔接点 (v0, 宽度 s)；K 个衔接 → K+1 段
MAP_FIT_RANGE = (2.0, 17.0)                  # 曲线拟合使用的 value 范围（含低段以标定四个转折点）
MAP_SLOPE_FLOOR = 0.06                       # 段斜率下界：保证单调且避免斜率塌陷放大方差
MAP_SLOPE_CEIL = 0.25                        # 段斜率上界：避免某段（尤其低端渐近线）过陡


def _softplus(z: float) -> float:
    return float(np.logaddexp(0.0, z))


def _phi(v: float, v0: float, s: float) -> float:
    """单个 softplus 转折基 s·softplus((v−v0)/s)。"""
    return s * _softplus((v - v0) / s)


def map_basis(v: float, knots=None) -> list[float]:
    """σ 空间的设计基：[1, ψ_0(v), …, ψ_K(v)]，g(v) = c + Σ σ_j ψ_j(v)。"""
    knots = MAP_KNOTS if knots is None else knots
    phi = [_phi(v, v0, s) for v0, s in knots]
    K = len(knots)
    w = [v - phi[0]]
    for j in range(1, K):
        w.append(phi[j - 1] - phi[j])
    w.append(phi[K - 1])
    return [1.0] + w


def map_forward(fit: dict, v: float) -> float:
    """映射曲线 g(v)（连续、两端线性渐近、严格单调）。"""
    knots = fit.get("map_knots", MAP_KNOTS)
    basis = map_basis(v, knots)
    nlead = 1 + len(knots) + 1  # c + σ_0..σ_K
    return sum(c * b for c, b in zip(fit["map_coef"][:nlead], basis))


def map_invert(fit: dict, target: float) -> float:
    """解 g(v) = target（g 单调 → 二分），返回原始 value。"""
    lo, hi = -60.0, 120.0
    for _ in range(90):
        mid = (lo + hi) * 0.5
        if map_forward(fit, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5



# ===========================================================================
# 不规则度特征（纯时间戳，不需要 BPM / 节拍元数据）
#
# 取全体轨道去重起音序列的相邻间距 g；对每个滑窗：x = log2(g)，
# z = (x - min) / (max - min)（窗口归一化），用高斯核估计其微分熵（核密度交叉熵）
#     H = -mean_i log( (1/n) Σ_j N(z_i; z_j, h) )
# 再对所有滑窗取均值。它衡量“局部间隔取值的多样性”，全程连续、无分箱、无分段。
# ===========================================================================
IRR_WINDOW = 32        # 全局滑窗包含的间隔数
IRR_STEP = 8           # 全局滑窗步长
IRR_BANDWIDTH = 0.08   # 归一化 log2 间隔坐标下的高斯核带宽
IRR_LOCAL_WINDOW = 16  # 逐键局部邻域窗口（间隔数）
IRR_LOCAL_STRIDE = 4   # 逐键局部熵的滑窗步长（批量加速；>1 时档间线性插值回逐键）
_IRR_CACHE: dict = {}
_IRR_LOCAL_CACHE: dict = {}

_INV2H2 = 1.0 / (2.0 * IRR_BANDWIDTH * IRR_BANDWIDTH)
_LOG_DEN = math.log(IRR_BANDWIDTH * math.sqrt(2.0 * math.pi))


def _window_entropies(lg, window: int, stride: int, f32: bool = True, chunk: int = 4096):
    """对 log2 间隔序列做滑窗，返回每窗的 KDE 交叉熵及其起始下标（向量化、分块）。

    窗口熵 = -mean_i log( (1/W)·Σ_j exp(-½((z_i−z_j)/h)²) / den )
           = log(den) − mean_i log( (1/W)·Σ_j exp(-½((z_i−z_j)/h)²) )
    z 为窗口内归一化 log2 间隔。核密度均值 ≥ 1/W > 0，故 log 恒有定义（无需截断）。
    批量实现：一次构造 (块, W, W) 张量，避免 Python 逐窗循环。
    """
    n = lg.size
    if n < window:
        return np.zeros(0, dtype=np.float32 if f32 else np.float64), np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n - window + 1, stride)
    dt = np.float32 if f32 else np.float64
    coef, log_den = dt(-0.5 * _INV2H2), dt(_LOG_DEN)
    view = np.lib.stride_tricks.sliding_window_view(lg, window)
    out = np.empty(starts.size, dtype=dt)
    for b in range(0, starts.size, chunk):  # 分块以限制 (M, W, W) 峰值内存
        win = view[starts[b:b + chunk]].astype(dt, copy=False)
        lo = win.min(axis=1, keepdims=True)
        rng = win.max(axis=1, keepdims=True) - lo
        z = np.divide(win - lo, rng, out=np.zeros_like(win), where=(rng > 1e-12))
        d = z[:, :, None] - z[:, None, :]
        q = np.exp(coef * (d * d)).mean(axis=2)
        out[b:b + chunk] = log_den - np.log(q).mean(axis=1)
    return out, starts


def _onsets_of(columns) -> np.ndarray:
    """全轨去重起音序列（升序）。"""
    return np.asarray(sorted({float(t) for col in columns for t, _ in col}), dtype=np.float64)


def irregularity(columns) -> float:
    """从物件时间戳提取的“局部间隔多样性”特征（连续、无分箱、无条件分段）。

    仅依赖物件时间戳，不使用任何 BPM / 节拍线信息。
    对滑窗逐窗计算归一化 log2 间隔坐标下的 KDE 交叉熵，再对所有窗口取均值。
    """
    onsets = _onsets_of(columns)
    if onsets.size < 2:
        return 0.0
    diffs = np.diff(onsets)
    g = diffs[diffs > TIME_EPS]
    if g.size < IRR_WINDOW:
        return 0.0
    ents, _starts = _window_entropies(np.log2(g), IRR_WINDOW, IRR_STEP, f32=False)
    return float(ents.mean()) if ents.size else 0.0


def irregularity_cached(folder: str, name: str, columns) -> float:
    """带缓存的 irregularity（同谱面只算一次，调参时零开销）。"""
    key = (folder, name, len(columns))
    if key not in _IRR_CACHE:
        _IRR_CACHE[key] = irregularity(columns)
    return _IRR_CACHE[key]


def chart_duration_s(columns) -> float:
    """谱面有效时长（秒）= 最晚物件结束 − 最早物件开始；无物件时为 0。"""
    t0 = min(t for c in columns for t, _ in c) if any(columns) else 0.0
    t1 = max(t + l for c in columns for t, l in c) if any(columns) else 0.0
    return max((t1 - t0) / 1000.0, 0.0)


def chord_ratio(columns) -> float:
    """和弦度 = 音符总数 / 去重起音数（平均同时按键数，≥ 1）。

    仅依赖物件时间戳；纯单键流为 1，和弦越密越大（实测范围约 1.0–3.0）。
    与 irregularity 在语义上相反：和弦越密 → 起音间隔越规整 → irr 越低。
    """
    notes = sum(len(col) for col in columns)
    onsets = {float(t) for col in columns for t, _ in col}
    return notes / max(len(onsets), 1)


def local_irregularity(onsets: np.ndarray, window: int = IRR_LOCAL_WINDOW,
                       stride: int = IRR_LOCAL_STRIDE) -> np.ndarray:
    """逐起音的局部邻域熵（数组版）：返回与 onsets 等长的逐键熵。

    与全局 irregularity 同一 KDE 交叉熵口径，但窗口更小（默认 16 个间隔）、
    以每个起音为中心；stride > 1 时先按档计算再线性插值回逐键（批量加速）。
    """
    if onsets.size < 3:
        return np.zeros(onsets.size, dtype=np.float32)
    diffs = np.diff(onsets)
    g = diffs[diffs > TIME_EPS]
    if g.size < window:
        return np.zeros(onsets.size, dtype=np.float32)
    ents, starts = _window_entropies(np.log2(g), window, stride, f32=True)
    centers = starts + window // 2
    return np.interp(np.arange(onsets.size), centers, ents).astype(np.float32)


def local_irregularity_cached(columns, window: int = IRR_LOCAL_WINDOW,
                              stride: int = IRR_LOCAL_STRIDE):
    """带缓存的逐键局部邻域熵；返回 (onsets, per_onset_entropy) 等长数组。

    以谱面起音序列内容为缓存键，保证同一谱面在各候选参数下只计算一次。
    """
    onsets = _onsets_of(columns)
    key = (window, stride, onsets.size, onsets.tobytes())
    hit = _IRR_LOCAL_CACHE.get(key)
    if hit is None:
        hit = (onsets, local_irregularity(onsets, window, stride))
        _IRR_LOCAL_CACHE[key] = hit
    return hit


def entropy_at(onsets: np.ndarray, ent: np.ndarray, times: np.ndarray) -> np.ndarray:
    """在给定时刻上取逐键局部熵（精确命中，容忍 TIME_EPS；未命中取 0）。"""
    if onsets.size == 0 or times.size == 0:
        return np.zeros(times.size, dtype=np.float64)
    k0 = np.searchsorted(onsets, times)
    kk = np.minimum(k0, onsets.size - 1)
    hit = (k0 < onsets.size) & (np.abs(onsets[kk] - times) < TIME_EPS)
    return np.where(hit, ent[kk], 0.0)
