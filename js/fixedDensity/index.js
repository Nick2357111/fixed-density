// js/fixedDensity/index.js
// Fixed Density v1（含 LN 参考难度）— osu!mania 难度计算。
//
// 移植 benchmark_test.py / benchmark_ln_report.py 在「冻结最优参数」下的算法：
//   RC 通道：物件赋值序列（逐物件 0.5^(dt/H) 衰减）+ 配对轨邻域额外 contribution +
//           单轨局部熵乘子（贡献层）+ 全谱局部熵乘子（最终合并层）+ 双轨/二次 softmax 合并
//           + 分位数二分求 n；再经连续映射曲线反解
//               g(rc_value) = ln(n) − E·chord − D·ln N，rc_value = g⁻¹(·)
//   LN 参考难度：ln 倍率 k = F(ρ)·(γ_rel·S_rel + γ_jack·S_jack + γ_hold·S_hold)，
//           F(ρ) = lin·ρ + (1−lin)·(1+c^h)·ρ^h/(ρ^h+c^h)（Hill + 线性 toe），
//           S* 含逐根长度权重（OD 判定窗）、长度因子 ℓ/(ℓ+ℓ₀)、整谱松手吸收折扣 w
//           与时长有效性门 v（见 lnEventScores / lnStructureScores）；
//           ln_value = a·ln(1 + b·k·n)
//
// 关于「被参数取消的逻辑」（不实现）：
//   * soft_beta = 0.706887 > 0 ⇒ 合并算子为 softmax 加权平均 w·a+(1−w)·b
//     （w = ½(1+tanh(½β(a−b)))）；RC 通道的 Lp 范数分支与 merge_power / merge2_power /
//     contrib_power 不参与。
//     （注：原 LN 通道内的 Lp 合并与最终 Lp 合并因 ln_gain = 0、LN 通道关闭而不参与。）
//   * cross_lane_mix = 0 ⇒ 贡献间距仅取同轨间距，跨轨混合逻辑不参与。
//   * irr_local_gain = 0、irr_gain_pair = 0 ⇒ 全谱熵@贡献层、轨道对熵@配对合并层不参与。
//   * ln_gain = 0 ⇒ LN 通道（松手流 / 控手 / 按住占用）关闭，n_LN 不参与 n。
//   * 拟合模型 C（irr 系数）= 0（--no-irr）；ln 倍率的 γ_cross = 0（S_cross 项不参与）。
//
// DT/HT/可调变速：所有物件时间戳按 1/rate 缩放后再参与计算（rate=1.5 为 DT、0.75 为 HT，
// lazer 任意正速同理）。
//
// 模块约束：DOM-free / state-free，Node 与浏览器一致。

import { OsuFileParser } from "../parser/osuFileParser.js";

const TIME_EPS = 1e-6;
const INV_DIST_SCALE = 1000.0; // 相邻物件间距倒数缩放：1000 / dt
const OVERLAP_PENALTY = 20000.0; // 同轨同时间戳重叠物件的额外惩罚
const N_PRECISION = 1e-12; // n 值二分精度
const TYPE_HOLD = 1 << 7; // osu!mania 长条类型位

// ---- 冻结最优参数（RC 通道）----
const HALF_LIFE_MS = 306.625;
const QUANTILE = 0.91162709;
const EXTRA_RATE = 2.0524655999999997;
const EXTRA_B = 2.337995528948387;
const EXTRA_LOG_BASE = 4.012384;
const SOFT_BETA = 0.706887; // > 0 ⇒ softmax 加权平均（Lp 范数分支不参与）
const IRR_GAIN_LANE = 0.1; // 单轨局部熵 @ 贡献层
const IRR_GAIN_GLOBAL = 0.06; // 全谱局部熵 @ 最终合并层

const LOG_BASE_DENOM = Math.log(EXTRA_LOG_BASE);
const HALF_BETA = 0.5 * SOFT_BETA;

// ---- 连续映射曲线（softplus 基，两端线性渐近、全程光滑且严格单调）----
const MAP_KNOTS = [
    [0.3, 1.6], [2.45, 1.6], [3.85, 0.9], [4.5, 2.0],
    [7.1, 1.6], [12.05, 0.9], [12.65, 0.9],
];
const MAP_COEF = [
    2.2337506713451534,
    0.24999999999999997,
    0.06000000000000313,
    0.24999999999958641,
    0.06000000000000012,
    0.23074460079088877,
    0.06000000000733659,
    0.07373131195591297,
    0.17858296244098779,
    -0.03290469294378953,
    -0.028511990541329264,
];
const MAP_LEAD = 1 + MAP_KNOTS.length + 1; // c + σ_0..σ_K 的个数
const FIT_E_CHORD = MAP_COEF[MAP_LEAD]; // chord 系数（irr 系数 C = 0，不参与）
const FIT_D_LNN = MAP_COEF[MAP_LEAD + 1]; // ln N 系数

// 局部间隔多样性（irregularity）参数
const IRR_LOCAL_WINDOW = 16;
const IRR_LOCAL_STRIDE = 4;
const IRR_BANDWIDTH = 0.08;
const IRR_INV2H2 = 1.0 / (2.0 * IRR_BANDWIDTH * IRR_BANDWIDTH);
const IRR_LOG_DEN = Math.log(IRR_BANDWIDTH * Math.sqrt(2.0 * Math.PI));
const IRR_ENT_COEF = -0.5 * IRR_INV2H2;

// ---- ln 参考难度参数 ----
const LN_MULT_REL = 0.1; // γ_rel（S_rel）
const LN_MULT_JACK = 0.4; // γ_jack（S_jack）
const LN_MULT_HOLD = 0.2; // γ_hold（S_hold）
const LN_MULT_EXP = 6.0;
const LN_MULT_RHO_C = 0.33;
const LN_MULT_RHO_LIN = 0.1; // 低 ρ 段线性 toe 权重
const LN_ABSORB_GAIN = 0.25; // 松手吸收折扣 λ
const ABSORB_DIST_SCALE = 0.5; // 吸收判定距离 = 该倍数 × 本地节拍
const LN_LEN_WEIGHT_GAIN = 1.0; // ln_len_weight_gain：长度权重增益
const LN_HOLD_REF_MS = 40.0; // ln_hold_ref_ms：长度因子 ℓ/(ℓ+ℓ₀) 的参考时长
const LN_A = 5.44200375179058;
const LN_B = 0.0960408821250538;

// LN 头/尾判定窗与长度有效性门（OD 相关）
const LN_HEAD_WINDOW_A = 64.0; // 头判窗总宽 = A − B·OD（ms）
const LN_HEAD_WINDOW_B = 3.0;
const LN_TAIL_WINDOW_RATIO = 1.5; // 尾判窗宽 / 头判窗宽
const LN_MIN_HOLD_MS = 20.0; // 一次点击的物理最短时长补足
const LN_VALID_OD_GAIN = 0.4; // 有效性门 OD 项系数
const LN_VALID_RAMP = 1.3; // 有效性门过渡宽度系数

function lowerBound(arr, target) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

function upperBound(arr, target) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// 合并两个升序分组时间序列并去重，同时输出与并集时间对齐的两侧物件数。
function mergeGroupSizes(timesA, sizesA, timesB, sizesB) {
    const times = [];
    const countsA = [];
    const countsB = [];
    let i = 0;
    let j = 0;
    while (i < timesA.length && j < timesB.length) {
        const ta = timesA[i];
        const tb = timesB[j];
        if (ta < tb) {
            times.push(ta); countsA.push(sizesA[i]); countsB.push(0); i += 1;
        } else if (ta > tb) {
            times.push(tb); countsA.push(0); countsB.push(sizesB[j]); j += 1;
        } else {
            times.push(ta); countsA.push(sizesA[i]); countsB.push(sizesB[j]); i += 1; j += 1;
        }
    }
    while (i < timesA.length) { times.push(timesA[i]); countsA.push(sizesA[i]); countsB.push(0); i += 1; }
    while (j < timesB.length) { times.push(timesB[j]); countsA.push(0); countsB.push(sizesB[j]); j += 1; }
    return { times, countsA, countsB };
}

// 单轨按相同时间戳分组 → [{ t, size, prevDt, nextDt }]，按时间升序。
function groupLane(times) {
    const groups = [];
    let i = 0;
    while (i < times.length) {
        let j = i;
        while (j + 1 < times.length && Math.abs(times[j + 1] - times[i]) < TIME_EPS) {
            j += 1;
        }
        groups.push({ t: times[i], size: j - i + 1 });
        i = j + 1;
    }
    const result = [];
    for (let idx = 0; idx < groups.length; idx += 1) {
        const prevDt = idx > 0 ? groups[idx].t - groups[idx - 1].t : null;
        const nextDt = idx + 1 < groups.length ? groups[idx + 1].t - groups[idx].t : null;
        result.push({ t: groups[idx].t, size: groups[idx].size, prevDt, nextDt });
    }
    return result;
}

// 升序去重（容忍 TIME_EPS）。
function uniqueSorted(times) {
    const out = [];
    for (const t of times) {
        if (out.length === 0 || Math.abs(t - out[out.length - 1]) >= TIME_EPS) {
            out.push(t);
        }
    }
    return out;
}

// 合并算子（soft_beta > 0）：softmax 加权平均（光滑的 max）。
//   w = ½(1+tanh(½β(a−b)))，M = w·a + (1−w)·b
function mergeOp(a, b) {
    const w = 0.5 * (1.0 + Math.tanh(HALF_BETA * (a - b)));
    return w * a + (1.0 - w) * b;
}

// 逐组注入贡献：base = 单件 M(prev_inv, next_inv)；多件按“超限替代”累加；
// extra = Σ_{配对轨键, |x|∈[0,4a]} size·exp(−((log_base(x/a)+extra_b)²)/2)，a = 1000/base；
// contribution = base·(1 + extra_rate·extra)
function laneContribs(groups, mateTimes, mateSizes) {
    const n = groups.length;
    const contribs = new Float64Array(n);
    if (n === 0) {
        return contribs;
    }
    const penPen = mergeOp(OVERLAP_PENALTY, OVERLAP_PENALTY);
    for (let k = 0; k < n; k += 1) {
        const g = groups[k];
        const prevInv = (g.prevDt != null && g.prevDt > TIME_EPS) ? INV_DIST_SCALE / g.prevDt : 0.0;
        const nextInv = (g.nextDt != null && g.nextDt > TIME_EPS) ? INV_DIST_SCALE / g.nextDt : 0.0;
        let base = mergeOp(prevInv, nextInv);
        if (g.size > 1) {
            base = (g.size - 2) * penPen
                + mergeOp(prevInv, OVERLAP_PENALTY)
                + mergeOp(OVERLAP_PENALTY, nextInv);
        }

        let extra = 0.0;
        if (EXTRA_RATE !== 0.0 && base > 0.0 && mateTimes.length > 0) {
            const a = INV_DIST_SCALE / base;
            const lo = lowerBound(mateTimes, g.t - 4.0 * a);
            const hi = upperBound(mateTimes, g.t + 4.0 * a);
            if (hi > lo) {
                const constTerm = EXTRA_B - Math.log(a) / LOG_BASE_DENOM;
                let total = 0.0;
                for (let m = lo; m < hi; m += 1) {
                    const x = Math.abs(mateTimes[m] - g.t);
                    if (x <= 0.0) {
                        continue;
                    }
                    const z = Math.log(x) / LOG_BASE_DENOM + constTerm;
                    total += mateSizes[m] * Math.exp(-(z * z) * 0.5);
                }
                extra = total;
            }
        }
        contribs[k] = base * (1.0 + EXTRA_RATE * extra);
    }
    return contribs;
}

// 单轨衰减序列在 unionTimes 各时刻的取值（下标对齐）：value ← value·0.5^(dt/H) + contribution。
function laneCurve(groupTimes, contribs, unionTimes) {
    const out = new Float64Array(unionTimes.length);
    let value = 0.0;
    let gi = 0;
    let lastT = null;
    for (let ti = 0; ti < unionTimes.length; ti += 1) {
        const t = unionTimes[ti];
        if (lastT !== null) {
            value *= Math.pow(0.5, (t - lastT) / HALF_LIFE_MS);
        }
        while (gi < groupTimes.length && Math.abs(groupTimes[gi] - t) < TIME_EPS) {
            value += contribs[gi];
            gi += 1;
        }
        out[ti] = value;
        lastT = t;
    }
    return out;
}

// 把“赋值处有突变的时间衰减函数”在给定时刻采样（替换语义）。
function sampleDecay(seq, times) {
    const out = new Float64Array(times.length);
    let value = 0.0;
    let si = 0;
    let lastT = null;
    for (let ti = 0; ti < times.length; ti += 1) {
        const t = times[ti];
        if (lastT !== null) {
            value *= Math.pow(0.5, (t - lastT) / HALF_LIFE_MS);
        }
        while (si < seq.length && Math.abs(seq[si][0] - t) < TIME_EPS) {
            value = seq[si][1];
            si += 1;
        }
        out[ti] = value;
        lastT = t;
    }
    return out;
}

// 合并两条“赋值突变衰减”序列：时间戳并集上采样后逐时刻 softmax 合并。
function mergeDecaySequences(seqA, seqB) {
    if (seqA.length === 0) {
        return seqB;
    }
    if (seqB.length === 0) {
        return seqA;
    }
    const lenA = seqA.length;
    const lenB = seqB.length;
    const times = new Float64Array(lenA + lenB);
    const cntA = new Int32Array(lenA + lenB);
    const cntB = new Int32Array(lenA + lenB);
    let i = 0;
    let j = 0;
    let m = 0;
    while (i < lenA && j < lenB) {
        const ta = seqA[i][0];
        const tb = seqB[j][0];
        if (ta < tb) { times[m] = ta; cntA[m] = seqA[i][2]; i += 1; }
        else if (ta > tb) { times[m] = tb; cntB[m] = seqB[j][2]; j += 1; }
        else { times[m] = ta; cntA[m] = seqA[i][2]; cntB[m] = seqB[j][2]; i += 1; j += 1; }
        m += 1;
    }
    while (i < lenA) { times[m] = seqA[i][0]; cntA[m] = seqA[i][2]; i += 1; m += 1; }
    while (j < lenB) { times[m] = seqB[j][0]; cntB[m] = seqB[j][2]; j += 1; m += 1; }

    const unionTimes = times.subarray(0, m);
    const fa = sampleDecay(seqA, unionTimes);
    const fb = sampleDecay(seqB, unionTimes);
    const out = new Array(m);
    for (let k = 0; k < m; k += 1) {
        const c = cntA[k] + cntB[k];
        if (c <= 0) {
            continue;
        }
        out[k] = [unionTimes[k], mergeOp(fa[k], fb[k]), c];
    }
    return out;
}

// numpy 默认线性插值分位数。
function quantileLinear(sorted, q) {
    const n = sorted.length;
    if (n === 0) {
        return 0.0;
    }
    if (n === 1) {
        return sorted[0];
    }
    const pos = q * (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) {
        return sorted[lo];
    }
    const frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// 求 n 使 mean(min(1, n/a_i)) == targetAvg（排序 + 1/a 前缀和 + 二分，分位数作上界初值）。
function findNQuantile(values, targetAvg) {
    const a = values instanceof Float64Array ? values : Float64Array.from(values);
    const n = a.length;
    if (n === 0) {
        throw new Error("赋值列表为空，无法计算 n");
    }
    if (targetAvg <= 0.0) {
        return 0.0;
    }
    a.sort();
    if (targetAvg >= 1.0) {
        return a[n - 1];
    }
    const prefixInv = new Float64Array(n + 1);
    for (let i = 0; i < n; i += 1) {
        prefixInv[i + 1] = prefixInv[i] + (a[i] > 0.0 ? 1.0 / a[i] : 0.0);
    }
    const totalInv = prefixInv[n];
    const targetSum = targetAvg * n;
    const partialSum = (value) => {
        const k = upperBound(a, value);
        return k + value * (totalInv - prefixInv[k]);
    };
    let lo = 0.0;
    let hi = quantileLinear(a, targetAvg);
    if (!(hi > 0.0) || partialSum(hi) < targetSum) {
        hi = a[n - 1];
    }
    // 二分求解。注意：hi/lo 收敛到 1 ULP 后 (lo+hi)*0.5 会取整回端点而不再推进，
    // 此时 hi-lo ≈ ULP 可能仍大于绝对容差 N_PRECISION（大 n 值时 ULP ≈ 3.6e-12），
    // 仅按绝对容差判停会死循环（超高物量谱面即触发）。故加“无进展即停”保护 + 迭代上限。
    for (let iter = 0; iter < 200; iter += 1) {
        if (hi - lo <= N_PRECISION) {
            break;
        }
        const mid = (lo + hi) * 0.5;
        if (mid === lo || mid === hi) {
            break;
        }
        if (partialSum(mid) < targetSum) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) * 0.5;
}

function softplus(z) {
    if (z > 0.0) {
        return z + Math.log1p(Math.exp(-z));
    }
    return Math.log1p(Math.exp(z));
}

function mapBasis(v) {
    const K = MAP_KNOTS.length;
    const phi = new Array(K);
    for (let i = 0; i < K; i += 1) {
        const [v0, s] = MAP_KNOTS[i];
        phi[i] = s * softplus((v - v0) / s);
    }
    const w = new Array(K + 1);
    w[0] = v - phi[0];
    for (let j = 1; j < K; j += 1) {
        w[j] = phi[j - 1] - phi[j];
    }
    w[K] = phi[K - 1];
    return w;
}

function mapForward(v) {
    const w = mapBasis(v);
    let sum = MAP_COEF[0];
    for (let i = 0; i < w.length; i += 1) {
        sum += MAP_COEF[i + 1] * w[i];
    }
    return sum;
}

// 解 g(v) = target（g 单调 → 二分），返回原始 value。
function mapInvert(target) {
    let lo = -60.0;
    let hi = 120.0;
    for (let i = 0; i < 90; i += 1) {
        const mid = (lo + hi) * 0.5;
        if (mapForward(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) * 0.5;
}

// 滑窗 KDE 交叉熵：返回每窗熵与起始下标。
function windowEntropies(lg, window, stride) {
    const n = lg.length;
    if (n < window) {
        return { ents: new Float64Array(0), starts: new Int32Array(0) };
    }
    const startList = [];
    for (let s = 0; s + window <= n; s += stride) {
        startList.push(s);
    }
    const ents = new Float64Array(startList.length);
    const z = new Float64Array(window);
    for (let si = 0; si < startList.length; si += 1) {
        const s = startList[si];
        let lo = Infinity;
        let hi = -Infinity;
        for (let k = 0; k < window; k += 1) {
            const x = lg[s + k];
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        const rng = hi - lo;
        if (rng > 1e-12) {
            const inv = 1.0 / rng;
            for (let k = 0; k < window; k += 1) {
                z[k] = (lg[s + k] - lo) * inv;
            }
        } else {
            z.fill(0.0);
        }
        let sumLogQ = 0.0;
        for (let i = 0; i < window; i += 1) {
            const zi = z[i];
            let acc = 0.0;
            for (let j = 0; j < window; j += 1) {
                const d = zi - z[j];
                acc += Math.exp(IRR_ENT_COEF * (d * d));
            }
            sumLogQ += Math.log(acc / window);
        }
        ents[si] = IRR_LOG_DEN - sumLogQ / window;
    }
    return { ents, starts: Int32Array.from(startList) };
}

// 逐起音的局部邻域熵（窗口 16、步长 4，档间线性插值回逐键）。
function localIrregularity(onsets) {
    const size = onsets.length;
    const out = new Float64Array(size);
    if (size < 3) {
        return out;
    }
    const gaps = [];
    for (let i = 1; i < size; i += 1) {
        const d = onsets[i] - onsets[i - 1];
        if (d > TIME_EPS) {
            gaps.push(d);
        }
    }
    if (gaps.length < IRR_LOCAL_WINDOW) {
        return out;
    }
    const lg = Float64Array.from(gaps, (g) => Math.log2(g));
    const { ents, starts } = windowEntropies(lg, IRR_LOCAL_WINDOW, IRR_LOCAL_STRIDE);
    if (starts.length === 0) {
        return out;
    }
    const half = IRR_LOCAL_WINDOW >> 1;
    const centers = new Float64Array(starts.length);
    for (let i = 0; i < starts.length; i += 1) {
        centers[i] = starts[i] + half;
    }
    const last = centers.length - 1;
    for (let x = 0; x < size; x += 1) {
        if (x <= centers[0]) {
            out[x] = ents[0];
        } else if (x >= centers[last]) {
            out[x] = ents[last];
        } else {
            let lo = 0;
            let hi = last;
            while (hi - lo > 1) {
                const mid = (lo + hi) >> 1;
                if (centers[mid] <= x) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            const x0 = centers[lo];
            const x1 = centers[hi];
            const t = x1 > x0 ? (x - x0) / (x1 - x0) : 0.0;
            out[x] = ents[lo] + (ents[hi] - ents[lo]) * t;
        }
    }
    return out;
}

// 在给定时刻上取逐键局部熵（精确命中，容忍 TIME_EPS；未命中取 0）。
function entropyAt(onsets, ent, times) {
    const out = new Float64Array(times.length);
    if (onsets.length === 0 || times.length === 0) {
        return out;
    }
    for (let i = 0; i < times.length; i += 1) {
        const t = times[i];
        const k0 = lowerBound(onsets, t);
        if (k0 < onsets.length && Math.abs(onsets[k0] - t) < TIME_EPS) {
            out[i] = ent[k0];
        }
    }
    return out;
}

// ---- LN 参考难度：OD 判定窗 / 长度有效性门 ----

// 该 OD 下 LN 头/尾判定的半宽 (W_head, W_tail)，单位 ms。
function lnOdWindows(od) {
    const wHead = Math.max((LN_HEAD_WINDOW_A - LN_HEAD_WINDOW_B * od) / 2.0, 0.0);
    return [wHead, wHead * LN_TAIL_WINDOW_RATIO];
}

// 长度有效性门的临界区间 (lo, hi)，单位 ms：ℓ ≤ lo 视为单键、ℓ ≥ hi 视为完整长条。
function lnValidBand(od) {
    const [wHead, wTail] = lnOdWindows(od);
    const lo = LN_MIN_HOLD_MS + LN_VALID_OD_GAIN * (wHead + wTail);
    return [lo, lo * LN_VALID_RAMP];
}

function smoothstep(x, lo, hi) {
    if (hi - lo <= TIME_EPS) {
        return x >= hi ? 1.0 : 0.0;
    }
    let t = (x - lo) / (hi - lo);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return t * t * (3.0 - 2.0 * t);
}

function median(values) {
    const n = values.length;
    if (n === 0) {
        return 0.0;
    }
    const a = Float64Array.from(values);
    a.sort();
    const mid = n >> 1;
    return n % 2 === 1 ? a[mid] : (a[mid - 1] + a[mid]) * 0.5;
}

// 单根长条的长度权重 u = smoothstep(ℓ; lo, hi) ∈ [0,1]：过短视为单键、足够长视为完整长条。
function lnLengthWeight(lengthMs, od) {
    const [lo, hi] = lnValidBand(od);
    return smoothstep(lengthMs, lo, hi);
}

// 长度权重施加上增益：w = 1 − clip(gain)·(1 − u)；gain ≤ 0 时恒为 1。
function lnLenWeight(lengthMs, od) {
    if (LN_LEN_WEIGHT_GAIN <= 0.0) {
        return 1.0;
    }
    const gain = Math.min(Math.max(LN_LEN_WEIGHT_GAIN, 0.0), 1.0);
    return 1.0 - gain * (1.0 - lnLengthWeight(lengthMs, od));
}

// 全谱 LN 时长有效性 v = smoothstep(ℓ_med; lo, hi) ∈ [0,1]（无长条时为 0）：
// 长条短到判定上等同单键时整层结构量清零。
function lnHoldValidity(lanes, od) {
    const lens = [];
    for (const lane of lanes) {
        for (const n of lane) {
            if (n.len > 0) {
                lens.push(n.len);
            }
        }
    }
    if (lens.length === 0) {
        return 0.0;
    }
    const [lo, hi] = lnValidBand(od);
    return smoothstep(median(lens), lo, hi);
}

// ---- LN 参考难度：结构得分与 ln 倍率 ----
//
// 冻结参数下 γ_cross = 0，故 S_cross 项不参与（不实现）；
// 参与 k 的是 S_rel（同轨相邻松手间距倒数，按松手分组）、S_jack（松手→同轨下一次按下）
// 与 S_hold（保持时长 / 本地物件平均间隔，按每根长条）。
// 逐灰长度权重 w/u 与长度因子 h 见 lnEventScores；整谱折扣 w(吸收) / v(有效性) 见 lnStructureScores。

function invGap(d) {
    return d > TIME_EPS ? INV_DIST_SCALE / d : 0.0;
}

// 单轨 LN 事件的三项结构量；该轨无 LN 时返回 null。
// 逐根长度权重 w = lnLenWeight(ℓ, od)；S_rel / S_jack 再乘长度因子 h = ℓ/(ℓ+ℓ₀)
// （恢复“有效长条中越长越难”的单调性），S_hold 只乘 w。
function lnEventScores(col, od) {
    const holds = col.filter((n) => n.len > 0);
    if (holds.length === 0) {
        return null;
    }
    const tt = uniqueSorted(holds.map((n) => n.t + n.len).sort((a, b) => a - b));
    const presses = col.map((n) => n.t).sort((a, b) => a - b);

    // 每个松手时刻的平均长度 → (长度权重 w, 长度因子 h)
    const lenAcc = new Map();
    for (const n of holds) {
        const key = n.t + n.len;
        const e = lenAcc.get(key);
        if (e) {
            e[0] += n.len;
            e[1] += 1;
        } else {
            lenAcc.set(key, [n.len, 1]);
        }
    }
    const w = new Float64Array(tt.length);
    const h = new Float64Array(tt.length);
    for (let i = 0; i < tt.length; i += 1) {
        const e = lenAcc.get(tt[i]);
        const m = e ? e[0] / e[1] : 0.0;
        w[i] = lnLenWeight(m, od);
        h[i] = LN_HOLD_REF_MS <= 0.0 ? 1.0 : m / (m + LN_HOLD_REF_MS);
    }

    // S_rel：同轨相邻松手间距倒数（Lp 范数 p = ln_rel_power = 2）
    const sRel = new Float64Array(tt.length);
    for (let i = 0; i < tt.length; i += 1) {
        const pd = i > 0 ? tt[i] - tt[i - 1] : -1.0;
        const nd = i + 1 < tt.length ? tt[i + 1] - tt[i] : -1.0;
        sRel[i] = Math.hypot(invGap(pd), invGap(nd)) * w[i] * h[i];
    }

    // S_jack：松手 → 同轨下一次按下间距倒数
    const sJack = new Float64Array(tt.length);
    for (let i = 0; i < tt.length; i += 1) {
        const k = upperBound(presses, tt[i]);
        const d = k < presses.length ? presses[k] - tt[i] : -1.0;
        sJack[i] = invGap(d) * w[i] * h[i];
    }

    // S_hold：保持时长 / 该轨物件平均间隔（无量纲，一根长条跨越多少个本地节拍）× w
    const lens = holds.map((n) => n.len);
    let baseGap = 0.0;
    if (presses.length > 1) {
        let sum = 0.0;
        for (let i = 1; i < presses.length; i += 1) {
            sum += presses[i] - presses[i - 1];
        }
        baseGap = sum / (presses.length - 1);
    }
    const sHold = (baseGap > TIME_EPS && lens.length > 0)
        ? Float64Array.from(lens, (l) => (l / baseGap) * lnLenWeight(l, od))
        : new Float64Array(lens.length);

    return { sRel, sJack, sHold };
}

// 全谱 LN 事件上的均值，并施以整谱折扣：S_rel / S_jack × w(松手吸收) × v(时长有效性)，
// S_hold × v。
function lnStructureScores(lanes, od) {
    const rels = [];
    const jacks = [];
    const holdsArr = [];
    for (const lane of lanes) {
        const got = lnEventScores(lane, od);
        if (got) {
            for (const v of got.sRel) {
                rels.push(v);
            }
            for (const v of got.sJack) {
                jacks.push(v);
            }
            for (const v of got.sHold) {
                holdsArr.push(v);
            }
        }
    }
    const mean = (arr) => (arr.length === 0 ? 0.0 : arr.reduce((s, x) => s + x, 0) / arr.length);
    const absorb = lnReleaseAbsorb(lanes, LN_ABSORB_GAIN);
    const valid = lnHoldValidity(lanes, od);
    return {
        sRel: mean(rels) * absorb * valid,
        sJack: mean(jacks) * absorb * valid,
        sHold: mean(holdsArr) * valid,
    };
}

// 全谱松手吸收折扣 w = 1 − λ·ā ∈ [0,1]（λ ≤ 0 时恒为 1）。
// 单次松手吸收度 a = clip(1 − d/(0.5·g), 0, 1)：d 为松手时刻到最近“其它起音”的距离，
// g 为本地节拍（全谱正间隔中位数），ā 为全谱松手吸收度均值。
function lnReleaseAbsorb(lanes, gain) {
    if (gain <= 0.0) {
        return 1.0;
    }
    const onsetSet = new Set();
    for (const lane of lanes) {
        for (const n of lane) {
            onsetSet.add(n.t);
        }
    }
    const onsets = Array.from(onsetSet).sort((a, b) => a - b);
    if (onsets.length === 0) {
        return 1.0;
    }
    const diffs = [];
    for (let i = 1; i < onsets.length; i += 1) {
        const d = onsets[i] - onsets[i - 1];
        if (d > TIME_EPS) {
            diffs.push(d);
        }
    }
    if (diffs.length === 0) {
        return 1.0;
    }
    diffs.sort((a, b) => a - b);
    const mid = diffs.length >> 1;
    const gap = diffs.length % 2 === 1 ? diffs[mid] : (diffs[mid - 1] + diffs[mid]) * 0.5;
    if (gap <= TIME_EPS) {
        return 1.0;
    }
    const rels = [];
    for (const lane of lanes) {
        for (const n of lane) {
            if (n.len > 0) {
                rels.push(n.t + n.len);
            }
        }
    }
    if (rels.length === 0) {
        return 1.0;
    }
    const scale = ABSORB_DIST_SCALE * gap;
    let sum = 0.0;
    for (const r of rels) {
        const k0 = lowerBound(onsets, r);
        let d = Infinity;
        for (let sh = -1; sh <= 0; sh += 1) {
            const kk = Math.min(Math.max(k0 + sh, 0), onsets.length - 1);
            const dd = Math.abs(onsets[kk] - r);
            if (dd < d) {
                d = dd;
            }
        }
        sum += Math.min(Math.max(1.0 - d / scale, 0.0), 1.0);
    }
    const aBar = sum / rels.length;
    return 1.0 - Math.min(Math.max(gain, 0.0), 1.0) * aBar;
}

// ρ 饱和曲线 F(ρ) = lin·ρ + (1−lin)·(1+c^h)·ρ^h/(ρ^h+c^h)。
// 纯 Hill（lin=0）在 ρ≪c 段按 ρ^h 塌向 0；叠一条线性 toe（lin>0）后低 ρ 段近似线性，
// 少量长条的谱面不再被归零。仍满足 F(0)=0、F(1)=1、单调不减。
function lnRhoCurve(rho, h, c, lin) {
    if (rho <= 0.0) {
        return 0.0;
    }
    const k = Math.min(Math.max(lin, 0.0), 1.0);
    const rh = Math.pow(rho, h);
    const ch = Math.pow(c, h);
    const hill = (1.0 + ch) * rh / (rh + ch);
    return k * rho + (1.0 - k) * hill;
}

// 长条占比 ρ = Σ 长度权重 / 物件总数（无物件时为 0）。
function lnHoldRatio(lanes, od) {
    let total = 0;
    let holds = 0.0;
    for (const lane of lanes) {
        total += lane.length;
        for (const n of lane) {
            if (n.len > 0) {
                holds += lnLenWeight(n.len, od);
            }
        }
    }
    return total === 0 ? 0.0 : holds / total;
}

// ln 倍率 k = F(ρ)·(γ_rel·S_rel + γ_jack·S_jack + γ_hold·S_hold)（γ_cross = 0 不参与）。
// S_rel / S_jack 已含整谱折扣；rate > 1（DT）时这两项与物件时间同倍缩放 → 乘 rate。
function lnMultiplier(lanes, od, rate) {
    const rho = lnHoldRatio(lanes, od);
    if (rho <= 0.0) {
        return 0.0;
    }
    const { sRel, sJack, sHold } = lnStructureScores(lanes, od);
    const total = LN_MULT_REL * (sRel * rate)
        + LN_MULT_JACK * (sJack * rate)
        + LN_MULT_HOLD * sHold;
    if (total < 0.0) {
        return 0.0;
    }
    return lnRhoCurve(rho, LN_MULT_EXP, LN_MULT_RHO_C, LN_MULT_RHO_LIN) * total;
}

function collectLanes(parsed) {
    const keyCount = Number(parsed?.columnCount) || 0;
    const columns = Array.isArray(parsed?.columns) ? parsed.columns : [];
    const starts = Array.isArray(parsed?.noteStarts) ? parsed.noteStarts : [];
    const ends = Array.isArray(parsed?.noteEnds) ? parsed.noteEnds : [];
    const types = Array.isArray(parsed?.noteTypes) ? parsed.noteTypes : [];
    const lanes = Array.from({ length: keyCount }, () => []);
    for (let i = 0; i < columns.length; i += 1) {
        const key = Number(columns[i]);
        const t = Number(starts[i]);
        if (!Number.isFinite(key) || key < 0 || key >= keyCount || !Number.isFinite(t)) {
            continue;
        }
        let len = 0;
        if ((Number(types[i]) & TYPE_HOLD) !== 0) {
            const end = Number(ends[i]);
            if (Number.isFinite(end) && end > t) {
                len = end - t;
            }
        }
        lanes[key].push({ t, len });
    }
    for (const lane of lanes) {
        lane.sort((a, b) => a.t - b.t);
    }
    return lanes;
}

function resolveParsed(source) {
    if (source && typeof source === "object" && typeof source.getParsedData === "function") {
        return source.getParsedData();
    }
    if (typeof source === "string") {
        const parser = new OsuFileParser(source);
        parser.process();
        return parser.getParsedData();
    }
    throw new Error("Unsupported Fixed Density source.");
}

function normalizeRate(rate) {
    const value = Number(rate);
    if (!Number.isFinite(value) || value <= 0) {
        return 1.0;
    }
    return value;
}

/**
 * 计算 Fixed Density v1 的 rc_value 与 ln_value。
 *
 * @param {string|object} source .osu 文本 或 已 process() 的 OsuFileParser 实例
 * @param {number} [rate=1] 变速倍率（DT=1.5、HT=0.75、lazer 任意正速）；时间戳按 1/rate 缩放
 * @returns {{ rcValue: number, lnValue: number }} 无有效物件时两者均为 NaN
 */
export function calculateFixedDensity(source, rate = 1.0) {
    const parsed = resolveParsed(source);
    if (parsed?.status === "NotMania") {
        throw new Error("Beatmap mode is not mania");
    }
    if (parsed?.status === "Fail") {
        throw new Error("Beatmap parse failed");
    }

    const rawLanes = collectLanes(parsed);
    const kTotal = rawLanes.length;
    if (kTotal < 1) {
        return { rcValue: Number.NaN, lnValue: Number.NaN };
    }

    // LN 长度门 / 判定窗依赖 OD；缺失（<0）时回退 8.0（与 benchmark 一致）。
    const odRaw = Number(parsed?.od);
    const od = (Number.isFinite(odRaw) && odRaw >= 0.0) ? odRaw : 8.0;

    // 变速：时间戳按 1/rate 缩放（rate=1 时不变）。
    const rateNorm = normalizeRate(rate);
    const invRate = 1.0 / rateNorm;
    const lanes = invRate === 1.0
        ? rawLanes
        : rawLanes.map((lane) => lane.map((n) => ({ t: n.t * invRate, len: n.len * invRate })));

    let noteCount = 0;
    const onsetSet = new Set();
    for (const lane of lanes) {
        noteCount += lane.length;
        for (const n of lane) {
            onsetSet.add(n.t);
        }
    }
    const allOnsets = Array.from(onsetSet).sort((a, b) => a - b);
    const onsetCount = allOnsets.length;
    const chord = noteCount / Math.max(onsetCount, 1);
    const lnN = Math.log(Math.max(noteCount, 1));

    const laneTimes = lanes.map((lane) => lane.map((n) => n.t));
    const laneGroups = laneTimes.map(groupLane);
    const laneGroupTimes = laneGroups.map((gs) => gs.map((g) => g.t));
    const laneGroupSizes = laneGroups.map((gs) => gs.map((g) => g.size));

    // RC 通道：逐轨贡献（softmax 合并 + 邻域额外 contribution）
    const laneContribsList = [];
    for (let i = 0; i < kTotal; i += 1) {
        const mate = i % 2 === 0 ? i + 1 : i - 1;
        const mateIdx = (mate >= 0 && mate < kTotal) ? mate : i;
        laneContribsList.push(laneContribs(
            laneGroups[i],
            laneGroupTimes[mateIdx],
            laneGroupSizes[mateIdx],
        ));
    }

    // 单轨局部熵乘子（贡献层）：contribution *= exp(γ_lane · irr_lane(t))
    if (IRR_GAIN_LANE !== 0.0) {
        for (let i = 0; i < kTotal; i += 1) {
            const tt = laneGroupTimes[i];
            if (tt.length === 0) {
                continue;
            }
            const ent = localIrregularity(tt);
            const contribs = laneContribsList[i];
            for (let k = 0; k < tt.length; k += 1) {
                contribs[k] *= Math.exp(IRR_GAIN_LANE * ent[k]);
            }
        }
    }

    // 第一级合并：配对双轨 softmax 合并为一条赋值序列
    const pairSequences = [];
    for (let start = 0; start < kTotal; start += 2) {
        const mate = start + 1 < kTotal ? start + 1 : start;
        const groupsA = laneGroups[start];
        const groupsB = laneGroups[mate];
        if (groupsA.length === 0 && groupsB.length === 0) {
            continue;
        }
        const timesA = laneGroupTimes[start];
        const sizesA = laneGroupSizes[start];

        let unionTimes;
        let counts;
        let curveA;
        let curveB;
        if (mate === start) {
            unionTimes = timesA;
            counts = sizesA;
            curveA = laneCurve(timesA, laneContribsList[start], unionTimes);
            curveB = curveA;
        } else {
            const merged = mergeGroupSizes(timesA, sizesA, laneGroupTimes[mate], laneGroupSizes[mate]);
            unionTimes = merged.times;
            counts = merged.countsA.map((v, idx) => v + merged.countsB[idx]);
            curveA = laneCurve(timesA, laneContribsList[start], unionTimes);
            curveB = laneCurve(laneGroupTimes[mate], laneContribsList[mate], unionTimes);
        }

        const seq = [];
        if (mate === start) {
            for (let k = 0; k < unionTimes.length; k += 1) {
                if (counts[k] > 0) {
                    seq.push([unionTimes[k], curveA[k], counts[k]]);
                }
            }
        } else {
            for (let k = 0; k < unionTimes.length; k += 1) {
                const c = counts[k];
                if (c <= 0) {
                    continue;
                }
                seq.push([unionTimes[k], mergeOp(curveA[k], curveB[k]), c]);
            }
        }
        pairSequences.push(seq);
    }

    // 第二级合并：各配对序列视为赋值突变衰减函数，两两 softmax 合并
    let mergedSeq = [];
    for (const seq of pairSequences) {
        mergedSeq = mergeDecaySequences(mergedSeq, seq);
    }

    // 全谱局部熵乘子（最终合并层）：v(t) *= exp(γ_global · irr_global(t))
    if (IRR_GAIN_GLOBAL !== 0.0 && mergedSeq.length > 0) {
        const mergedTimes = Float64Array.from(mergedSeq, (row) => row[0]);
        const gEnt = localIrregularity(allOnsets);
        const e = entropyAt(allOnsets, gEnt, mergedTimes);
        for (let k = 0; k < mergedSeq.length; k += 1) {
            mergedSeq[k][1] *= Math.exp(IRR_GAIN_GLOBAL * e[k]);
        }
    }

    let totalCount = 0;
    for (const row of mergedSeq) {
        totalCount += row[2];
    }
    if (totalCount === 0) {
        return { rcValue: Number.NaN, lnValue: Number.NaN };
    }

    const mergedValues = new Float64Array(totalCount);
    let idx = 0;
    for (const row of mergedSeq) {
        const v = row[1];
        const c = row[2];
        for (let k = 0; k < c; k += 1) {
            mergedValues[idx] = v;
            idx += 1;
        }
    }

    const n = findNQuantile(mergedValues, QUANTILE);
    if (!Number.isFinite(n) || n <= 0.0) {
        return { rcValue: Number.NaN, lnValue: Number.NaN };
    }

    const gPred = Math.log(n) - FIT_E_CHORD * chord - FIT_D_LNN * lnN;
    const rcValue = mapInvert(gPred);

    const k = lnMultiplier(rawLanes, od, rateNorm);
    const lnValue = LN_A * Math.log1p(LN_B * k * n);

    return { rcValue, lnValue };
}

export default calculateFixedDensity;

// ---- 数字上方的文字说明：段位名 + 分段名 ----

// 希腊字母表小写：11→α、12→β … 34→ω
const FD_GREEK_LOWER = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
];

// 分段名：按小数部分以 .1/.3/.5/.7/.9 分段
// [.1,.3)→"+"、[.3,.5)→"++"、[.5,.7)→"--"、[.7,.9)→"-"、其余（[.9,1)∪[0,.1)）→""
function fixedDensitySegment(value) {
    const frac = value - Math.floor(value);
    if (frac < 0.1) {
        return "";
    }
    if (frac < 0.3) {
        return "+";
    }
    if (frac < 0.5) {
        return "++";
    }
    if (frac < 0.7) {
        return "--";
    }
    if (frac < 0.9) {
        return "-";
    }
    return "";
}

// rc 段位名：≤10 → "rf<k>"；≥11 → 希腊小写（11→α、14→δ …）；≥34.5 → 直接 ">ω++"。
export function fixedDensityRcLabel(value) {
    if (!Number.isFinite(value)) {
        return "";
    }
    const rounded = Math.round(value);
    if (value >= 34.5 || rounded - 11 >= FD_GREEK_LOWER.length) {
        return ">ω++";
    }
    const tier = rounded <= 10 ? `rf${rounded}` : FD_GREEK_LOWER[rounded - 11];
    return tier + fixedDensitySegment(value);
}

// ln 段位名："ln<k>"。
export function fixedDensityLnLabel(value) {
    if (!Number.isFinite(value)) {
        return "";
    }
    return `ln${Math.round(value)}${fixedDensitySegment(value)}`;
}
