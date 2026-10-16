"""osu!mania 4K · LN 参考难度（ln_value）benchmark 报告
=====================================================

仿照 fixed_density.benchmark 的 RC 报告制作，口径换成 LN：

    k        = γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross   （ln 倍率，无常数项）
    n_LN     = k · n_rc                                        （RC 的 n 按倍率缩放）
    ln_value = a · ln(b · n_LN + 1)                            （含下界对数标定）

同时给出 rc_value（现有 RC 标定反算）作为对照，两者共同构成谱面的参考难度。

用法
----
  python -m fixed_density.ln_benchmark           # γ 由网格搜索自动确定（默认）
  python -m fixed_density.ln_benchmark --fix-gamma --gamma-jack 0.4 --gamma-cross 1.0
  python -m fixed_density.ln_benchmark --output my.html
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import benchmark as bt
from .benchmark import (
    DEFAULT_CSV_PATH,
    GRADE_BOUND_TXT,
    GRADE_COLORS,
    GRADE_LABELS,
    ROOT_DIR,
    _fig_to_base64,
    _grade_badge,
    _setup_chinese_font,
    fit_exponential,
    grade_of,
    ln_log_value,
    ln_structure_scores,
    map_invert,
    predict_mapped,
    run_benchmark,
    select_calibration,
    summarize,
)

# LN 子类的三桶分组（用于“控手 / 密度 / 混合”对照）
BUCKETS = {
    "控手/抓准 (release+inverse)": ("release", "inverse"),
    "密度 (density+speed)": ("density", "speed"),
    "混合 (hybrid+tech+Unsigned)": ("hybrid", "tech", "Unsigned"),
}
BUCKET_COLORS = ("#e74c3c", "#3498db", "#e67e22")
BUCKET_ORDER = tuple(BUCKETS)

GAMMA_REL = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
GAMMA_JACK = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
GAMMA_CROSS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
GAMMA_HOLD = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)


def bucket_of(sub: str) -> str:
    for name, subs in BUCKETS.items():
        if sub in subs:
            return name
    return BUCKET_ORDER[0]


def _rmse(d) -> float:
    d = np.asarray(d, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def build_records(algo, params, rows):
    """跑一遍 benchmark，并把 RC 口径与 LN 口径的两套 value 都算出来。"""
    results = run_benchmark(algo, params, rows, source="extracted")
    calib = select_calibration(results)
    fit_rc = fit_exponential(calib)

    recs = [r for r in results if r["n"] is not None]
    for r in recs:
        cols = bt.load_extracted_columns(r["folder"], r["name"])
        od = bt.load_extracted_od(r["folder"], r["name"])
        r["od"] = od
        r["rho"] = bt.ln_hold_ratio(cols, od, float(params.get("ln_len_weight_gain", 0.0)))
        r["S_rel"], r["S_jack"], r["S_cross"], r["S_hold"] = ln_structure_scores(cols, params, od)
        r["has_ln"] = r["rho"] > 0.0
        r["rc_value"] = map_invert(
            fit_rc, predict_mapped(r["n"], fit_rc, r["irr"], r["ln_N"], r["chord"],
                                   r.get("ln_dur", 0.0)))
        r["dvalue_rc"] = r["rc_value"] - r["value"]
        r["grade_rc"] = grade_of(abs(r["dvalue_rc"]))
    return recs, fit_rc, calib


def search_gamma(ln_recs, fit_gamma: bool, fixed: tuple, h: float, c: float, lin: float = 0.0):
    """确定 ln 倍率权重：默认在网格上搜索 (γ_rel, γ_jack, γ_cross, γ_hold)，--fix-gamma 用给定值。

    F(ρ) 的形态参数 (h, c, lin) 不参与搜索：形态已由形态对比实验选定为饱和曲线
    （在 LN 谱 RMSE 与非 LN 谱尾部两个目标上同时优于幂律 ρ^m），见 --gamma-exp / --rho-c / --rho-lin。
    """
    if not fit_gamma:
        return fixed, None
    n_rc = np.array([r["n"] for r in ln_recs])
    v = np.array([r["value"] for r in ln_recs])
    f = np.array([bt.ln_rho_curve(r["rho"], h, c, lin) for r in ln_recs])
    sr = np.array([r["S_rel"] for r in ln_recs])
    sj = np.array([r["S_jack"] for r in ln_recs])
    sc = np.array([r["S_cross"] for r in ln_recs])
    sh = np.array([r["S_hold"] for r in ln_recs])
    best = None
    for gr, gj, gc, gh in itertools.product(GAMMA_REL, GAMMA_JACK, GAMMA_CROSS, GAMMA_HOLD):
        if gr == gj == gc == gh == 0.0:
            continue
        k = f * (gr * sr + gj * sj + gc * sc + gh * sh)
        if (k > 0).sum() < 5:
            continue
        a, b, _ = bt.fit_ln_log(k * n_rc, v)
        rmse = _rmse(a * np.log1p(b * k * n_rc) - v)
        if best is None or rmse < best[0]:
            best = (rmse, (gr, gj, gc, gh))
    return best[1], best[0]


def scatter_panel(ax, recs, xkey, ykey, title, show_legend=True, color_key="bucket"):
    groups = BUCKET_ORDER if color_key == "bucket" else sorted({r[color_key] for r in recs})
    for i, g in enumerate(groups):
        sub = [r for r in recs if r[color_key] == g]
        if not sub:
            continue
        ax.scatter([r[xkey] for r in sub], [r[ykey] for r in sub], s=42,
                   color=BUCKET_COLORS[i % len(BUCKET_COLORS)], marker=("o", "s", "^")[i % 3],
                   alpha=0.85, edgecolors="white", linewidths=0.6, label=g)
    xs = np.array([r[xkey] for r in recs], dtype=np.float64)
    ys = np.array([r[ykey] for r in recs], dtype=np.float64)
    lo = min(xs.min(), ys.min()) - 0.6
    hi = max(xs.max(), ys.max()) + 0.6
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.2, label="y = x")
    if recs and len(recs) >= 2:
        k1, k0 = np.polyfit(xs, ys, 1)
        grid = np.linspace(lo, hi, 100)
        ax.plot(grid, k1 * grid + k0, color="#c0392b", linewidth=1.6,
                label=f"线性拟合 y={k1:.3f}x+{k0:.3f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("实际 value")
    ax.set_ylabel("预测参考难度")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left")


def plot_overview(ln_recs, rice_recs) -> str:
    """左：原始 value 口径的预测值 点阵；右：ln 倍率 k 分布（非 LN 谱应贴近 0）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    ax = axes[0]
    for i, g in enumerate(BUCKET_ORDER):
        sub = [r for r in ln_recs if bucket_of(r["pattern"]) == g]
        if not sub:
            continue
        ax.scatter([r["value"] for r in sub], [r["ln_value_orig"] for r in sub], s=46,
                   color=BUCKET_COLORS[i], marker=("o", "s", "^")[i], alpha=0.88,
                   edgecolors="white", linewidths=0.6, label=g)
    lim = (0.0, 18.0)
    ax.plot(lim, lim, "--", color="#555555", linewidth=1.2, label="y = x")
    k1, k0 = np.polyfit([r["value"] for r in ln_recs],
                        [r["ln_value_orig"] for r in ln_recs], 1)
    grid = np.linspace(lim[0], lim[1], 100)
    ax.plot(grid, k1 * grid + k0, color="#c0392b", linewidth=1.6,
            label=f"线性拟合 y={k1:.3f}x+{k0:.3f}")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("实际 value")
    ax.set_ylabel("预测 value（反解回原始口径）")
    d_ln = np.array([r["dvalue_ln"] for r in ln_recs])
    ax.set_title("LN 谱面：原始 value 口径点阵 / 线性拟合 / y = x", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.03, 0.97, "原始口径 RMSE=%.2f  mean=%+.2f" % (_rmse(d_ln), d_ln.mean()),
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7", alpha=0.92))

    ax = axes[1]
    for i, g in enumerate(BUCKET_ORDER):
        sub = [r for r in ln_recs if bucket_of(r["pattern"]) == g]
        if not sub:
            continue
        ax.scatter([r["value"] for r in sub], [r["k"] for r in sub], s=46,
                   color=BUCKET_COLORS[i], marker=("o", "s", "^")[i], alpha=0.88,
                   edgecolors="white", linewidths=0.6, label=g)
    if rice_recs:
        ax.scatter([r["value"] for r in rice_recs], [r["k"] for r in rice_recs],
                   s=18, c="#bdc3c7", marker=".", alpha=0.5, label="非 LN 谱")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_xlabel("实际 value")
    ax.set_ylabel("ln 倍率 k")
    ax.set_title("ln 倍率 k 分布（非 LN 谱应贴近 0）", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_ln_map(fit_map) -> str:
    """LN 的 value 变换曲线 g_ln（横轴原始 value 0-18），含 y=x 参照与拟合范围。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    xs = np.linspace(0.0, 18.0, 300)
    ys = [bt.map_forward(fit_map, float(x)) for x in xs]
    ax.plot(xs, ys, color="#2980b9", linewidth=2.0, label="g_ln(value) 变换曲线")
    ax.plot(xs, xs, "--", color="#555555", linewidth=1.1, label="y = x")
    ax.axvspan(bt.LN_MAP_FIT_RANGE[0], bt.LN_MAP_FIT_RANGE[1], color="#f1c42f",
               alpha=0.16, label="拟合范围 [%g, %g]" % bt.LN_MAP_FIT_RANGE)
    for v0, _s in fit_map.get("map_knots", bt.LN_MAP_KNOTS):
        ax.axvline(v0, color="#bdc3c7", linewidth=0.9, linestyle=":")
    ax.set_xlim(0.0, 18.0)
    ax.set_ylim(0.0, 18.0)
    ax.set_xlabel("原始 value")
    ax.set_ylabel("变换后的值 g_ln(value)")
    ax.set_title("LN 的 value 变换曲线 g_ln", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_facets(ln_recs) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({r["pattern"] for r in ln_recs},
                    key=lambda p: (-sum(1 for r in ln_recs if r["pattern"] == p), p))
    ncol = 3
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.2 * nrow), squeeze=False)
    for i, g in enumerate(groups):
        sub = [r for r in ln_recs if r["pattern"] == g]
        scatter_panel(axes[i // ncol][i % ncol], sub, "value", "ln_value_orig",
                      f"subPattern = {g}（n={len(sub)}）", color_key="pattern")
    for j in range(len(groups), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_log_curve(ln_recs, a: float, b: float) -> str:
    """含下界对数标定曲线：n_LN → ln_value。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array([max(r["k"] * r["n"], 1e-6) for r in ln_recs])
    ys = np.array([r["value"] for r in ln_recs])
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.scatter(xs, ys, s=34, color="#3498db", alpha=0.75, edgecolors="white",
               linewidths=0.5, label="LN 谱面")
    grid = np.logspace(np.log10(max(xs.min(), 1e-3)), np.log10(xs.max()), 300)
    ax.plot(grid, a * np.log1p(b * grid), color="#c0392b", linewidth=1.9,
            label=f"拟合 ln_value = {a:.3f}·ln({b:.5f}·n_LN + 1)")
    ax.set_xscale("log")
    ax.set_xlabel("n_LN = k · n_rc（对数轴）")
    ax.set_ylabel("ln_value")
    ax.set_title("含下界对数标定：n_LN → ln_value", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


# 前端脚本：按内嵌 payload 绘制等级占比饼图（conic-gradient）与 dvalue 分布柱状图（canvas）。
# 直接继承 fixed_density.benchmark 的 RC 报告实现（DYNAMIC_REPORT_SCRIPT 中的 renderGrades /
# renderHist），去掉滑块联动后只做一次静态渲染。
LN_DYNAMIC_SCRIPT = """<script>
(function () {
  var CFG = JSON.parse(document.getElementById('payload').textContent);
  var L = CFG.labels, G = CFG.grades, rows = CFG.rows;
  function gradeOf(d) {
    var a = Math.abs(d);
    return a <= 0.2 ? G.labels[0] : a <= 0.5 ? G.labels[1] : a <= 1.0 ? G.labels[2] : G.labels[3];
  }
  function gIdx(lab) { return G.labels.indexOf(lab); }

  var gc = {}, total = rows.length;
  G.labels.forEach(function (l) { gc[l] = 0; });
  rows.forEach(function (r) { gc[gradeOf(r.d)]++; });
  var acc = 0, stops = [];
  G.labels.forEach(function (lab, i) {
    var frac = total ? gc[lab] / total : 0;
    stops.push(G.colors[i] + ' ' + (acc * 360).toFixed(2) + 'deg '
      + ((acc + frac) * 360).toFixed(2) + 'deg');
    acc += frac;
  });
  document.getElementById('grade-pie').style.background =
    total ? 'conic-gradient(' + stops.join(',') + ')' : 'none';

  var cv = document.getElementById('dhist'), ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '15px sans-serif'; ctx.fillStyle = '#222'; ctx.textAlign = 'left';
  var padL = 56, padR = 18, padT = 40, padB = 52;
  var ds = rows.map(function (r) { return r.d; });
  if (!ds.length) {
    ctx.fillStyle = '#888';
    ctx.fillText(L.d + ' 分布（无谱面）', padL, 24);
    return;
  }
  var lo = Math.floor(Math.min.apply(null, ds) / 0.1) * 0.1;
  var hi = Math.ceil(Math.max.apply(null, ds) / 0.1) * 0.1;
  var nb = Math.round((hi - lo) / 0.1) + 1;
  var counts = new Array(nb), i;
  for (i = 0; i < nb; i++) counts[i] = 0;
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
  ctx.textAlign = 'left'; ctx.font = '13px sans-serif';
  ctx.fillText(L.d + ' = ' + L.p + ' − ' + L.v + '（柱按区间着色，分度 0.1）', padL, base + 40);
  ctx.font = '15px sans-serif'; ctx.fillStyle = '#222';
  ctx.fillText(L.d + ' 分布图（分度区间 0.1，N=' + ds.length + '）', padL, 24);
  var lx = W - padR - 480;
  G.labels.forEach(function (lab, i) {
    ctx.fillStyle = G.colors[i]; ctx.fillRect(lx, 14, 12, 12);
    ctx.fillStyle = '#333'; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(lab + '（' + G.bounds[i] + '）', lx + 16, 24);
    lx += 122;
  });
})();
</script>"""


def render_html(ctx: dict) -> str:
    ln_recs = ctx["ln_recs"]
    rice_recs = ctx["rice_recs"]
    d_ln = np.array([r["dvalue_ln"] for r in ln_recs])

    # 误差区间表（ln_value 口径）
    overall = summarize(d_ln)
    total_g = sum(overall["grade_counts"].values())
    grade_rows = "".join(
        f"<tr><td>{_grade_badge(lb)}</td><td>{txt}</td><td>{overall['grade_counts'][lb]}</td>"
        f"<td>{100.0 * overall['grade_counts'][lb] / total_g:.2f}%</td></tr>"
        for lb, txt in zip(GRADE_LABELS, GRADE_BOUND_TXT))

    # 分 subPattern 表
    pats = sorted({r["pattern"] for r in ln_recs},
                  key=lambda p: (-sum(1 for r in ln_recs if r["pattern"] == p), p))
    pat_rows = ""
    for p in pats:
        sub = [r for r in ln_recs if r["pattern"] == p]
        s = summarize([r["dvalue_ln"] for r in sub])
        gc = s["grade_counts"]
        pat_rows += (
            f"<tr><td><b>{p}</b></td><td>{len(sub)}</td>"
            f"<td>{s['mean']:+.3f}</td><td>{s['var']:.3f}</td>"
            f"<td>{s['abs_mean']:.3f}</td><td>{s['rmse']:.3f}</td>"
            f"<td style='color:{GRADE_COLORS[0]}'>{gc['精确']}</td>"
            f"<td style='color:{GRADE_COLORS[1]}'>{gc['接近']}</td>"
            f"<td style='color:{GRADE_COLORS[2]}'>{gc['一般']}</td>"
            f"<td style='color:{GRADE_COLORS[3]}'>{gc['误判']}</td></tr>")

    # 三个桶表
    bucket_rows = ""
    for b in BUCKET_ORDER:
        sub = [r for r in ln_recs if bucket_of(r["pattern"]) == b]
        s = summarize([r["dvalue_ln"] for r in sub])
        bucket_rows += (f"<tr><td><b>{b}</b></td><td>{len(sub)}</td>"
                        f"<td>{s['mean']:+.3f}</td><td>{s['var']:.3f}</td>"
                        f"<td>{s['abs_mean']:.3f}</td><td>{s['rmse']:.3f}</td></tr>")

    # 明细表
    detail_rows = "".join(
        f"<tr><td>{r['pattern']}</td><td style='text-align:left'>{r['name']}</td>"
        f"<td>{r['value']:g}</td><td>{r['rho']:.3f}</td><td>{r['n']:.5g}</td>"
        f"<td>{r['k'] * r['n']:.5g}</td>"
        f"<td>{r['S_rel']:.3f}</td><td>{r['S_jack']:.3f}</td><td>{r['S_cross']:.3f}</td>"
        f"<td>{r['S_hold']:.3f}</td><td>{r['k']:.4f}</td>"
        f"<td>{r['ln_value']:.3f}</td><td>{r['dvalue_map']:+.3f}</td>"
        f"<td>{r['ln_value_orig']:.3f}</td><td>{r['dvalue_ln']:+.3f}</td>"
        f"<td><span style='color:{GRADE_COLORS[GRADE_LABELS.index(r['grade_ln'])]}'>"
        f"{r['grade_ln']}</span></td></tr>"
        for r in sorted(ln_recs, key=lambda x: (x["pattern"], x["name"])))

    g_rel, g_jack, g_cross, g_hold, g_exp, rho_c, rho_lin = ctx["gamma"]
    fit_map = ctx["fit_map"]
    absorb_gain = float(ctx["params"].get("ln_absorb_gain", 0.0))
    hold_ref = float(ctx["params"].get("ln_hold_ref_ms", 0.0))
    min_hold, od_gain, ramp = bt.LN_MIN_HOLD_MS, bt.LN_VALID_OD_GAIN, bt.LN_VALID_RAMP

    # 冻结的 RC 参数（非默认项）+ 完整复现命令
    diff = [(k, v) for k, v in ctx["params"].items()
            if abs(v - ctx["default_params"][k]) > 1e-12]
    set_args = " ".join("--set {}={!r}".format(k, v) for k, v in diff)
    repro = ("python -m fixed_density.ln_benchmark"
             + (" --no-irr" if ctx["no_irr"] else "")
             + (" " + set_args if set_args else "")
             + " --fix-gamma --gamma-rel {!r} --gamma-jack {!r} --gamma-cross {!r}"
               " --gamma-hold {!r} --gamma-exp {!r} --rho-c {!r} --rho-lin {!r}".format(
                   g_rel, g_jack, g_cross, g_hold, g_exp, rho_c, rho_lin)
             + (" --ln-a {!r} --ln-b {!r}".format(ctx["a"], ctx["b"])
                if ctx.get("pinned") else ""))
    rc_rows = ("".join(
        f"<tr><td><code>{k}</code></td><td>{v!r}</td><td>{ctx['default_params'][k]!r}</td></tr>"
        for k, v in sorted(diff)) or
        "<tr><td colspan='3' style='text-align:center;color:#888'>全部为默认值</td></tr>")
    fit_feats = ("chord + lnN，且 C=0（--no-irr，irr 信息由逐键局部熵三层增益承担）"
                 if ctx["no_irr"] else "irr、chord + lnN")

    payload = json.dumps({
        "labels": {"v": "实际 value", "p": "预测 value（原始口径）", "d": "dvalue"},
        "grades": {"labels": GRADE_LABELS, "bounds": GRADE_BOUND_TXT, "colors": GRADE_COLORS},
        "rows": [{"pa": r["pattern"], "nm": r["name"], "v": r["value"], "n": r["n"],
                  "pr": r["ln_value_orig"], "d": r["dvalue_ln"]} for r in ln_recs],
    }, ensure_ascii=False, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>osu!mania 4K · LN 参考难度（ln_value）Benchmark 报告</title>
<style>
  body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; margin: 0; background:#f4f6f8; color:#222; }}
  header {{ background:#1f2d3d; color:#fff; padding:26px 36px; }}
  header h1 {{ margin:0 0 6px; font-size:24px; }}
  header .sub {{ color:#bcc7d3; font-size:13px; }}
  main {{ max-width:1180px; margin:0 auto; padding:24px 28px 60px; }}
  section {{ background:#fff; border-radius:10px; padding:22px 26px; margin:18px 0;
            box-shadow:0 1px 4px rgba(0,0,0,.07); }}
  h2 {{ font-size:18px; border-left:5px solid #3498db; padding-left:10px; margin:0 0 16px; }}
  h3 {{ font-size:15px; margin:18px 0 8px; color:#2c3e50; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ border:1px solid #e0e4e8; padding:7px 10px; text-align:center; }}
  th {{ background:#f0f3f6; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .card {{ flex:1 1 150px; background:#f8fafc; border:1px solid #e3e8ee; border-radius:8px;
          padding:14px; text-align:center; }}
  .card .num {{ font-size:22px; font-weight:700; color:#1f2d3d; }}
  .card .lab {{ font-size:12px; color:#778592; margin-top:4px; }}
  .badge {{ color:#fff; padding:2px 10px; border-radius:10px; font-size:12px; }}
  img {{ max-width:100%; display:block; margin:10px auto; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }}
  details {{ margin-top:8px; }}
  summary {{ cursor:pointer; font-weight:600; color:#346; }}
  .scroll {{ max-height:560px; overflow:auto; margin-top:10px; }}
  .note {{ font-size:12px; color:#778592; }}
  .fml {{ background:#f8fafc; border:1px solid #e3e8ee; border-radius:8px; padding:12px 16px;
         font-family: Consolas, monospace; font-size:13px; line-height:1.9; }}
  .two-col {{ display:flex; gap:20px; flex-wrap:wrap; align-items:center; }}
  .two-col > div {{ flex:1 1 380px; }}
  .pie {{ width:230px; height:230px; border-radius:50%; margin:8px auto; border:1px solid #e0e4e8; }}
  #dhist {{ max-width:100%; display:block; margin:10px auto; }}
</style>
</head>
<body>
<header>
  <h1>osu!mania 4K · LN 参考难度（ln_value）Benchmark 报告</h1>
  <div class="sub">
    数据：{ctx['csv_name']}　|　LN 谱面 {len(ln_recs)} 张，非 LN 谱面 {len(rice_recs)} 张
    　|　生成时间：{ctx['now']}
  </div>
</header>
<main>

<section>
  <h2>1. 模型与参数</h2>
  <div class="fml">
    k = F(ρ) · (γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross + γ_hold·S_hold) &nbsp;（ln 倍率，<b>无常数项</b>）<br>
    F(ρ) = lin · ρ + (1 - lin) · (1 + c<sup>h</sup>) · ρ<sup>h</sup> / (ρ<sup>h</sup> + c<sup>h</sup>)
    &nbsp;（在线性 toe 与 Hill 型饱和之间混合：低 ρ 近似线性、高 ρ 趋平）<br>
    n_LN = k · n_rc &nbsp;（RC 求得的 n 直接按倍率缩放）<br>
    ln_value = a · ln(b · n_LN + 1) &nbsp;（含下界对数标定，即<b>变换后的值</b>）<br>
    g_ln(value) ≈ ln_value &nbsp;（value 变换曲线，仿 RC 的 g：softplus 分段、两端线性渐近、严格单调）<br>
    当前：γ_rel = {g_rel:g}，γ_jack = {g_jack:g}，γ_cross = {g_cross:g}，γ_hold = {g_hold:g}，
    h = {g_exp:g}，c = {rho_c:g}，lin = {rho_lin:g}，a = {ctx['a']:.4f}，b = {ctx['b']:.5f}<br>
    变换曲线：σ = {[round(x, 3) for x in fit_map['map_coef'][1:]]}，衔接点
    {[(round(v0, 2), s) for v0, s in fit_map['map_knots']]}，拟合范围
    [{bt.LN_MAP_FIT_RANGE[0]:g}, {bt.LN_MAP_FIT_RANGE[1]:g}]，
    斜率界 [{bt.LN_MAP_SLOPE_FLOOR:g}, {bt.LN_MAP_SLOPE_CEIL:g}]<br>
    两侧 R²：变换后口径 {fit_map['r2_log']:.4f} / 原始口径 {fit_map['r2_raw']:.4f}
    （拟合集 N = {ctx['n_fit']}，{'排除' if ctx['exclude_course'] else '含'}段位单曲 {ctx['n_course']} 张）
  </div>
  <p class="note">
    ρ = 长条占比（长条物件数 / 物件总数）；<br>
    S_rel（抓准）= 全谱 LN 松手事件“相邻松手间距倒数”的均值；<br>
    S_jack（控手）= 全谱 LN 松手事件“松手→同轨下一次按下间距倒数”的均值；<br>
    S_cross（跨轨）= 全谱 LN 松手事件“超出本地基线期望的跨轨动作速率”的均值；<br>
    S_hold（保持时长）= 保持时长 / 该轨物件平均间隔 的均值（无量纲，一根长条跨越多少个本地节拍）。<br>
    长度因子 h = ℓ/(ℓ+ℓ₀)（ℓ₀ = {hold_ref:g}ms）：S_rel / S_jack 逐根再乘 h，恢复“有效长条中
    越长越难”的单调性——纯长度门会把这条信息整段削掉。ℓ₀ ≤ 0 时关闭。<br>
    松手吸收折扣 λ = {absorb_gain:g}：roll 反键里 LN 头挨得很近，松手时刻距邻近起音不足
    半个本地节拍时，OD9 判定窗下可直接按两头中点打（当成高密度米），不需要独立控制；
    此时该次松手对 S_rel / S_jack 的贡献按 w = 1 - λ·a 打折（a∈[0,1] 为吸收度）。<br>
    时长有效性门 v = smoothstep(ℓ_med; lo, hi)（ms）：lo = {min_hold:g} + {od_gain:g}·(W_head+W_tail)，
    hi = {ramp:g}·lo，其中 W_head = (64-3·OD)/2、W_tail = 1.5·W_head（LN 尾判为 1.5 倍判定）。
    长条时长中位数过短的谱面（roll 反键 / 高速 LN 流，判定上等同单键）整层结构量清零，
    四个 S 同乘 v。OD8 时门为 [45, 49.5]ms、OD0 时 [60, 66]ms。<br>
    各项 S 只描述“每根 LN 有多强”，无 LN 时 ρ = 0 且各项皆 0 ⇒ k = 0 ⇒ n_LN = 0 ⇒ ln_value = 0，
    即“几乎没有 ln 的谱面显示 0”。<br>
    F(ρ) 用 Hill 型而非幂律 ρ<sup>m</sup>：幂律在高 ρ 区仍在快速变化，而 LN 谱残差与 ρ 正相关
    （响应偏强）；Hill 型高 ρ 趋平、低 ρ 陡降，可同时改善 LN 拟合与压低非 LN 谱尾部
    （实测 h=4,c=0.4：LN RMSE 1.002→0.884，非 LN 谱最大 7.95→0.63）。<br>
    纯 Hill 在 ρ ≪ c 时按 ρ<sup>h</sup> 塌向 0（h=6 时 ρ=1% 的 F 仅 1e-11），
    实战中“长条占比很小但确有 LN 难度”的谱面会被直接归零，故叠一条线性 toe
    （lin = {rho_lin:g}）：低 ρ 段近似线性（斜率 lin），仍保留 ρ 落在
    非 LN 谱 max≈0.14 与 LN 谱 min≈0.25 之间时的陡峭过渡。实测 lin=0.10 时
    LN RMSE 与纯 Hill 持平，lin=0.20 才略升（代价见报告末尾参数说明）。<br>
    S_hold 用“相对本地节拍”而非 1000/ℓ：后者会被搜索归零（又是密度代理），
    前者使 LN RMSE 0.884→0.864 且三个特征桶同时改善。
    a、b 由下方 LN 谱面最小二乘拟合（给定 b 时 a 有闭式解）。<br>
    <b>双口径</b>（与 RC 报告一致）：<b>变换后口径</b>直接比较 ln_value 与 g_ln(value)；
    <b>原始口径</b>把 ln_value 经 g_ln<sup>-1</sup> 反解回 value 再与实际 value 比较，
    正文的 RMSE / 等级 / 分桶统计均以原始口径为准。
  </p>
</section>

<section>
  <h2>2. 总览（LN 谱面 · 原始 value 口径）</h2>
  <div class="cards">
    <div class="card"><div class="num">{len(ln_recs)}</div><div class="lab">LN 谱面数</div></div>
    <div class="card"><div class="num">{overall['mean']:+.4f}</div><div class="lab">dvalue 均值</div></div>
    <div class="card"><div class="num">{overall['var']:.4f}</div><div class="lab">dvalue 方差</div></div>
    <div class="card"><div class="num">{overall['abs_mean']:.4f}</div><div class="lab">|dvalue| 均值</div></div>
    <div class="card"><div class="num">{overall['rmse']:.4f}</div><div class="lab">dvalue RMSE</div></div>
  </div>
  <div class="two-col" style="margin-top:18px">
    <div>
      <h3>误差区间分布</h3>
      <table>
        <tr><th>等级</th><th>区间</th><th>谱面数</th><th>占比</th></tr>
        {grade_rows}
      </table>
    </div>
    <div>
      <div id="grade-pie" class="pie"></div>
      <p class="note" style="text-align:center">dvalue 区间占比</p>
    </div>
  </div>
</section>

<section>
  <h2>3. 点阵图</h2>
  <img src="{ctx['fig_overview']}" alt="overview">
  <p class="note">左：原始口径的预测 value 点阵（虚线为 y = x，红实线为线性拟合）。
    右：ln 倍率 k 与实际 value 的关系，灰点为非 LN 谱面（按设计应贴近 0）。</p>
</section>

<section>
  <h2>4. 按 subPattern 分面</h2>
  <img src="{ctx['fig_facets']}" alt="facets">
  <h3>分 subPattern 统计</h3>
  <table>
    <tr><th>subPattern</th><th>N</th><th>dvalue 均值</th><th>dvalue 方差</th>
        <th>|dvalue| 均值</th><th>dvalue RMSE</th>
        <th>精确</th><th>接近</th><th>一般</th><th>误判</th></tr>
    {pat_rows}
  </table>
  <h3>按特征桶汇总</h3>
  <table>
    <tr><th>特征桶</th><th>N</th><th>dvalue 均值</th><th>dvalue 方差</th>
        <th>|dvalue| 均值</th><th>dvalue RMSE</th></tr>
    {bucket_rows}
  </table>
</section>

<section>
  <h2>5. 标定曲线与 value 变换</h2>
  <img src="{ctx['fig_logcurve']}" alt="logcurve">
  <p class="note">含下界对数标定 ln_value = a·ln(b·n_LN + 1) 的拟合曲线；横轴 n_LN = k·n_rc（对数轴）。</p>
  <img src="{ctx['fig_lnmap']}" alt="lnmap">
  <p class="note">LN 的 value 变换曲线 g_ln（仿 RC 的映射逻辑）：softplus 分段、两端线性渐近、全程光滑且严格单调，
    斜率限制在 [{bt.LN_MAP_SLOPE_FLOOR:g}, {bt.LN_MAP_SLOPE_CEIL:g}]。
    原始口径的预测 value 由 ln_value 经 g_ln<sup>-1</sup> 反解得到（二分求逆）。</p>
</section>

<section>
  <h2>6. dvalue 分布（分度 0.1）</h2>
  <canvas id="dhist" width="1080" height="300"></canvas>
</section>

<section>
  <h2>7. 逐谱明细</h2>
  <details open>
    <summary>展开 / 收起（{len(ln_recs)} 行）</summary>
    <div class="scroll">
      <table>
        <tr><th>subPattern</th><th>谱面</th><th>实际 value</th><th>ρ</th><th>n_rc</th><th>n_LN</th>
            <th>S_rel</th><th>S_jack</th><th>S_cross</th><th>S_hold</th><th>k</th>
            <th>ln_value<br><span style="font-weight:400;color:#888">变换后</span></th>
            <th>dvalue<br><span style="font-weight:400;color:#888">变换后</span></th>
            <th>预测 value<br><span style="font-weight:400;color:#888">原始口径</span></th>
            <th>dvalue<br><span style="font-weight:400;color:#888">原始口径</span></th>
            <th>等级</th></tr>
        {detail_rows}
      </table>
    </div>
  </details>
</section>

<section>
  <h2>8. 复现命令</h2>
  <div class="fml">{repro}</div>
  <h3>冻结的 RC 参数（与默认值不同的项）</h3>
  <table>
    <tr><th>参数</th><th>本次取值</th><th>默认值</th></tr>
    {rc_rows}
  </table>
  <p class="note">RC 侧拟合模型特征：{fit_feats}。</p>
  <p class="note">
    不带 <code>--fix-gamma</code> 时脚本会自行在网格上搜索 γ（本报告直接采用最终模型的 γ）。<br>
    注：a、b 默认在全部 {len(ln_recs)} 张 LN 谱上做样本内拟合；本报告用
    <code>--ln-a / --ln-b</code> 固定为最终模型（tosu 插件内硬编码）的取值，
    以便逐谱 ln_value 与插件输出严格一致。γ 已由调参阶段确定；尚未做留出验证。</p>
</section>

</main>
<script id="payload" type="application/json">{payload}</script>
{LN_DYNAMIC_SCRIPT}
</body>
</html>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="生成 LN 参考难度（ln_value）benchmark 报告")
    p.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="benchmark CSV 路径")
    p.add_argument("--output", default=str(ROOT_DIR / "report" / "benchmark_ln_report.html"),
                   help="输出 HTML 路径")
    p.add_argument("--gamma-rel", type=float, default=0.0, help="ln 倍率·抓准权重（配合 --fix-gamma）")
    p.add_argument("--gamma-jack", type=float, default=0.4, help="ln 倍率·控手权重（配合 --fix-gamma）")
    p.add_argument("--gamma-cross", type=float, default=1.0, help="ln 倍率·跨轨权重（配合 --fix-gamma）")
    p.add_argument("--gamma-hold", type=float, default=1.0, help="ln 倍率·保持时长权重（配合 --fix-gamma）")
    p.add_argument("--gamma-exp", type=float, default=6.0,
                   help="ln 倍率·ρ 饱和陡度 h（Hill 型 F(ρ)=(1+c^h)·ρ^h/(ρ^h+c^h)，默认 6.0）")
    p.add_argument("--rho-c", type=float, default=0.33,
                   help="ln 倍率·ρ 半饱和位置 c（默认 0.33，由 (h,c) 二维扫描确定）")
    p.add_argument("--rho-lin", type=float, default=0.1,
                   help="ln 倍率·ρ 线性 toe 权重 lin（默认 0.1：低 ρ 段近似线性，"
                        "少量长条的谱面不再被归零；0 退回纯 Hill）")
    p.add_argument("--fix-gamma", action="store_true",
                   help="直接使用 --gamma-* 给定值，不做网格搜索（默认搜索）")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="NAME=VALUE", help="算法参数覆盖，可多次使用")
    p.add_argument("--params-json", default=None,
                   help="从 JSON 的 params 字段读取算法参数覆盖"
                        "（如 fixed_density/params_final.json，用于复现最终模型）")
    p.add_argument("--ln-a", type=float, default=None,
                   help="固定 ln 标定 a（须与 --ln-b 同时给出）：跳过 a/b 拟合，"
                        "用于让报告与已冻结的最终模型严格一致")
    p.add_argument("--ln-b", type=float, default=None, help="固定 ln 标定 b")
    p.add_argument("--no-irr", action="store_true",
                   help="拟合模型去掉 irr 项（C=0），与 fixed_density.benchmark 的同名开关一致")
    p.add_argument("--exclude-course", action="store_true",
                   help="LN 拟合集排除“段位单曲”（dan course 马拉松）。默认包含：它们是低 "
                        "value 段（1~6）唯一的样本来源，排除后曲线在中低段会整体飘高")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    _setup_chinese_font()
    if args.no_irr:                     # 与 fixed_density.benchmark 的 main() 一致：C=0
        bt.FIT_USE_IRR = False

    algo = bt._REGISTRY["object_sequence"]
    params = algo.default_params()
    for item in args.overrides:
        key, raw = item.split("=", 1)
        params[key.strip()] = float(raw)
    if args.params_json:
        data = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        params.update({k: float(v) for k, v in data.get("params", {}).items()})

    rows = bt.load_csv(Path(args.csv))
    recs, fit_rc, calib = build_records(algo, params, rows)
    ln_recs = [r for r in recs if r["folder"] == "ln"]
    rice_recs = [r for r in recs if r["folder"] != "ln"]
    print("RC 标定集 N=%d  R2(log)=%.4f | LN 谱 %d 张 | 非 LN 谱 %d 张"
          % (len(calib), fit_rc["r2_log"], len(ln_recs), len(rice_recs)))

    g_exp = float(args.gamma_exp)
    rho_c = float(args.rho_c)
    rho_lin = float(args.rho_lin)
    # 拟合口径：默认把“段位单曲”（dan course 马拉松）也算进去——它们是低 value 段
    # （1~6）唯一的样本来源，且本身就是可玩的 LN 谱；排除后标定集只剩 38 张且集中在
    # 高 value 区，低 n_LN 段没有点，曲线在中低段会整体飘高（见标定曲线图）。
    # --exclude-course 可切回“只拟合非段位谱”用于对照。
    fit_ln = ([r for r in ln_recs if not bt.is_ln_course(r["name"])]
              if args.exclude_course else ln_recs)
    n_course = sum(1 for r in ln_recs if bt.is_ln_course(r["name"]))
    print("LN 拟合口径：%d 张（%s段位单曲 %d 张）"
          % (len(fit_ln), "排除" if args.exclude_course else "含", n_course))
    gamma, gamma_rmse = search_gamma(
        fit_ln, not args.fix_gamma,
        (args.gamma_rel, args.gamma_jack, args.gamma_cross, args.gamma_hold), g_exp, rho_c, rho_lin)
    if gamma_rmse is not None:
        print("倍率网格搜索最优: γ_rel=%.3f γ_jack=%.3f γ_cross=%.3f γ_hold=%.3f"
              "（h 固定 %.2f, c 固定 %.2f, lin 固定 %.2f）RMSE=%.4f"
              % (*gamma, g_exp, rho_c, rho_lin, gamma_rmse))
    else:
        print("倍率使用给定值: γ_rel=%.3f γ_jack=%.3f γ_cross=%.3f γ_hold=%.3f  h=%.2f c=%.2f lin=%.2f"
              % (*gamma, g_exp, rho_c, rho_lin))

    g_rel, g_jack, g_cross, g_hold = gamma
    n_rc = np.array([r["n"] for r in recs])
    k_all = np.array([bt.ln_rho_curve(r["rho"], g_exp, rho_c, rho_lin)
                      * (g_rel * r["S_rel"] + g_jack * r["S_jack"]
                         + g_cross * r["S_cross"] + g_hold * r["S_hold"])
                      for r in recs])
    fit_keys = {(r["folder"], r["name"]) for r in fit_ln}
    ln_idx = [i for i, r in enumerate(recs) if (r["folder"], r["name"]) in fit_keys]
    fit_map = bt.fit_ln_map(k_all[ln_idx] * n_rc[ln_idx],
                            np.array([recs[i]["value"] for i in ln_idx]),
                            pin_a=args.ln_a, pin_b=args.ln_b)
    a, b = fit_map["a"], fit_map["b"]
    for i, r in enumerate(recs):
        r["k"] = float(k_all[i])
        r["ln_value"] = ln_log_value(r["n"], r["k"], a, b)        # 变换后的值
        r["ln_map"] = bt.map_forward(fit_map, r["value"])         # 该谱 value 的变换目标 g_ln(value)
        r["dvalue_map"] = r["ln_value"] - r["ln_map"]             # 变换后口径残差
        r["ln_value_orig"] = bt.map_invert(fit_map, r["ln_value"])  # 反解回原始 value 口径
        r["dvalue_ln"] = r["ln_value_orig"] - r["value"]          # 原始口径残差
        r["grade_ln"] = grade_of(abs(r["dvalue_ln"]))
    print("拟合 a=%.4f  b=%.5f  变换后 R2=%.4f / 原始 R2=%.4f  曲线 σ=%s  （拟合集 N=%d）"
          % (a, b, fit_map["r2_log"], fit_map["r2_raw"],
             [round(x, 3) for x in fit_map["map_coef"][1:]], len(ln_idx)))

    d_ln = np.array([r["dvalue_ln"] for r in ln_recs])
    d_map = np.array([r["dvalue_map"] for r in ln_recs])
    d_rc = np.array([r["dvalue_rc"] for r in ln_recs])
    real_recs = [r for r in ln_recs if not bt.is_ln_course(r["name"])]
    cour_recs = [r for r in ln_recs if bt.is_ln_course(r["name"])]
    print("LN 谱: rc_value RMSE=%.3f mean=%+.3f" % (_rmse(d_rc), d_rc.mean()))
    print("  原始 value 口径: RMSE=%.3f mean=%+.3f | 变换后口径: RMSE=%.3f mean=%+.3f"
          % (_rmse(d_ln), d_ln.mean(), _rmse(d_map), d_map.mean()))
    for tag, sub in (("非合成谱", real_recs), ("合成谱", cour_recs)):
        print("  %-8s N=%2d  原始 RMSE=%.3f / 变换后 RMSE=%.3f"
              % (tag, len(sub), _rmse([r["dvalue_ln"] for r in sub]),
                 _rmse([r["dvalue_map"] for r in sub])))
    for bkt in BUCKET_ORDER:
        sub = [r for r in ln_recs if bucket_of(r["pattern"]) == bkt]
        print("  %-30s N=%2d  rc RMSE=%.3f -> ln RMSE=%.3f"
              % (bkt, len(sub), _rmse([r["dvalue_rc"] for r in sub]),
                 _rmse([r["dvalue_ln"] for r in sub])))
    v_rice = np.array([r["ln_value"] for r in rice_recs])
    k_rice = np.array([r["k"] for r in rice_recs])
    rho_rice = np.array([r["rho"] for r in rice_recs])
    rho_ln = np.array([r["rho"] for r in ln_recs])
    print("长条占比 ρ: LN 谱中位=%.3f | 非 LN 谱中位=%.4f max=%.3f"
          % (np.median(rho_ln), np.median(rho_rice), rho_rice.max()))
    print("非 LN 谱: k 中位=%.3f max=%.3f | ln_value 中位=%.3f max=%.3f"
          % (np.median(k_rice), k_rice.max(), np.median(v_rice), v_rice.max()))

    ctx = {
        "csv_name": Path(args.csv).name,
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ln_recs": ln_recs, "rice_recs": rice_recs,
        "gamma": (g_rel, g_jack, g_cross, g_hold, g_exp, rho_c, rho_lin), "a": a, "b": b,
        "fit_map": fit_map, "n_fit": len(ln_idx), "n_course": n_course,
        "exclude_course": bool(args.exclude_course),
        "params": params, "default_params": algo.default_params(), "no_irr": bool(args.no_irr),
        "pinned": bool(args.ln_a is not None and args.ln_b is not None),
        "fig_overview": plot_overview(ln_recs, rice_recs),
        "fig_facets": plot_facets(ln_recs),
        "fig_logcurve": plot_log_curve(ln_recs, a, b),
        "fig_lnmap": plot_ln_map(fit_map),
    }
    out = Path(args.output)
    out.write_text(render_html(ctx), encoding="utf-8")
    print("报告已生成：%s" % out)


if __name__ == "__main__":
    main()
