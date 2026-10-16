"""Fixed Density v1 —— 端到端求值入口。

两种等价用法：

1. **从谱面文件到 value**：``evaluate_osu("xxx.osu")``
2. **从 (t, l) 元组列表到 value**：``evaluate(columns, od)``

其中 ``columns`` 就是 ``reader.read_mania_beatmap`` 的输出——逐轨的
``[(开始时间ms, 长度ms), ...]`` 列表（单键长度 0）。算法只吃这份数据，
不需要 BPM / 节拍线，因此任何能产出该列表的前端（osu! / Malody / 自研解析器）
都可以直接复用本模块。

返回值：

- ``rc_value``：RC（米）通道的难度值，量纲为原始 value（dan 标尺）；
- ``ln_value``：LN 参考难度，量纲为 g_ln(value)（变换后标尺，与 tosu 插件显示一致）；
- ``n`` / ``k``：中间量（分位数 n、LN 倍率 k），便于排查。

冻结参数与映射曲线来自 ``params_final.json``（与 tosu 插件逐字一致）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .core import (
    algo_object_sequence,
    chord_ratio,
    ln_multiplier,
    map_invert,
)
from .reader import read_mania_beatmap_full

PARAMS_PATH = Path(__file__).resolve().parent / "params_final.json"


def load_final_model(path: str | Path | None = None) -> dict:
    """读取冻结模型（算法参数 + 映射曲线 + LN 标定）。"""
    with open(Path(path) if path else PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _empty() -> dict:
    nan = float("nan")
    return {"n": nan, "rc_value": nan, "ln_value": nan, "k": 0.0, "chord": 1.0}


def evaluate(
    columns: list[list[tuple[float, float]]],
    od: float = 8.0,
    rate: float = 1.0,
    model: dict | None = None,
) -> dict:
    """从逐轨 ``(时间, 长度)`` 元组列表计算 Fixed Density 的 rc_value / ln_value。

    Args:
        columns: 长度为 K 的列表，第 k 项是该轨 ``[(t, l), ...]``（升序）。
        od: 谱面 OverallDifficulty（用于 LN 判定窗与长度有效性门；缺失时用 8.0）。
        rate: 变速倍率（1=原速、1.5=DT、0.75=HT，lazer 任意正速同理）。
        model: 冻结模型字典；默认读取 params_final.json。

    Returns:
        dict(rc_value, ln_value, n, k, chord)。无有效物件时 rc_value/ln_value 为 NaN。
    """
    model = model or load_final_model()
    params = dict(model["params"])
    fit = dict(model["fit"])
    ln_a = float(model["ln_calibration"]["a"])
    ln_b = float(model["ln_calibration"]["b"])

    inv_rate = 1.0 / rate if rate and rate > 0 else 1.0
    lanes = [[(float(t), float(l)) for t, l in col] for col in columns]
    if sum(len(c) for c in lanes) == 0:
        return _empty()

    # RC 通道：变速时时间戳与长度一并按 1/rate 缩放
    rc_lanes = lanes
    if inv_rate != 1.0:
        rc_lanes = [[(t * inv_rate, l * inv_rate) for t, l in col] for col in lanes]

    n = float(algo_object_sequence(rc_lanes, params))
    if not math.isfinite(n) or n <= 0.0:
        return _empty()

    chord = chord_ratio(rc_lanes)
    ln_n = math.log(max(sum(len(c) for c in rc_lanes), 1))
    # 映射曲线：g_pred = ln n − E·chord − D·ln N（irr 系数 C = 0，见 params_final.json）
    g_pred = math.log(n) - float(fit["E"]) * chord - float(fit["D"]) * ln_n
    rc_value = map_invert(fit, g_pred)

    # LN 参考难度：k 用原始（未缩放）轨道计算，S_rel / S_jack 随 rate 同倍缩放
    k = ln_multiplier(lanes, params, float(od), rate=rate)
    ln_value = ln_a * math.log1p(ln_b * k * n)

    return {"n": n, "rc_value": rc_value, "ln_value": ln_value, "k": k, "chord": chord}


def evaluate_osu(path: str | Path, rate: float = 1.0, model: dict | None = None) -> dict:
    """从 ``.osu`` 文件计算 Fixed Density 的 rc_value / ln_value（自动取 OD）。"""
    beatmap, columns = read_mania_beatmap_full(path)
    return evaluate(columns, od=beatmap.od, rate=rate, model=model)


def _fmt(res: dict) -> str:
    return (f"rc_value = {res['rc_value']:.4f}   ln_value = {res['ln_value']:.4f}   "
            f"n = {res['n']:.6f}   k = {res['k']:.6f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Fixed Density v1 端到端求值：.osu 谱面 → rc_value / ln_value")
    p.add_argument("paths", nargs="+", help=".osu 文件或包含它们的目录")
    p.add_argument("--rate", type=float, default=1.0, help="变速倍率（DT=1.5、HT=0.75，默认 1）")
    p.add_argument("--model", default=None, help="冻结模型 JSON（默认 fixed_density/params_final.json）")
    args = p.parse_args(argv)

    model = load_final_model(args.model)
    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        files.extend(sorted(path.rglob("*.osu")) if path.is_dir() else [path])

    for osu_file in files:
        try:
            res = evaluate_osu(osu_file, rate=args.rate, model=model)
        except Exception as exc:  # noqa: BLE001
            print(f"{osu_file.name}: 失败 {type(exc).__name__}: {exc}")
            continue
        print(f"{osu_file.name}\n    {_fmt(res)}")


if __name__ == "__main__":
    main()
