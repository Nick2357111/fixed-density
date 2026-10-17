"""Fixed Density v1 —— 完整工作流命令行（``.osu`` → JSON）。

输入：一个或多个 ``.osu`` 谱面文件（也可传入目录，递归处理其中的 ``.osu``）。
输出：JSON。对 RC 与 LN 两个通道分别给出：

===========  ==================================================================
字段         含义
===========  ==================================================================
``n``        分位数 n。RC 通道为 ``n_RC``；LN 通道为 ``n_LN = k · n_RC``
``fitted``   拟合结果（映射曲线空间）。RC 为 ``g_pred = ln(n) − C·chord − E·ln(N)``；
             LN 为 ``a · ln(1 + b · k · n)``
``mapped``   经映射曲线反解回原始 value 尺度后的值：RC 为 ``g⁻¹(g_pred)``，
             LN 为 ``g_ln⁻¹(ln_value)``（无 LN 时为 0）
===========  ==================================================================

另附 ``od``、``rate``、``chord``，以及 LN 通道的倍率 ``k``，便于核对中间量。

用法::

    python -m fixed_density.workflow "song.osu"
    python -m fixed_density.workflow "song.osu" --rate 1.5 --out result.json
    python -m fixed_density.workflow maps/ --out all.json
    python -m fixed_density.workflow "song.osu" --compact

说明：``--rate 1.5`` 表示 DT、``0.75`` 表示 HT；时间戳按 ``1/rate`` 缩放，
与 ``fixed_density.value`` 及 tosu 插件的处理方式一致。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .core import algo_object_sequence, chord_ratio, ln_multiplier, map_invert
from .reader import read_mania_beatmap_full
from .value import load_final_model


def _num(value: float):
    """JSON 不支持 NaN/Inf，非有限值统一输出为 null。"""
    value = float(value)
    return value if math.isfinite(value) else None


def _scale(columns: list[list[tuple[float, float]]], inv_rate: float):
    if inv_rate == 1.0:
        return columns
    return [[(t * inv_rate, l * inv_rate) for t, l in col] for col in columns]


def evaluate_file(path: str | Path, rate: float = 1.0, model: dict | None = None) -> dict:
    """对单个 ``.osu`` 谱面执行完整工作流，返回结果字典。"""
    model = model or load_final_model()
    params = dict(model["params"])
    fit = dict(model["fit"])
    ln_fit = model.get("ln_fit")
    ln_a = float(model["ln_calibration"]["a"])
    ln_b = float(model["ln_calibration"]["b"])

    rate_norm = float(rate) if rate and float(rate) > 0 else 1.0
    beatmap, columns = read_mania_beatmap_full(path)
    od = float(beatmap.od)
    lanes = [[(float(t), float(l)) for t, l in col] for col in columns]
    note_count = sum(len(col) for col in lanes)

    # RC 通道（变速时时间戳与长度按 1/rate 缩放）
    rc_lanes = _scale(lanes, 1.0 / rate_norm)
    n_rc = float(algo_object_sequence(rc_lanes, params)) if note_count else 0.0
    if not math.isfinite(n_rc) or n_rc <= 0.0:
        return {
            "file": str(path),
            "od": od,
            "rate": rate_norm,
            "rc": {"n": 0.0, "fitted": None, "mapped": None},
            "ln": {"n": 0.0, "fitted": 0.0, "mapped": 0.0, "k": 0.0},
            "chord": None,
        }

    chord = chord_ratio(rc_lanes)
    ln_n = math.log(max(note_count, 1))
    # 最终模型 C = 0：g_pred = ln n − E·chord − D·ln N
    g_pred = math.log(n_rc) - float(fit["E"]) * chord - float(fit["D"]) * ln_n
    rc_mapped = map_invert(fit, g_pred)

    # LN 通道：k 用原始轨道计算，S_rel / S_jack 随 rate 同倍缩放
    k = float(ln_multiplier(lanes, params, od, rate=rate_norm))
    n_ln = k * n_rc
    ln_fitted = ln_a * math.log1p(ln_b * n_ln)
    if k > 0.0 and ln_fit:
        ln_mapped = map_invert(dict(ln_fit), ln_fitted)
    else:
        ln_mapped = 0.0

    return {
        "file": str(path),
        "od": od,
        "rate": rate_norm,
        "rc": {"n": _num(n_rc), "fitted": _num(g_pred), "mapped": _num(rc_mapped)},
        "ln": {"n": _num(n_ln), "fitted": _num(ln_fitted), "mapped": _num(ln_mapped), "k": _num(k)},
        "chord": _num(chord),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Fixed Density v1 完整工作流：.osu → JSON（rc/ln 的 n、拟合值、映射后值）")
    parser.add_argument("paths", nargs="+", help=".osu 文件或包含 .osu 的目录")
    parser.add_argument("--rate", type=float, default=1.0, help="变速倍率（DT=1.5、HT=0.75，默认 1）")
    parser.add_argument("--model", default=None,
                        help="冻结模型 JSON（默认 fixed_density/params_final.json）")
    parser.add_argument("--out", default=None, help="输出 JSON 文件；省略时打印到标准输出")
    parser.add_argument("--compact", action="store_true", help="输出紧凑 JSON（不缩进）")
    args = parser.parse_args(argv)

    model = load_final_model(args.model)
    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        files.extend(sorted(path.rglob("*.osu")) if path.is_dir() else [path])

    results = []
    for osu_file in files:
        try:
            results.append(evaluate_file(osu_file, rate=args.rate, model=model))
        except Exception as exc:  # noqa: BLE001 —— 单个谱面失败不影响批量输出
            results.append({"file": str(osu_file), "error": f"{type(exc).__name__}: {exc}"})

    payload = results[0] if len(results) == 1 else results
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      indent=None if args.compact else 2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {out_path}（{len(results)} 个谱面）")
    else:
        print(text)


if __name__ == "__main__":
    main()
