"""benchmark 谱面提取脚本
=========================
把 benchmark/ 中 data.csv 引用的所有“有效谱面”提取为“物件 (头时间戳, 长度)”，
按原 folder/name 结构写入 benchmark_extracted/，供 fixed_density.benchmark 直接读取
（fixed_density.benchmark --source extracted，为默认数据源）。

输出结构（每个谱面一个 JSON）：
    benchmark_extracted/<folder>/<sanitize_stem(name)>.json
    {
      "folder": "...", "name": "...", "source": "<相对 benchmark 的 .osu 路径>",
      "match": "<定位方式>", "lane_count": K, "note_count": N,
      "columns": [[[t0, l0], [t1, l1], ...], ...]   # 每轨的 (头时间戳, 长度)（整数 ms，升序）
    }
    单键长度为 0，长条长度为 结束时间 - 开始时间。

“有效谱面”指在 data.csv 中、能被 resolve_osu_path 唯一定位且成功解析的谱面；
定位失败或解析失败的谱面会被跳过并在末尾汇总（与 benchmark 运行时的跳过集合一致）。

用法：
    python extract_benchmark.py
    python extract_benchmark.py --csv benchmark/data.csv --out benchmark_extracted
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .benchmark import (
    BENCHMARK_DIR,
    DEFAULT_CSV_PATH,
    EXTRACTED_DIR,
    load_csv,
    resolve_osu_path,
    sanitize_stem,
)
from .reader import read_mania_beatmap_full


def extract_note_times(path: Path) -> tuple[list[list[list[int]]], float]:
    """读取 .osu 谱面，返回 (每轨的 (头时间戳, 长度)（升序整数 ms）, OD)。"""
    beatmap, columns = read_mania_beatmap_full(path)
    return ([[[int(t), int(l)] for t, l in col] for col in columns],
            float(beatmap.overall_difficulty))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="提取 benchmark 谱面的物件 (头时间戳, 长度)")
    parser.add_argument("--csv", dest="csv_path", default=str(DEFAULT_CSV_PATH),
                        help="benchmark CSV 路径（提供待提取谱面清单）")
    parser.add_argument("--out", dest="out_dir", default=str(EXTRACTED_DIR),
                        help="输出目录，默认 benchmark_extracted")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rows = load_csv(Path(args.csv_path))
    print(f"读取 CSV：{args.csv_path}（{len(rows)} 行）")

    ok = 0
    skipped: list[tuple[str, str, str]] = []
    seen: dict[tuple[str, str], str] = {}          # (folder, 文件 stem) -> CSV 名
    conflicts: list[tuple[str, str, str]] = []     # 文件名冲突（会互相覆盖）
    t0 = time.perf_counter()

    for idx, row in enumerate(rows, 1):
        name, folder = row["name"], row["folder"]
        path, match = resolve_osu_path(folder, name)
        if path is None:
            skipped.append((folder, name, match))
            print(f"[{idx}/{len(rows)}] 跳过 {folder}/{name}：{match}")
            continue
        try:
            columns, od = extract_note_times(path)
        except Exception as exc:  # 单个谱面失败不影响整体提取
            skipped.append((folder, name, f"{type(exc).__name__}: {exc}"))
            print(f"[{idx}/{len(rows)}] 失败 {folder}/{name}：{type(exc).__name__}: {exc}")
            continue

        stem = sanitize_stem(name)
        key = (folder, stem.casefold())
        if key in seen and seen[key] != name:
            conflicts.append((folder, seen[key], name))
        seen[key] = name

        payload = {
            "folder": folder,
            "name": name,
            "source": path.relative_to(BENCHMARK_DIR).as_posix(),
            "match": match,
            "lane_count": len(columns),
            "note_count": sum(len(c) for c in columns),
            "od": od,
            "columns": columns,
        }
        out_path = out_dir / folder / f"{stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        ok += 1

    print(f"\n完成：提取 {ok} 个谱面 → {out_dir}"
          f"；跳过 {len(skipped)} 个；耗时 {time.perf_counter() - t0:.1f}s")
    if conflicts:
        print(f"警告：{len(conflicts)} 组文件名冲突（后写覆盖先写，请改名后重跑）：")
        for folder, a, b in conflicts:
            print(f"  [{folder}] {a}  与  {b}")
    if skipped:
        print("跳过明细：")
        for folder, name, reason in skipped:
            print(f"  [{folder}] {name}：{reason}")


if __name__ == "__main__":
    main()
