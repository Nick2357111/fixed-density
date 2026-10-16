"""osu!mania 谱面读取器

把 ``.osu`` 文件解析为“逐轨 (时间, 长度) 列表”：

- 返回一个长度为 K（键数）的列表，第 k 个元素是第 k 轨的 ``[(开始时间ms, 长度ms), ...]``，按时间升序；
- 单键（circle）长度为 0；
- 长条（hold note）长度为 ``结束时间 − 开始时间``。

这是 Fixed Density 的输入格式：算法只需要 (时间, 长度) 元组列表，
不需要 BPM / 节拍线等任何其它元数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# osu! HitObject type 位标志（仅 mania 长条需要）
TYPE_HOLD = 1 << 7  # 128


@dataclass
class ManiaBeatmap:
    """解析后的 osu!mania 谱面数据。"""

    circle_size: int = 4  # 键数 K
    overall_difficulty: float = 8.0  # OD，用于 LN 判定窗
    hit_objects: list[dict] = field(default_factory=list)  # 原始 hitobject 列表

    @property
    def column_count(self) -> int:
        return self.circle_size

    @property
    def od(self) -> float:
        return self.overall_difficulty


def _parse_int(value: str, default: int = 0) -> int:
    """安全地把字符串解析成整数，支持 '5' 或 '5.0' 这种写法。"""
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return default


def read_mania_beatmap(file_path: str | Path) -> list[list[tuple[int, int]]]:
    """读取 osu!mania 谱面文件，返回每个轨道的 (时间, 长度) 列表。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到谱面文件: {path}")
    return _build_columns(_parse_osu_file(path))


def read_mania_beatmap_full(
    file_path: str | Path,
) -> tuple[ManiaBeatmap, list[list[tuple[int, int]]]]:
    """读取谱面，额外返回 ManiaBeatmap（含 OD / 键数）。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到谱面文件: {path}")
    beatmap = _parse_osu_file(path)
    return beatmap, _build_columns(beatmap)


def _parse_osu_file(file_path: Path) -> ManiaBeatmap:
    """解析 .osu 文件，提取 CircleSize / OverallDifficulty / HitObjects。"""
    beatmap = ManiaBeatmap()
    current_section = ""

    # osu 文件统一 UTF-8（可能带 BOM）
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1]
                continue

            if current_section == "Difficulty":
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    if key == "CircleSize":
                        beatmap.circle_size = _parse_int(value, 4)
                    elif key == "OverallDifficulty":
                        try:
                            beatmap.overall_difficulty = float(value.strip())
                        except ValueError:
                            beatmap.overall_difficulty = 8.0
            elif current_section == "HitObjects":
                obj = _parse_hit_object(line, beatmap.circle_size)
                if obj is not None:
                    beatmap.hit_objects.append(obj)

    return beatmap


def _parse_hit_object(line: str, key_count: int) -> dict | None:
    """解析一行 HitObject：``x,y,time,type,hitSample[,endTime,...]``。"""
    parts = line.split(",")
    if len(parts) < 5:
        return None
    try:
        x = int(float(parts[0]))
        time = int(float(parts[2]))
        obj_type = int(float(parts[3]))
    except (ValueError, IndexError):
        return None

    # 轨道索引 = floor(x * K / 512)，裁剪到 [0, K-1]
    column = (x * key_count) // 512
    column = 0 if column < 0 else (key_count - 1 if column >= key_count else column)

    length = 0
    if obj_type & TYPE_HOLD:  # 长条：parts[5] 为 endTime（可能带 ":hitSample" 后缀）
        end_field = parts[5] if len(parts) > 5 else ""
        try:
            end_time = int(float(end_field.split(":")[0]))
            length = max(end_time - time, 0)
        except ValueError:
            length = 0

    return {"column": column, "time": time, "length": length, "type": obj_type}


def _build_columns(beatmap: ManiaBeatmap) -> list[list[tuple[int, int]]]:
    """把 hitobject 列表按轨道分组，返回 K 个按时间升序的列表。"""
    columns: list[list[tuple[int, int]]] = [[] for _ in range(beatmap.column_count)]
    for obj in beatmap.hit_objects:
        col = obj["column"]
        if 0 <= col < beatmap.column_count:
            columns[col].append((obj["time"], obj["length"]))
    for col_notes in columns:
        col_notes.sort(key=lambda note: note[0])
    return columns


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "examples"
    target = Path(path)
    files = sorted(target.rglob("*.osu")) if target.is_dir() else [target]
    for osu_file in files:
        try:
            beatmap, columns = read_mania_beatmap_full(osu_file)
        except Exception as exc:  # noqa: BLE001
            print(f"{osu_file.name}: 解析失败 {type(exc).__name__}: {exc}")
            continue
        counts = "  ".join(f"轨{i + 1}={len(c)}" for i, c in enumerate(columns))
        print(f"{osu_file.name}: {beatmap.column_count}K  OD={beatmap.od:g}  {counts}")
