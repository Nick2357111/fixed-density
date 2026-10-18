"""Malody 谱面（.mc / .mcz）解析，并转换为 osu!mania 的 .osu 文本。

.tosu 插件只吃 .osu，所以这里把 Malody 谱面转成一份临时的 .osu 再交给渲染器。

格式依据（对真实 .mcz 逆向 + 与 rmstZ / rconv 参考实现交叉验证）：
  - beat 三元组 [a, b, c] 表示 a + b/c 拍
  - 键数取 meta.mode_ext.column，meta.mode = 0 为 Key 模式
  - note[] 中 type==1 且无 column 的条目是 BGM 事件，其 offset 决定 AudioLeadIn
  - 长按用 endbeat 表示，输出 osu!mania 的 type 128
  - effect[].scroll 为正卷速 → osu 绿线（uninherited=0，beatLength = -100/scroll）

时间戳按 lazer 形式输出：保留浮点毫秒，不做取整。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .osu_source import OsuSourceError, decode_osu_text

logger = logging.getLogger(__name__)

MALODY_MODE_KEY = 0
SUPPORTED_KEYS = 4


def parse_beat(value: Any) -> float | None:
    """Malody 的拍位三元组 [小节, snap 索引, snap 大小] → 浮点拍数 a + b/c。"""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        measure, index, size = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (measure, index, size)) or size <= 0:
        return None
    return measure + index / size


def _format_time(value: float) -> str:
    """laz 形式：保留浮点，去掉多余的尾随零（整数值不带小数点）。"""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in ("", "-", "-0"):
        return "0"
    return text


class TempoMap:
    """把 Malody 的拍位映射到毫秒（逐段线性，非正 BPM 视为 SV 不计入）。"""

    def __init__(self, time_list: Any) -> None:
        if time_list is not None and not isinstance(time_list, list):
            raise OsuSourceError("Malody 谱面的 time 字段必须是数组。")

        points: list[tuple[float, float, int]] = []
        for entry in time_list or []:
            if not isinstance(entry, dict):
                continue
            beat = parse_beat(entry.get("beat"))
            try:
                bpm = float(entry.get("bpm"))
            except (TypeError, ValueError):
                continue
            if beat is None or not math.isfinite(bpm) or bpm <= 0:
                continue  # 非正 BPM 是变速（SV），不是真实 BPM
            try:
                meter = int(entry.get("signature") or 4)
            except (TypeError, ValueError):
                meter = 4
            points.append((beat, bpm, meter if meter > 0 else 4))

        points.sort(key=lambda item: item[0])
        deduped: list[tuple[float, float, int]] = []
        for point in points:
            if deduped and abs(deduped[-1][0] - point[0]) < 1e-9:
                deduped[-1] = point
            else:
                deduped.append(point)
        if not deduped:
            deduped = [(0.0, 120.0, 4)]
        if abs(deduped[0][0]) > 1e-9:
            deduped.insert(0, (0.0, deduped[0][1], deduped[0][2]))

        self.points = deduped
        self._ms_at = [0.0]
        for i in range(1, len(deduped)):
            beat0, bpm0, _ = deduped[i - 1]
            self._ms_at.append(self._ms_at[i - 1] + (deduped[i][0] - beat0) * 60000.0 / bpm0)

    def to_ms(self, beat: float) -> float:
        points = self.points
        if beat <= points[0][0]:
            return (beat - points[0][0]) * 60000.0 / points[0][1]
        for i in range(1, len(points)):
            if beat <= points[i][0]:
                return self._ms_at[i - 1] + (beat - points[i - 1][0]) * 60000.0 / points[i - 1][1]
        last = points[-1]
        return self._ms_at[-1] + (beat - last[0]) * 60000.0 / last[1]


@dataclass
class ConvertedChart:
    """一份转换结果；error 非空时 osu_text 为空。"""

    osu_text: str = ""
    title: str = ""
    artist: str = ""
    version: str = ""
    creator: str = ""
    key: int = 0
    notes: int = 0
    holds: int = 0
    error: str | None = None

    @property
    def display_name(self) -> str:
        name = " - ".join(part for part in (self.artist, self.title) if part) or "未知谱面"
        if self.version:
            name = f"{name} [{self.version}]"
        if self.creator:
            name = f"{name} // {self.creator}"
        return name

    @property
    def ok(self) -> bool:
        return self.error is None


def _read_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_key(mc: dict[str, Any]) -> int:
    """meta 里没有 mode_ext.column 时，用音符的最大列号推断键数。"""
    best = 0
    for note in mc.get("note") or []:
        if not isinstance(note, dict) or note.get("column") is None:
            continue
        best = max(best, _read_int(note.get("column"), -1) + 1)
    return best


def convert_mc(
    mc: dict[str, Any],
    *,
    audio_name: str = "",
    background_name: str = "",
    shift_ms: float = 0.0,
) -> ConvertedChart:
    """把解析后的 .mc JSON 转成 osu!mania 谱面文本。"""
    meta = mc.get("meta") or {}
    song = meta.get("song") or {}
    mode_ext = meta.get("mode_ext") or {}

    title = song.get("title") or song.get("titleorg") or "Unknown Title"
    artist = song.get("artist") or song.get("artistorg") or "Unknown Artist"
    creator = meta.get("creator") or "Unknown"
    version = str(meta.get("version") or "Converted")

    chart = ConvertedChart(title=str(title), artist=str(artist), version=version, creator=str(creator))

    # ---- 准入校验：模式与轨道数 ----
    raw_mode = meta.get("mode")
    mode = _read_int(raw_mode, -1) if raw_mode is not None else -1
    key = _read_int(mode_ext.get("column"), 0)
    if key <= 0:
        key = _infer_key(mc)

    if raw_mode is not None and mode != MALODY_MODE_KEY:
        chart.error = (
            f"该谱面不是 Malody 键模式（mode={mode}），本工具仅支持 Key 模式谱面。"
        )
        return chart
    if key <= 0:
        chart.error = "该 Malody 谱面缺少键数信息，无法判断轨道数。"
        return chart
    if key != SUPPORTED_KEYS:
        chart.error = f"该谱面为 {key}K，本工具仅支持 4K mania 谱面。"
        return chart
    chart.key = key

    raw_notes = mc.get("note")
    if not isinstance(raw_notes, list) or not raw_notes:
        chart.error = "该 Malody 谱面没有音符数据。"
        return chart
    if any(isinstance(note, (list, tuple)) for note in raw_notes):
        chart.error = "该 Malody 谱面使用了旧版数组式音符格式，暂不支持转换。"
        return chart

    tempo = TempoMap(mc.get("time"))

    # BGM 事件的 offset：Malody 语义是「延迟多久开始播放音频」，
    # 音频位置 = 谱面时间 - offset，因此物件时间整体前移 offset。
    audio_lead_in = 0
    for note in raw_notes:
        if not isinstance(note, dict):
            continue
        if _read_int(note.get("type"), 0) == 1 and note.get("column") is None:
            try:
                audio_lead_in = int(round(float(note.get("offset"))))
            except (TypeError, ValueError):
                continue
    time_shift = -audio_lead_in + shift_ms

    # ---- 音符 ----
    rows: list[tuple[float, int, int, float]] = []  # (time, x, type, end_time)
    skipped = 0
    for note in raw_notes:
        if not isinstance(note, dict):
            continue
        beat = parse_beat(note.get("beat"))
        if beat is None:
            skipped += 1
            continue
        if note.get("column") is None:
            continue  # 无列号的条目是 BGM 事件，已单独处理
        column = _read_int(note.get("column"), -1)
        if column < 0 or column >= key:
            skipped += 1
            continue

        time = tempo.to_ms(beat) + time_shift
        x = int(round(((column * 2 + 1) / (key * 2)) * 512))
        x = min(511, max(0, x))

        hit_type = 1
        end_time = time
        end_beat = parse_beat(note.get("endbeat"))
        if end_beat is not None and end_beat > beat:
            candidate = tempo.to_ms(end_beat) + time_shift
            if candidate > time:
                hit_type = 128
                end_time = candidate
        rows.append((time, x, hit_type, end_time))

    if skipped:
        logger.info("  Malody 转换：跳过 %d 个无效音符（beat 缺失或列号越界）", skipped)
    if not rows:
        chart.error = "该 Malody 谱面没有可用的音符。"
        return chart

    rows.sort(key=lambda row: (row[0], row[1]))
    chart.notes = len(rows)
    chart.holds = sum(1 for row in rows if row[2] == 128)
    chart.osu_text = _build_osu(
        title=str(title),
        artist=str(artist),
        creator=str(creator),
        version=version,
        key=key,
        audio_name=audio_name or str(song.get("file") or "audio.mp3"),
        background_name=background_name,
        audio_lead_in=audio_lead_in,
        tempo=tempo,
        effects=mc.get("effect") or [],
        rows=rows,
        time_shift=time_shift,
    )
    return chart


def _build_osu(
    *,
    title: str,
    artist: str,
    creator: str,
    version: str,
    key: int,
    audio_name: str,
    background_name: str,
    audio_lead_in: int,
    tempo: TempoMap,
    effects: list[Any],
    rows: list[tuple[float, int, int, float]],
    time_shift: float,
) -> str:
    lines: list[str] = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_name or 'audio.mp3'}",
        f"AudioLeadIn: {audio_lead_in}",
        "PreviewTime: -1",
        "Countdown: 0",
        "SampleSet: Soft",
        "StackLeniency: 0.7",
        "Mode: 3",
        "LetterboxInBreaks: 0",
        "SpecialStyle: 0",
        "WidescreenStoryboard: 0",
        "",
        "[Editor]",
        "Bookmarks: ",
        "DistanceSpacing: 1",
        "BeatDivisor: 4",
        "GridSize: 4",
        "TimelineZoom: 1",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"TitleUnicode:{title}",
        f"Artist:{artist}",
        f"ArtistUnicode:{artist}",
        f"Creator:{creator}",
        f"Version:{version}",
        "Source:",
        "Tags:malody converted",
        "BeatmapID:0",
        "BeatmapSetID:-1",
        "",
        "[Difficulty]",
        "HPDrainRate:8",
        f"CircleSize:{key}",
        "OverallDifficulty:8",
        "ApproachRate:5",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[Events]",
        "//Background and Video events",
    ]
    if background_name:
        lines.append(f'0,0,"{background_name}",0,0')
    lines += [
        "//Break Periods",
        "//Storyboard Layer 0 (Background)",
        "//Storyboard Layer 1 (Fail)",
        "//Storyboard Layer 2 (Pass)",
        "//Storyboard Layer 3 (Foreground)",
        "//Storyboard Sound Samples",
        "",
        "[TimingPoints]",
    ]

    # 红线（BPM）+ 绿线（SV）合并按时间排序；时间同样保留浮点
    points: list[tuple[float, float, int, int]] = []  # (time, beatLength, meter, uninherited)
    for beat, bpm, meter in tempo.points:
        points.append((tempo.to_ms(beat) + time_shift, 60000.0 / bpm, meter, 1))
    for effect in effects:
        if not isinstance(effect, dict):
            continue
        raw_scroll = effect.get("scroll")
        if not isinstance(raw_scroll, (int, float)) or not math.isfinite(raw_scroll):
            continue
        scroll = float(raw_scroll)
        if scroll <= 0:
            continue  # 负卷速是非标准效果，不转绿线
        beat = parse_beat(effect.get("beat"))
        if beat is None:
            continue
        points.append((tempo.to_ms(beat) + time_shift, -100.0 / scroll, 4, 0))
    points.sort(key=lambda item: item[0])
    for time, beat_length, meter, uninherited in points:
        lines.append(
            ",".join(
                [
                    _format_time(time),
                    _format_time(beat_length),
                    str(meter),
                    "2",
                    "0",
                    "100",
                    str(uninherited),
                    "0",
                ]
            )
        )

    lines += ["", "[HitObjects]"]
    for time, x, hit_type, end_time in rows:
        if hit_type == 128:
            lines.append(f"{x},192,{_format_time(time)},128,0,{_format_time(end_time)}:0:0:0:0:")
        else:
            lines.append(f"{x},192,{_format_time(time)},1,0,0:0:0:0:")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 文件读取
# ---------------------------------------------------------------------------


def _decode_json(raw: bytes) -> dict[str, Any]:
    return json.loads(decode_osu_text(raw))


def read_malody_chart(
    data: bytes, filename: str, *, background_name: str = ""
) -> ConvertedChart:
    """读取单个 .mc 字节流并转换。"""
    try:
        mc = _decode_json(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OsuSourceError(f"{filename} 不是有效的 Malody 谱面（JSON 解析失败：{exc}）。") from exc
    if not isinstance(mc, dict):
        raise OsuSourceError(f"{filename} 不是有效的 Malody 谱面。")
    return convert_mc(
        mc,
        audio_name=pick_audio_name(mc),
        background_name=background_name,
    )


def pick_audio_name(mc: dict[str, Any]) -> str:
    """BGM 事件的 sound 字段是最权威的音频名。"""
    for note in mc.get("note") or []:
        if isinstance(note, dict) and _read_int(note.get("type"), 0) == 1 and note.get("sound"):
            return Path(str(note["sound"])).name
    return ""


def background_name_of(mc: dict[str, Any]) -> str:
    """取 .mc 里声明的背景图文件名（仅基名）。"""
    name = str((mc.get("meta") or {}).get("background") or "")
    return Path(name).name if name else ""


def is_malody_payload(data: bytes) -> bool:
    """内容嗅探：Malody 的 .mc 是 JSON 且带 meta/time/note 结构。"""
    head = data.lstrip()[:1]
    if head not in (b"{", b"["):
        return False
    try:
        obj = json.loads(decode_osu_text(data[:200000]))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(obj, dict) and "note" in obj and ("meta" in obj or "time" in obj)
