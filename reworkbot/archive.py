"""统一的谱面容器扫描。

把 .zip / .osz / .mcz（以及被改名的同内容文件）统一当作压缩包处理：
递归遍历包内**所有目录**，收集其中的
  - `.osu`  → 直接作为谱面
  - `.mc`   → 转换为 osu!mania 谱面
  - `.zip` / `.osz` / `.mcz` → 继续递归（支持嵌套压缩包）

对压缩炸弹做了条目数 / 解压体积 / 嵌套层数限制。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import zipfile
from pathlib import Path

from . import malody
from .osu_source import OSU_MAGIC, BeatmapFile, OsuSourceError, decode_osu_text

logger = logging.getLogger(__name__)

MAX_DEPTH = 4
MAX_ENTRIES = 20000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 ** 3  # 2 GB

_ZIP_SUFFIXES = {".zip", ".osz", ".mcz"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def is_zip(data: bytes) -> bool:
    """按文件头判断是否为 zip 容器（.osz / .mcz / .zip 都是）。"""
    return data[:2] == b"PK"


class _Budget:
    """压缩包扫描预算，防止深层嵌套 / 压缩炸弹。"""

    def __init__(self) -> None:
        self.entries = 0
        self.size = 0
        self.exhausted = False

    def take(self, size: int) -> bool:
        if self.exhausted:
            return False
        self.entries += 1
        self.size += max(0, size)
        if self.entries > MAX_ENTRIES or self.size > MAX_UNCOMPRESSED_BYTES:
            self.exhausted = True
            return False
        return True


def collect(data: bytes, filename: str) -> list[BeatmapFile]:
    """把任意谱面文件/容器统一解析为 BeatmapFile 列表。

    支持：.osu、.mc、以及 .zip / .osz / .mcz（含嵌套）。
    """
    suffix = Path(filename).suffix.lower()

    if is_zip(data):
        return collect_from_zip(data, filename)

    if suffix == ".mc" or malody.is_malody_payload(data):
        chart = malody.read_malody_chart(data, filename)
        return [_chart_to_beatmap(chart, filename)]

    text = decode_osu_text(data)
    if text.lstrip().startswith(OSU_MAGIC):
        return [BeatmapFile(filename=filename, text=text)]

    raise OsuSourceError(f"{filename} 不是有效的 .osu / .mc / 压缩包。")


def collect_from_zip(data: bytes, source_name: str) -> list[BeatmapFile]:
    """递归扫描压缩包，返回其中所有可分析的谱面。"""
    results: list[BeatmapFile] = []
    seen: set[str] = set()
    budget = _Budget()
    try:
        _scan(data, source_name, 0, budget, results, seen)
    except zipfile.BadZipFile as exc:
        raise OsuSourceError(f"{source_name} 不是有效的压缩包。") from exc

    if not results:
        raise OsuSourceError(f"{source_name} 内没有找到任何 .osu / .mc 谱面文件。")

    logger.info(
        "  %s：扫描完成，共 %d 张谱面（检查 %d 个条目）",
        source_name, len(results), budget.entries,
    )
    return results


def _scan(
    data: bytes,
    label: str,
    depth: int,
    budget: _Budget,
    results: list[BeatmapFile],
    seen: set[str],
) -> None:
    if depth > MAX_DEPTH:
        logger.warning("  %s：嵌套层数超过 %d，已跳过", label, MAX_DEPTH)
        return

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = [
            info
            for info in archive.infolist()
            if not info.is_dir() and not info.filename.startswith("__MACOSX")
        ]

        # 包内最大的图片作为兜底背景（按目录就近取更好，这里保持简单）
        images = [info for info in entries if Path(info.filename).suffix.lower() in _IMAGE_SUFFIXES]
        images.sort(key=lambda info: info.file_size, reverse=True)
        fallback_background = archive.read(images[0].filename) if images else None

        for info in entries:
            name = info.filename
            if not budget.take(info.file_size):
                logger.warning("  %s：条目过多或解压体积过大，停止扫描", label)
                return

            suffix = Path(name).suffix.lower()

            if suffix in _ZIP_SUFFIXES:
                _recurse(archive, name, label, depth, budget, results, seen)
                continue

            if suffix == ".osu":
                text = decode_osu_text(archive.read(name))
                if not text.lstrip().startswith(OSU_MAGIC):
                    logger.warning("  %s/%s：不是有效的 .osu，已跳过", label, name)
                    continue
                _append_valid(results, seen, text, fallback_background, Path(name).name)
                continue

            if suffix == ".mc":
                _collect_mc(archive, info, entries, label, fallback_background, results, seen)
                continue

            # 扩展名不认识：可能是被改名的容器，嗅探文件头
            if suffix not in _IMAGE_SUFFIXES and _looks_like_zip(archive, name):
                _recurse(archive, name, label, depth, budget, results, seen)


def _recurse(
    archive: zipfile.ZipFile,
    name: str,
    label: str,
    depth: int,
    budget: _Budget,
    results: list[BeatmapFile],
    seen: set[str],
) -> None:
    try:
        nested = archive.read(name)
    except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
        logger.warning("  %s/%s：读取失败，已跳过（%s）", label, name, exc)
        return
    if not is_zip(nested):
        logger.warning("  %s/%s：不是有效的压缩包，已跳过", label, name)
        return
    _scan(nested, f"{label}/{name}", depth + 1, budget, results, seen)


def _looks_like_zip(archive: zipfile.ZipFile, name: str) -> bool:
    try:
        with archive.open(name) as handle:
            return handle.read(2) == b"PK"
    except (RuntimeError, zipfile.BadZipFile, OSError):
        return False


def _collect_mc(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    entries: list[zipfile.ZipInfo],
    label: str,
    fallback_background: bytes | None,
    results: list[BeatmapFile],
    seen: set[str],
) -> None:
    name = info.filename
    raw = archive.read(name)
    try:
        mc = json.loads(decode_osu_text(raw))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("  %s/%s：Malody JSON 解析失败，已跳过（%s）", label, name, exc)
        return
    if not isinstance(mc, dict):
        logger.warning("  %s/%s：不是有效的 Malody 谱面，已跳过", label, name)
        return

    background_name = malody.background_name_of(mc)
    background = _read_image(archive, entries, background_name) or fallback_background
    chart = malody.convert_mc(
        mc,
        audio_name=malody.pick_audio_name(mc),
        background_name=background_name if background else "",
    )
    chart_file = _chart_to_beatmap(chart, name, background)
    if chart.ok:
        _append_valid(results, seen, chart_file.text, background, name, extra=chart_file)
        return
    results.append(chart_file)


def _read_image(
    archive: zipfile.ZipFile, entries: list[zipfile.ZipInfo], wanted: str
) -> bytes | None:
    if not wanted:
        return None
    target = Path(wanted).name.lower()
    for info in entries:
        if Path(info.filename).name.lower() == target:
            try:
                return archive.read(info.filename)
            except (RuntimeError, zipfile.BadZipFile, OSError):
                return None
    return None


def _append_valid(
    results: list[BeatmapFile],
    seen: set[str],
    osu_text: str,
    background: bytes | None,
    filename: str,
    *,
    extra: BeatmapFile | None = None,
) -> None:
    """按谱面文本去重后追加（同一谱面包里重复的谱面只保留一份）。"""
    key = hashlib.sha256(osu_text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    if key in seen:
        logger.info("  %s：重复谱面，已合并", filename)
        return
    seen.add(key)

    if extra is not None:
        extra.background = background
        results.append(extra)
    else:
        results.append(BeatmapFile(filename=filename, text=osu_text, background=background))


def _chart_to_beatmap(
    chart: malody.ConvertedChart, filename: str, background: bytes | None = None
) -> BeatmapFile:
    if chart.ok:
        return BeatmapFile(
            filename=Path(filename).name,
            text=chart.osu_text,
            background=background,
            name=chart.display_name,
        )
    return BeatmapFile(
        filename=Path(filename).name,
        text="",
        name=chart.display_name,
        error=chart.error,
    )
