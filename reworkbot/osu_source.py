"""谱面来源：镜像站下载、压缩包解包、回复附件读取。"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from .config import OsuConfig

logger = logging.getLogger(__name__)

OSU_MAGIC = "osu file format"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_OSU_URL_RE = re.compile(r"osu\.ppy\.sh/(?:beatmapsets|s)/(\d+)(?:[#/](?:mania|osu|taiko|fruits)?/?(\d+))?")
_OSU_B_RE = re.compile(r"osu\.ppy\.sh/(?:beatmaps|b)/(\d+)")


class OsuSourceError(RuntimeError):
    """谱面获取失败。"""


@dataclass
class BeatmapFile:
    """一份谱面文本及其可选背景图。

    error 非空表示这份谱面在读取阶段就已被判定不可用（例如 Malody 模式/轨道数不符），
    此时 text 可以为空，调用方应直接产出错误提示。
    """

    filename: str
    text: str
    background: bytes | None = None
    name: str | None = None
    error: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or Path(self.filename).stem


def parse_osu_url(url: str) -> tuple[str, int] | None:
    """解析 osu! 官网链接，返回 (kind, id)。

    kind 为 "beatmap" 或 "set"。无法识别时返回 None。
    """
    match = _OSU_B_RE.search(url)
    if match:
        return "beatmap", int(match.group(1))
    match = _OSU_URL_RE.search(url)
    if match:
        set_id, beatmap_id = match.group(1), match.group(2)
        if beatmap_id:
            return "beatmap", int(beatmap_id)
        return "set", int(set_id)
    return None


async def _get_bytes(
    session: aiohttp.ClientSession,
    url: str,
    cfg: OsuConfig,
    *,
    allow_html: bool = False,
) -> bytes:
    headers = {"User-Agent": cfg.user_agent, "Accept": "*/*"}
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_sec)
    try:
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                raise OsuSourceError(f"下载失败（HTTP {resp.status}）：{url}")
            data = await resp.read()
    except aiohttp.ClientError as exc:
        raise OsuSourceError(f"下载出错：{url}（{exc}）") from exc

    if not allow_html and data.lstrip()[:1] == b"<":
        logger.warning("  镜像返回了 HTML 而非谱面文件：%s", url)
        raise OsuSourceError(f"镜像站返回的不是谱面文件（可能被限流或 ID 不存在）：{url}")
    return data


def decode_osu_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _validate_osu_text(text: str) -> None:
    if not text.lstrip().startswith(OSU_MAGIC):
        raise OsuSourceError("下载到的内容不是有效的 .osu 谱面文件。")


async def fetch_beatmap(
    session: aiohttp.ClientSession, cfg: OsuConfig, beatmap_id: int
) -> BeatmapFile:
    """按谱面 ID 拉取单个 .osu。"""
    url = cfg.beatmap_url.format(id=beatmap_id)
    logger.info("  下载 .osu：%s", url)
    data = await _get_bytes(session, url, cfg)
    text = decode_osu_text(data)
    _validate_osu_text(text)
    logger.info("  .osu 下载完成：%.1f KB", len(data) / 1024)
    return BeatmapFile(filename=f"{beatmap_id}.osu", text=text)


async def fetch_beatmapset(
    session: aiohttp.ClientSession, cfg: OsuConfig, set_id: int
) -> list[BeatmapFile]:
    """按谱包 ID 下载 .osz 并解包出全部 .osu。"""
    url = cfg.beatmapset_url.format(id=set_id)
    logger.info("  下载 .osz：%s", url)
    data = await _get_bytes(session, url, cfg)
    if data[:2] != b"PK":
        raise OsuSourceError("镜像站返回的不是 .osz 压缩包（可能被限流或 ID 不存在）。")
    logger.info("  .osz 下载完成：%.1f KB，正在解包…", len(data) / 1024)
    beatmaps = extract_beatmaps_from_zip(data)
    if not beatmaps:
        raise OsuSourceError("压缩包内没有找到任何 .osu 谱面文件。")
    logger.info("  解包完成：%d 张谱面", len(beatmaps))
    return beatmaps


def extract_beatmaps_from_zip(data: bytes) -> list[BeatmapFile]:
    """从 .osz / .zip 字节流中解包出全部谱面，并附带各自背景图。"""
    results: list[BeatmapFile] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".osu") and not name.startswith("__MACOSX")
            ]
            if not entries:
                return results

            image_entries = [
                name
                for name in archive.namelist()
                if Path(name).suffix.lower() in _IMAGE_SUFFIXES and not name.startswith("__MACOSX")
            ]
            image_entries.sort(key=lambda name: archive.getinfo(name).file_size, reverse=True)

            for name in entries:
                raw = archive.read(name)
                text = decode_osu_text(raw)
                if not text.lstrip().startswith(OSU_MAGIC):
                    continue
                background: bytes | None = None
                for image_name in image_entries:
                    if Path(image_name).name.lower() == Path(name).name.lower():
                        continue
                    if _background_matches(text, Path(image_name).name):
                        background = archive.read(image_name)
                        break
                if background is None and image_entries:
                    background = archive.read(image_entries[0])
                results.append(
                    BeatmapFile(filename=Path(name).name, text=text, background=background)
                )
    except zipfile.BadZipFile as exc:
        raise OsuSourceError("压缩包已损坏或格式不受支持。") from exc
    return results


def _background_matches(osu_text: str, image_name: str) -> bool:
    section = ""
    for raw_line in osu_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "Events":
            continue
        if line.startswith("0,0,"):
            match = re.search(r'"([^"]+)"', line)
            if match:
                return Path(match.group(1)).name.lower() == image_name.lower()
    return False


def read_local_beatmap(path: str | Path) -> list[BeatmapFile]:
    """读取本地 .osu / .osz / .zip / .mc / .mcz，统一返回 BeatmapFile 列表。"""
    from . import malody  # 局部导入避免与 malody 模块循环依赖

    file_path = Path(path)
    if not file_path.exists():
        raise OsuSourceError(f"文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix in (".mc", ".mcz"):
        return _malody_entries_to_beatmaps(malody.read_local_malody(file_path))

    if suffix in (".osz", ".zip"):
        data = file_path.read_bytes()
        # 容器里可能是 .osu（.osz）也可能是 .mc（被改名的 .mcz）
        try:
            entries = malody.read_malody_zip(data)
        except OsuSourceError:
            entries = []
        if entries:
            return _malody_entries_to_beatmaps(entries)

        beatmaps = extract_beatmaps_from_zip(data)
        if not beatmaps:
            raise OsuSourceError("压缩包内没有找到任何 .osu / .mc 谱面文件。")
        return beatmaps

    text = decode_osu_text(file_path.read_bytes())
    background = None
    sibling = _find_sibling_background(file_path)
    if sibling is not None:
        background = sibling.read_bytes()
    return [BeatmapFile(filename=file_path.name, text=text, background=background)]


def _malody_entries_to_beatmaps(
    entries: list[tuple[str, object, bytes | None]]
) -> list[BeatmapFile]:
    results: list[BeatmapFile] = []
    for entry_name, chart, background in entries:
        if chart.ok:  # type: ignore[attr-defined]
            results.append(
                BeatmapFile(
                    filename=entry_name,
                    text=chart.osu_text,  # type: ignore[attr-defined]
                    background=background,
                    name=chart.display_name,  # type: ignore[attr-defined]
                )
            )
        else:
            results.append(
                BeatmapFile(
                    filename=entry_name,
                    text="",
                    name=chart.display_name,  # type: ignore[attr-defined]
                    error=chart.error,  # type: ignore[attr-defined]
                )
            )
    if not results:
        raise OsuSourceError("Malody 谱包内没有可用的谱面。")
    return results


def _find_sibling_background(osu_path: Path) -> Path | None:
    """在同目录寻找谱面引用的背景图（本地谱面文件夹场景）。"""
    try:
        text = decode_osu_text(osu_path.read_bytes())
    except OSError:
        return None
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "Events" and line.startswith("0,0,"):
            match = re.search(r'"([^"]+)"', line)
            if match:
                candidate = osu_path.parent / Path(match.group(1)).name
                if candidate.exists():
                    return candidate
    return None
