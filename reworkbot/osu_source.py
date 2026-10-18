"""谱面来源：镜像站下载、容器解包、回复附件读取。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from .config import OsuConfig

logger = logging.getLogger(__name__)

OSU_MAGIC = "osu file format"
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
    from . import archive  # 局部导入避免循环依赖

    beatmaps = archive.collect_from_zip(data, f"set_{set_id}.osz")
    logger.info("  解包完成：%d 张谱面", len(beatmaps))
    return beatmaps


def read_local_beatmap(path: str | Path) -> list[BeatmapFile]:
    """读取本地谱面文件：.osu / .mc / 任意 zip（.osz / .mcz / .zip，可嵌套）。"""
    from . import archive  # 局部导入避免循环依赖

    file_path = Path(path)
    if not file_path.exists():
        raise OsuSourceError(f"文件不存在：{file_path}")

    data = file_path.read_bytes()
    beatmaps = archive.collect(data, file_path.name)

    # 本地单文件：同目录的背景图
    if not archive.is_zip(data):
        background = _find_local_background(file_path)
        if background:
            for beatmap in beatmaps:
                if beatmap.background is None:
                    beatmap.background = background
    return beatmaps


def _find_local_background(path: Path) -> bytes | None:
    """为本地 .osu / .mc 找同目录的背景图。"""
    suffix = path.suffix.lower()
    if suffix == ".osu":
        sibling = _find_sibling_background(path)
        return sibling.read_bytes() if sibling is not None else None

    if suffix == ".mc":
        try:
            mc = json.loads(decode_osu_text(path.read_bytes()))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        name = str((mc.get("meta") or {}).get("background") or "")
        if not name:
            return None
        sibling = path.parent / Path(name).name
        return sibling.read_bytes() if sibling.exists() else None

    return None


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
