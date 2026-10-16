"""命令解析与业务编排。"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import aiohttp

from . import malody, osu_source
from .beatmap import BeatmapInfo, parse_beatmap
from .config import AppConfig
from .errorcard import render_error_card
from .onebot import (
    OneBotClient,
    extract_plain_text,
    forward_node,
    image_segment,
    text_segment,
)
from .osu_source import BeatmapFile, OsuSourceError
from .renderer import RenderError, Renderer

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "【Rework 谱面分析】\n"
    "/rework <谱面ID>       分析单张谱面\n"
    "/rework bid<谱面ID>    同上（显式指定谱面，亦可 b<ID>）\n"
    "/rework sid<谱包ID>    分析整个谱包（合并转发发送，亦可 s<ID>）\n"
    "/rework <osu! 链接>   支持官网谱面 / 谱包链接\n"
    "回复一条谱面文件并发送 /rework\n"
    "  支持 .osu / .osz（osu!）与 .mc / .mcz（Malody）\n"
    "（仅支持 4K mania / Malody Key 4K 谱面）"
)

_ID_RE = re.compile(r"^\d+$")
# b / bid / s / sid 与数字之间不再需要空格：bid123、bid 123、s456 均有效
_KIND_ID_RE = re.compile(r"^(bid|sid|b|s)\s*(\d+)$", re.IGNORECASE)
_request_seq = itertools.count(1)
_KIND_ALIASES = {
    "b": "beatmap", "bid": "beatmap", "beatmap": "beatmap",
    "s": "set", "sid": "set", "set": "set", "setid": "set",
}


@dataclass
class Command:
    kind: str  # beatmap / set / auto / reply / help
    value: int | None = None
    payload: "BeatmapFile | None" = None  # 探测阶段已下载的谱面，避免重复请求


@dataclass
class ResultItem:
    name: str
    info: BeatmapInfo | None
    png: bytes | None = None
    error: str | None = None


def parse_command(text: str, prefix: str) -> Command | None:
    """解析 /rework 指令，非指令返回 None。"""
    normalized = text.strip().replace("／", "/")
    pattern = re.compile(rf"^{re.escape(prefix)}(\s+.*)?$", re.IGNORECASE)
    match = pattern.match(normalized)
    if not match:
        return None

    rest = normalized[len(prefix):].strip()
    if not rest:
        return Command(kind="reply")

    url_kind = osu_source.parse_osu_url(rest)
    if url_kind is not None:
        return Command(kind=url_kind[0], value=url_kind[1])

    kind_match = _KIND_ID_RE.match(rest)
    if kind_match:
        return Command(kind=_KIND_ALIASES[kind_match.group(1).lower()], value=int(kind_match.group(2)))

    tokens = rest.split()
    head = tokens[0].lower()
    if head in _KIND_ALIASES and len(tokens) >= 2 and tokens[1].isdigit():
        return Command(kind=_KIND_ALIASES[head], value=int(tokens[1]))

    if _ID_RE.match(tokens[0]):
        return Command(kind="auto", value=int(tokens[0]))

    return Command(kind="help")


class ReworkService:
    def __init__(self, cfg: AppConfig, client: OneBotClient, renderer: Renderer) -> None:
        self.cfg = cfg
        self.client = client
        self.renderer = renderer
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------ 事件入口
    async def handle_event(self, event: dict[str, Any]) -> None:
        if event.get("user_id") == event.get("self_id"):
            return
        if not self._allowed(event):
            return

        message = event.get("message")
        text = extract_plain_text(message if isinstance(message, list) else None)
        command = parse_command(text, self.cfg.onebot.command_prefix)
        if command is None:
            return

        message_type = event.get("message_type", "private")
        group_id = event.get("group_id")
        user_id = event.get("user_id")

        tag = f"[#{next(_request_seq)}]"
        scope = f"群 {group_id}" if message_type == "group" else f"用户 {user_id}"
        started = time.perf_counter()
        logger.info("%s 收到指令：%r（来自%s）", tag, text.strip(), scope)

        async def reply(message_segments: list[dict[str, Any]]) -> None:
            await self.client.send_msg(
                message_type=message_type,
                group_id=group_id,
                user_id=user_id,
                message=message_segments,
            )

        try:
            command = await self._resolve_auto(command, tag)
            if command.kind == "help":
                logger.info("%s 返回用法说明", tag)
                await reply([text_segment(HELP_TEXT)])
                return

            files, ack = await self._collect_sources(command, event, tag)
            if ack:
                await reply([text_segment(ack)])
            if not files:
                logger.warning("%s 没有获取到任何谱面文件", tag)
                await reply([text_segment("没有获取到任何谱面文件。\n\n" + HELP_TEXT)])
                return

            await self._render_and_send(files, message_type, group_id, user_id, event, tag)
            logger.info("%s 处理完成，总用时 %.1fs", tag, time.perf_counter() - started)
        except OsuSourceError as exc:
            logger.warning("%s 获取谱面失败：%s", tag, exc)
            await reply([text_segment(f"获取谱面失败：{exc}")])
        except RenderError as exc:
            logger.error("%s 渲染失败：%s", tag, exc)
            await reply([text_segment(f"渲染失败：{exc}")])
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s 处理 /rework 请求时发生异常", tag)
            await reply([text_segment(f"处理失败：{exc}")])

    def _allowed(self, event: dict[str, Any]) -> bool:
        access = self.cfg.access
        message_type = event.get("message_type", "private")
        if message_type == "group":
            if not access.allow_group:
                return False
            allowlist = [str(g) for g in access.group_allowlist]
            if allowlist and str(event.get("group_id")) not in allowlist:
                return False
        elif not access.allow_private:
            return False
        denylist = [str(u) for u in access.user_denylist]
        return str(event.get("user_id")) not in denylist

    # ------------------------------------------------------------------ 谱面来源
    async def _resolve_auto(self, command: Command, tag: str) -> Command:
        """裸 ID：先按谱面 ID 试，失败再按谱包 ID 试。"""
        if command.kind != "auto" or command.value is None:
            return command
        assert self._session is not None
        logger.info("%s ID=%s 未指定类型，先按谱面 ID 探测…", tag, command.value)
        try:
            fetched = await osu_source.fetch_beatmap(self._session, self.cfg.osu, command.value)
            logger.info("%s ID=%s 识别为单张谱面", tag, command.value)
            return Command(kind="beatmap", value=command.value, payload=fetched)
        except OsuSourceError as exc:
            logger.info("%s ID=%s 不是可用谱面（%s），改按谱包处理", tag, command.value, exc)
            return Command(kind="set", value=command.value)

    async def _collect_sources(
        self, command: Command, event: dict[str, Any], tag: str
    ) -> tuple[list[BeatmapFile], str | None]:
        if command.kind in ("beatmap", "set"):
            assert self._session is not None and command.value is not None
            if command.kind == "beatmap":
                if command.payload is not None:
                    return [command.payload], None
                logger.info("%s 正在下载谱面 ID=%s …", tag, command.value)
                return [await osu_source.fetch_beatmap(self._session, self.cfg.osu, command.value)], None

            logger.info("%s 正在下载谱包 ID=%s …", tag, command.value)
            beatmaps = await osu_source.fetch_beatmapset(self._session, self.cfg.osu, command.value)
            limit = self.cfg.osu.max_beatmaps_per_pack
            logger.info("%s 谱包解析完成：共 %d 张谱面（上限 %d）", tag, len(beatmaps), limit)
            if len(beatmaps) > limit:
                raise OsuSourceError(
                    f"该谱包包含 {len(beatmaps)} 张谱面，超过单次上限 {limit} 张，请改用谱面 ID。"
                )
            return beatmaps, f"谱包共 {len(beatmaps)} 张谱面，正在分析…"

        # 回复文件
        logger.info("%s 解析回复消息中的附件…", tag)
        files = await self._files_from_reply(event, tag)
        if not files:
            return [], None
        beatmaps: list[BeatmapFile] = []
        for filename, data in files:
            extracted = self._beatmaps_from_attachment(filename, data)
            logger.info("%s 附件 %s：%d 字节 -> %d 张谱面", tag, filename, len(data), len(extracted))
            beatmaps.extend(extracted)
        limit = self.cfg.osu.max_beatmaps_per_pack
        if len(beatmaps) > limit:
            raise OsuSourceError(f"文件包含 {len(beatmaps)} 张谱面，超过单次上限 {limit} 张。")
        ack = f"共 {len(beatmaps)} 张谱面，正在分析…" if len(beatmaps) > 1 else None
        return beatmaps, ack

    def _beatmaps_from_attachment(self, filename: str, data: bytes) -> list[BeatmapFile]:
        suffix = Path(filename).suffix.lower()

        # Malody 谱面：.mc（单张）/ .mcz（谱包，zip）
        if suffix == ".mcz":
            return self._malody_to_beatmaps(filename, malody.read_malody_zip(data))
        if suffix == ".mc":
            return self._malody_to_beatmaps(
                filename, [(filename, malody.read_malody_chart(data, filename), None)]
            )

        if suffix in (".osz", ".zip") or data[:2] == b"PK":
            # 容器里的实际内容可能是 .osu（.osz）也可能是 .mc（被改名的 .mcz）
            malody_entries: list[tuple[str, Any, bytes | None]] = []
            try:
                malody_entries = malody.read_malody_zip(data)
            except OsuSourceError:
                malody_entries = []
            if malody_entries:
                return self._malody_to_beatmaps(filename, malody_entries)

            beatmaps = osu_source.extract_beatmaps_from_zip(data)
            if not beatmaps:
                raise OsuSourceError(f"{filename} 内没有找到任何 .osu / .mc 谱面文件。")
            return beatmaps

        text = osu_source.decode_osu_text(data)
        if text.lstrip().startswith(osu_source.OSU_MAGIC):
            return [BeatmapFile(filename=filename, text=text)]

        # 扩展名不可靠时按内容嗅探 Malody JSON
        if malody.is_malody_payload(data):
            return self._malody_to_beatmaps(
                filename, [(filename, malody.read_malody_chart(data, filename), None)]
            )

        raise OsuSourceError(f"{filename} 不是有效的 .osu / .mc 谱面文件。")

    def _malody_to_beatmaps(
        self, filename: str, entries: list[tuple[str, Any, bytes | None]]
    ) -> list[BeatmapFile]:
        """把 Malody 转换结果包装成统一的 BeatmapFile（含已判定不可用的条目）。"""
        results: list[BeatmapFile] = []
        for entry_name, chart, background in entries:
            name = entry_name or filename
            if chart.ok:
                logger.info(
                    "  Malody 转换完成：%s（%dK，%d 音符 / %d 长按）",
                    chart.display_name, chart.key, chart.notes, chart.holds,
                )
                self._save_temp_text(Path(name).stem, chart.osu_text, ".osu")
                results.append(
                    BeatmapFile(
                        filename=name,
                        text=chart.osu_text,
                        background=background,
                        name=chart.display_name,
                    )
                )
            else:
                logger.warning("  Malody 谱面不可用：%s -> %s", chart.display_name, chart.error)
                results.append(
                    BeatmapFile(filename=name, text="", name=chart.display_name, error=chart.error)
                )
        if not results:
            raise OsuSourceError(f"{filename} 内没有可用的 Malody 谱面。")
        return results

    async def _files_from_reply(self, event: dict[str, Any], tag: str) -> list[tuple[str, bytes]]:
        reply_id = None
        for segment in event.get("message") or []:
            if segment.get("type") == "reply":
                reply_id = segment.get("data", {}).get("id")
                break
        if reply_id is None:
            logger.info("%s 未检测到回复消息", tag)
            return []

        assert self._session is not None
        logger.info("%s 读取被回复消息 %s 的内容…", tag, reply_id)
        reply_event = await self.client.get_msg(reply_id)
        data = reply_event.get("data") or {}
        segments = data.get("message") or []
        message_type = event.get("message_type", "private")

        results: list[tuple[str, bytes]] = []
        for segment in segments:
            if segment.get("type") != "file":
                continue
            payload = segment.get("data") or {}
            filename = payload.get("file") or payload.get("name") or "attachment.osu"
            url = payload.get("url")
            if not url and payload.get("file_id"):
                try:
                    if message_type == "group":
                        fetched = await self.client.get_group_file_url(
                            event.get("group_id"), payload["file_id"]
                        )
                    else:
                        fetched = await self.client.get_private_file_url(
                            event.get("user_id"), payload["file_id"]
                        )
                    url = (fetched.get("data") or {}).get("url")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s 获取文件下载地址失败：%s", tag, exc)
            if not url:
                logger.warning("%s 附件 %s 没有可用的下载地址，已跳过", tag, filename)
                continue
            logger.info("%s 下载附件 %s …", tag, filename)
            data_bytes = await self._download(url)
            logger.info("%s 附件 %s 下载完成：%.1f KB", tag, filename, len(data_bytes) / 1024)
            results.append((filename, data_bytes))
        return results

    async def _download(self, url: str) -> bytes:
        if url.startswith("file://"):
            parsed = urlparse(url)
            raw_path = unquote(parsed.path)
            if parsed.netloc:
                raw_path = f"//{parsed.netloc}{raw_path}"
            return await asyncio.to_thread(Path(url2pathname(raw_path)).read_bytes)
        assert self._session is not None
        timeout = aiohttp.ClientTimeout(total=self.cfg.osu.timeout_sec)
        async with self._session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                raise OsuSourceError(f"下载附件失败（HTTP {resp.status}）。")
            return await resp.read()

    # ------------------------------------------------------------------ 渲染与发送
    async def _render_and_send(
        self,
        files: list[BeatmapFile],
        message_type: str,
        group_id: Any,
        user_id: Any,
        event: dict[str, Any],
        tag: str,
    ) -> None:
        total = len(files)
        render_started = time.perf_counter()
        logger.info("%s 开始处理 %d 张谱面（串行渲染）", tag, total)

        items: list[ResultItem] = []
        for index, beatmap_file in enumerate(files, 1):
            # 读取阶段就已判定不可用（例如 Malody 模式/轨道数不符）
            if beatmap_file.error:
                name = beatmap_file.display_name
                logger.info("%s [%d/%d] %s", tag, index, total, name)
                logger.warning("%s [%d/%d] 不可分析，改为错误提示：%s", tag, index, total, beatmap_file.error)
                items.append(
                    ResultItem(
                        name=name,
                        info=None,
                        error=beatmap_file.error,
                        png=render_error_card(beatmap_file.error, subtitle=name),
                    )
                )
                continue

            info = parse_beatmap(beatmap_file.text)
            name = info.display_name if info.title or info.artist else beatmap_file.display_name
            logger.info("%s [%d/%d] %s", tag, index, total, name)

            if not info.ok:
                logger.warning("%s [%d/%d] 不可分析，改为错误提示：%s", tag, index, total, info.error)
                items.append(
                    ResultItem(
                        name=name,
                        info=info,
                        error=info.error,
                        png=render_error_card(info.error or "无法分析该谱面。", subtitle=name),
                    )
                )
                continue

            background = beatmap_file.background if self.cfg.osu.include_background else None
            logger.info(
                "%s [%d/%d] 校验通过：%dK mania，背景图 %s",
                tag, index, total, info.column_count,
                f"{len(background) / 1024:.0f} KB" if background else "无",
            )
            item_started = time.perf_counter()
            png = await self.renderer.render(beatmap_file.text, background)
            self._save_temp(name, png)
            logger.info(
                "%s [%d/%d] 完成，用时 %.1fs（累计 %.1fs）",
                tag, index, total,
                time.perf_counter() - item_started,
                time.perf_counter() - render_started,
            )
            items.append(ResultItem(name=name, info=info, png=png))

        ok_count = sum(1 for item in items if not item.error)
        logger.info(
            "%s 渲染阶段结束：成功 %d 张 / 失败 %d 张，用时 %.1fs",
            tag, ok_count, len(items) - ok_count, time.perf_counter() - render_started,
        )

        if len(items) == 1:
            item = items[0]
            logger.info("%s 发送单张消息（%s）", tag, "错误提示" if item.error else "分析卡片")
            await self.client.send_msg(
                message_type=message_type,
                group_id=group_id,
                user_id=user_id,
                message=self._single_message(item),
            )
            return

        nodes = [
            forward_node(
                self.cfg.output.forward_name,
                self.cfg.output.forward_uin,
                [text_segment(f"Rework 分析结果 · 共 {len(items)} 张")],
            )
        ]
        for item in items:
            content: list[dict[str, Any]] = [text_segment(item.name)]
            if item.png:
                content.append(image_segment(item.png))
            else:
                content.append(text_segment(item.error or "分析失败"))
            sender = (event.get("sender") or {}).get("card") or (event.get("sender") or {}).get("nickname")
            nodes.append(
                forward_node(sender or self.cfg.output.forward_name, self.cfg.output.forward_uin, content)
            )

        if not self.cfg.output.send_as_forward:
            logger.info("%s 逐条发送 %d 张图片（合并转发已关闭）", tag, len(nodes) - 1)
            for node in nodes[1:]:
                await self.client.send_msg(
                    message_type=message_type,
                    group_id=group_id,
                    user_id=user_id,
                    message=node["data"]["content"],
                )
            return

        logger.info("%s 发送合并转发：%d 个节点…", tag, len(nodes))
        await self.client.send_forward(
            message_type=message_type, group_id=group_id, user_id=user_id, nodes=nodes
        )
        logger.info("%s 合并转发已发送", tag)

    def _single_message(self, item: ResultItem) -> list[dict[str, Any]]:
        if item.error:
            return [
                text_segment(f"❌ {item.name}\n{item.error}"),
                image_segment(item.png) if item.png else text_segment(""),
            ]
        return [image_segment(item.png)] if item.png else [text_segment("分析失败")]

    def _temp_path(self, name: str, suffix: str) -> Path:
        safe = re.sub(r"[^\w\u4e00-\u9fff.\[\]()\-\s]+", "_", name)[:80] or "beatmap"
        return self.cfg.temp_dir / f"{safe}{suffix}"

    def _save_temp(self, name: str, png: bytes) -> None:
        if not self.cfg.output.keep_files:
            return
        try:
            self._temp_path(name, ".png").write_bytes(png)
        except OSError as exc:
            logger.debug("保存临时图片失败：%s", exc)

    def _save_temp_text(self, name: str, text: str, suffix: str) -> None:
        """把中间产物（如 Malody 转换出的临时 .osu）落到 temp 目录，便于排查。"""
        if not self.cfg.output.keep_files:
            return
        try:
            path = self._temp_path(name, suffix)
            path.write_text(text, encoding="utf-8")
            logger.debug("已保存临时谱面：%s", path)
        except OSError as exc:
            logger.debug("保存临时谱面失败：%s", exc)
