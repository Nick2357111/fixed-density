"""极简 OneBot v11 客户端（正向 WebSocket）。

适用于 NapCat / Lagrange.Core / go-cqhttp 等实现了 OneBot v11 的 QQ 端。
机器人作为客户端连接它们的正向 WebSocket 服务。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Awaitable, Callable

import aiohttp

from .config import OneBotConfig

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


def text_segment(text: str) -> dict[str, Any]:
    return {"type": "text", "data": {"text": text}}


def image_segment(png: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(png).decode("ascii")
    return {"type": "image", "data": {"file": f"base64://{encoded}"}}


def forward_node(name: str, uin: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "node", "data": {"name": name, "uin": str(uin), "content": content}}


def extract_plain_text(message: list[dict[str, Any]] | None) -> str:
    if not message:
        return ""
    chunks = [
        str(seg.get("data", {}).get("text", ""))
        for seg in message
        if seg.get("type") == "text"
    ]
    return "".join(chunks)


class OneBotClient:
    def __init__(self, cfg: OneBotConfig, on_message: MessageHandler) -> None:
        self.cfg = cfg
        self._on_message = on_message
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._counter = 0
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------ 连接
    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        try:
            while True:
                try:
                    await self._connect_and_listen()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning("与 OneBot 的连接中断：%s", exc)
                self._fail_pending(ConnectionError("与 OneBot 的连接已断开"))
                self._connected.clear()
                await asyncio.sleep(self.cfg.reconnect_delay_sec)
        finally:
            await self._session.close()

    async def _connect_and_listen(self) -> None:
        headers = {}
        if self.cfg.access_token:
            headers["Authorization"] = f"Bearer {self.cfg.access_token}"
        logger.info("正在连接 OneBot：%s", self.cfg.ws_url)
        async with self._session.ws_connect(  # type: ignore[union-attr]
            self.cfg.ws_url, headers=headers, heartbeat=30, max_msg_size=0
        ) as ws:
            self._ws = ws
            self._connected.set()
            logger.info("已连接到 OneBot。")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_payload(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        self._ws = None

    def _handle_payload(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("收到无法解析的负载：%s", raw[:200])
            return

        echo = payload.get("echo")
        if echo is not None and echo in self._pending:
            future = self._pending.pop(echo)
            if not future.done():
                future.set_result(payload)
            return

        if payload.get("post_type") == "message":
            asyncio.create_task(self._dispatch(payload))

    async def _dispatch(self, event: dict[str, Any]) -> None:
        try:
            await self._on_message(event)
        except Exception:  # noqa: BLE001
            logger.exception("处理消息事件时发生异常")

    def _fail_pending(self, exc: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    # ------------------------------------------------------------------ 调用
    async def call(self, action: str, **params: Any) -> dict[str, Any]:
        if self._ws is None or self._ws.closed:
            raise ConnectionError("尚未连接到 OneBot 服务。")
        self._counter += 1
        echo = str(self._counter)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[echo] = future
        await self._ws.send_json({"action": action, "params": params, "echo": echo})
        logger.debug("API 调用 %s", action)
        try:
            result = await asyncio.wait_for(future, timeout=self.cfg.request_timeout_sec)
        except asyncio.TimeoutError:
            logger.warning("API %s 超时（%.0fs）", action, self.cfg.request_timeout_sec)
            raise
        finally:
            self._pending.pop(echo, None)

        if str(result.get("status", "ok")) not in ("ok", "async"):
            logger.warning("API %s 返回异常：%s", action, result)
        return result

    # ------------------------------------------------------------------ 语义化封装
    async def send_group_msg(self, group_id: Any, message: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.call("send_group_msg", group_id=group_id, message=message)

    async def send_private_msg(self, user_id: Any, message: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.call("send_private_msg", user_id=user_id, message=message)

    async def send_msg(
        self, *, message_type: str, group_id: Any = None, user_id: Any = None,
        message: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if message_type == "group":
            return await self.send_group_msg(group_id, message)
        return await self.send_private_msg(user_id, message)

    async def send_forward(
        self,
        *,
        message_type: str,
        group_id: Any = None,
        user_id: Any = None,
        nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """发送合并转发（聊天记录）消息，按不同实现依次尝试多种 action。"""
        if message_type == "group":
            attempts = [
                ("send_group_forward_msg", {"group_id": group_id, "messages": nodes}),
                ("send_forward_msg", {"message_type": "group", "group_id": group_id, "messages": nodes}),
            ]
        else:
            attempts = [
                ("send_private_forward_msg", {"user_id": user_id, "messages": nodes}),
                ("send_forward_msg", {"message_type": "private", "user_id": user_id, "messages": nodes}),
            ]

        last_error: Exception | None = None
        for action, params in attempts:
            try:
                result = await self.call(action, **params)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            if str(result.get("status", "ok")) in ("ok", "async"):
                return result
            last_error = RuntimeError(str(result))
        raise RuntimeError(f"发送合并转发失败：{last_error}")

    async def get_msg(self, message_id: Any) -> dict[str, Any]:
        return await self.call("get_msg", message_id=message_id)

    async def get_group_file_url(self, group_id: Any, file_id: str) -> dict[str, Any]:
        return await self.call("get_group_file_url", group_id=group_id, file_id=file_id)

    async def get_private_file_url(self, user_id: Any, file_id: str) -> dict[str, Any]:
        return await self.call("get_private_file_url", user_id=user_id, file_id=file_id)
