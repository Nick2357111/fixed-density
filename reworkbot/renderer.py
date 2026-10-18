"""离屏渲染：用本地静态服务 + 无头浏览器复现 tosu 插件的分析卡片并截图。

原理
----
ManiaMapAnalyser 是 tosu 的前端插件：页面加载后会直接向
``http://<socketHost>/files/beatmap/file`` 取 .osu 原文并分析渲染，不依赖
tosu 的 WebSocket（断连只会后台重试）。因此我们：

1. 用本地 HTTP 服务把插件目录当作站点根目录托管；
2. 拦截 ``/files/beatmap/file`` 返回目标谱面文本、``/settings.json`` 返回
   注入过覆盖项的设置；
3. 用无头浏览器打开 ``index.html``，等卡片渲染完成后对 ``.main-card`` 截图。

设置覆盖项对应需求方给出的插件设置截图（Modules Customization /
Theme & Effects / Functionality Options）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import RenderConfig

logger = logging.getLogger(__name__)

# 与需求方截图一致的插件设置（uniqueID -> value）。
# 除下列键外的设置沿用插件自带 settings.json。
PRESET_OVERRIDES: dict[str, Any] = {
    # Modules Customization
    "contentBar": "Pattern",              # Card Body Content
    "srText": "Fixed Density v1",         # Top-left Capsule Text
    "diffText": "Difficulty",             # Top-right Content
    "showModeTagCapsule": True,           # Map Tag Capsule
    # Theme & Effects
    "enableOsuTheme": True,               # osu!Lazer Card Theme
    "useOsuFont": True,                   # osu! Font
    "enableFloatingTriangles": True,      # Floating Triangles Animation
    "enableCoverArt": True,               # Cover Art Background
    "customBackgroundColor": "#000000",
    "enableEtternaRainbowBars": False,    # Rainbow Bars
    "enableStatusMarquee": True,          # Metadata Marquee
    "enableNumericDifficulty": True,      # Numeric Difficulty
    "enableLNDifficulty": True,           # LN Star Rating
    "reverseCardExtendDirection": False,  # Reverse Card Extension
    "cardVisibility": "Always",
    "cardOpacity": "95%",
    "cardBgBlur": "4px",
    "cardRadius": "Medium",
    # Functionality Options
    "enablePauseDetection": True,         # Pause Detection
    "VibroDetection": True,               # Vibro Detection
    "useSvDetection": False,              # SV Detection
    "display6kLevel": True,               # Show 6K Constant Rating
    "extendedEstimationRange": False,     # Extended Estimation Range
    "forceSunnyWindow": True,             # Improve Sunny LN Estimation
    "enableAnalyzeLN": True,              # Analyze LN Parts（HB/LN/RC/Mix 胶囊）
    "pauseDetectionThreshold": "500",
    "estimatorAlgorithm": "Mixed",        # Estimator Algorithm
    "etternaVersion": "0.72.3",           # Global Etterna Version
    "companellaEtternaVersion": "0.74.0",  # Companella Etterna Version
    # 离屏渲染专用：关闭联网行为，避免卡顿与外部请求
    "enableUpdateCheck": False,
    "enableTelemetry": False,
    "preset": "LastSavedPreset",          # 避免预设回写覆盖上面的设置
}

# 页面状态探针：返回星等 / 状态行 / 遮罩信息，供轮询等待与进度日志使用。
_READY_CHECK_JS = """
() => {
  const star = document.querySelector('#rework-star');
  const status = document.querySelector('#status');
  const overlay = document.querySelector('#card-overlay');
  const text = star ? (star.textContent || '').trim() : '';
  const overlayVisible = Boolean(overlay && !overlay.hidden);
  return {
    ready: Boolean(star) && text !== '' && text !== '-' && !star.classList.contains('loading'),
    overlayVisible: overlayVisible,
    star: text,
    status: status ? (status.textContent || '').trim() : '',
    overlayTitle: overlayVisible
      ? ((document.querySelector('#overlay-title') || {}).textContent || '').trim() : '',
    overlayMessage: overlayVisible
      ? ((document.querySelector('#overlay-message') || {}).textContent || '').trim() : '',
  };
}
"""


class RenderError(RuntimeError):
    """渲染失败。"""


@dataclass
class RenderState:
    """静态服务与页面共享的可变状态。"""

    osu_text: str = ""
    background: bytes | None = None
    settings_json: str = ""


def _image_content_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


class _StaticHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".otf": "font/otf",
        ".ttf": "font/ttf",
        ".json": "application/json",
        ".css": "text/css",
        ".svg": "image/svg+xml",
    }
    state: RenderState

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("static: " + fmt, *args)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]

        if route == "/files/beatmap/file":
            self._send_bytes(self.state.osu_text.encode("utf-8"), "text/plain; charset=utf-8")
            return

        if route == "/files/beatmap/background":
            if self.state.background:
                self._send_bytes(self.state.background, _image_content_type(self.state.background))
            else:
                self.send_error(404, "no background")
            return

        if route == "/settings.json":
            self._send_bytes(self.state.settings_json.encode("utf-8"), "application/json; charset=utf-8")
            return

        return super().do_GET()


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """浏览器中断连接（换页/关闭）时不要打印堆栈。"""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:  # noqa: D102
        logger.debug("静态服务连接异常（通常为浏览器提前断开）", exc_info=True)


def _ensure_port_available(host: str, port: int) -> None:
    """确认端口没有被别的进程占用。

    Windows 下 SO_REUSEADDR 允许两个进程绑定同一端口，若已有实例在跑，
    浏览器会连到那一个并取到**它的**谱面，症状是「渲染出来的是别的歌」。
    这里先探一次，冲突就直接报错，避免静默串台。
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        in_use = probe.connect_ex((host, port)) == 0
    finally:
        probe.close()
    if in_use:
        raise RenderError(
            f"端口 {host}:{port} 已被占用：可能已有一个机器人实例在运行。\n"
            "同一个端口上跑两个实例会互相串台（渲染出别的谱面）。\n"
            "请先关闭已有实例，或把 config.json 里的 render.port 改成其它端口。"
        )


class Renderer:
    """插件卡片离屏渲染器（进程内单例，串行渲染）。"""

    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self.base_url = f"http://{cfg.host}:{cfg.port}/index.html"
        self._state = RenderState(settings_json=self._build_settings())
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ 设置注入
    def _build_settings(self) -> str:
        settings_path = Path(self.cfg.plugin_dir) / "settings.json"
        data = json.loads(settings_path.read_text(encoding="utf-8"))

        # 关键：把插件的谱面来源指向我们自己的本地静态服务。
        # 若沿用默认值 localhost:24050，页面会去读本机 tosu 当前打开的谱面。
        overrides = dict(PRESET_OVERRIDES)
        overrides["wsEndpoint"] = f"{self.cfg.host}:{self.cfg.port}"

        applied: list[str] = []
        for item in data:
            uid = item.get("uniqueID")
            if uid in overrides:
                item["value"] = overrides[uid]
                applied.append(uid)
        missing = set(PRESET_OVERRIDES) - set(applied)
        if missing:
            logger.warning("以下设置项在 settings.json 中不存在，已跳过：%s", sorted(missing))
        return json.dumps(data, ensure_ascii=False, indent="\t")

    # ------------------------------------------------------------------ 生命周期
    def start_http(self) -> None:
        _ensure_port_available(self.cfg.host, self.cfg.port)
        handler_cls = type("_BoundStaticHandler", (_StaticHandler,), {"state": self._state})
        handler = partial(handler_cls, directory=str(self.cfg.plugin_dir))
        self._server = _QuietThreadingHTTPServer(
            (self.cfg.host, self.cfg.port), handler  # type: ignore[arg-type]
        )
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info("静态服务已启动：http://%s:%s/", self.cfg.host, self.cfg.port)

    async def start(self) -> None:
        self.start_http()

    async def _ensure_browser(self) -> None:
        if self._page is not None:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        launch_kwargs: dict[str, Any] = {
            "headless": self.cfg.headless,
            "args": ["--disable-gpu", "--hide-scrollbars", "--mute-audio"],
        }
        channel = (self.cfg.browser_channel or "").strip()
        browser = None
        if channel:
            try:
                browser = await self._playwright.chromium.launch(channel=channel, **launch_kwargs)
                logger.info("浏览器已启动（channel=%s）", channel)
            except Exception as exc:  # noqa: BLE001
                logger.warning("使用 channel=%s 启动失败（%s），回退到内置 Chromium", channel, exc)
        if browser is None:
            try:
                browser = await self._playwright.chromium.launch(**launch_kwargs)
                logger.info("浏览器已启动（内置 Chromium）")
            except Exception as exc:  # noqa: BLE001
                raise RenderError(
                    "无法启动无头浏览器。请执行 `python -m playwright install chromium`，"
                    f"或在 config.json 中把 render.browser_channel 设为已安装的浏览器（如 msedge）。原始错误：{exc}"
                ) from exc

        self._browser = browser
        context = await browser.new_context(
            viewport={"width": 780, "height": 1000},
            device_scale_factor=max(1, int(self.cfg.device_scale_factor)),
        )
        self._page = await context.new_page()
        self._page.on("pageerror", lambda err: logger.debug("page error: %s", err))

    async def close(self) -> None:
        # 先取锁：保证正在进行的渲染结束后再关闭浏览器。
        async with self._lock:
            try:
                if self._browser is not None:
                    await self._browser.close()
                if self._playwright is not None:
                    await self._playwright.stop()
            finally:
                self._browser = None
                self._page = None
                self._playwright = None
                if self._server is not None:
                    self._server.shutdown()
                    self._server.server_close()
                    self._server = None

    # ------------------------------------------------------------------ 渲染
    async def render(self, osu_text: str, background: bytes | None = None) -> bytes:
        """渲染单张谱面，返回卡片 PNG 字节。"""
        async with self._lock:
            await self._ensure_browser()
            self._state.osu_text = osu_text
            self._state.background = background

            started = time.perf_counter()
            logger.info(
                "开始渲染：谱面文本 %d 字符，背景图 %s",
                len(osu_text),
                f"{len(background) / 1024:.0f} KB" if background else "无",
            )
            try:
                png = await self._render_once()
            except RenderError:
                logger.error("渲染失败（用时 %.1fs）", time.perf_counter() - started)
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("渲染出错，重试一次：%s", exc)
                try:
                    png = await self._render_once()
                except Exception as retry_exc:  # noqa: BLE001
                    logger.error("重试仍失败（用时 %.1fs）：%s", time.perf_counter() - started, retry_exc)
                    raise

            logger.info(
                "渲染完成：输出 %.1f KB，用时 %.1fs",
                len(png) / 1024,
                time.perf_counter() - started,
            )
            return png

    async def _wait_until_ready(self, page: Any) -> dict[str, Any]:
        """轮询页面状态直到分析完成，并在等待期间输出进度。

        星等与胶囊数值有 ~400ms 的入场动画，因此必须连续两次读数一致
        才认为稳定，否则会读到动画中间值。
        """
        timeout_sec = self.cfg.page_timeout_ms / 1000
        start = time.perf_counter()
        last_status: str | None = None
        prev_star: str | None = None
        next_heartbeat = start + 5.0

        while True:
            info = await page.evaluate(_READY_CHECK_JS)
            if info["overlayVisible"]:
                return info
            if info["ready"]:
                if info["star"] == prev_star:
                    return info
                prev_star = info["star"]

            now = time.perf_counter()
            status = info.get("status") or "(空)"
            if status != last_status:
                logger.info("  等待分析… 页面状态：%s", status)
                last_status = status
            elif now >= next_heartbeat:
                logger.info("  仍在分析…（已等待 %.0f 秒）", now - start)
                next_heartbeat = now + 5.0

            if now - start >= timeout_sec:
                raise RenderError(
                    f"等待分析结果超时（{timeout_sec:.0f} 秒），页面最后状态：{status}"
                )
            await asyncio.sleep(0.5)

    async def _render_once(self) -> bytes:
        page = self._page

        load_started = time.perf_counter()
        await page.goto(self.base_url, wait_until="load", timeout=self.cfg.page_timeout_ms)
        logger.info("  页面已加载（%.1fs），等待插件分析…", time.perf_counter() - load_started)

        info = await self._wait_until_ready(page)
        if info["overlayVisible"]:
            raise RenderError(
                f"插件分析失败：{info['overlayTitle']} {info['overlayMessage']}".strip()
            )

        star = str(info["star"]).strip()
        if not star or star == "-":
            raise RenderError("插件未能产出分析结果（星等为空）。")
        # 插件内部报错时仍会画出星等，但状态行以 [Error] 开头 —— 这种结果是假的，不能发出去
        status_text = str(info.get("status") or "").strip()
        if status_text.startswith("[Error]"):
            raise RenderError(status_text)
        logger.info("  分析完成：星等=%s（用时 %.1fs）", star, time.perf_counter() - load_started)

        # 封面主题：正常运行时由 tosu 的 socket payload 触发，离线渲染没有 socket，
        # 这里直接调用插件自身的 coverTheme 模块，从 /files/beatmap/background 取色垫图。
        if self._state.background:
            logger.info("  正在应用封面背景…")
            try:
                await page.evaluate(
                    "async () => {"
                    " const m = await import('./js/app/coverTheme.js');"
                    " await m.applyCoverThemeForBeatmap('qqbot-cover');"
                    " }"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("应用封面主题失败（不影响出图）：%s", exc)

        logger.info("  等待动画结束（%.0f ms）…", self.cfg.settle_ms)
        await page.wait_for_timeout(self.cfg.settle_ms)
        await page.evaluate("() => { document.body.style.background = '#000000'; }")

        final = await page.evaluate(_READY_CHECK_JS)
        final_status = str(final.get("status") or "")
        if final_status.startswith("[Error]"):
            raise RenderError(final_status)
        logger.info(
            "  卡片就绪：谱面=%s  星等=%s",
            final_status or "未知",
            final.get("star") or "?",
        )

        card = await page.query_selector(".main-card")
        if card is None:
            raise RenderError("页面上找不到分析卡片容器 .main-card。")
        box = await card.bounding_box()
        if not box or box["width"] < 1 or box["height"] < 1:
            raise RenderError("分析卡片尺寸异常。")

        logger.info(
            "  正在截图：卡片 %.0f×%.0f（倍率 %s）…",
            box["width"],
            box["height"],
            self.cfg.device_scale_factor,
        )
        return await page.screenshot(
            clip={
                "x": round(box["x"]),
                "y": round(box["y"]),
                "width": round(box["width"]),
                "height": round(box["height"]),
            },
            type="png",
        )
