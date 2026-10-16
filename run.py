"""qqbot 入口。

用法：
    python run.py                          # 启动 QQ 机器人（读取 config.json）
    python run.py --config other.json      # 指定配置文件
    python run.py --render map.osu         # 本地渲染，不启动机器人（自测用）
    python run.py --render pack.osz --out ./out
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from reworkbot.beatmap import parse_beatmap
from reworkbot.config import AppConfig, load_config
from reworkbot.errorcard import render_error_card
from reworkbot.onebot import OneBotClient
from reworkbot.osu_source import OsuSourceError, read_local_beatmap
from reworkbot.renderer import Renderer, RenderError
from reworkbot.service import ReworkService

logger = logging.getLogger("qqbot")


async def run_bot(cfg: AppConfig) -> None:
    renderer = Renderer(cfg.render)
    await renderer.start()

    service_ref: list[ReworkService] = []

    async def handle_event(event: dict) -> None:
        if service_ref:
            await service_ref[0].handle_event(event)

    client = OneBotClient(cfg.onebot, handle_event)
    service = ReworkService(cfg, client, renderer)
    service_ref.append(service)
    await service.start()

    logger.info("机器人已就绪，等待指令。命令前缀：%s", cfg.onebot.command_prefix)
    try:
        await client.run()
    finally:
        await service.close()
        await renderer.close()


async def render_local(cfg: AppConfig, target: str, out: str | None) -> int:
    beatmaps = read_local_beatmap(target)
    out_path = Path(out) if out else None

    renderer = Renderer(cfg.render)
    await renderer.start()
    exit_code = 0
    total = len(beatmaps)
    logger.info("本地渲染：共 %d 张谱面，来源 %s", total, target)
    try:
        for index, beatmap_file in enumerate(beatmaps, 1):
            if beatmap_file.error:
                name = beatmap_file.display_name
                logger.info("[%d/%d] %s", index, total, name)
                logger.warning("[%d/%d] 不可分析，输出错误提示：%s", index, total, beatmap_file.error)
                png = render_error_card(beatmap_file.error, subtitle=name)
                exit_code = 2
            else:
                info = parse_beatmap(beatmap_file.text)
                name = info.display_name if (info.title or info.artist) else beatmap_file.display_name
                logger.info("[%d/%d] %s", index, total, name)
                if not info.ok:
                    png = render_error_card(info.error or "无法分析该谱面。", subtitle=name)
                    logger.warning("[%d/%d] 不可分析，输出错误提示：%s", index, total, info.error)
                    exit_code = 2
                else:
                    png = await renderer.render(beatmap_file.text, beatmap_file.background)
                    logger.info("[%d/%d] 渲染完成（%.1f KB）", index, total, len(png) / 1024)

            if out_path is None:
                destination = cfg.temp_dir / f"{Path(beatmap_file.filename).stem}.png"
            elif total == 1 and out_path.suffix.lower() == ".png":
                destination = out_path
            else:
                destination = out_path / f"{Path(beatmap_file.filename).stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(png)
            logger.info("[%d/%d] 已保存：%s", index, total, destination)
    finally:
        await renderer.close()
    logger.info("本地渲染结束，退出码 %d", exit_code)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="osu!mania 4K Rework 谱面分析 QQ 机器人")
    parser.add_argument("--config", help="配置文件路径（默认 qqbot/config.json）")
    parser.add_argument("--render", metavar="FILE", help="本地渲染 .osu / .osz，不启动机器人")
    parser.add_argument("--out", help="--render 的输出文件或目录")
    parser.add_argument("--log-level", default="INFO", help="日志级别 DEBUG/INFO/WARNING（默认 INFO）")
    parser.add_argument("--log-file", help="同时把日志写入该文件（滚动，单文件 5MB × 3）")
    return parser


def _setup_logging(level_name: str, log_file: str | None) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # 第三方库的噪音压到 WARNING
    for noisy in ("aiohttp", "asyncio", "playwright"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _log_config_summary(cfg: AppConfig) -> None:
    logger.info("插件目录：%s", cfg.plugin_dir)
    logger.info(
        "渲染服务：http://%s:%s/（截图倍率 %s，浏览器 %s）",
        cfg.render.host,
        cfg.render.port,
        cfg.render.device_scale_factor,
        cfg.render.browser_channel or "内置 Chromium",
    )
    logger.info(
        "OneBot：%s（Token %s）",
        cfg.onebot.ws_url,
        "已设置" if cfg.onebot.access_token else "未设置",
    )
    logger.info("谱面镜像：%s", cfg.osu.beatmap_url)
    logger.info("产物目录：%s（保留文件：%s）", cfg.temp_dir, cfg.output.keep_files)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level, args.log_file)

    if args.log_file:
        logger.info("日志同时写入：%s", Path(args.log_file).resolve())

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    _log_config_summary(cfg)

    try:
        if args.render:
            return asyncio.run(render_local(cfg, args.render, args.out))
        asyncio.run(run_bot(cfg))
    except OsuSourceError as exc:
        logger.error("谱面获取失败：%s", exc)
        return 1
    except RenderError as exc:
        logger.error("渲染失败：%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
