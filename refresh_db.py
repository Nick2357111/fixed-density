"""谱面库刷新入口。

重新渲染数据库中所有已缓存谱面，并用新图片覆盖原图。
适合在修改插件设置（`PRESET_OVERRIDES`）或升级插件后统一重算。

用法：
    python refresh_db.py                 # 刷新整个数据库
    python refresh_db.py --dry-run       # 只列出条目，不渲染
    python refresh_db.py --config other.json --log-file logs/refresh.log
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from reworkbot.cache import ChartCache
from reworkbot.config import load_config
from reworkbot.renderer import RenderError, Renderer
from run import _log_config_summary, _setup_logging

logger = logging.getLogger("refresh")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="刷新 Rework 谱面库（重新渲染所有已缓存谱面）")
    parser.add_argument("--config", help="配置文件路径（默认 qqbot/config.json）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要刷新的条目")
    parser.add_argument("--log-level", default="INFO", help="日志级别（默认 INFO）")
    parser.add_argument("--log-file", help="同时把日志写入该文件")
    return parser


async def refresh(cfg, dry_run: bool) -> int:
    cache = ChartCache(cfg.data_dir, enabled=True)
    entries = list(cache.entries())
    charts, images = cache.stats()
    logger.info("谱面库位置：%s", cache.root)
    logger.info("当前共 %d 条谱面记录 / %d 张图片", charts, images)

    if not entries:
        logger.warning("数据库为空，没有可刷新的谱面。")
        return 0

    if dry_run:
        for index, entry in enumerate(entries, 1):
            flag = "" if entry.has_image else "（缺图片）"
            logger.info("[%d/%d] %s  %s%s", index, len(entries), entry.key[:12], entry.name, flag)
        logger.info("dry-run 结束，未做任何修改。")
        return 0

    renderer = Renderer(cfg.render)
    await renderer.start()
    succeeded = failed = 0
    started = time.perf_counter()
    try:
        for index, entry in enumerate(entries, 1):
            try:
                osu_text = entry.osu_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("[%d/%d] %s 读取谱面失败：%s", index, len(entries), entry.key[:12], exc)
                failed += 1
                continue

            background = cache.get_background(entry.key) if cfg.osu.include_background else None
            item_started = time.perf_counter()
            try:
                png = await renderer.render(osu_text, background)
            except RenderError as exc:
                logger.error("[%d/%d] %s 渲染失败：%s", index, len(entries), entry.name, exc)
                failed += 1
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%d/%d] %s 渲染异常：%s", index, len(entries), entry.name, exc)
                failed += 1
                continue

            cache.put(
                entry.key,
                osu_text,
                png,
                name=entry.name,
                source=str(entry.meta.get("source") or ""),
                background=background,
            )
            succeeded += 1
            logger.info(
                "[%d/%d] %s 已刷新（%.1f KB，%.1fs）",
                index, len(entries), entry.name, len(png) / 1024,
                time.perf_counter() - item_started,
            )
    finally:
        await renderer.close()

    logger.info(
        "刷新完成：成功 %d / 失败 %d，总用时 %.1fs",
        succeeded, failed, time.perf_counter() - started,
    )
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level, args.log_file)
    if args.log_file:
        logger.info("日志同时写入：%s", args.log_file)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    _log_config_summary(cfg)
    try:
        return asyncio.run(refresh(cfg, args.dry_run))
    except KeyboardInterrupt:
        logger.info("已中断。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
