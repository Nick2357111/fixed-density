"""按内容哈希存储的谱面 / 图片数据库。

每个谱面以其最终 `.osu` 文本的 sha256 为键，落盘为：

    <data_dir>/charts/<hash>.osu    谱面文本（Malody 则为转换后的结果）
    <data_dir>/charts/<hash>.png    生成的分析卡片
    <data_dir>/charts/<hash>.bg     可选背景图（刷新时用来还原封面）
    <data_dir>/charts/<hash>.json   元信息（名称、来源、时间）

新查询命中同一 hash 时直接复用已有 png，不再渲染。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """统一换行，保证同一谱面不同换行风格得到同一哈希。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class CacheEntry:
    key: str
    name: str
    osu_path: Path
    png_path: Path
    bg_path: Path
    meta: dict[str, Any]

    @property
    def has_image(self) -> bool:
        return self.png_path.is_file()


class ChartCache:
    """内容寻址的谱面 / 图片库。"""

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        self.root = Path(root) / "charts"
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 键
    @staticmethod
    def key_for(osu_text: str) -> str:
        return hashlib.sha256(normalize_text(osu_text).encode("utf-8")).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path, Path, Path]:
        return (
            self.root / f"{key}.osu",
            self.root / f"{key}.png",
            self.root / f"{key}.bg",
            self.root / f"{key}.json",
        )

    # ------------------------------------------------------------------ 读写
    def get_image(self, key: str) -> bytes | None:
        if not self.enabled:
            return None
        _, png_path, _, _ = self._paths(key)
        try:
            return png_path.read_bytes() if png_path.is_file() else None
        except OSError as exc:
            logger.debug("读取缓存图片失败 %s：%s", key[:12], exc)
            return None

    def get_background(self, key: str) -> bytes | None:
        if not self.enabled:
            return None
        _, _, bg_path, _ = self._paths(key)
        try:
            return bg_path.read_bytes() if bg_path.is_file() else None
        except OSError:
            return None

    def has(self, key: str) -> bool:
        if not self.enabled:
            return False
        _, png_path, _, _ = self._paths(key)
        return png_path.is_file() and (self.root / f"{key}.osu").is_file()

    def put(
        self,
        key: str,
        osu_text: str,
        png: bytes,
        *,
        name: str,
        source: str = "",
        background: bytes | None = None,
    ) -> None:
        if not self.enabled:
            return
        osu_path, png_path, bg_path, meta_path = self._paths(key)
        now = time.time()
        try:
            _write_atomic(osu_path, normalize_text(osu_text).encode("utf-8"))
            _write_atomic(png_path, png)
            if background:
                _write_atomic(bg_path, background)
            old = self._read_meta(meta_path)
            meta = {
                "key": key,
                "name": name,
                "source": source,
                "created_at": old.get("created_at", now),
                "updated_at": now,
                "png_bytes": len(png),
            }
            _write_atomic(meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
        except OSError as exc:
            logger.warning("写入缓存失败 %s：%s", key[:12], exc)

    def _read_meta(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ------------------------------------------------------------------ 遍历
    def entries(self) -> Iterator[CacheEntry]:
        if not self.root.is_dir():
            return
        for osu_path in sorted(self.root.glob("*.osu")):
            key = osu_path.stem
            _, png_path, bg_path, meta_path = self._paths(key)
            meta = self._read_meta(meta_path)
            yield CacheEntry(
                key=key,
                name=str(meta.get("name") or key),
                osu_path=osu_path,
                png_path=png_path,
                bg_path=bg_path,
                meta=meta,
            )

    def stats(self) -> tuple[int, int]:
        """返回 (谱面数, 已有图片数)。"""
        if not self.root.is_dir():
            return 0, 0
        charts = sum(1 for _ in self.root.glob("*.osu"))
        images = sum(1 for _ in self.root.glob("*.png"))
        return charts, images


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
