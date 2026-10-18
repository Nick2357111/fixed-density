"""配置加载。

配置文件为 JSON（默认 qqbot/config.json），若不存在则自动从
config.example.json 复制一份。相对路径一律相对于配置文件所在目录解析。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_CONFIG_NAME = "config.json"
EXAMPLE_CONFIG_NAME = "config.example.json"


@dataclass
class OneBotConfig:
    ws_url: str = "ws://127.0.0.1:3001"
    access_token: str = ""
    reconnect_delay_sec: float = 5.0
    request_timeout_sec: float = 30.0
    command_prefix: str = "/rework"


@dataclass
class RenderConfig:
    plugin_dir: str = "../ManiaMapAnalyser.by.Leo_Black"
    host: str = "127.0.0.1"
    port: int = 24750
    browser_channel: str = "msedge"
    headless: bool = True
    device_scale_factor: int = 2
    page_timeout_ms: int = 180000
    settle_ms: int = 1500


@dataclass
class OsuConfig:
    beatmap_url: str = "https://osu.direct/api/osu/{id}"
    beatmapset_url: str = "https://osu.direct/api/d/{id}"
    user_agent: str = "qqbot-rework/1.0"
    timeout_sec: float = 60.0
    max_beatmaps_per_pack: int = 40
    include_background: bool = True


@dataclass
class OutputConfig:
    temp_dir: str = "temp"
    keep_files: bool = True
    send_as_forward: bool = True
    forward_name: str = "Rework 谱面分析"
    forward_uin: str = "10000"


@dataclass
class AccessConfig:
    allow_group: bool = True
    allow_private: bool = True
    group_allowlist: list[str] = field(default_factory=list)
    user_denylist: list[str] = field(default_factory=list)


@dataclass
class StorageConfig:
    data_dir: str = "data"
    enable_cache: bool = True
    max_sent_records: int = 5000


@dataclass
class AppConfig:
    onebot: OneBotConfig = field(default_factory=OneBotConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    osu: OsuConfig = field(default_factory=OsuConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    access: AccessConfig = field(default_factory=AccessConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    base_dir: Path = field(default_factory=Path.cwd)
    source_path: Path | None = None

    @property
    def temp_dir(self) -> Path:
        return Path(self.output.temp_dir)

    @property
    def plugin_dir(self) -> Path:
        return Path(self.render.plugin_dir)

    @property
    def data_dir(self) -> Path:
        return Path(self.storage.data_dir)


T = TypeVar("T")


def _section(cls: type[T], data: Any, path_keys: tuple[str, ...], base: Path) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key not in known:
                continue
            if key in path_keys and isinstance(value, str) and not Path(value).is_absolute():
                value = str((base / value).resolve())
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载配置；文件不存在时自动从示例配置复制。"""
    here = Path(__file__).resolve().parent.parent  # qqbot/
    cfg_path = Path(path).resolve() if path else here / DEFAULT_CONFIG_NAME

    if not cfg_path.exists():
        example = here / EXAMPLE_CONFIG_NAME
        if not path and example.exists():
            shutil.copyfile(example, cfg_path)
            print(f"[config] 未找到 {cfg_path.name}，已从 {example.name} 复制一份默认配置。")
        else:
            raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    cfg = AppConfig(
        onebot=_section(OneBotConfig, raw.get("onebot"), (), base),
        render=_section(RenderConfig, raw.get("render"), ("plugin_dir",), base),
        osu=_section(OsuConfig, raw.get("osu"), (), base),
        output=_section(OutputConfig, raw.get("output"), ("temp_dir",), base),
        access=_section(AccessConfig, raw.get("access"), (), base),
        storage=_section(StorageConfig, raw.get("storage"), ("data_dir",), base),
        base_dir=base,
        source_path=cfg_path,
    )

    if not cfg.plugin_dir.is_dir():
        raise FileNotFoundError(
            f"渲染配置 render.plugin_dir 指向的目录不存在：{cfg.plugin_dir}\n"
            "请确认 ManiaMapAnalyser.by.Leo_Black 插件目录的位置。"
        )
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
