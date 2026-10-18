"""用户反馈与「已发送消息 → 谱面」索引。

- SentMessageIndex：记录机器人发出的每条图片消息对应哪些谱面哈希，
  `/report` 回复时据此定位用户想反馈的是哪一张（合并转发则按图片编号取）。
- FeedbackStore：保存用户实际体感难度，键为 (用户, 谱面哈希)，
  同一用户对同一谱面只保留最新一条。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _write_atomic(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class SentRecord:
    """一条已发送消息对应的谱面列表。"""

    message_id: str
    keys: list[str | None] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    message_type: str = "private"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "keys": self.keys,
            "names": self.names,
            "message_type": self.message_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentRecord":
        return cls(
            message_id=str(data.get("message_id") or ""),
            keys=list(data.get("keys") or []),
            names=list(data.get("names") or []),
            message_type=str(data.get("message_type") or "private"),
            created_at=float(data.get("created_at") or 0.0),
        )


class SentMessageIndex:
    """message_id → 谱面哈希列表，落盘保存并限制条数。"""

    def __init__(self, path: str | Path, *, max_records: int = 5000, enabled: bool = True) -> None:
        self.path = Path(path)
        self.max_records = max(100, int(max_records))
        self.enabled = enabled
        self._records: dict[str, SentRecord] = {}
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if isinstance(item, dict):
                record = SentRecord.from_dict(item)
                if record.message_id:
                    self._records[record.message_id] = record

    def _save(self) -> None:
        if not self.enabled:
            return
        records = sorted(self._records.values(), key=lambda r: r.created_at)
        if len(records) > self.max_records:
            records = records[-self.max_records :]
            self._records = {record.message_id: record for record in records}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(
                self.path,
                json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=1),
            )
        except OSError as exc:
            logger.warning("保存发送索引失败：%s", exc)

    def add(
        self,
        message_id: Any,
        keys: list[str | None],
        names: list[str],
        *,
        message_type: str = "private",
    ) -> None:
        if not self.enabled or message_id is None:
            return
        self._records[str(message_id)] = SentRecord(
            message_id=str(message_id),
            keys=list(keys),
            names=list(names),
            message_type=message_type,
            created_at=time.time(),
        )
        self._save()

    def get(self, message_id: Any) -> SentRecord | None:
        return self._records.get(str(message_id))

    def __len__(self) -> int:
        return len(self._records)


class FeedbackStore:
    """用户实际体感难度反馈，(用户, 谱面哈希) 唯一。"""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._records: dict[str, dict[str, Any]] = {}
        if self.enabled:
            self._load()

    @staticmethod
    def _key(user_id: Any, chart_key: str) -> str:
        return f"{user_id}|{chart_key}"

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            self._records = {str(k): v for k, v in raw.items() if isinstance(v, dict)}

    def _save(self) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(self.path, json.dumps(self._records, ensure_ascii=False, indent=1))
        except OSError as exc:
            logger.warning("保存反馈失败：%s", exc)

    def upsert(
        self,
        user_id: Any,
        chart_key: str,
        *,
        name: str,
        rc: float | None,
        ln: float | None,
    ) -> bool:
        """写入/覆盖一条反馈；返回是否为新建。"""
        key = self._key(user_id, chart_key)
        created = key not in self._records
        self._records[key] = {
            "user_id": str(user_id),
            "chart": chart_key,
            "name": name,
            "rc": rc,
            "ln": ln,
            "updated_at": time.time(),
        }
        self._save()
        return created

    def get(self, user_id: Any, chart_key: str) -> dict[str, Any] | None:
        return self._records.get(self._key(user_id, chart_key))

    def all_records(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)
