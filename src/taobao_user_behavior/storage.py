"""提供文件哈希、JSON读写和临时统计连接。

主要接口：file_hash、save_json、read_json、connect。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

from taobao_user_behavior.settings import Settings

RAW_COLUMNS = ("time", "user_id", "item_id", "item_category", "behavior_type")


def file_hash(path: Path) -> str:
    """分块计算文件的SHA-256。

    Args:
        path: 要读取的文件路径。

    Returns:
        小写十六进制哈希值。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, document: Any) -> None:
    """将JSON写入临时文件，写完后替换目标文件。

    Args:
        path: 保存位置，父目录不存在时创建。
        document: 要保存的内容，特殊标量按字符串写入。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """读取UTF-8格式的JSON文件。

    Args:
        path: 已存在的报告文件。

    Returns:
        解析后的JSON内容。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    """建立临时内存连接，供查询标准事件使用。

    Args:
        settings: 内存、线程数和事件文件位置。

    Returns:
        含events视图的连接，调用方负责关闭。
    """
    connection = duckdb.connect()
    connection.execute("SET memory_limit = ?", [settings.memory_limit])
    connection.execute("SET threads = ?", [settings.threads])
    connection.execute("SET preserve_insertion_order = false")
    connection.read_parquet(str(settings.data / "events.parquet")).create_view("events")
    return connection
