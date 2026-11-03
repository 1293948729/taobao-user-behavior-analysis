"""逐块读取ZIP内的CSV，保存标准事件与两类隔离记录。

主要接口：ingest。
"""

import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from taobao_user_behavior.data.hourly_filter import quarantine_hourly_events
from taobao_user_behavior.data.normalization import (
    EVENT_SCHEMA,
    REJECT_SCHEMA,
    normalize_events,
)
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import RAW_COLUMNS, file_hash, save_json

LOGGER = logging.getLogger(__name__)
RULE_VERSION = 4


def ingest(archive_path: Path, settings: Settings) -> dict:
    """逐块读取CSV，按字段规则和小时重复阈值分别保存结果。

    Args:
        archive_path: 根目录含user_behavior.csv的原始压缩包。
        settings: 分块行数、小时重复阈值和结果保存位置。

    Returns:
        读取行数、最终保留行数、两类隔离数量和源文件哈希。

    Raises:
        ValueError: CSV成员或表头不符合要求，或没有可用事件。
    """
    fingerprint = file_hash(archive_path)
    manifest_path = settings.outputs / "ingestion.json"
    outputs = [settings.data / "events.parquet", settings.data / "rejected.parquet"]
    totals: Counter = Counter()
    reasons: Counter = Counter()
    with ZipFile(archive_path) as archive:
        if archive.namelist().count("user_behavior.csv") != 1:
            raise ValueError("Archive must contain one root user_behavior.csv.")
        with archive.open("user_behavior.csv") as stream:
            header = tuple(stream.readline().decode("utf-8-sig").strip().split(","))
        if header != RAW_COLUMNS:
            raise ValueError(f"Expected {RAW_COLUMNS}; received {header}.")
        with (
            archive.open("user_behavior.csv") as stream,
            pq.ParquetWriter(
                str(outputs[0]) + ".tmp", EVENT_SCHEMA, compression="zstd"
            ) as accepted_writer,
            pq.ParquetWriter(
                str(outputs[1]) + ".tmp", REJECT_SCHEMA, compression="zstd"
            ) as rejected_writer,
        ):
            chunks = pd.read_csv(
                stream,
                dtype="string",
                keep_default_na=False,
                encoding="utf-8-sig",
                chunksize=settings.chunk_size,
                on_bad_lines="error",
            )
            for chunk in chunks:
                accepted, rejected = normalize_events(chunk)
                totals["raw_rows"] += len(accepted) + len(rejected)
                totals["accepted_rows"] += len(accepted)
                totals["rejected_rows"] += len(rejected)
                totals["missing_category_rows"] += int(
                    accepted["category_missing"].sum()
                )
                for text in rejected["reason"]:
                    reasons.update(text.split(";"))
                accepted_writer.write_table(
                    pa.Table.from_pandas(
                        accepted, schema=EVENT_SCHEMA, preserve_index=False
                    )
                )
                if not rejected.empty:
                    rejected_writer.write_table(
                        pa.Table.from_pandas(
                            rejected, schema=REJECT_SCHEMA, preserve_index=False
                        )
                    )
                LOGGER.info("Read %s events", f"{totals['raw_rows']:,}")
    if totals["accepted_rows"] == 0:
        raise ValueError("No usable events. Temporary files retained for inspection.")
    hourly = quarantine_hourly_events(Path(str(outputs[0]) + ".tmp"), settings)
    Path(str(outputs[1]) + ".tmp").replace(outputs[1])
    totals["field_valid_rows"] = totals["accepted_rows"]
    totals["accepted_rows"] -= hourly.flagged_rows
    totals["hourly_rejected_rows"] = hourly.flagged_rows
    manifest = {
        **dict(totals),
        "source_sha256": fingerprint,
        "source_name": archive_path.name,
        "rule_version": RULE_VERSION,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "timezone": "Asia/Shanghai",
        "invalid_reason_counts": dict(reasons),
        "hourly_rule": {
            **asdict(hourly),
            "threshold": settings.hourly_repeat_limit,
            "group_fields": ["user_id", "item_id", "behavior_type", "event_time"],
            "comparison": ">",
            "action": "quarantine entire group; retain groups at or below threshold",
        },
        "duplicates": "retained at or below hourly threshold; not collapsed",
        "missing_category": "retained with a flag and __missing__ category",
        "missing_category_scope": "field-valid rows before hourly isolation",
        "csv_syntax": "strict; malformed records stop ingestion",
        "processing": {
            "executor": "sequential chunks",
            "chunk_size": settings.chunk_size,
        },
    }
    save_json(manifest_path, manifest)
    return manifest
