"""按块读取行为记录，检查缺失、重复和用户活跃度。

主要接口：normalize_events、ingest、audit。
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import (
    RAW_COLUMNS,
    connect,
    file_hash,
    save_json,
)

LOGGER = logging.getLogger(__name__)
RULE_VERSION = 3
EVENT_SCHEMA = pa.schema(
    [
        ("event_time", pa.timestamp("ns")),
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("category_id", pa.string()),
        ("behavior_type", pa.int8()),
        ("category_missing", pa.bool_()),
    ]
)
REJECT_SCHEMA = pa.schema(
    [(column, pa.string()) for column in RAW_COLUMNS] + [("reason", pa.string())]
)


def normalize_events(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一字段格式，分开返回有效记录和无效记录。

    Args:
        raw: 按原顺序排列的五列数据，各列按字符串读取。

    Returns:
        标准事件，以及附有原因的无效原始记录；不改动输入。

    Raises:
        ValueError: 输入字段或字段顺序不符合要求。
    """
    if tuple(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected columns: {tuple(raw.columns)}")
    cleaned = raw.apply(lambda column: column.astype("string").str.strip())
    timestamps = pd.to_datetime(
        cleaned["time"], format="%Y-%m-%d %H", errors="coerce", exact=True
    )
    invalid_time = (
        ~cleaned["time"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}", na=False)
        | timestamps.isna()
    )
    rules = {
        "missing_user": cleaned["user_id"].fillna("").eq(""),
        "missing_item": cleaned["item_id"].fillna("").eq(""),
        "invalid_time": invalid_time,
        "invalid_behavior": ~cleaned["behavior_type"].isin(["1", "2", "3", "4"]),
    }
    reasons = pd.Series("", index=raw.index, dtype="string")
    for reason, mask in rules.items():
        reasons.loc[mask] = reasons.loc[mask] + reason + ";"
    invalid = reasons.ne("")
    rejected = raw.loc[invalid].copy()
    rejected["reason"] = reasons.loc[invalid].str.rstrip(";")
    category_missing = cleaned["item_category"].fillna("").eq("")
    accepted = pd.DataFrame(
        {
            "event_time": timestamps.loc[~invalid],
            "user_id": cleaned.loc[~invalid, "user_id"],
            "item_id": cleaned.loc[~invalid, "item_id"],
            "category_id": cleaned.loc[~invalid, "item_category"].mask(
                category_missing.loc[~invalid], "__missing__"
            ),
            "behavior_type": cleaned.loc[~invalid, "behavior_type"].astype("int8"),
            "category_missing": category_missing.loc[~invalid],
        }
    )
    return accepted, rejected


def ingest(archive_path: Path, settings: Settings) -> dict:
    """逐块读取CSV，将有效和无效记录分别保存为Parquet。

    Args:
        archive_path: 根目录含user_behavior.csv的原始压缩包。
        settings: 分块行数和结果保存位置。

    Returns:
        读取行数、隔离原因和源文件哈希。

    Raises:
        ValueError: CSV成员或表头不符合要求，或没有可用事件。
    """
    fingerprint = file_hash(archive_path)
    manifest_path = settings.outputs / "ingestion.json"
    outputs = [settings.data / "events.parquet", settings.data / "rejected.parquet"]
    totals: Counter = Counter()
    reasons: Counter = Counter()
    with ZipFile(archive_path) as archive:
        if "user_behavior.csv" not in archive.namelist():
            raise ValueError("Archive must contain user_behavior.csv at its root.")
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
    for output in outputs:
        Path(str(output) + ".tmp").replace(output)
    manifest = {
        **dict(totals),
        "source_sha256": fingerprint,
        "source_name": archive_path.name,
        "rule_version": RULE_VERSION,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "timezone": "Asia/Shanghai",
        "invalid_reason_counts": dict(reasons),
        "duplicates": "retained; hour-level records cannot prove ingestion duplicates",
        "missing_category": "retained with a flag and __missing__ category",
        "csv_syntax": "strict; malformed records stop ingestion",
        "processing": {
            "executor": "sequential chunks",
            "chunk_size": settings.chunk_size,
        },
    }
    save_json(manifest_path, manifest)
    return manifest


def audit(settings: Settings) -> dict:
    """统计重复记录和用户事件量，不删除有效事件。

    Args:
        settings: 标准事件位置、报告目录和查询资源限制。

    Returns:
        事件量、重复计数、活跃度分布和IQR检查结果。

    Raises:
        ValueError: 没有可供检查的有效事件。
    """
    with connect(settings) as connection:
        total = connection.execute("SELECT count(*) FROM events").fetchone()
        if total is None or total[0] == 0:
            raise ValueError("Source has no accepted events.")
        rows = int(total[0])
        duplicate_summary = (
            connection.execute(
                """
            SELECT count(*) AS repeated_groups,
                coalesce(sum(event_count - 1), 0) AS repeated_surplus
            FROM (
                SELECT count(*) AS event_count
                FROM events
                GROUP BY user_id, item_id, category_id, event_time, behavior_type
                HAVING count(*) > 1
            ) AS repeated
            """,
            )
            .df()
            .iloc[0]
            .to_dict()
        )
        activity = connection.execute(
            "SELECT user_id, count(*) AS events FROM events GROUP BY 1"
        ).df()
        lower, upper = activity["events"].quantile([0.25, 0.75])
        threshold = float(upper + 1.5 * (upper - lower))
        # 次数多只能说明活跃，单凭这个阈值不能删除用户记录。
        summary = {
            "rows": rows,
            **duplicate_summary,
            "activity_minimum": int(activity["events"].min()),
            "activity_median": float(activity["events"].median()),
            "activity_maximum": int(activity["events"].max()),
            "activity_q1": float(lower),
            "activity_q3": float(upper),
            "activity_iqr_upper": threshold,
            "users_above_iqr": int(activity["events"].gt(threshold).sum()),
            "outlier_action": "retain counts; statistical threshold only",
            "deduplication_action": "no rows deleted",
        }
    save_json(settings.outputs / "quality.json", summary)
    return summary
