"""对正式样本执行全表结构、主键、有限性与业务范围检查。

主要接口：validate_sample。
"""

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from taobao_user_behavior.features.dictionary import field_dictionary
from taobao_user_behavior.settings import Settings


def _numeric_batch(batch: pa.RecordBatch) -> dict[str, np.ndarray]:
    columns = {}
    for index, field in enumerate(batch.schema):
        column = batch.column(index)
        if column.null_count:
            raise ValueError(f"Null values in {field.name}.")
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            values = column.to_numpy(zero_copy_only=False)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {field.name}.")
            columns[field.name] = values
        elif pa.types.is_boolean(field.type):
            columns[field.name] = column.to_numpy(zero_copy_only=False)
    return columns


def _check_relations(columns: dict[str, np.ndarray]) -> None:
    for prefix in ["pair", "user", "item", "category"]:
        total = columns[f"{prefix}_events"]
        parts = sum(
            columns[f"{prefix}_{action}"]
            for action in ["views", "favorites", "carts", "purchases"]
        )
        if not np.equal(total, parts).all() or not (total > 0).all():
            raise ValueError(f"Behavior conservation failed: {prefix}")
    for prefix in ["pair", "user", "item"]:
        if not (
            (columns[f"{prefix}_events_1d"] <= columns[f"{prefix}_events_3d"])
            & (columns[f"{prefix}_events_3d"] <= columns[f"{prefix}_events"])
        ).all():
            raise ValueError(f"Recent-window counts disagree: {prefix}")
    if not np.equal(columns["pair_no_views"], columns["pair_views"] == 0).all():
        raise ValueError("Missing-view flag disagrees.")
    observed = columns["pair_purchase_observed"]
    recency = columns["pair_purchase_recency_hours"]
    if not np.equal(observed, columns["pair_purchases"] > 0).all():
        raise ValueError("Historical purchase flag disagrees.")
    if not np.where(observed, (recency >= 1) & (recency <= 168), recency == -1).all():
        raise ValueError("Historical purchase recency disagrees.")


def _check_ranges(columns: dict[str, np.ndarray]) -> None:
    for name, values in columns.items():
        if name != "pair_purchase_recency_hours" and not (values >= 0).all():
            raise ValueError(f"Negative values in {name}.")
        if name.endswith("_share") and not ((values > 0) & (values <= 1)).all():
            raise ValueError(f"Share outside (0, 1]: {name}")
        if name.endswith("_active_days") and not ((values >= 1) & (values <= 7)).all():
            raise ValueError(f"Active days outside history: {name}")
        if name in {"pair_recency_hours", "user_recency_hours"}:
            if not ((values >= 1) & (values <= 168)).all():
                raise ValueError(f"Recency outside history: {name}")
        if name == "cutoff_weekday" and not (values <= 6).all():
            raise ValueError("Invalid weekday.")
        if name == "label" and not np.isin(values, [0, 1]).all():
            raise ValueError("Invalid binary label.")


def validate_sample(path: Path, labeled: bool, settings: Settings) -> dict:
    """逐批检查全部行，不将抽样检查说成全表检查。

    Args:
        path: 已落盘的正式样本Parquet。
        labeled: 是否必须包含监督标签。
        settings: 唯一键检查的内存与线程限制。

    Returns:
        行数、列数、schema以及全部数值列范围。

    Raises:
        ValueError: 字段、类型、空值、键或业务范围不符合契约。
    """
    dictionary = [
        row for row in field_dictionary() if labeled or row["role"] != "label"
    ]
    source = pq.ParquetFile(path)
    types = {
        "VARCHAR": pa.string(),
        "TIMESTAMP": pa.timestamp("us"),
        "BIGINT": pa.int64(),
        "DOUBLE": pa.float64(),
        "BOOLEAN": pa.bool_(),
    }
    expected = pa.schema([(row["name"], types[row["dtype"]]) for row in dictionary])
    if not source.schema_arrow.equals(expected, check_metadata=False):
        raise ValueError("Feature schema differs from field dictionary.")
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = ?", [settings.memory_limit])
        connection.execute("SET threads = ?", [settings.threads])
        connection.read_parquet(str(path)).create_view("checked")
        counts = connection.execute(
            "SELECT count(*), count(DISTINCT (user_id, item_id, cutoff)) FROM checked"
        ).fetchone()
    if counts is None or counts[0] == 0 or counts[0] != counts[1]:
        raise ValueError("Empty samples or duplicate sample keys.")
    ranges: dict[str, list[float]] = {}
    for batch in source.iter_batches(batch_size=50000):
        columns = _numeric_batch(batch)
        _check_ranges(columns)
        _check_relations(columns)
        for name, values in columns.items():
            low, high = float(values.min()), float(values.max())
            previous = ranges.get(name, [low, high])
            ranges[name] = [min(previous[0], low), max(previous[1], high)]
    return {
        "passed": True,
        "rows": int(counts[0]),
        "columns": len(expected),
        "unique_keys": int(counts[1]),
        "nulls": 0,
        "nonfinite": 0,
        "numeric_ranges": ranges,
    }
