"""标准化原始字段并按原因隔离无效行。

主要接口：normalize_events；标准事件及隔离结构供接入流程使用。
"""

import pandas as pd
import pyarrow as pa

from taobao_user_behavior.storage import RAW_COLUMNS

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
