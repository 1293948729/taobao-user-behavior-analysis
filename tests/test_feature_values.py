"""检查候选、标签、字段值与防止未来信息泄漏的业务契约。

测试入口：本模块test_开头的函数。
"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taobao_user_behavior.data.normalization import EVENT_SCHEMA
from taobao_user_behavior.features import queries
from taobao_user_behavior.features.pipeline import build_features, prepare_window
from taobao_user_behavior.features.windows import FeatureWindow
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import connect


def _events() -> pd.DataFrame:
    rows = [
        ("2025-11-30 23", "001", "002", "old", 1),
        ("2025-12-01 00", "001", "002", "c2", 1),
        ("2025-12-05 01", "001", "002", "c3", 3),
        ("2025-12-07 12", "001", "002", "c2", 4),
        ("2025-12-07 23", "001", "002", "c2", 3),
        ("2025-12-07 23", "001", "002", "c1", 2),
        ("2025-12-07 21", "001", "009", "__missing__", 4),
        ("2025-12-07 22", "003", "008", "c1", 1),
        ("2025-12-07 20", "004", "008", "c1", 1),
        ("2025-12-08 00", "001", "002", "future", 4),
        ("2025-12-08 01", "new", "new", "future", 4),
        ("2025-12-15 00", "004", "008", "c1", 4),
    ]
    for day in range(28):
        time = datetime(2025, 11, 24) + timedelta(days=day)
        rows.append((str(time), "anchor", "anchor", "anchor", 1))
    rows.append(("2025-12-21 23", "anchor", "anchor", "anchor", 1))
    frame = pd.DataFrame(
        rows,
        columns=["event_time", "user_id", "item_id", "category_id", "behavior_type"],
    )
    frame["event_time"] = frame["event_time"].map(pd.Timestamp)
    frame["category_missing"] = frame["category_id"].eq("__missing__")
    return frame


def _sample(frame: pd.DataFrame) -> pd.DataFrame:
    with duckdb.connect() as connection:
        connection.register("events", frame)
        prepare_window(
            connection,
            FeatureWindow(
                "test",
                datetime(2025, 12, 1),
                datetime(2025, 12, 8),
                datetime(2025, 12, 15),
            ),
        )
        return connection.sql(queries.LABELED).df()


def test_boundaries_category_ties_and_exact_values() -> None:
    """历史起点包含、截点排除，类目仅取历史且同小时稳定选择。"""
    sample = _sample(_events()).set_index(["user_id", "item_id"])
    row = sample.loc[("001", "002")]
    assert row["pair_events"] == 5
    assert row["pair_views"] == 1
    assert row["pair_carts"] == 2
    assert row["pair_favorites"] == row["pair_purchases"] == 1
    assert row["pair_events_1d"] == 3
    assert row["pair_events_3d"] == 4
    assert row["pair_active_days"] == 3
    assert row["pair_recency_hours"] == 1
    assert row["pair_purchase_recency_hours"] == 12
    assert row["category_id"] == "c1"
    assert row["user_events"] == 6
    assert row["user_items"] == 2
    assert row["user_categories"] == 3
    assert row["category_events"] == 3
    assert row["user_category_events"] == 1
    assert row["user_category_share"] == pytest.approx(1 / 6)
    assert row["pair_user_share"] == pytest.approx(5 / 6)
    assert row["pair_item_share"] == 1
    assert row["pair_cart_view_ratio"] == 2
    assert row["label"] == 1
    assert sample.loc[("004", "008"), "label"] == 0
    assert ("new", "new") not in sample.index


def test_future_changes_leave_all_features_unchanged() -> None:
    """只改未来购买与类目时，历史字段完全相同而标签可以变化。"""
    events = _events()
    before = _sample(events)
    modified = events.copy()
    future = modified["event_time"].ge(pd.Timestamp("2025-12-08"))
    modified.loc[future, "behavior_type"] = 1
    modified.loc[future, "category_id"] = "changed_future"
    after = _sample(modified)
    pd.testing.assert_frame_equal(
        before.drop(columns="label"), after.drop(columns="label")
    )
    assert before["label"].sum() == 1
    assert after["label"].sum() == 0


def test_zero_views_missing_category_and_no_purchase() -> None:
    """无浏览购买与类目缺失保留，未购买使用独立缺省值和标记。"""
    sample = _sample(_events()).set_index(["user_id", "item_id"])
    missing = sample.loc[("001", "009")]
    assert missing["category_id"] == "__missing__"
    assert missing["category_missing"]
    assert missing["category_events"] == 1
    assert missing["pair_no_views"]
    assert missing["pair_purchase_view_ratio"] == 0
    assert missing["pair_purchase_observed"]
    assert missing["pair_purchase_recency_hours"] == 3
    negative = sample.loc[("003", "008")]
    assert not negative["pair_purchase_observed"]
    assert negative["pair_purchase_recency_hours"] == -1
    assert negative["item_users"] == 2


def test_literal_missing_text_is_separate_from_missing_group() -> None:
    """原始类目恰好等于缺失占位文本时，仍以缺失标记区分统计组。"""
    events = _events()
    events.loc[events["item_id"].eq("008"), "category_id"] = "__missing__"
    sample = _sample(events).set_index(["user_id", "item_id"])
    assert sample.loc[("001", "009"), "category_events"] == 1
    assert sample.loc[("003", "008"), "category_events"] == 2
    assert not sample.loc[("003", "008"), "category_missing"]


def _settings(tmp_path: Path, events: pd.DataFrame) -> Settings:
    settings = Settings(tmp_path / "artifacts/test", project_root=tmp_path, threads=1)
    settings.data.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(events, schema=EVENT_SCHEMA, preserve_index=False),
        settings.data / "events.parquet",
    )
    return settings


def test_export_has_full_schema_and_latest_has_no_label(tmp_path: Path) -> None:
    """四套导出保留字符串ID，并对全部行验证schema和键。

    Args:
        tmp_path: 当前测试独立的临时目录。
    """
    settings = _settings(tmp_path, _events())
    document = build_features(settings)
    assert document["input_features"] == 48
    assert document["metadata_fields"] == 4
    assert document["label_fields"] == 1
    assert len(document["datasets"]) == 4
    with connect(settings) as connection:
        train = connection.read_parquet(str(settings.data / "features/train.parquet"))
        latest = connection.read_parquet(str(settings.data / "features/latest.parquet"))
        assert len(train.columns) == 53
        assert len(latest.columns) == 52
        assert "label" not in latest.columns
        assert "user_id" in latest.columns
    assert all(row["checks"]["passed"] for row in document["datasets"])


def test_empty_history_is_rejected_instead_of_silent_export(tmp_path: Path) -> None:
    """完整日范围并不能代替实际候选，空历史不导出成功记录。

    Args:
        tmp_path: 当前测试独立的临时目录。
    """
    events = _events().iloc[[0, -1]].copy()
    events.iloc[0, events.columns.get_loc("event_time")] = pd.Timestamp("2025-11-01")
    settings = _settings(tmp_path, events)
    with pytest.raises(ValueError, match="No historical candidates"):
        build_features(settings)
