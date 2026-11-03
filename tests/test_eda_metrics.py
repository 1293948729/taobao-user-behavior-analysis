"""核对SQL守恒、覆盖率分母、漏斗与留存的业务边界。

测试入口：本模块test_开头的函数。
"""

import pandas as pd
import pytest

from taobao_user_behavior.analysis.cohorts import label_tables, user_tables
from taobao_user_behavior.analysis.metrics import overview_metrics, validate_aggregates
from taobao_user_behavior.analysis.temporal import temporal_tables
from taobao_user_behavior.analysis.warehouse import open_warehouse
from taobao_user_behavior.settings import Settings


def test_aggregate_conservation_and_coverage(eda_settings: Settings) -> None:
    """事件不去重，覆盖率按用户商品对计算。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        summary = overview_metrics(connection)
        checked = validate_aggregates(connection)

    assert checked["passed"]
    assert set(checked["totals"].values()) == {13}
    assert summary["users"] == 4
    assert summary["categories"] == 3
    assert summary["missing_category_rows"] == 1
    assert summary["views"] == 5
    assert summary["purchases"] == 5
    assert summary["viewed_pairs"] == 4
    assert summary["bought_viewed_pairs"] == 2
    assert summary["viewed_pair_coverage"] == 0.5
    assert summary["buyer_coverage"] == 0.75


def test_observation_window_excludes_new_pairs(eda_settings: Settings) -> None:
    """后续窗口从边界当天起算，未来首次出现的对不进入历史分母。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        tables = label_tables(connection)
        raw_candidates = connection.execute(
            "SELECT * FROM candidates ORDER BY user_id, item_id"
        ).df()

    window = tables["label_window"].iloc[0]
    assert str(window["label_start"])[:10] == "2025-12-04"
    assert str(window["label_end"])[:10] == "2025-12-11"
    assert window["candidates"] == 3
    assert window["positives"] == 2
    assert window["negatives"] == 1
    assert window["all_label_pairs"] == 3
    assert raw_candidates.loc[raw_candidates["user_id"].eq("001"), "views"].sum() == 3
    assert "003" not in raw_candidates["user_id"].tolist()
    assert tables["depth_purchase"].groupby("behavior")["candidates"].sum().eq(3).all()


def test_retention_has_right_censoring(eda_settings: Settings) -> None:
    """末日新观察用户的D1与D7为空，不能按零留存计入分母。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        retained = temporal_tables(connection)["retention"]
    last = retained.loc[retained["first_date"].eq(pd.Timestamp("2025-12-10"))]
    d1 = last.loc[last["lag"].eq(1)].iloc[0]
    assert d1["eligible"] == 0
    assert pd.isna(d1["retention"])
    start_d1 = retained.loc[
        retained["first_date"].eq(pd.Timestamp("2025-12-01")) & retained["lag"].eq(1)
    ].iloc[0]
    assert start_d1["eligible"] == start_d1["returned"] == 1


def test_daily_calendar_includes_zero_days(eda_settings: Settings) -> None:
    """无日志的日期补四类零值，不将缺失日连接成连续活跃。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        daily = temporal_tables(connection)["daily"]
    missing = daily.loc[daily["event_date"].eq(pd.Timestamp("2025-12-05"))]
    assert len(daily) == 40
    assert len(missing) == 4
    assert missing["events"].sum() == 0


def test_funnel_is_nested_cooccurrence(eda_settings: Settings) -> None:
    """漏斗按行为交集递减，不把有购买但无浏览的对计入末级。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        tables = user_tables(connection)
    assert tables["funnel"]["pairs"].tolist() == [4, 2, 1, 1]
    assert tables["independent_coverage"].iloc[0]["purchased"] == 3
    assert tables["users"]["user_id"].tolist() == ["001", "002", "003", "004"]


def test_incomplete_label_window_is_rejected(eda_settings: Settings) -> None:
    """不足八个日期的窗口不得强行构造七日标签。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        connection.execute(
            """
            CREATE TEMP TABLE source_subset AS SELECT * FROM events
            WHERE event_time < TIMESTAMP '2025-12-04 00:00:00'
            """,
        )
        connection.execute("CREATE TEMP VIEW events AS SELECT * FROM source_subset")
        with pytest.raises(ValueError, match="eight observed"):
            label_tables(connection)
