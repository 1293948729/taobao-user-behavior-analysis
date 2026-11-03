"""验证跨小时集合转移、支持量和空分母。

测试入口：本模块test_开头的函数。
"""

from pathlib import Path

import pandas as pd
import pytest

from taobao_user_behavior.analysis.metrics import metric_dictionary
from taobao_user_behavior.analysis.products import product_tables
from taobao_user_behavior.analysis.temporal import temporal_tables
from taobao_user_behavior.analysis.warehouse import build_warehouse, open_warehouse
from taobao_user_behavior.quality import normalize_events
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import RAW_COLUMNS


def test_transition_weights_and_normalization(eda_settings: Settings) -> None:
    """同小时多行为不排序，每次跨小时关系总权重为1。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        tables = temporal_tables(connection)
    transitions = tables["transitions"]
    row_sums = transitions.groupby("current_code")["probability"].sum()
    assert row_sums.to_numpy() == pytest.approx([1] * len(row_sums))
    assert transitions["weight"].sum() == pytest.approx(6)
    assert tables["transition_gaps"].iloc[0]["transitions"] == 6


def test_product_support_does_not_invent_rankings(eda_settings: Settings) -> None:
    """微型数据未达到20个浏览对时，覆盖排名与四象限样本应为空。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        tables = product_tables(connection)
    assert tables["item_conversion"].empty
    assert tables["category_conversion"].empty
    assert tables["quadrant_sample"].empty
    assert tables["quadrant_thresholds"].iloc[0]["eligible_items"] == 0
    assert tables["item_flow"]["events"].sum() == 13


def test_no_browse_has_null_purchase_coverage(eda_settings: Settings) -> None:
    """只有购买记录的商品没有浏览分母，覆盖率应为空而不是0或100%。

    Args:
        eda_settings: 已建库的微型输入。
    """
    with open_warehouse(eda_settings) as connection:
        items = product_tables(connection)["item_flow"].set_index("item_id")
    assert pd.isna(items.loc["004", "purchase_coverage"])
    assert items.loc["004", "purchases"] == 1


def test_empty_retained_data_stops_analysis(tmp_path: Path) -> None:
    """全被隔离后的空表不能生成成功EDA报告。

    Args:
        tmp_path: 本测试独立目录。
    """
    settings = Settings(tmp_path, threads=1)
    settings.data.mkdir()
    raw = pd.DataFrame(columns=RAW_COLUMNS)
    accepted, _ = normalize_events(raw)
    accepted.to_parquet(settings.data / "events.parquet", index=False)
    with pytest.raises(ValueError, match="at least one"):
        build_warehouse(settings)
    assert not (settings.data / "analysis.duckdb").exists()


def test_metric_definitions_are_explicit() -> None:
    """指标说明必须区分真实时长、候选分母和购买事件。"""
    dictionary = metric_dictionary().set_index("metric")
    assert "不是在线或停留时长" in dictionary.loc["日活跃小时代理", "limitation"]
    assert "历史候选" in dictionary.loc["七日候选正例率", "limitation"]
    assert "非可靠订单" in dictionary.loc["购买事件", "limitation"]


def test_twenty_viewed_pairs_pass_support_boundary(tmp_path: Path) -> None:
    """20个不同浏览对达到支持量，重复浏览不增加覆盖分母。

    Args:
        tmp_path: 当前测试独立目录。
    """
    settings = Settings(tmp_path, threads=1)
    settings.data.mkdir()
    rows = [["2025-12-01 00", f"u{index}", "i1", "c1", "1"] for index in range(20)]
    rows.extend([["2025-12-01 00", "u0", "i1", "c1", "1"]] * 5)
    rows.extend(["2025-12-01 01", f"u{index}", "i1", "c1", "4"] for index in range(5))
    accepted, _ = normalize_events(pd.DataFrame(rows, columns=RAW_COLUMNS))
    accepted.to_parquet(settings.data / "events.parquet", index=False)

    build_warehouse(settings)
    with open_warehouse(settings) as connection:
        tables = product_tables(connection)

    ranked = tables["item_conversion"].iloc[0]
    assert ranked["viewed_pairs"] == 20
    assert ranked["bought_viewed_pairs"] == 5
    assert ranked["purchase_coverage"] == 0.25
    assert tables["quadrant_counts"]["items"].sum() == 1
