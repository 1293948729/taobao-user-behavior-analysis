"""验证重复保留、缺失处理和数据量核对。

测试入口：test_repeated_purchases_and_missing_category_are_preserved、
test_invalid_essential_fields_are_quarantined_with_reasons、
test_ingestion_and_audit_reconcile_all_rows、test_high_activity_is_retained。
配置入口：isolated_settings。
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from taobao_user_behavior.quality import audit, ingest, normalize_events
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import RAW_COLUMNS


@pytest.fixture(name="settings")
def isolated_settings(tmp_path: Path) -> Settings:
    """为每个测试创建独立的数据和报告目录。

    Args:
        tmp_path: pytest提供的临时目录。

    Returns:
        已创建输出目录的测试配置。
    """
    configuration = Settings(root=tmp_path, chunk_size=2, threads=1)
    configuration.data.mkdir()
    configuration.outputs.mkdir()
    return configuration


def test_repeated_purchases_and_missing_category_are_preserved() -> None:
    """同小时购买和缺少类目的事件都应保留。"""
    raw = pd.DataFrame(
        [
            ["2025-12-01 10", "u1", "i1", "c1", "4"],
            ["2025-12-01 10", "u1", "i1", "c1", "4"],
            ["2025-12-01 11", "u2", "i2", "", "1"],
        ],
        columns=RAW_COLUMNS,
    )
    accepted, rejected = normalize_events(raw)
    assert len(accepted) == 3
    assert rejected.empty
    assert accepted["category_missing"].sum() == 1


def test_invalid_essential_fields_are_quarantined_with_reasons() -> None:
    """无效时间、缺失编号和错误行为应附有隔离原因。"""
    raw = pd.DataFrame(
        [
            ["2025-02-30 10", "u1", "i1", "c1", "1"],
            ["2025-12-01 10", "", "i1", "c1", "9"],
        ],
        columns=RAW_COLUMNS,
    )
    accepted, rejected = normalize_events(raw)
    assert accepted.empty
    assert "invalid_time" in rejected.iloc[0]["reason"]
    assert {"missing_user", "invalid_behavior"} == set(
        rejected.iloc[1]["reason"].split(";")
    )


def test_ingestion_and_audit_reconcile_all_rows(settings: Settings) -> None:
    """分块写入后，接受与隔离行数之和应等于输入行数。

    Args:
        settings: 测试专用的目录和分块配置。
    """
    archive_path = settings.root / "source.zip"
    rows = [
        ",".join(RAW_COLUMNS),
        "2025-12-01 10,u1,i1,c1,4",
        "2025-12-01 10,u1,i1,c1,4",
        "2025-12-01 11,u2,i2,,1",
        "invalid,u3,i3,c3,1",
    ]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", "\n".join(rows) + "\n")
    manifest = ingest(archive_path, settings)
    report = audit(settings)
    assert manifest["raw_rows"] == manifest["accepted_rows"] + manifest["rejected_rows"]
    assert report["rows"] == 3
    assert report["repeated_surplus"] == 1
    assert report["repeated_groups"] == 1
    assert len(pd.read_parquet(settings.data / "rejected.parquet")) == 1


def test_high_activity_is_retained(settings: Settings) -> None:
    """次数明显偏高的用户仍保留全部记录。

    Args:
        settings: 测试专用的输出目录和分块配置。
    """
    archive_path = settings.root / "active.zip"
    rows = [",".join(RAW_COLUMNS)]
    rows.extend(["2025-12-01 10,u0,i0,c1,4"] * 100)
    rows.extend(f"2025-12-01 10,u{index},i1,c1,1" for index in range(1, 10))
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", "\n".join(rows) + "\n")
    ingest(archive_path, settings)
    events_path = settings.data / "events.parquet"
    before = events_path.read_bytes()

    report = audit(settings)

    assert report["activity_maximum"] == 100
    assert report["users_above_iqr"] == 1
    assert report["rows"] == 109
    assert events_path.read_bytes() == before
    assert not (settings.data / "unique_quadruples.parquet").exists()
