"""验证编号、时间格式和重新读取的处理规则。

测试入口：test_ids_remain_strings_and_input_is_unchanged、
test_intra_hour_time_is_rejected、test_header_mismatch_stops_ingestion、
test_changed_source_is_read_again。
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from taobao_user_behavior.quality import ingest, normalize_events
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import RAW_COLUMNS


def test_ids_remain_strings_and_input_is_unchanged() -> None:
    """去掉两端空白时保留编号前导零，不改动输入表。"""
    raw = pd.DataFrame(
        [["2025-12-01 10", " 001 ", "002", "", " 4 "]], columns=RAW_COLUMNS
    )
    original = raw.copy(deep=True)

    accepted, rejected = normalize_events(raw)

    assert accepted.iloc[0]["user_id"] == "001"
    assert accepted.iloc[0]["item_id"] == "002"
    assert accepted.iloc[0]["category_id"] == "__missing__"
    assert rejected.empty
    pd.testing.assert_frame_equal(raw, original)


def test_intra_hour_time_is_rejected() -> None:
    """分钟精度的时间不能被悄悄截断为小时。"""
    raw = pd.DataFrame(
        [["2025-12-01 10:30", "u1", "i1", "c1", "1"]], columns=RAW_COLUMNS
    )

    accepted, rejected = normalize_events(raw)

    assert accepted.empty
    assert rejected.iloc[0]["reason"] == "invalid_time"


def test_header_mismatch_stops_ingestion(tmp_path: Path) -> None:
    """表头不符合要求时，不能生成正式事件文件。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    archive_path = tmp_path / "bad_header.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", "time,user_id\n2025-12-01 10,u1\n")

    with pytest.raises(ValueError, match="Expected"):
        ingest(archive_path, Settings(tmp_path))
    assert not (tmp_path / "data/events.parquet").exists()


def test_changed_source_is_read_again(tmp_path: Path) -> None:
    """源文件改变后应重新读取，不能沿用旧计数。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    settings = Settings(tmp_path)
    settings.data.mkdir()
    settings.outputs.mkdir()
    archive_path = tmp_path / "source.zip"
    header = ",".join(RAW_COLUMNS) + "\n"
    record = "2025-12-01 10,001,002,c1,4\n"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", header + record)
    previous = ingest(archive_path, settings)

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", header + record * 2)
    current = ingest(archive_path, settings)

    assert previous["source_sha256"] != current["source_sha256"]
    assert current["accepted_rows"] == 2
    assert len(pd.read_parquet(settings.data / "events.parquet")) == 2
