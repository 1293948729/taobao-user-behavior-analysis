"""验证小时重复阈值、全局分组和隔离结果。

测试入口：test_boundary_and_cross_chunk_conservation、
test_group_keys_are_independent、test_category_does_not_split_group、
test_all_suspicious_rows_can_be_quarantined、test_limit_is_configurable。
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from taobao_user_behavior.quality import audit, ingest
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import RAW_COLUMNS


def _run(
    root: Path, rows: list[str], limit: int = 100
) -> tuple[Settings, dict, pd.DataFrame, pd.DataFrame]:
    settings = Settings(root, chunk_size=60, threads=1, hourly_repeat_limit=limit)
    settings.data.mkdir()
    settings.outputs.mkdir()
    archive_path = root / "input.zip"
    with ZipFile(archive_path, "w") as archive:
        contents = "\n".join([",".join(RAW_COLUMNS), *rows]) + "\n"
        archive.writestr("user_behavior.csv", contents)
    manifest = ingest(archive_path, settings)
    accepted = pd.read_parquet(settings.data / "events.parquet")
    rejected = pd.read_parquet(settings.data / "hourly_rejected.parquet")
    return settings, manifest, accepted, rejected


def test_boundary_and_cross_chunk_conservation(tmp_path: Path) -> None:
    """100次保留，跨块的101次整组隔离，另计字段无效行。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    rows = ["2025-12-01 10,u1,i1,c1,4"] * 100
    rows += ["2025-12-01 10,u2,i1,c1,4"] * 101
    rows += ["invalid,u3,i1,c1,4"]
    settings, manifest, accepted, rejected = _run(tmp_path, rows)

    assert len(accepted) == 100
    assert accepted["user_id"].eq("u1").all()
    assert len(rejected) == 101
    assert rejected["user_id"].eq("u2").all()
    assert rejected["hourly_count"].eq(101).all()
    assert rejected["reason"].eq("hourly_repeat_limit_exceeded").all()
    assert manifest["hourly_rule"]["flagged_groups"] == 1
    invalid = pd.read_parquet(settings.data / "rejected.parquet")
    assert invalid["reason"].tolist() == ["invalid_time"]
    assert manifest["raw_rows"] == len(accepted) + len(rejected) + len(invalid)
    assert manifest["accepted_rows"] == 100
    assert manifest["rejected_rows"] == 1
    assert manifest["hourly_rejected_rows"] == 101
    assert audit(settings)["rows"] == 100


@pytest.mark.parametrize(
    "different",
    [
        "2025-12-01 10,u2,i1,c1,1",
        "2025-12-01 10,u1,i2,c1,1",
        "2025-12-01 10,u1,i1,c1,4",
        "2025-12-01 11,u1,i1,c1,1",
        "2025-12-02 10,u1,i1,c1,1",
    ],
)
def test_group_keys_are_independent(tmp_path: Path, different: str) -> None:
    """用户、商品、行为或小时不同的记录不能合并计数。

    Args:
        tmp_path: 本测试独立使用的临时目录。
        different: 与基准记录只有一个分组字段不同的记录。
    """
    rows = ["2025-12-01 10,u1,i1,c1,1"] * 60 + [different] * 60
    _, manifest, accepted, rejected = _run(tmp_path, rows)

    assert len(accepted) == 120
    assert rejected.empty
    assert manifest["hourly_rule"]["maximum_count"] == 60
    assert manifest["hourly_rule"]["total_groups"] == 2


def test_category_does_not_split_group(tmp_path: Path) -> None:
    """不同类目或缺类目不能拆开同一小时重复组。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    rows = ["2025-12-01 10,u1,i1,c1,1"] * 60
    rows += ["2025-12-01 10,u1,i1,c2,1"] * 40
    rows += ["2025-12-01 10, u1 , i1 ,,1"]
    _, manifest, accepted, rejected = _run(tmp_path, rows)

    assert accepted.empty
    assert len(rejected) == 101
    assert set(rejected["category_id"]) == {"c1", "c2", "__missing__"}
    assert rejected["hourly_count"].eq(101).all()
    assert manifest["hourly_rule"]["flagged_groups"] == 1


def test_all_suspicious_rows_can_be_quarantined(tmp_path: Path) -> None:
    """全组超限时也应完成隔离，空保留数据的统计不能写成NaN。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    settings, manifest, accepted, rejected = _run(
        tmp_path, ["2025-12-01 10,u1,i1,c1,1"] * 101
    )
    report = audit(settings)

    assert accepted.empty
    assert len(rejected) == manifest["raw_rows"] == 101
    assert report["rows"] == 0
    assert report["repeated_groups"] == report["repeated_surplus"] == 0
    assert report["activity_iqr_upper"] is None
    assert report["activity_maximum"] is None
    assert report["users_above_iqr"] == 0
    assert "NaN" not in (settings.outputs / "quality.json").read_text(encoding="utf-8")


def test_limit_is_configurable(tmp_path: Path) -> None:
    """配置阈值应参与处理，不应在SQL中固定写为100。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    _, manifest, accepted, rejected = _run(
        tmp_path, ["2025-12-01 10,u1,i1,c1,4"] * 3, limit=2
    )

    assert accepted.empty
    assert len(rejected) == 3
    assert manifest["hourly_rule"]["threshold"] == 2
