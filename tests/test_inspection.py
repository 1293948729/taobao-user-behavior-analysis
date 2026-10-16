"""验证样本读取上限、字段顺序和压缩成员限制。

测试入口：test_sample_limit_and_leading_zero_ids、
test_unexpected_header_is_rejected、test_missing_member_is_rejected、
test_bad_row_width_is_rejected、test_nonpositive_limit_is_rejected。
"""

from pathlib import Path
from zipfile import ZipFile

import pytest

from taobao_user_behavior.inspection import inspect_source


def test_sample_limit_and_leading_zero_ids(tmp_path: Path) -> None:
    """只读取指定行数，并记录缺失类目。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    archive_path = tmp_path / "input.zip"
    source = (
        "time,user_id,item_id,item_category,behavior_type\n"
        + "2025-12-01 10,001,002,,1\n"
        + "2025-12-01 11,003,004,c1,4\n"
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", source)
    original = archive_path.read_bytes()

    profile = inspect_source(archive_path, 1)

    assert profile.sampled_rows == 1
    assert profile.blank_counts["item_category"] == 1
    assert profile.behavior_counts == {"1": 1}
    assert archive_path.read_bytes() == original


def test_unexpected_header_is_rejected(tmp_path: Path) -> None:
    """字段顺序不正确时停止读取。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    archive_path = tmp_path / "bad.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("user_behavior.csv", "user_id,time\nu1,2025-12-01 10\n")

    with pytest.raises(ValueError, match="Unexpected columns"):
        inspect_source(archive_path)


def test_missing_member_is_rejected(tmp_path: Path) -> None:
    """只读取固定成员，不解压任意路径。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    archive_path = tmp_path / "nested.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../user_behavior.csv", "ignored")

    with pytest.raises(ValueError, match="exactly one"):
        inspect_source(archive_path)
    assert not (tmp_path.parent / "user_behavior.csv").exists()


def test_bad_row_width_is_rejected(tmp_path: Path) -> None:
    """字段数量不足的行不能作为正常样本。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    archive_path = tmp_path / "short.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "user_behavior.csv",
            "time,user_id,item_id,item_category,behavior_type\nbad,u1\n",
        )

    with pytest.raises(ValueError, match="Invalid CSV width"):
        inspect_source(archive_path)


def test_nonpositive_limit_is_rejected(tmp_path: Path) -> None:
    """先检查行数参数，再打开文件。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    with pytest.raises(ValueError, match="positive"):
        inspect_source(tmp_path / "unused.zip", 0)
