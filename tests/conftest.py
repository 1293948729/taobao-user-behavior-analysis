"""为探索分析测试提供有完整七日后续窗口的微型数据。

主要接口：eda_settings。
"""

from pathlib import Path
from zipfile import ZipFile

import pytest

from taobao_user_behavior.analysis.warehouse import build_warehouse
from taobao_user_behavior.quality import audit, ingest
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import RAW_COLUMNS


@pytest.fixture
def eda_settings(tmp_path: Path) -> Settings:
    """创建固定输入并建立分析库，不使用真实数据。

    Args:
        tmp_path: 当前测试独立的临时目录。

    Returns:
        指向微型分析库的配置。
    """
    settings = Settings(tmp_path, threads=1)
    settings.data.mkdir()
    settings.outputs.mkdir()
    rows = [
        "2025-12-01 00,001,002,c1,1",
        "2025-12-01 00,001,002,c1,1",
        "2025-12-01 00,001,002,c1,2",
        "2025-12-01 01,001,002,c1,3",
        "2025-12-01 01,001,002,c1,4",
        "2025-12-02 12,001,003,c2,1",
        "2025-12-02 12,002,002,c1,1",
        "2025-12-03 12,002,002,c1,3",
        "2025-12-04 00,001,002,c1,4",
        "2025-12-04 00,003,004,c3,4",
        "2025-12-08 10,001,002,c1,4",
        "2025-12-09 10,002,002,c1,4",
        "2025-12-10 23,004,009,,1",
    ]
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "user_behavior.csv",
            "\n".join([",".join(RAW_COLUMNS), *rows]) + "\n",
        )
    ingest(archive_path, settings)
    audit(settings)
    build_warehouse(settings)
    return settings
