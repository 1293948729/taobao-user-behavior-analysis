"""以完整合成日期流程检查准备阶段与原特征产物的边界。

测试入口：本模块test_开头的函数。
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taobao_user_behavior.data.normalization import EVENT_SCHEMA
from taobao_user_behavior.features.pipeline import build_features
from taobao_user_behavior.preparation.common import checked_inputs, output_folder
from taobao_user_behavior.preparation.pipeline import prepare_data
from taobao_user_behavior.preparation.temporal import review_windows
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import file_hash, read_json


@pytest.fixture(name="prepared_source")
def fixture_prepared_source(tmp_path: Path) -> Settings:
    """建立31完整日、正负样本并存的独立特征输入。

    Args:
        tmp_path: 测试专用项目位置。

    Returns:
        已生成四套宽表的运行配置。
    """
    config = tmp_path / "configs"
    config.mkdir()
    (config / "project.toml").write_text(
        "[preparation]\nnegative_ratio = 2\n", encoding="utf-8"
    )
    settings = Settings(tmp_path / "artifacts", project_root=tmp_path)
    settings.data.mkdir(parents=True)
    rows = []
    for day in pd.date_range("2025-11-18", periods=31):
        for user in range(8):
            for occurrence in range(1 + (day.day + user) % 3):
                rows.append(
                    (
                        day + pd.Timedelta(hours=occurrence),
                        f"{user:03}",
                        f"{user % 3:03}",
                        f"{user % 2:02}",
                        1,
                        False,
                    )
                )
            if user % 2 == 0 and day.day % 4 == 0:
                rows.append(
                    (
                        day + pd.Timedelta(hours=20),
                        f"{user:03}",
                        f"{user % 3:03}",
                        f"{user % 2:02}",
                        4,
                        False,
                    )
                )
    rows.append((pd.Timestamp("2025-12-18 23"), "007", "001", "01", 1, False))
    frame = pd.DataFrame(rows, columns=EVENT_SCHEMA.names)
    pq.write_table(
        pa.Table.from_pandas(frame, schema=EVENT_SCHEMA, preserve_index=False),
        settings.data / "events.parquet",
    )
    build_features(settings)
    return settings


def test_full_preparation_preserves_original_and_projects_consistently(
    prepared_source: Settings,
) -> None:
    """准备后四原表不变，同一筛选清单应用四集合，latest无标签。

    Args:
        prepared_source: 已构建四套宽表的测试配置。
    """
    inputs = list((prepared_source.data / "features").glob("*.parquet"))
    original = {path: file_hash(path) for path in inputs}
    result = prepare_data(prepared_source)
    assert all(file_hash(path) == expected for path, expected in original.items())
    selected = result["selection"]["selected_features"]
    assert selected
    assert not result["model_training_performed"]
    assert not result["selection"]["labels_used_for_selection"]
    for row in result["projected"]:
        source = pd.read_parquet(
            prepared_source.data / "features" / f"{row['dataset']}.parquet"
        )
        output = pd.read_parquet(prepared_source.root / row["file"])
        pd.testing.assert_frame_equal(output, source[row["columns"]])
        assert ("label" in output) == (row["dataset"] != "latest")
    daily = pd.read_csv(prepared_source.outputs / "preparation/window_daily.csv")
    assert daily.groupby(["dataset", "phase"]).size().eq(7).all()
    assert result["weights"]["fit_dataset"] == "full_natural_train"


def test_changed_input_or_time_plan_is_rejected(prepared_source: Settings) -> None:
    """文件篡改或窗口定义变动均不得沿用旧结果。

    Args:
        prepared_source: 已构建四套宽表的测试配置。
    """
    manifest = read_json(prepared_source.outputs / "features/manifest.json")
    manifest["datasets"][0]["history_start"] = "2025-11-22 00:00:00"
    with pytest.raises(ValueError, match="Changed temporal plan"):
        review_windows(prepared_source, manifest)
    path = prepared_source.data / "features/validation.parquet"
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="Changed feature data"):
        checked_inputs(prepared_source)


def test_preparation_output_rejects_escape(tmp_path: Path) -> None:
    """输出目录不能解析到运行根之外。

    Args:
        tmp_path: 测试专用位置。
    """
    with pytest.raises(ValueError, match="inside"):
        output_folder(Settings(tmp_path / "artifacts"), "../outside")
