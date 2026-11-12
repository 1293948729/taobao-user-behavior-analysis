"""提供数据准备参数、字段定义和输入来源检查。

主要接口：PreparationOptions、load_options、feature_names、checked_inputs、
output_folder、write_table。
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from taobao_user_behavior.features.dictionary import field_dictionary
from taobao_user_behavior.settings import Settings, tomllib
from taobao_user_behavior.storage import file_hash, read_json

KEYS = ["user_id", "item_id", "cutoff"]
DATASETS = ("train", "validation", "test", "latest")


@dataclass(frozen=True)
class PreparationOptions:
    """保存从训练数据确定处理规则所需的参数。

    Attributes:
        seed: 排序后负样本抽样使用的PCG64种子。
        negative_ratio: 负例数量相对于全部正例数量的上限。
        correlation_limit: 高Spearman相关的绝对阈值。
        near_constant_share: 仅标记、不自动删除的众数比例阈值。

    Examples:
        options = PreparationOptions()
    """

    seed: int = 20261107
    negative_ratio: int = 100
    correlation_limit: float = 0.98
    near_constant_share: float = 0.999

    def __post_init__(self) -> None:
        """拒绝不合法种子、负正比和概率阈值。

        Raises:
            ValueError: 整数参数或阈值不符合约束。
        """
        for name in ("seed", "negative_ratio"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid preparation parameter: {name}")
        for name in ("correlation_limit", "near_constant_share"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0 < value <= 1:
                raise ValueError(f"Invalid preparation parameter: {name}")


def load_options(project: Path) -> PreparationOptions:
    """从项目配置读取准备阶段参数。

    Args:
        project: 包含configs/project.toml的目录。

    Returns:
        已验证的参数。
    """
    with (project / "configs/project.toml").open("rb") as stream:
        values = tomllib.load(stream).get("preparation", {})
    return PreparationOptions(**values)


def feature_names() -> list[str]:
    """取得固定顺序的48项候选特征，不包含标识与标签。

    Returns:
        字段字典中角色为feature的名称。
    """
    return [row["name"] for row in field_dictionary() if row["role"] == "feature"]


def checked_inputs(settings: Settings) -> dict:
    """检查四套样本和标准事件与特征构建记录是否一致。

    Args:
        settings: 当前项目和结果目录。

    Returns:
        四套特征构建记录。

    Raises:
        ValueError: 数据集、原始事件或参与构建的源码与指纹不符。
    """
    document = read_json(settings.outputs / "features/manifest.json")
    names = [row["name"] for row in document["datasets"]]
    if names != list(DATASETS) or document["input_features"] != len(feature_names()):
        raise ValueError("Expected four ordered datasets and matching feature fields.")
    if file_hash(settings.data / "events.parquet") != document["events_sha256"]:
        raise ValueError("Standard events changed; rebuild features first.")
    for relative, expected in document["source_sha256"].items():
        if file_hash(settings.project / relative) != expected:
            raise ValueError(f"Feature source changed: {relative}")
    for row in document["datasets"]:
        expected_path = settings.data / "features" / f"{row['name']}.parquet"
        recorded = (settings.root / row["file"]).resolve()
        if recorded != expected_path.resolve():
            raise ValueError("Unexpected feature data location.")
        if file_hash(expected_path) != row["sha256"]:
            raise ValueError(f"Changed feature data: {row['name']}")
    return document


def output_folder(settings: Settings, relative: str) -> Path:
    """创建运行根内的输出目录并拒绝目录链接外跳。

    Args:
        settings: 当前运行位置。
        relative: 运行根内的相对目录。

    Returns:
        已存在且通过边界检查的目录。

    Raises:
        ValueError: 目录或现有子文件解析到运行根外。
    """
    root = settings.root.resolve()
    folder = (root / relative).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("Preparation output must stay inside the run directory.")
    if folder.exists() and any(
        not path.resolve().is_relative_to(root) for path in folder.rglob("*")
    ):
        raise ValueError("Preparation output contains an external link.")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_table(folder: Path, name: str, frame: pd.DataFrame) -> None:
    """以UTF-8 CSV保存一项分析，索引不作为隐含列输出。

    Args:
        folder: 已验证的输出目录。
        name: 不带扩展名的表名。
        frame: 包含显式字段的结果。
    """
    frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
