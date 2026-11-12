"""用训练参考分箱描述各集合分布差异，不据测试数据调整筛选。

主要接口：distribution_reference、compare_distribution。
"""

import numpy as np
import pandas as pd

from taobao_user_behavior.preparation.selection import feature_profile


def _probabilities(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(float)
    proportions = counts / len(values) + 1e-6
    return proportions / proportions.sum()


def distribution_reference(frame: pd.DataFrame) -> dict:
    """在自然训练特征上拟合分箱和参考分布。

    Args:
        frame: 完整训练的数值或布尔特征。

    Returns:
        仅包含训练统计的参考字典。
    """
    profile = feature_profile(frame).set_index("feature")
    reference = {}
    for name in frame:
        values = frame[name].to_numpy(dtype=float)
        row = profile.loc[name]
        cuts = np.unique(np.quantile(values, np.arange(0.1, 1, 0.1)))
        edges = np.concatenate(([-np.inf], cuts, [np.inf]))
        reference[name] = {
            "mean": float(row["mean"]),
            "std": float(np.std(values)),
            "constant": bool(row["unique_values"] == 1),
            "constant_value": float(values[0]),
            "edges": edges,
            "probabilities": _probabilities(values, edges),
        }
    return reference


def compare_distribution(
    frame: pd.DataFrame, reference: dict, dataset: str
) -> pd.DataFrame:
    """报告分布差异，不产生特征选择或采样决策。

    Args:
        frame: 待描述的全部候选特征。
        reference: 已冻结的训练分箱与统计。
        dataset: 结果中的集合名称。

    Returns:
        含分位数、标准化均值差、PSI与常量变化的逐字段统计。
    """
    profile = feature_profile(frame)
    rows = []
    for record in profile.to_dict("records"):
        name = record["feature"]
        base = reference[name]
        values = frame[name].to_numpy(dtype=float)
        difference = record["mean"] - base["mean"]
        if base["constant"]:
            psi = None
            change = float(np.mean(values != base["constant_value"]))
        else:
            actual = _probabilities(values, base["edges"])
            expected = base["probabilities"]
            psi = float(np.sum((actual - expected) * np.log(actual / expected)))
            change = None
        rows.append(
            {
                "dataset": dataset,
                **record,
                "smd_from_train": difference / base["std"] if base["std"] else None,
                "psi_from_train": psi,
                "changed_from_train_constant_share": change,
            }
        )
    return pd.DataFrame(rows)
