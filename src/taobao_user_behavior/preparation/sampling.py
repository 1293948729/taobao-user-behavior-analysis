"""仅从训练负例无放回抽样，保留全部正例并计算独立权重方案。

主要接口：sample_training、class_weights、sampling_comparison。
"""

import numpy as np
import pandas as pd

from taobao_user_behavior.preparation.common import KEYS, PreparationOptions


def _ordered_training(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(KEYS + ["label"]).issubset(frame.columns):
        raise ValueError("Training requires keys and binary labels.")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate training keys.")
    if frame["cutoff"].nunique() != 1:
        raise ValueError("Sampling expects one training cutoff.")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Labels must be zero or one.")
    if frame["label"].nunique() != 2:
        raise ValueError("Sampling requires both classes.")
    return frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def sample_training(
    frame: pd.DataFrame, options: PreparationOptions
) -> tuple[pd.DataFrame, dict]:
    """以排序后的行号作可追溯键，抽取训练负例。

    Args:
        frame: 一套完整自然训练样本，不接受混合截点。
        options: 种子与负正比上限。

    Returns:
        保持原字段的训练子集，以及独立保存的抽样参数和计数。
    """
    ordered = _ordered_training(frame)
    positive = np.flatnonzero(ordered["label"].to_numpy() == 1)
    negative = np.flatnonzero(ordered["label"].to_numpy() == 0)
    requested = len(positive) * options.negative_ratio
    count = min(len(negative), requested)
    generator = np.random.Generator(np.random.PCG64(options.seed))
    chosen = generator.permutation(negative)[:count]
    positions = np.sort(np.concatenate([positive, chosen]))
    sampled = ordered.iloc[positions].reset_index(drop=True)
    return sampled, {
        "algorithm": "sorted keys + numpy PCG64 permutation without replacement",
        "seed": options.seed,
        "negative_ratio_limit": options.negative_ratio,
        "before_rows": len(ordered),
        "after_rows": len(sampled),
        "positive_rows": len(positive),
        "negative_before": len(negative),
        "negative_after": count,
        "negative_inclusion_probability": count / len(negative),
        "positive_inclusion_probability": 1.0,
        "positive_rate_before": len(positive) / len(ordered),
        "positive_rate_after": len(positive) / len(sampled),
        "retained_fraction": len(sampled) / len(ordered),
    }


def class_weights(labels: pd.Series) -> dict:
    """从完整自然训练标签计算平衡权重，不叠加到负采样方案。

    Args:
        labels: 完整训练标签。

    Returns:
        分类权重、正类倍率及基准样本量。

    Raises:
        ValueError: 标签不完整、不合法或只有一类。
    """
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("Class weights require both binary classes.")
    positive = int(labels.sum())
    negative = len(labels) - positive
    return {
        "fit_dataset": "full_natural_train",
        "rows": len(labels),
        "negative": negative,
        "positive": positive,
        "balanced_class_weight": {
            "0": len(labels) / (2 * negative),
            "1": len(labels) / (2 * positive),
        },
        "positive_class_multiplier": negative / positive,
        "combine_with_downsampling": False,
        "probability_calibration_evaluated": False,
    }


def sampling_comparison(
    original: pd.DataFrame, sampled: pd.DataFrame, names: list[str]
) -> tuple[dict, pd.DataFrame]:
    """比较覆盖和负例分布，不把有意改变的类别比例当作漂移。

    Args:
        original: 完整自然训练样本。
        sampled: 训练子集。
        names: 需要检查的全部候选特征。

    Returns:
        用户/商品/类目覆盖，以及负例特征均值标准化差异。
    """
    coverage = {}
    for column in ("user_id", "item_id", "category_id"):
        before = original[column].nunique()
        after = sampled[column].nunique()
        coverage[column] = {
            "before": int(before),
            "after": int(after),
            "coverage": float(after / before),
        }
    full = original.loc[original["label"].eq(0), names].astype("float64")
    part = sampled.loc[sampled["label"].eq(0), names].astype("float64")
    deviation = full.std(ddof=0)
    difference = part.mean() - full.mean()
    scale = deviation.replace(0, np.nan)
    smd = difference / scale
    smd.loc[deviation.eq(0) & difference.eq(0)] = 0.0
    comparison = pd.DataFrame(
        {
            "feature": names,
            "negative_mean_before": full.mean().to_numpy(),
            "negative_mean_after": part.mean().to_numpy(),
            "standardized_mean_difference": smd.to_numpy(),
            "absolute_smd": smd.abs().to_numpy(),
        }
    )
    return coverage, comparison
