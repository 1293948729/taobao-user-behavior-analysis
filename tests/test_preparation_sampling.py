"""检查训练负采样的可复现性、正例保留、边界和权重。

测试入口：本模块test_开头的函数。
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from taobao_user_behavior.preparation.common import KEYS, PreparationOptions
from taobao_user_behavior.preparation.sampling import (
    class_weights,
    sample_training,
    sampling_comparison,
)


def _training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"{index:04}" for index in range(1000)],
            "item_id": ["0001"] * 1000,
            "category_id": ["001"] * 1000,
            "cutoff": [pd.Timestamp("2025-11-28")] * 1000,
            "signal": np.arange(1000),
            "label": [1] * 10 + [0] * 990,
        }
    )


def test_sampling_retains_positives_and_is_order_independent() -> None:
    """改变输入行序不改变抽样键，且全部正例保留。"""
    train = _training()
    before = train.copy(deep=True)
    options = PreparationOptions(negative_ratio=5)
    sample, statistics = sample_training(train, options)
    reordered, _ = sample_training(train.sample(frac=1, random_state=44), options)
    pd.testing.assert_frame_equal(sample, reordered)
    pd.testing.assert_frame_equal(train, before)
    assert len(sample) == 60
    assert sample["label"].sum() == 10
    assert not sample.duplicated(KEYS).any()
    assert statistics["negative_inclusion_probability"] == pytest.approx(50 / 990)
    assert sample["user_id"].str.len().eq(4).all()


def test_sampling_seed_changes_negative_selection() -> None:
    """不同种子改变负例集合，而正例集合保持。"""
    options = PreparationOptions(negative_ratio=5)
    first, _ = sample_training(_training(), options)
    second, _ = sample_training(_training(), replace(options, seed=4))
    assert set(first.loc[first["label"].eq(0), "user_id"]) != set(
        second.loc[second["label"].eq(0), "user_id"]
    )
    assert set(first.loc[first["label"].eq(1), "user_id"]) == set(
        second.loc[second["label"].eq(1), "user_id"]
    )


def test_sampling_caps_at_available_negatives_and_comparison() -> None:
    """请求数量超过负例总数时全保留，分布对比应为零。"""
    train = _training()
    sample, statistics = sample_training(train, PreparationOptions())
    assert statistics["negative_after"] == 990
    coverage, differences = sampling_comparison(train, sample, ["signal"])
    assert coverage["user_id"]["coverage"] == 1
    assert differences["absolute_smd"].max() == 0


@pytest.mark.parametrize("problem", ["duplicate", "single", "bad_label", "cutoff"])
def test_sampling_rejects_invalid_training(problem: str) -> None:
    """拒绝重复主键、单类、非法标签和混合截点。

    Args:
        problem: 本次制造的输入错误。
    """
    frame = _training()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif problem == "single":
        frame["label"] = 0
    elif problem == "bad_label":
        frame.loc[0, "label"] = 2
    else:
        frame.loc[0, "cutoff"] = pd.Timestamp("2025-12-05")
    with pytest.raises(ValueError):
        sample_training(frame, PreparationOptions())


def test_balanced_weights_reconcile_and_are_separate() -> None:
    """两类加权总量相同，权重只对应完整自然训练。"""
    result = class_weights(_training()["label"])
    weights = result["balanced_class_weight"]
    assert 10 * weights["1"] == pytest.approx(990 * weights["0"])
    assert weights["1"] == 50
    assert result["positive_class_multiplier"] == 99
    assert not result["combine_with_downsampling"]


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": -1},
        {"seed": True},
        {"negative_ratio": 0},
        {"negative_ratio": 2.5},
        {"correlation_limit": 0},
        {"correlation_limit": float("nan")},
        {"near_constant_share": 1.1},
    ],
)
def test_options_validate_ranges(changes: dict) -> None:
    """错误配置必须立即失败。

    Args:
        changes: 要替换的参数。
    """
    with pytest.raises(ValueError):
        PreparationOptions(**changes)
