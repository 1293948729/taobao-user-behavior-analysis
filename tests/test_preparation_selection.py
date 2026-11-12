"""检查筛选只使用训练特征，并保留必要的稀疏行为信息。

测试入口：本模块test_开头的函数。
"""

import numpy as np
import pandas as pd
import pytest

from taobao_user_behavior.preparation.common import PreparationOptions
from taobao_user_behavior.preparation.distribution import (
    compare_distribution,
    distribution_reference,
)
from taobao_user_behavior.preparation.selection import select_features


def test_constant_duplicate_and_rank_redundancy() -> None:
    """去除常量和重复，固定顺序保留强秩相关关系的第一个特征。"""
    frame = pd.DataFrame(
        {
            "constant": [4] * 100,
            "original": np.arange(100),
            "duplicate": np.arange(100),
            "nonlinear": np.arange(100) ** 3,
            "independent": np.random.default_rng(11).permutation(100),
        }
    )
    result = select_features(frame, PreparationOptions())
    assert result["selected"] == ["original", "independent"]
    decisions = result["decisions"].set_index("feature")
    assert decisions.loc["duplicate", "reason"] == "完全重复"
    assert decisions.loc["nonlinear", "related_feature"] == "original"


def test_behavior_totals_are_verified_before_exclusion() -> None:
    """只有真实满足四类行为之和的总量才按代数关系排除。"""
    generator = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            f"pair_{name}": generator.integers(0, 10, 500)
            for name in ("views", "favorites", "carts", "purchases")
        }
    )
    frame.insert(0, "pair_events", frame.sum(axis=1))
    result = select_features(frame, PreparationOptions())
    assert "pair_events" not in result["selected"]
    assert result["identities"].iloc[0]["maximum_absolute_residual"] == 0
    frame.loc[0, "pair_events"] += 1
    after = select_features(frame, PreparationOptions())
    assert "pair_events" in after["selected"]
    assert after["identities"].iloc[0]["maximum_absolute_residual"] == 1


def test_sparse_signal_is_flagged_but_retained() -> None:
    """近常量只作提示，不机械删除少数购买信号。"""
    frame = pd.DataFrame(
        {
            "sparse": [0] * 1999 + [1],
            "activity": np.random.default_rng(9).integers(1, 100, 2000),
        }
    )
    result = select_features(frame, PreparationOptions())
    assert "sparse" in result["selected"]
    assert result["decisions"].iloc[0]["near_constant"]


def test_high_correlation_preserves_distinct_business_signals() -> None:
    """次数、时间距离、相对占比及跨层特征不能仅凭高秩相关混同。"""
    values = np.arange(1, 101)
    frame = pd.DataFrame(
        {
            "pair_purchases": values,
            "pair_purchase_recency_hours": values**2,
            "pair_purchase_view_ratio": values / 2,
            "item_users": values**3,
        }
    )
    result = select_features(frame, PreparationOptions())
    assert result["selected"] == list(frame)
    assert result["decisions"]["reason"].eq("保留不同业务含义").sum() == 3


@pytest.mark.parametrize("field", ["label", "user_id", "item_id", "cutoff"])
def test_selection_rejects_outcome_or_identifiers(field: str) -> None:
    """禁止把答案或标识传入拟合矩阵。

    Args:
        field: 不允许用于此筛选器的字段。
    """
    with pytest.raises(ValueError, match="feature columns only"):
        select_features(pd.DataFrame({field: [0, 1]}), PreparationOptions())


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_selection_rejects_nonfinite_values(invalid: float) -> None:
    """缺失和无穷值须明确失败。

    Args:
        invalid: 非有限数值。
    """
    with pytest.raises(ValueError, match="finite"):
        select_features(pd.DataFrame({"x": [0, invalid]}), PreparationOptions())


def test_distribution_uses_frozen_training_bins() -> None:
    """描述后续漂移不修改训练参考或选择规则。"""
    train = pd.DataFrame({"activity": range(100), "flag": [0] * 100})
    reference = distribution_reference(train)
    edges = reference["activity"]["edges"].copy()
    later = pd.DataFrame({"activity": range(1000, 1100), "flag": [1] * 100})
    report = compare_distribution(later, reference, "validation").set_index("feature")
    assert report.loc["activity", "psi_from_train"] > 1
    assert report.loc["flag", "changed_from_train_constant_share"] == 1
    np.testing.assert_array_equal(reference["activity"]["edges"], edges)
