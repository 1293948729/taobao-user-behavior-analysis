"""仅从自然训练数据筛选常量、重复、总量冗余和高相关特征。

主要接口：feature_profile、select_features。
"""

import hashlib

import numpy as np
import pandas as pd

from taobao_user_behavior.preparation.common import PreparationOptions

COMPONENTS = ("views", "favorites", "carts", "purchases")
TOTAL_PREFIXES = ("pair", "user", "item", "category")
DISTINCT_SIGNALS = ("_recency_hours", "_share", "_ratio", "_active_days")


def _comparable(left: str, right: str) -> bool:
    if left.endswith(DISTINCT_SIGNALS) or right.endswith(DISTINCT_SIGNALS):
        return False
    left_entity, right_entity = left.split("_")[0], right.split("_")[0]
    known = set(TOTAL_PREFIXES)
    return not ({left_entity, right_entity} & known) or left_entity == right_entity


def _correlation_decision(
    decision: dict, kept: list[str], correlation: pd.DataFrame, threshold: float
) -> None:
    name = decision["feature"]
    related = [
        other for other in kept if abs(correlation.loc[name, other]) >= threshold
    ]
    comparable = next((other for other in related if _comparable(name, other)), None)
    if comparable is not None:
        decision.update(
            selected=False,
            reason="与同层保留特征高度秩相关",
            related_feature=comparable,
            abs_spearman=float(abs(correlation.loc[name, comparable])),
        )
    else:
        kept.append(name)
        if related:
            other = related[0]
            decision.update(
                reason="保留不同业务含义",
                related_feature=other,
                abs_spearman=float(abs(correlation.loc[name, other])),
            )


def feature_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """统计数值列的分布和近常量诊断所需信息。

    Args:
        frame: 只包含候选特征的数据。

    Returns:
        每个字段一行的描述统计。

    Raises:
        ValueError: 输入为空、字段重复、含缺失或非有限值。
    """
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Feature matrix must be nonempty with unique columns.")
    values = frame.astype("float64")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Features must be finite and nonmissing.")
    summary = values.describe(percentiles=[0.5, 0.9, 0.99]).T
    summary = summary.rename(columns={"50%": "p50", "90%": "p90", "99%": "p99"}).drop(
        columns="count"
    )
    summary["unique_values"] = values.nunique()
    summary["mode_share"] = [
        float(values[column].value_counts(normalize=True).iloc[0]) for column in values
    ]
    summary["feature"] = summary.index
    return summary.reset_index(drop=True)


def _base_decisions(
    values: pd.DataFrame, profile: pd.DataFrame, options: PreparationOptions
) -> tuple[list[dict], list[dict]]:
    decisions = []
    identities = []
    hashes: dict[str, str] = {}
    statistics = profile.set_index("feature")
    for name in values:
        row = statistics.loc[name]
        decision = {
            "feature": name,
            "selected": True,
            "reason": "保留",
            "related_feature": "",
            "abs_spearman": None,
            "near_constant": bool(row["mode_share"] >= options.near_constant_share),
        }
        if row["unique_values"] == 1:
            decision.update(selected=False, reason="训练集常量")
        signature = hashlib.sha256(values[name].to_numpy().tobytes()).hexdigest()
        previous = hashes.get(signature)
        if decision["selected"] and previous is not None:
            if np.array_equal(values[name].to_numpy(), values[previous].to_numpy()):
                decision.update(
                    selected=False, reason="完全重复", related_feature=previous
                )
        if decision["selected"]:
            hashes[signature] = name
        if name in [f"{prefix}_events" for prefix in TOTAL_PREFIXES]:
            prefix = name.removesuffix("_events")
            members = [f"{prefix}_{suffix}" for suffix in COMPONENTS]
            if set(members).issubset(values.columns):
                residual = float(
                    (values[name] - values[members].sum(axis=1)).abs().max()
                )
                identities.append(
                    {
                        "total": name,
                        "components": " + ".join(members),
                        "maximum_absolute_residual": residual,
                    }
                )
                if residual == 0 and all(
                    statistics.loc[field, "unique_values"] > 1 for field in members
                ):
                    decision.update(
                        selected=False,
                        reason="等于四类行为之和",
                        related_feature=" + ".join(members),
                    )
        decisions.append(decision)
    return decisions, identities


def select_features(frame: pd.DataFrame, options: PreparationOptions) -> dict:
    """按固定顺序筛选，不接收标签或其他集合进行拟合。

    Args:
        frame: 完整自然训练集的候选特征。
        options: 筛选阈值。

    Returns:
        决策、描述统计、Pearson/Spearman矩阵与代数核验结果。

    Raises:
        ValueError: 输入包含标签、标识或非有限数值，或没有可保留特征。
    """
    forbidden = {"label", "user_id", "item_id", "category_id", "cutoff"}
    if forbidden.intersection(frame.columns):
        raise ValueError("Selection accepts feature columns only.")
    profile = feature_profile(frame)
    values = frame.astype("float64")
    pearson = values.corr()
    spearman = values.rank(method="average").corr()
    decisions, identities = _base_decisions(values, profile, options)
    kept: list[str] = []
    for decision in decisions:
        if not decision["selected"]:
            continue
        _correlation_decision(decision, kept, spearman, options.correlation_limit)
    if not kept:
        raise ValueError("No informative features remain.")
    return {
        "selected": kept,
        "decisions": pd.DataFrame(decisions),
        "profile": profile,
        "pearson": pearson,
        "spearman": spearman,
        "identities": pd.DataFrame(identities),
    }
