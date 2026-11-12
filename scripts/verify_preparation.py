"""独立检查筛选投影、训练抽样、权重、窗口及来源守恒。

主要入口：main。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from taobao_user_behavior.features.dictionary import field_dictionary
from taobao_user_behavior.settings import Settings, load_runtime_settings
from taobao_user_behavior.source_inventory import source_fingerprints
from taobao_user_behavior.storage import connect, file_hash, read_json, save_json

KEYS = ["user_id", "item_id", "cutoff"]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _selection_check(train: pd.DataFrame, report: dict, folder: Path) -> dict:
    chosen = report["selection"]["selected_features"]
    names = [row["name"] for row in field_dictionary() if row["role"] == "feature"]
    _check(set(chosen).issubset(names), "Unknown selected feature")
    _check(len(set(chosen)) == len(chosen), "Duplicate selected feature")
    decisions = pd.read_csv(folder / "feature_decisions.csv").fillna("")
    _check(decisions["feature"].tolist() == names, "Decision order differs")
    _check(
        decisions.loc[decisions["selected"], "feature"].tolist() == chosen,
        "Decisions and selected fields differ",
    )
    _check(not report["selection"]["labels_used_for_selection"], "Labels used")
    correlation = train[names].astype(float).rank(method="average").corr()
    threshold = report["selection"]["parameters"]["correlation_limit"]
    for row in decisions.to_dict("records"):
        feature = row["feature"]
        reason = row["reason"]
        other = row["related_feature"]
        if reason == "训练集常量":
            _check(train[feature].nunique() == 1, f"Not constant: {feature}")
        elif reason == "完全重复":
            _check(
                np.array_equal(train[feature].to_numpy(), train[other].to_numpy()),
                f"Not duplicate: {feature}",
            )
        elif reason == "等于四类行为之和":
            parts = other.split(" + ")
            _check(
                np.array_equal(train[feature].to_numpy(), train[parts].sum(axis=1)),
                f"Not an identity: {feature}",
            )
        elif reason == "与同层保留特征高度秩相关":
            _check(other in chosen, "Correlation representative was removed")
            _check(
                abs(correlation.loc[feature, other]) >= threshold,
                f"Correlation threshold not reached: {feature}",
            )
        else:
            _check(
                reason in {"保留", "保留不同业务含义"} and row["selected"],
                "Unknown decision",
            )
    for index, name in enumerate(chosen):
        _check(train[name].nunique() > 1, "Constant was retained")
        for other in chosen[:index]:
            protected = ("_recency_hours", "_share", "_ratio", "_active_days")
            same_level = name.split("_")[0] == other.split("_")[0]
            if (
                same_level
                and not name.endswith(protected)
                and not other.endswith(protected)
            ):
                _check(
                    abs(correlation.loc[name, other]) < threshold,
                    "Retained comparable features exceed threshold",
                )
    return {"input_features": len(names), "selected_features": len(chosen)}


def _projection_check(settings: Settings, report: dict) -> list[dict]:
    checked = []
    selected = report["selection"]["selected_features"]
    for row in report["projected"]:
        name = row["dataset"]
        original = pd.read_parquet(settings.data / "features" / f"{name}.parquet")
        output = pd.read_parquet(settings.root / row["file"])
        expected_columns = ["user_id", "item_id", "category_id", "cutoff", *selected]
        if name != "latest":
            expected_columns.append("label")
        _check(row["columns"] == expected_columns, "Projection columns differ")
        pd.testing.assert_frame_equal(original[expected_columns], output)
        _check(not output.duplicated(KEYS).any(), "Projection contains duplicate keys")
        _check(
            file_hash(settings.data / "features" / f"{name}.parquet")
            == report["input_sha256"][name],
            "Original feature data changed",
        )
        checked.append(
            {
                "dataset": name,
                "rows": len(output),
                "all_projected_cells_equal": True,
            }
        )
    return checked


def _sampling_check(train: pd.DataFrame, settings: Settings, report: dict) -> dict:
    ordered = train.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    label = ordered["label"].to_numpy()
    positive = np.where(label == 1)[0]
    negative = np.where(label == 0)[0]
    parameters = report["selection"]["parameters"]
    generator = np.random.Generator(np.random.PCG64(parameters["seed"]))
    permutation = generator.permutation(negative)
    amount = min(len(negative), len(positive) * parameters["negative_ratio"])
    indices = np.sort(np.r_[positive, permutation[:amount]])
    actual = pd.read_parquet(settings.data / "prepared/train_sampled.parquet")
    expected = ordered.iloc[indices][actual.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    _check(int(actual["label"].sum()) == len(positive), "Lost positive examples")
    weights = report["weights"]
    _check(weights["rows"] == len(train), "Weights used wrong dataset")
    expected_weights = [
        len(train) / (2 * len(negative)),
        len(train) / (2 * len(positive)),
    ]
    np.testing.assert_allclose(
        [weights["balanced_class_weight"]["0"], weights["balanced_class_weight"]["1"]],
        expected_weights,
        rtol=0,
        atol=1e-12,
    )
    _check(not weights["combine_with_downsampling"], "Natural weights double applied")
    return {
        "rows": len(actual),
        "positives": len(positive),
        "negatives": amount,
        "all_sampled_cells_equal": True,
        "sampling_independently_reproduced": True,
        "weights_checked": True,
    }


def _temporal_check(settings: Settings, folder: Path) -> dict:
    windows = pd.read_csv(folder / "window_summary.csv")
    daily = pd.read_csv(folder / "daily_events.csv", parse_dates=["event_date"])
    phases = pd.read_csv(folder / "window_daily.csv")
    with connect(settings) as connection:
        expected = connection.execute(
            """
            SELECT CAST(event_time AS DATE) AS event_date, count(*) AS events,
                count(DISTINCT user_id) AS users,
                sum(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS purchase_events
            FROM events GROUP BY 1 ORDER BY 1
            """,
        ).df()
    pd.testing.assert_frame_equal(daily, expected, check_dtype=False)
    for row in windows.to_dict("records"):
        start = pd.Timestamp(row["history_start"])
        cutoff = pd.Timestamp(row["cutoff"])
        _check((cutoff - start).days == 7, "History must remain seven days")
        days = pd.date_range(start, periods=7)
        weekend_days = sum(day.weekday() >= 5 for day in days)
        _check(weekend_days == 2, "Incorrect weekend coverage")
        if row["name"] != "latest":
            end = pd.Timestamp(row["label_end"])
            _check((end - cutoff).days == 7, "Label must remain seven days")
            daily_positive = phases.loc[
                phases["dataset"].eq(row["name"]) & phases["phase"].eq("label"),
                "positive_pairs",
            ].sum()
            _check(daily_positive == row["positives"], "Daily positive count differs")
    _check(
        phases.groupby(["dataset", "phase"]).size().eq(7).all(), "Incomplete calendar"
    )
    return {"daily_rows": len(daily), "window_phase_days": len(phases)}


def main() -> None:
    """独立核验当前准备结果，失败时不输出通过记录。"""
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    settings = load_runtime_settings(project, parser.parse_args().run_root)
    folder = settings.outputs / "preparation"
    target = folder / "verification.json"
    target.unlink(missing_ok=True)
    report = read_json(folder / "manifest.json")
    _check(
        report["source_sha256"] == source_fingerprints(project), "Stale source manifest"
    )
    _check(
        file_hash(project / "configs/project.toml") == report["configuration_sha256"],
        "Changed preparation configuration",
    )
    for relative, expected in report["output_sha256"].items():
        _check(file_hash(settings.root / relative) == expected, f"Changed {relative}")
    train = pd.read_parquet(settings.data / "features/train.parquet")
    document = {
        "passed": True,
        "selection": _selection_check(train, report, folder),
        "sampling": _sampling_check(train, settings, report),
        "projection": _projection_check(settings, report),
        "temporal": _temporal_check(settings, folder),
        "models_trained": False,
    }
    save_json(target, document)
    print(document)


if __name__ == "__main__":
    main()
