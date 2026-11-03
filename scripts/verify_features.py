"""从标准事件独立复算候选、标签及固定位置样本的全部特征。

主要入口：main。不调用生产特征聚合、窗口推导或字段计算函数。
"""

import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from taobao_user_behavior.settings import Settings, load_runtime_settings
from taobao_user_behavior.source_inventory import source_fingerprints
from taobao_user_behavior.storage import connect, file_hash, read_json, save_json


def _global_checks(
    connection: duckdb.DuckDBPyConnection, dataset: dict
) -> dict[str, int]:
    connection.execute(
        """
        CREATE TEMP TABLE observed AS SELECT DISTINCT user_id, item_id
        FROM events WHERE event_time >= ? AND event_time < ?
        """,
        [dataset["history_start"], dataset["cutoff"]],
    )
    differences = connection.execute(
        """
        SELECT count(*) FROM (
            (SELECT user_id, item_id FROM observed
             EXCEPT SELECT user_id, item_id FROM exported)
            UNION ALL
            (SELECT user_id, item_id FROM exported
             EXCEPT SELECT user_id, item_id FROM observed)
        )
        """,
    ).fetchone()
    counts = connection.execute(
        """
        SELECT count(*), count(DISTINCT (user_id, item_id, cutoff)),
            count(*) FILTER (WHERE cutoff != ?::TIMESTAMP) FROM exported
        """,
        [dataset["cutoff"]],
    ).fetchone()
    if differences is None or counts is None:
        raise ValueError("Candidate verification returned no counts.")
    if differences[0] or counts[0] != counts[1] or counts[2]:
        raise ValueError("Exported candidates, keys or cutoff disagree.")
    if counts[0] != dataset["candidates"]:
        raise ValueError("Manifest candidate count disagrees.")
    result = {"rows": int(counts[0]), "candidate_differences": 0, "label_errors": 0}
    if dataset["label_end"] is None:
        if "label" in connection.table("exported").columns:
            raise ValueError("Latest samples contain an unobserved label.")
        return result
    connection.execute(
        """
        CREATE TEMP TABLE purchased AS SELECT DISTINCT user_id, item_id
        FROM events
        WHERE event_time >= ? AND event_time < ? AND behavior_type = 4
        """,
        [dataset["cutoff"], dataset["label_end"]],
    )
    labels = connection.execute(
        """
        SELECT count(*) FILTER (WHERE
            label != CASE WHEN purchased.user_id IS NULL THEN 0 ELSE 1 END),
            sum(label) FROM exported LEFT JOIN purchased USING (user_id, item_id)
        """,
    ).fetchone()
    future = connection.execute("SELECT count(*) FROM purchased").fetchone()
    if labels is None or future is None:
        raise ValueError("Label verification returned no counts.")
    if labels[0] or labels[1] != dataset["positives"]:
        raise ValueError("Full-table labels disagree.")
    if future[0] != dataset["future_purchase_pairs"]:
        raise ValueError("Future-pair coverage disagrees.")
    result["positives"] = int(labels[1])
    result["future_purchase_pairs"] = int(future[0])
    return result


def _select_samples(
    connection: duckdb.DuckDBPyConnection, dataset: dict
) -> pd.DataFrame:
    positions = np.unique(np.linspace(1, dataset["candidates"], 64, dtype=int))
    chosen = connection.execute(
        """
        SELECT * EXCLUDE (position) FROM (
            SELECT *, row_number() OVER (ORDER BY user_id, item_id) AS position
            FROM exported
        ) WHERE position IN (SELECT unnest(?)) ORDER BY user_id, item_id
        """,
        [positions.tolist()],
    ).df()
    if dataset["label_end"] is not None:
        positives = connection.execute(
            """
            SELECT * FROM exported WHERE label = 1
            ORDER BY user_id, item_id LIMIT 8
            """,
        ).df()
        chosen = pd.concat([chosen, positives]).drop_duplicates(["user_id", "item_id"])
    return chosen.reset_index(drop=True)


def _base_values(frame: pd.DataFrame, prefix: str, cutoff: pd.Timestamp) -> dict:
    counts = frame["behavior_type"].value_counts()
    values = {
        f"{prefix}_events": len(frame),
        f"{prefix}_views": int(counts.get(1, 0)),
        f"{prefix}_favorites": int(counts.get(2, 0)),
        f"{prefix}_carts": int(counts.get(3, 0)),
        f"{prefix}_purchases": int(counts.get(4, 0)),
    }
    if prefix != "category":
        values.update(
            {
                f"{prefix}_active_days": frame["event_time"].dt.normalize().nunique(),
                f"{prefix}_events_1d": int(
                    frame["event_time"].ge(cutoff - pd.Timedelta(days=1)).sum()
                ),
                f"{prefix}_events_3d": int(
                    frame["event_time"].ge(cutoff - pd.Timedelta(days=3)).sum()
                ),
            }
        )
    if prefix in {"pair", "user"}:
        values[f"{prefix}_recency_hours"] = int(
            (cutoff - frame["event_time"].max()).total_seconds() / 3600
        )
    if prefix in {"item", "category"}:
        values[f"{prefix}_users"] = frame["user_id"].nunique()
    if prefix in {"user", "category"}:
        values[f"{prefix}_items"] = frame["item_id"].nunique()
    if prefix == "user":
        values["user_categories"] = frame.loc[
            ~frame["category_missing"], "category_id"
        ].nunique()
    if prefix == "pair":
        bought = frame.loc[frame["behavior_type"].eq(4), "event_time"]
        values["pair_purchase_observed"] = not bought.empty
        values["pair_purchase_recency_hours"] = (
            int((cutoff - bought.max()).total_seconds() / 3600)
            if not bought.empty
            else -1
        )
    return values


def _row_values(row: pd.Series, groups: dict, dataset: dict) -> dict:
    cutoff = pd.Timestamp(dataset["cutoff"])
    pair = groups["pair"].get_group((row["user_id"], row["item_id"]))
    recent = pair.sort_values(
        ["event_time", "category_id", "category_missing"],
        ascending=[False, True, True],
    ).iloc[0]
    category_key = (recent["category_id"], bool(recent["category_missing"]))
    entities = {
        "pair": pair,
        "user": groups["user"].get_group(row["user_id"]),
        "item": groups["item"].get_group(row["item_id"]),
        "category": groups["category"].get_group(category_key),
    }
    values = {
        "user_id": row["user_id"],
        "item_id": row["item_id"],
        "category_id": recent["category_id"],
        "cutoff": cutoff,
    }
    for prefix, frame in entities.items():
        values.update(_base_values(frame, prefix, cutoff))
    user = entities["user"]
    preference = int(
        (
            user["category_id"].eq(recent["category_id"])
            & user["category_missing"].eq(recent["category_missing"])
        ).sum()
    )
    values.update(
        {
            "user_category_events": preference,
            "user_category_share": preference / len(user),
            "pair_user_share": len(pair) / len(user),
            "pair_item_share": len(pair) / len(entities["item"]),
            "pair_no_views": values["pair_views"] == 0,
            "cutoff_weekday": cutoff.weekday(),
            "category_missing": bool(recent["category_missing"]),
        }
    )
    for action, target in [
        ("carts", "cart"),
        ("favorites", "favorite"),
        ("purchases", "purchase"),
    ]:
        values[f"pair_{target}_view_ratio"] = (
            values[f"pair_{action}"] / values["pair_views"]
            if values["pair_views"]
            else 0.0
        )
    return values


def _compare(row: pd.Series, expected: dict) -> None:
    for name, value in expected.items():
        actual = row[name]
        if isinstance(value, (str, bool, datetime, pd.Timestamp)):
            equal = actual == value
        else:
            equal = np.isclose(actual, value, rtol=1e-12, atol=1e-12)
        if not equal:
            raise ValueError(
                f"Feature mismatch {row['user_id']}/{row['item_id']}/{name}: "
                + f"{actual} != {value}"
            )


def _verify_dataset(settings: Settings, dataset: dict) -> dict:
    path = settings.root / dataset["file"]
    if file_hash(path) != dataset["sha256"]:
        raise ValueError(f"Changed feature file: {path}")
    with connect(settings) as connection:
        connection.read_parquet(str(path)).create_view("exported")
        checks = _global_checks(connection, dataset)
        chosen = _select_samples(connection, dataset)
        connection.register("sampled_keys", chosen)
        history = connection.execute(
            """
            SELECT events.* FROM events
            WHERE event_time >= ? AND event_time < ? AND (
                user_id IN (SELECT user_id FROM sampled_keys)
                OR item_id IN (SELECT item_id FROM sampled_keys)
                OR (category_id, category_missing) IN (
                    SELECT category_id, category_missing FROM sampled_keys
                )
            )
            """,
            [dataset["history_start"], dataset["cutoff"]],
        ).df()
    groups = {
        "pair": history.groupby(["user_id", "item_id"], sort=False),
        "user": history.groupby("user_id", sort=False),
        "item": history.groupby("item_id", sort=False),
        "category": history.groupby(["category_id", "category_missing"], sort=False),
    }
    for _, row in chosen.iterrows():
        expected = _row_values(row, groups, dataset)
        if set(expected) != set(chosen.columns) - {"label"}:
            raise ValueError("Independent feature names disagree with schema.")
        _compare(row, expected)
    return {
        **checks,
        "sampled_rows": len(chosen),
        "recomputed_features_per_sample": 48,
        "sample_rule": "64 evenly spaced sorted positions plus first 8 positive keys",
        "sha256": file_hash(path),
    }


def main() -> None:
    """核对当前来源并独立验证四套正式特征与标签，保存证据。

    Raises:
        ValueError: 来源、候选、标签、字段或抽查数值不一致。
    """
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    settings = load_runtime_settings(project, parser.parse_args().run_root)
    document = read_json(settings.outputs / "features/manifest.json")
    sources = source_fingerprints(project)
    current = {name: sources.get(name) for name in document["source_sha256"]}
    if current != document["source_sha256"]:
        raise ValueError("Feature source fingerprints changed.")
    if file_hash(settings.data / "events.parquet") != document["events_sha256"]:
        raise ValueError("Retained events changed.")
    results = {
        dataset["name"]: _verify_dataset(settings, dataset)
        for dataset in document["datasets"]
    }
    save_json(
        settings.outputs / "features/independent_verification.json",
        {"passed": True, "datasets": results},
    )
    print(results)


if __name__ == "__main__":
    main()
