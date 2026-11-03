"""直接从标准事件独立复算EDA指标，检查图表与汇总表的来源。

主要入口：main。不调用分析聚合函数，不修改正式数据。
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from PIL import Image

from taobao_user_behavior.settings import Settings, load_runtime_settings
from taobao_user_behavior.storage import connect, file_hash, read_json, save_json


def _assert_close(actual: float, expected: float, label: str) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-10):
        raise ValueError(f"{label}: {actual} != {expected}")


def _read(settings: Settings, name: str) -> pd.DataFrame:
    return pd.read_csv(
        settings.outputs / f"eda/tables/{name}.csv",
        dtype={"user_id": str, "item_id": str, "category_id": str},
    )


def _overview(connection: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    row = connection.execute(
        """
        SELECT count(*), count(DISTINCT user_id), count(DISTINCT item_id),
            count(DISTINCT category_id) FILTER (WHERE NOT category_missing),
            count(*) FILTER (WHERE behavior_type = 4),
            count(DISTINCT user_id) FILTER (WHERE behavior_type = 4)
        FROM events
        """,
    ).fetchone()
    if row is None:
        raise ValueError("Independent overview returned no data.")
    fields = ["events", "users", "items", "categories", "purchases", "buyers"]
    result = dict(zip(fields, row, strict=True))
    for name, actual in result.items():
        _assert_close(actual, summary["overview"][name], name)
    return result


def _daily(connection: duckdb.DuckDBPyConnection, settings: Settings) -> dict:
    events = connection.execute(
        """
        SELECT CAST(event_time AS DATE) AS day, behavior_type, count(*) AS events
        FROM events GROUP BY 1, 2
        """,
    ).df()
    expected = _read(settings, "daily")
    events["day"] = events["day"].astype(str)
    indexed = expected.set_index(["event_date", "behavior_type"])["events"]
    for row in events.itertuples():
        if indexed.loc[(row.day, row.behavior_type)] != row.events:
            raise ValueError(f"Daily mismatch: {row.day}, {row.behavior_type}")
    if events["events"].sum() != expected["events"].sum():
        raise ValueError("Daily total mismatch.")
    return {"direct_groups": len(events), "calendar_rows": len(expected)}


def _retention(connection: duckdb.DuckDBPyConnection, settings: Settings) -> dict:
    observations = connection.execute(
        "SELECT DISTINCT user_id, CAST(event_time AS DATE) AS day FROM events"
    ).df()
    first = observations.groupby("user_id")["day"].min()
    end_time = connection.execute("SELECT max(event_time) FROM events").fetchone()
    if end_time is None:
        raise ValueError("No end time.")
    last_complete = (pd.Timestamp(end_time[0]) + pd.Timedelta(hours=1)).floor("D")
    last_complete -= pd.Timedelta(days=1)
    grouped = observations.groupby("day")["user_id"].agg(set).to_dict()
    expected = _read(settings, "retention")
    checked = 0
    for cohort, users in first.groupby(first):
        identities = set(users.index)
        for lag in [0, 1, 2, 3, 7, 14]:
            target = cohort + pd.Timedelta(days=lag)
            eligible = len(identities) if target <= last_complete else 0
            returned = len(identities & grouped.get(target, set())) if eligible else 0
            selected = expected.loc[
                expected["first_date"].eq(str(cohort.date())) & expected["lag"].eq(lag)
            ].iloc[0]
            if selected["eligible"] != eligible or selected["returned"] != returned:
                raise ValueError(f"Retention mismatch: {cohort}, D{lag}")
            checked += 1
    return {"cohort_offset_cells": checked}


def _labels(connection: duckdb.DuckDBPyConnection, settings: Settings) -> dict:
    window = _read(settings, "label_window").iloc[0]
    start, end = str(window["label_start"]), str(window["label_end"])
    direct = connection.execute(
        """
        WITH historical AS (
            SELECT DISTINCT user_id, item_id FROM events WHERE event_time < ?
        ), future AS (
            SELECT DISTINCT user_id, item_id FROM events
            WHERE event_time >= ? AND event_time < ? AND behavior_type = 4
        )
        SELECT count(*) AS candidates,
            count(future.user_id) AS positives
        FROM historical LEFT JOIN future USING (user_id, item_id)
        """,
        [start, start, end],
    ).fetchone()
    if direct is None:
        raise ValueError("Independent labels returned no result.")
    if direct[0] != window["candidates"] or direct[1] != window["positives"]:
        raise ValueError("Independent candidate labels disagree.")
    return {"candidates": direct[0], "positives": direct[1], "start": start, "end": end}


def _rankings(connection: duckdb.DuckDBPyConnection, settings: Settings) -> dict:
    checked = {}
    for entity in ["item", "category"]:
        frames = [
            _read(settings, f"{entity}_flow"),
            _read(settings, f"{entity}_conversion"),
        ]
        field = f"{entity}_id"
        expected = pd.concat(frames).drop_duplicates(field).set_index(field)
        identifiers = expected.index.tolist()
        if entity == "item":
            result = connection.execute(
                """
                SELECT item_id AS identity, user_id,
                    count(*) AS events,
                    count(*) FILTER (WHERE behavior_type = 1) AS views,
                    count(*) FILTER (WHERE behavior_type = 4) AS buys
                FROM events WHERE item_id IN (SELECT unnest(?)) GROUP BY 1, 2
                """,
                [identifiers],
            ).df()
        else:
            result = connection.execute(
                """
                SELECT category_id AS identity, user_id, item_id,
                    count(*) AS events,
                    count(*) FILTER (WHERE behavior_type = 1) AS views,
                    count(*) FILTER (WHERE behavior_type = 4) AS buys
                FROM events WHERE category_id IN (SELECT unnest(?)) GROUP BY 1, 2, 3
                """,
                [identifiers],
            ).df()
        for identity, rows in result.groupby("identity"):
            original = expected.loc[identity]
            _assert_close(rows["events"].sum(), original["events"], "ranking events")
            viewed = rows["views"].gt(0)
            bought = viewed & rows["buys"].gt(0)
            _assert_close(viewed.sum(), original["viewed_pairs"], "ranking denominator")
            _assert_close(
                bought.sum(), original["bought_viewed_pairs"], "ranking numerator"
            )
        checked[entity] = len(expected)
    return checked


def _artifacts(settings: Settings, summary: dict) -> dict:
    root = settings.outputs / "eda"
    for name, digest in summary["table_sha256"].items():
        if file_hash(root / "tables" / name) != digest:
            raise ValueError(f"Changed summary table: {name}")
    manifest = read_json(root / "chart_manifest.json")
    for key, digest in manifest["figures"].items():
        path = root / "figures" / f"{key}.png"
        if file_hash(path) != digest:
            raise ValueError(f"Changed chart: {key}")
        with Image.open(path) as image:
            image.verify()
    return {
        "charts": len(manifest["figures"]),
        "tables": len(summary["table_sha256"]),
    }


def main() -> None:
    """直接复算核心指标与来源，保存独立核验结果。

    Raises:
        ValueError: 聚合、留存、标签、排名或来源指纹不一致。
    """
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    settings = load_runtime_settings(project, parser.parse_args().run_root)
    summary = read_json(settings.outputs / "eda/summary.json")
    with connect(settings) as connection:
        checks = {
            "overview": _overview(connection, summary),
            "daily": _daily(connection, settings),
            "retention": _retention(connection, settings),
            "labels": _labels(connection, settings),
            "rankings": _rankings(connection, settings),
        }
    checks["artifacts"] = _artifacts(settings, summary)
    save_json(
        settings.outputs / "eda/independent_verification.json",
        {"passed": True, "checks": checks},
    )
    print(checks)


if __name__ == "__main__":
    main()
