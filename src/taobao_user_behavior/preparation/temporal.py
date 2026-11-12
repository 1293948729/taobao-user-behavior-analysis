"""核对原七日窗口并统计日期覆盖、每日行为和候选正例。

主要接口：review_windows。
"""

from dataclasses import asdict

import pandas as pd

from taobao_user_behavior.features.windows import FeatureWindow, plan_windows
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import connect

DAILY = """
SELECT CAST(event_time AS DATE) AS event_date, count(*) AS events,
    count(DISTINCT user_id) AS users,
    count(*) FILTER (WHERE behavior_type = 4) AS purchase_events
FROM events GROUP BY event_date ORDER BY event_date
"""
FIRST_PURCHASE = """
SELECT CAST(first_purchase AS DATE) AS event_date, count(*) AS positive_pairs
FROM (
    SELECT events.user_id, events.item_id, min(event_time) AS first_purchase
    FROM events JOIN candidate USING (user_id, item_id)
    WHERE event_time >= ? AND event_time < ? AND behavior_type = 4
    GROUP BY events.user_id, events.item_id
) GROUP BY event_date ORDER BY event_date
"""


def _check_plan(planned: FeatureWindow, recorded: dict) -> None:
    for key, value in asdict(planned).items():
        actual = recorded[key]
        if key != "name" and actual is not None:
            actual = pd.Timestamp(actual).to_pydatetime()
        if value != actual:
            raise ValueError(f"Changed temporal plan: {planned.name}/{key}")


def review_windows(settings: Settings, manifest: dict) -> dict[str, pd.DataFrame]:
    """核验窗口与旧规划完全一致，不变更任何样本或标签。

    Args:
        settings: 标准事件与完整样本所在位置。
        manifest: 已通过来源核对的特征清单。

    Returns:
        每日、窗口汇总及工作日周末分布。

    Raises:
        ValueError: 窗口不符、标签窗重叠或每日正例与样本数不守恒。
    """
    with connect(settings) as connection:
        bounds = connection.execute(
            "SELECT min(event_time), max(event_time) FROM events"
        ).fetchone()
        if bounds is None:
            raise ValueError("Missing event bounds.")
        expected = plan_windows(*bounds)
        daily = connection.sql(DAILY).df()
        daily["event_date"] = pd.to_datetime(daily["event_date"])
        windows = []
        details = []
        for planned, recorded in zip(expected, manifest["datasets"], strict=True):
            _check_plan(planned, recorded)
            connection.read_parquet(
                str(settings.data / "features" / f"{planned.name}.parquet")
            ).create_view("candidate", replace=True)
            if planned.label_end is None:
                positives = pd.DataFrame(columns=["event_date", "positive_pairs"])
            else:
                positives = connection.execute(
                    FIRST_PURCHASE, [planned.cutoff, planned.label_end]
                ).df()
                if int(positives["positive_pairs"].sum()) != recorded["positives"]:
                    raise ValueError("Daily candidate positives do not reconcile.")
            for phase, start, end in [
                ("history", planned.history_start, planned.cutoff),
                ("label", planned.cutoff, planned.label_end),
            ]:
                if end is None:
                    continue
                dates = pd.DataFrame(
                    {"event_date": pd.date_range(start, periods=7, freq="D")}
                )
                part = dates.merge(daily, on="event_date", how="left").fillna(0)
                part["dataset"] = planned.name
                part["phase"] = phase
                part["weekday"] = part["event_date"].dt.dayofweek
                part["weekend"] = part["weekday"].ge(5)
                if phase == "label":
                    part = part.merge(positives, on="event_date", how="left")
                    part["positive_pairs"] = (
                        part["positive_pairs"].fillna(0).astype("Float64")
                    )
                else:
                    part["positive_pairs"] = pd.Series(
                        pd.NA, index=part.index, dtype="Float64"
                    )
                details.append(part)
            windows.append(
                {
                    **{
                        key: recorded[key]
                        for key in (
                            "name",
                            "history_start",
                            "cutoff",
                            "label_end",
                            "candidates",
                            "positives",
                            "negatives",
                            "positive_rate",
                            "future_pair_coverage",
                        )
                    },
                    "history_calendar_days": 7,
                    "history_weekdays": 5,
                    "history_weekend_days": 2,
                    "label_calendar_days": 7 if planned.label_end else None,
                }
            )
    labeled = expected[:-1]
    if any(
        left.label_end != right.cutoff
        for left, right in zip(labeled[:-1], labeled[1:], strict=True)
    ):
        raise ValueError("Label windows must be adjacent without overlap.")
    detail = pd.concat(details, ignore_index=True)
    by_weekend = (
        detail.groupby(["dataset", "phase", "weekend"], sort=False)
        .agg(
            calendar_days=("event_date", "size"),
            events=("events", "sum"),
            purchase_events=("purchase_events", "sum"),
            positive_pairs=("positive_pairs", lambda values: values.sum(min_count=1)),
        )
        .reset_index()
    )
    return {
        "daily_events": daily,
        "window_summary": pd.DataFrame(windows),
        "window_daily": detail,
        "window_weekend": by_weekend,
    }
