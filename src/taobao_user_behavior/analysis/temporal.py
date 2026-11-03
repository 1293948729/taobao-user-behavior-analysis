"""聚合日趋势、星期小时分布及完整观察期留存。

主要接口：temporal_tables。
"""

import duckdb
import pandas as pd


def temporal_tables(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """计算时序汇总表，不修改事件记录。

    Args:
        connection: 含标准事件与基础聚合的分析连接。

    Returns:
        日趋势、小时分布、留存和小时频次诊断表。
    """
    daily = connection.execute(
        """
        WITH bounds AS (
            SELECT min(event_date) AS start_date, max(event_date) AS end_date
            FROM daily_behavior
        ), calendar AS (
            SELECT CAST(unnest(generate_series(start_date, end_date,
                INTERVAL 1 DAY)) AS DATE) AS event_date FROM bounds
        )
        SELECT event_date, behavior_type, coalesce(events, 0) AS events,
            coalesce(users, 0) AS users
        FROM calendar CROSS JOIN (VALUES (1), (2), (3), (4)) AS codes(behavior_type)
        LEFT JOIN daily_behavior USING (event_date, behavior_type)
        ORDER BY event_date, behavior_type
        """,
    ).df()
    hourly = connection.execute(
        """
        WITH bounds AS (
            SELECT min(event_date) AS first_day, max(event_date) AS last_day
            FROM daily_behavior
        ), calendar AS (
            SELECT unnest(generate_series(first_day, last_day, INTERVAL 1 DAY))
                AS event_date FROM bounds
        ), day_counts AS (
            SELECT isodow(event_date) AS weekday, count(*) AS days
            FROM calendar GROUP BY 1
        )
        SELECT isodow(event_time) AS weekday, hour(event_time) AS hour,
            behavior_type, count(*) AS events, count(*) * 1.0 / days AS per_day
        FROM events JOIN day_counts ON isodow(event_time) = weekday
        GROUP BY 1, 2, 3, days ORDER BY 1, 2, 3
        """,
    ).df()
    return {
        "daily": daily,
        "hourly": hourly,
        "retention": _retention(connection),
        "hourly_diagnostic": _hourly_diagnostic(connection),
        "transitions": _transitions(connection),
        "transition_gaps": connection.execute(
            """
            WITH adjacent AS (
                SELECT date_diff('hour', event_time,
                    lead(event_time) OVER (
                        PARTITION BY user_id ORDER BY event_time)) AS gap
                FROM user_hours
            )
            SELECT count(gap) AS transitions, median(gap) AS median_gap_hours,
                quantile_cont(gap, 0.9) AS p90_gap_hours,
                count(*) FILTER (WHERE gap > 24) AS gaps_over_24h
            FROM adjacent
            """,
        ).df(),
    }


def _retention(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH bounds AS (
            SELECT CAST(date_trunc('day', max(event_time) + INTERVAL 1 HOUR)
                AS DATE) - 1 AS end_date FROM events
        ), cohorts AS (
            SELECT user_id, first_date FROM user_summary
        ), offsets AS (
            SELECT unnest([0, 1, 2, 3, 7, 14]) AS lag
        )
        SELECT first_date, lag, count(*) AS cohort_users,
            count(*) FILTER (
                WHERE first_date + CAST(lag AS INTEGER) <= end_date) AS eligible,
            count(*) FILTER (WHERE user_days.user_id IS NOT NULL
                AND first_date + CAST(lag AS INTEGER) <= end_date) AS returned,
            count(*) FILTER (WHERE user_days.user_id IS NOT NULL
                AND first_date + CAST(lag AS INTEGER) <= end_date) * 1.0
                / nullif(count(*) FILTER (
                    WHERE first_date + CAST(lag AS INTEGER) <= end_date), 0)
                AS retention
        FROM cohorts CROSS JOIN offsets CROSS JOIN bounds
        LEFT JOIN user_days ON cohorts.user_id = user_days.user_id
            AND event_date = first_date + CAST(lag AS INTEGER)
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
    ).df()


def _hourly_diagnostic(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH pair_hours AS (
            SELECT user_id, item_id, event_time, behavior_type, count(*) AS events
            FROM events GROUP BY 1, 2, 3, 4
        )
        SELECT 'user_item_behavior_hour' AS scope, count(*) AS groups,
            max(events) AS maximum, count(*) FILTER (WHERE events > 100) AS over100,
            coalesce(sum(events) FILTER (WHERE events > 100), 0) AS rows_over100,
            count(DISTINCT user_id) FILTER (WHERE events > 100) AS users_over100
        FROM pair_hours
        UNION ALL
        SELECT 'user_hour', count(*), max(events),
            count(*) FILTER (WHERE events > 100),
            coalesce(sum(events) FILTER (WHERE events > 100), 0),
            count(DISTINCT user_id) FILTER (WHERE events > 100)
        FROM user_hours
        """,
    ).df()


def _transitions(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # 同小时内没有顺序；一次跨小时关系的权重在两个行为集合的笛卡尔积中平分。
    return connection.execute(
        """
        WITH adjacent AS (
            SELECT behaviors, lead(behaviors) OVER (
                PARTITION BY user_id ORDER BY event_time) AS next_behaviors
            FROM user_hours
        ), weighted AS (
            SELECT current_code, next_code,
                sum(1.0 / (len(behaviors) * len(next_behaviors))) AS weight
            FROM adjacent, unnest(behaviors) AS current_set(current_code),
                unnest(next_behaviors) AS next_set(next_code)
            WHERE next_behaviors IS NOT NULL GROUP BY 1, 2
        )
        SELECT *, weight / sum(weight) OVER (PARTITION BY current_code)
            AS probability
        FROM weighted ORDER BY 1, 2
        """,
    ).df()
