"""分析用户活跃分层、行为共现及完整七日观察窗口。

主要接口：user_tables、label_tables。
"""

import duckdb
import pandas as pd


def user_tables(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """汇总当前观察期的用户画像与行为交集。

    Args:
        connection: 含用户与用户商品聚合表的连接。

    Returns:
        用户明细、活跃分箱、购买次数分箱、偏好和共现漏斗。
    """
    users = connection.execute("SELECT * FROM user_summary ORDER BY user_id").df()
    users["activity_group"] = pd.cut(
        users["events"],
        [0, 99, 999, float("inf")],
        labels=["低活跃", "中活跃", "高活跃"],
    )
    activity = _group_users(
        users,
        "events",
        [0, 9, 29, 99, 299, 999, 2999, 9999, float("inf")],
        [
            "1-9",
            "10-29",
            "30-99",
            "100-299",
            "300-999",
            "1000-2999",
            "3000-9999",
            "10000+",
        ],
    )
    purchases = _group_users(
        users,
        "purchases",
        [-1, 0, 1, 3, 7, 15, float("inf")],
        ["0", "1", "2-3", "4-7", "8-15", "16+"],
    )
    return {
        "users": users,
        "activity": activity,
        "purchase_frequency": purchases,
        "preferences": _preferences(connection),
        "funnel": connection.execute(
            """
            SELECT 1 AS step, '浏览' AS behavior, count(*) AS pairs
            FROM pair_summary WHERE viewed
            UNION ALL SELECT 2, '浏览且加购', count(*) FROM pair_summary
                WHERE viewed AND carted
            UNION ALL SELECT 3, '再有收藏', count(*) FROM pair_summary
                WHERE viewed AND carted AND favored
            UNION ALL SELECT 4, '再有购买', count(*) FROM pair_summary
                WHERE viewed AND carted AND favored AND purchased
            ORDER BY step
            """,
        ).df(),
        "independent_coverage": connection.execute(
            """
            SELECT count(*) AS observed_pairs,
                count(*) FILTER (WHERE viewed) AS viewed,
                count(*) FILTER (WHERE carted) AS carted,
                count(*) FILTER (WHERE favored) AS favored,
                count(*) FILTER (WHERE purchased) AS purchased,
                count(*) FILTER (WHERE viewed AND purchased) AS view_buy,
                count(*) FILTER (WHERE carted AND purchased) AS cart_buy,
                count(*) FILTER (WHERE favored AND purchased) AS favorite_buy
            FROM pair_summary
            """,
        ).df(),
    }


def _group_users(
    users: pd.DataFrame, field: str, edges: list[float], labels: list[str]
) -> pd.DataFrame:
    assigned = users.assign(
        bin=pd.cut(users[field], edges, labels=labels),
        buyer=users["purchases"].gt(0),
    )
    grouped = (
        assigned.groupby("bin", observed=False)
        .agg(
            users=("user_id", "size"),
            buyers=("buyer", "sum"),
            events=("events", "sum"),
            purchases=("purchases", "sum"),
        )
        .reset_index()
    )
    grouped["buyer_coverage"] = grouped["buyers"] / grouped["users"].replace(
        0, float("nan")
    )
    return grouped


def _preferences(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH counts AS (
            SELECT CASE WHEN user_summary.events < 100 THEN '低活跃'
                WHEN user_summary.events < 1000 THEN '中活跃'
                ELSE '高活跃' END AS activity_group,
                category_id, count(*) AS events
            FROM events JOIN user_summary USING (user_id)
            WHERE NOT category_missing GROUP BY 1, 2
        ), ranked AS (
            SELECT *, events * 1.0 / sum(events) OVER (
                PARTITION BY activity_group) AS share,
                row_number() OVER (
                    PARTITION BY activity_group ORDER BY events DESC, category_id
                ) AS ranking
            FROM counts
        )
        SELECT * FROM ranked WHERE ranking <= 5 ORDER BY activity_group, ranking
        """,
    ).df()


def label_tables(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """用最后七个完整日作为标签窗，检查历史行为深度与未来购买的关系。

    候选只含标签窗前已经交互的用户商品对，不训练模型；
    临时创建候选表和标签表，不写入正式分析库。

    Args:
        connection: 可创建临时表的分析连接。

    Returns:
        观察窗说明、自然标签比例及三类历史行为分箱的后续购买率。

    Raises:
        ValueError: 日期跨度不足八天，不能分开历史和七日标签窗。
    """
    bounds = connection.execute(
        """
        SELECT min(CAST(event_time AS DATE)), max(CAST(event_time AS DATE)),
            date_diff('day', min(event_time),
                date_trunc('day', max(event_time) + INTERVAL 1 HOUR))
        FROM events
        """,
    ).fetchone()
    if bounds is None or bounds[2] < 8:
        raise ValueError("Seven-day labels need at least eight observed dates.")
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE label_window AS
        SELECT CAST(date_trunc('day', max(event_time) + INTERVAL 1 HOUR)
                AS DATE) - 7 AS label_start,
            CAST(date_trunc('day', max(event_time) + INTERVAL 1 HOUR)
                AS DATE) AS label_end,
            min(CAST(event_time AS DATE)) AS history_start
        FROM events
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE label_pairs AS
        SELECT DISTINCT user_id, item_id
        FROM events CROSS JOIN label_window
        WHERE event_time >= label_start AND event_time < label_end
            AND behavior_type = 4
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE candidates AS
        SELECT events.user_id, events.item_id,
            count(*) FILTER (WHERE behavior_type = 1) AS views,
            count(*) FILTER (WHERE behavior_type = 2) AS favorites,
            count(*) FILTER (WHERE behavior_type = 3) AS carts,
            bool_or(label_pairs.user_id IS NOT NULL) AS label
        FROM events CROSS JOIN label_window
        LEFT JOIN label_pairs USING (user_id, item_id)
        WHERE event_time < label_start GROUP BY 1, 2
        """,
    )
    summary = connection.execute(
        """
        SELECT first(history_start) AS history_start,
            first(label_start) AS label_start, first(label_end) AS label_end,
            count(*) AS candidates, count(*) FILTER (WHERE label) AS positives,
            count(*) FILTER (WHERE NOT label) AS negatives,
            avg(CAST(label AS INTEGER)) AS positive_rate,
            (SELECT count(*) FROM label_pairs) AS all_label_pairs
        FROM candidates CROSS JOIN label_window
        """,
    ).df()
    binned = connection.execute(
        """
        WITH depths AS (
            SELECT label, unnest(['浏览', '加购', '收藏']) AS behavior,
                unnest([views, carts, favorites]) AS depth FROM candidates
        ), buckets AS (
            SELECT *, CASE WHEN depth = 0 THEN 0 WHEN depth = 1 THEN 1
                WHEN depth <= 3 THEN 2 WHEN depth <= 7 THEN 3
                WHEN depth <= 15 THEN 4 ELSE 5 END AS bin
            FROM depths
        )
        SELECT behavior, bin, count(*) AS candidates,
            count(*) FILTER (WHERE label) AS positives,
            avg(CAST(label AS INTEGER)) AS purchase_rate
        FROM buckets GROUP BY 1, 2 ORDER BY 1, 2
        """,
    ).df()
    return {"label_window": summary, "depth_purchase": binned}
