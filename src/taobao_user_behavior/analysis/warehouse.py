"""构建本地SQL分析库，保存可复核的事件和基础聚合。

主要接口：build_warehouse、open_warehouse。
"""

from pathlib import Path

import duckdb

from taobao_user_behavior.settings import Settings


def open_warehouse(settings: Settings) -> duckdb.DuckDBPyConnection:
    """只读打开本次运行的分析库。

    Args:
        settings: 分析库位置与资源限制。

    Returns:
        需要由调用方关闭的只读连接。
    """
    connection = duckdb.connect(str(settings.data / "analysis.duckdb"), read_only=True)
    connection.execute("SET memory_limit = ?", [settings.memory_limit])
    connection.execute("SET threads = ?", [settings.threads])
    return connection


def build_warehouse(settings: Settings) -> Path:
    """从标准事件重建分析库，成功后替换本运行内的旧分析库。

    Args:
        settings: 输入Parquet及本次运行位置。

    Returns:
        已关闭并落盘的数据库路径。

    Raises:
        ValueError: 输入为空，没有可分析事件。
    """
    temporary = settings.data / "analysis.build.duckdb"
    if temporary.exists():
        temporary.unlink()
    with duckdb.connect(str(temporary)) as connection:
        connection.execute("SET memory_limit = ?", [settings.memory_limit])
        connection.execute("SET threads = ?", [settings.threads])
        connection.read_parquet(str(settings.data / "events.parquet")).to_table(
            "events"
        )
        count = connection.execute("SELECT count(*) FROM events").fetchone()
        if count is None or count[0] == 0:
            raise ValueError("EDA requires at least one retained event.")
        _build_time(connection)
        _build_users(connection)
        _build_products(connection)
        _build_pairs(connection)
        connection.execute("CHECKPOINT")
    target = settings.data / "analysis.duckdb"
    temporary.replace(target)
    return target


def _build_time(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE user_days AS
        SELECT user_id, CAST(event_time AS DATE) AS event_date,
            count(*) AS events, count(DISTINCT event_time) AS active_hours,
            count(*) FILTER (WHERE behavior_type = 4) AS purchases
        FROM events GROUP BY 1, 2
        """,
    )
    connection.execute(
        """
        CREATE TABLE daily_behavior AS
        SELECT CAST(event_time AS DATE) AS event_date, behavior_type,
            count(*) AS events, count(DISTINCT user_id) AS users
        FROM events GROUP BY 1, 2
        """,
    )
    connection.execute(
        """
        CREATE TABLE user_hours AS
        SELECT user_id, event_time, count(*) AS events,
            list_sort(list_distinct(list(behavior_type))) AS behaviors
        FROM events GROUP BY 1, 2
        """,
    )
    connection.execute("CREATE INDEX user_days_key ON user_days(user_id, event_date)")


def _build_users(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE user_summary AS
        SELECT user_id, count(*) AS events,
            count(*) FILTER (WHERE behavior_type = 1) AS views,
            count(*) FILTER (WHERE behavior_type = 2) AS favorites,
            count(*) FILTER (WHERE behavior_type = 3) AS carts,
            count(*) FILTER (WHERE behavior_type = 4) AS purchases,
            count(DISTINCT item_id) AS items,
            count(DISTINCT category_id)
                FILTER (WHERE NOT category_missing) AS categories,
            count(DISTINCT CAST(event_time AS DATE)) AS active_days,
            count(DISTINCT event_time) AS active_hours,
            min(CAST(event_time AS DATE)) AS first_date,
            max(CAST(event_time AS DATE)) AS last_date
        FROM events GROUP BY 1
        """,
    )
    connection.execute("CREATE INDEX user_key ON user_summary(user_id)")


def _build_products(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE item_categories AS
        SELECT item_id, category_id, count(*) AS events
        FROM events GROUP BY 1, 2
        """,
    )
    connection.execute(
        """
        CREATE TABLE item_summary AS
        WITH catalog AS (
            SELECT item_id,
                first(category_id ORDER BY events DESC, category_id) AS category_id,
                count(*) AS category_count
            FROM item_categories GROUP BY 1
        )
        SELECT item_id, first(catalog.category_id) AS category_id,
            first(category_count) AS category_count, count(*) AS events,
            count(*) FILTER (WHERE behavior_type = 1) AS views,
            count(*) FILTER (WHERE behavior_type = 4) AS purchases,
            count(DISTINCT user_id) AS users,
            count(DISTINCT user_id)
                FILTER (WHERE behavior_type = 1) AS viewers,
            count(DISTINCT user_id)
                FILTER (WHERE behavior_type = 4) AS buyers
        FROM events JOIN catalog USING (item_id) GROUP BY 1
        """,
    )
    connection.execute(
        """
        CREATE TABLE category_summary AS
        SELECT category_id, count(*) AS events, count(DISTINCT item_id) AS items,
            count(*) FILTER (WHERE behavior_type = 1) AS views,
            count(*) FILTER (WHERE behavior_type = 4) AS purchases,
            count(DISTINCT user_id) AS users
        FROM events GROUP BY 1
        """,
    )
    connection.execute("CREATE INDEX item_key ON item_summary(item_id)")


def _build_pairs(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE pair_summary AS
        SELECT user_id, item_id,
            bool_or(behavior_type = 1) AS viewed,
            bool_or(behavior_type = 2) AS favored,
            bool_or(behavior_type = 3) AS carted,
            bool_or(behavior_type = 4) AS purchased
        FROM events GROUP BY 1, 2
        """,
    )
    connection.execute(
        """
        CREATE TABLE item_conversion AS
        SELECT item_id, count(*) FILTER (WHERE viewed) AS viewed_pairs,
            count(*) FILTER (WHERE viewed AND purchased) AS bought_viewed_pairs,
            count(*) FILTER (WHERE viewed AND purchased) * 1.0
                / nullif(count(*) FILTER (WHERE viewed), 0) AS purchase_coverage
        FROM pair_summary GROUP BY 1
        """,
    )
