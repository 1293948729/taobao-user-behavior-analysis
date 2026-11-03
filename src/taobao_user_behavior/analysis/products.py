"""计算商品类目排名、长尾及浏览代理四象限。

主要接口：product_tables。
"""

import duckdb
import pandas as pd

MINIMUM_VIEWED_PAIRS = 20
SCATTER_LIMIT = 12000


def product_tables(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """计算全量排名与分层，散点只导出有界确定性样本。

    Args:
        connection: 含商品、类目及用户商品对聚合表的连接。

    Returns:
        两类双排名、类目长尾、全量四象限计数及散点样本。
    """
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW products AS
        SELECT * FROM item_summary JOIN item_conversion USING (item_id)
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW category_conversion AS
        WITH category_pairs AS (
            SELECT category_id, user_id, item_id,
                bool_or(behavior_type = 1) AS viewed,
                bool_or(behavior_type = 4) AS purchased
            FROM events GROUP BY 1, 2, 3
        )
        SELECT category_id, count(*) FILTER (WHERE viewed) AS viewed_pairs,
            count(*) FILTER (WHERE viewed AND purchased) AS bought_viewed_pairs,
            count(*) FILTER (WHERE viewed AND purchased) * 1.0
                / nullif(count(*) FILTER (WHERE viewed), 0) AS purchase_coverage
        FROM category_pairs GROUP BY 1
        """,
    )
    tables = _rankings(connection)
    tables["category_tail"] = connection.execute(
        """
        SELECT *, row_number() OVER (ORDER BY events DESC, category_id) AS ranking,
            sum(events) OVER (ORDER BY events DESC, category_id) * 1.0
                / sum(events) OVER () AS cumulative_events,
            sum(purchases) OVER (ORDER BY events DESC, category_id) * 1.0
                / nullif(sum(purchases) OVER (), 0) AS cumulative_purchases
        FROM category_summary WHERE category_id <> '__missing__'
        ORDER BY events DESC, category_id
        """,
    ).df()
    tables.update(_quadrants(connection))
    return tables


def _rankings(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    return {
        "item_flow": connection.execute(
            "SELECT * FROM products ORDER BY events DESC, item_id LIMIT 20"
        ).df(),
        "item_conversion": connection.execute(
            """
            SELECT * FROM products WHERE viewed_pairs >= ?
            ORDER BY purchase_coverage DESC, viewed_pairs DESC, item_id LIMIT 20
            """,
            [MINIMUM_VIEWED_PAIRS],
        ).df(),
        "category_flow": connection.execute(
            """
            SELECT * FROM category_summary JOIN category_conversion
                USING (category_id)
            WHERE category_id <> '__missing__'
            ORDER BY events DESC, category_id LIMIT 20
            """,
        ).df(),
        "category_conversion": connection.execute(
            """
            SELECT * FROM category_summary JOIN category_conversion
                USING (category_id)
            WHERE category_id <> '__missing__' AND viewed_pairs >= ?
            ORDER BY purchase_coverage DESC, viewed_pairs DESC, category_id LIMIT 20
            """,
            [MINIMUM_VIEWED_PAIRS],
        ).df(),
    }


def _quadrants(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    # 用有支持量商品的上四分位区分高低，避免中位数为零时划分退化。
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE quadrant_thresholds AS
        SELECT quantile_cont(views, 0.75) AS view_threshold,
            quantile_cont(purchase_coverage, 0.75) AS rate_threshold,
            count(*) AS eligible_items
        FROM products WHERE viewed_pairs >= ?
        """,
        [MINIMUM_VIEWED_PAIRS],
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE quadrants AS
        SELECT products.*,
            CASE WHEN views >= view_threshold
                AND purchase_coverage > rate_threshold THEN '明星型'
            WHEN views >= view_threshold THEN '流量型'
            WHEN purchase_coverage > rate_threshold THEN '潜力型'
            ELSE '长尾型' END AS quadrant
        FROM products CROSS JOIN quadrant_thresholds WHERE viewed_pairs >= ?
        """,
        [MINIMUM_VIEWED_PAIRS],
    )
    return {
        "quadrant_thresholds": connection.execute(
            "SELECT * FROM quadrant_thresholds"
        ).df(),
        "quadrant_counts": connection.execute(
            "SELECT quadrant, count(*) AS items FROM quadrants GROUP BY 1 ORDER BY 1"
        ).df(),
        "quadrant_sample": connection.execute(
            "SELECT * FROM quadrants ORDER BY hash(item_id), item_id LIMIT ?",
            [SCATTER_LIMIT],
        ).df(),
        "item_tail": connection.execute(
            """
            WITH ranked AS (
                SELECT *, row_number() OVER (
                    ORDER BY events DESC, item_id) AS ranking,
                    count(*) OVER () AS all_items,
                    sum(events) OVER () AS all_events
                FROM item_summary
            )
            SELECT count(*) AS items, sum(events) AS events,
                count(*) FILTER (WHERE purchases = 0) AS never_bought_items,
                sum(events) FILTER (
                    WHERE ranking <= ceil(all_items * 0.2)) * 1.0
                    / first(all_events) AS top20_event_share,
                count(*) FILTER (WHERE category_count > 1) AS multi_category_items
            FROM ranked
            """,
        ).df(),
    }
