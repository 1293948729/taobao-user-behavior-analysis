"""计算总体规模并复核聚合表的基本守恒关系。

主要接口：overview_metrics、validate_aggregates、metric_dictionary。
"""

import duckdb
import pandas as pd


def overview_metrics(connection: duckdb.DuckDBPyConnection) -> dict:
    """计算全量总体指标，分开记录事件比和用户覆盖。

    Args:
        connection: 当前分析库连接。

    Returns:
        总体规模、观察跨度、用户深度及购买关联指标。
    """
    scale = (
        connection.execute(
            """
        SELECT count(*) AS events, count(DISTINCT user_id) AS users,
            count(DISTINCT item_id) AS items,
            count(DISTINCT category_id)
                FILTER (WHERE NOT category_missing) AS categories,
            min(event_time) AS start_time, max(event_time) AS end_time,
            date_diff('day', min(event_time), max(event_time)) + 1 AS calendar_days,
            count(*) FILTER (WHERE category_missing) AS missing_category_rows,
            count(*) FILTER (WHERE behavior_type = 1) AS views,
            count(*) FILTER (WHERE behavior_type = 2) AS favorites,
            count(*) FILTER (WHERE behavior_type = 3) AS carts,
            count(*) FILTER (WHERE behavior_type = 4) AS purchases,
            count(DISTINCT user_id)
                FILTER (WHERE behavior_type = 4) AS buyers
        FROM events
        """,
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    depth = (
        connection.execute(
            """
        SELECT avg(events) AS events_per_user, avg(items) AS items_per_user,
            avg(categories) AS categories_per_user,
            avg(purchases) AS purchases_per_user,
            avg(active_hours * 1.0 / active_days) AS mean_active_hours_on_active_day
        FROM user_summary
        """,
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    pair = (
        connection.execute(
            """
        SELECT count(*) AS observed_pairs,
            count(*) FILTER (WHERE viewed) AS viewed_pairs,
            count(*) FILTER (WHERE viewed AND purchased) AS bought_viewed_pairs,
            count(*) FILTER (WHERE viewed AND purchased) * 1.0
                / nullif(count(*) FILTER (WHERE viewed), 0) AS viewed_pair_coverage
        FROM pair_summary
        """,
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    return {
        **scale,
        **depth,
        **pair,
        "buyer_coverage": scale["buyers"] / scale["users"],
        "purchase_view_event_ratio": (
            scale["purchases"] / scale["views"] if scale["views"] else None
        ),
        "mean_observed_category_coverage": (
            depth["categories_per_user"] / scale["categories"]
            if scale["categories"]
            else None
        ),
    }


def validate_aggregates(connection: duckdb.DuckDBPyConnection) -> dict:
    """核对事件、用户、商品、类目和日期聚合总量。

    Args:
        connection: 当前分析库连接。

    Returns:
        独立读取各聚合总数的检查记录。

    Raises:
        ValueError: 事件总数与任一聚合总数不相等。
    """
    counts = (
        connection.execute(
            """
        SELECT (SELECT count(*) FROM events) AS events,
            (SELECT sum(events) FROM user_summary) AS user_events,
            (SELECT sum(events) FROM item_summary) AS item_events,
            (SELECT sum(events) FROM category_summary) AS category_events,
            (SELECT sum(events) FROM daily_behavior) AS daily_events,
            (SELECT sum(events) FROM user_days) AS user_day_events,
            (SELECT sum(events) FROM user_hours) AS user_hour_events
        """,
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    if len(set(counts.values())) != 1:
        raise ValueError(f"Aggregate totals disagree: {counts}")
    return {"passed": True, "totals": counts}


def metric_dictionary() -> pd.DataFrame:
    """列出报告中主要指标的分母与解释边界。

    Returns:
        可保存为CSV的指标字典。
    """
    records = [
        ("行为次数", "标准事件行数", "保留重复；非会话数"),
        ("购买事件", "behavior_type=4行数", "非可靠订单；无金额"),
        ("用户购买覆盖", "购买用户数/观察用户数", "同窗覆盖；不是未来预测"),
        ("购买浏览事件比", "购买事件数/浏览事件数", "可大于1；不是漏斗转化概率"),
        (
            "浏览商品对购买覆盖",
            "有浏览且购买的用户商品对/有浏览用户商品对",
            "同窗共现；未保证先后顺序",
        ),
        (
            "品类覆盖代理",
            "人均已交互类目数/本数据已观察非缺失类目数",
            "非平台全类目覆盖；缺类目不纳入类目数",
        ),
        (
            "日活跃小时代理",
            "逐用户累计非空小时/活跃日，再对用户平均",
            "不是在线或停留时长",
        ),
        (
            "七日候选正例率",
            "历史候选中后续七日购买对数/历史已交互对数",
            "仅历史候选；新商品对另计；完整日窗口",
        ),
        (
            "Dk留存",
            "首次观察日后第k天有事件用户/窗口完整的cohort用户",
            "非注册新增；未完整观察显示空",
        ),
        (
            "共现链路",
            "浏览∩加购∩收藏∩购买用户商品对逐级交集",
            "不要求行为顺序，非真实路径",
        ),
        (
            "跨小时转移",
            "相邻非空小时行为集合加权关系/该源行为总关系",
            "多行为组合平分权重；排除末尾；无退出",
        ),
        (
            "商品四象限",
            "支持量>=20浏览商品对商品的浏览量及购买覆盖上四分位",
            "浏览代理曝光；高覆盖严格大于阈值；非经营标准",
        ),
        (
            "活跃分层",
            "全期行为<100、100-999、>=1000三组",
            "描述性分层；不删除低活跃或高活跃用户",
        ),
        (
            "重复阈值",
            "用户+商品+行为+小时全局次数>100整组隔离",
            "不是用户小时全部操作阈值；疑似异常非恶意证明",
        ),
    ]
    return pd.DataFrame(records, columns=["metric", "definition", "limitation"])
