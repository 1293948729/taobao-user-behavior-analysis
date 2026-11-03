"""统计保留记录的重复频次和用户活跃度，不二次删除。

主要接口：audit。
"""

from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import connect, save_json


def audit(settings: Settings) -> dict:
    """统计清洗后保留的重复记录和用户事件量，不再删除事件。

    Args:
        settings: 标准事件位置、报告目录和查询资源限制。

    Returns:
        事件量、重复计数、活跃度分布和IQR检查结果。

    Raises:
        ValueError: 事件计数查询未返回结果。
    """
    with connect(settings) as connection:
        total = connection.execute("SELECT count(*) FROM events").fetchone()
        if total is None:
            raise ValueError("Event count query returned no result.")
        rows = int(total[0])
        duplicate_summary = (
            connection.execute(
                """
            SELECT count(*) AS repeated_groups,
                coalesce(sum(event_count - 1), 0) AS repeated_surplus
            FROM (
                SELECT count(*) AS event_count
                FROM events
                GROUP BY user_id, item_id, event_time, behavior_type
                HAVING count(*) > 1
            ) AS repeated
            """,
            )
            .df()
            .iloc[0]
            .to_dict()
        )
        activity = connection.execute(
            "SELECT user_id, count(*) AS events FROM events GROUP BY 1"
        ).df()
        lower, upper = (
            activity["events"].quantile([0.25, 0.75])
            if not activity.empty
            else (0.0, 0.0)
        )
        threshold = float(upper + 1.5 * (upper - lower)) if rows else None
        # 次数多只能说明活跃，单凭这个阈值不能删除用户记录。
        summary = {
            "rows": rows,
            **duplicate_summary,
            "activity_minimum": int(activity["events"].min()) if rows else None,
            "activity_median": float(activity["events"].median()) if rows else None,
            "activity_maximum": int(activity["events"].max()) if rows else None,
            "activity_q1": float(lower) if rows else None,
            "activity_q3": float(upper) if rows else None,
            "activity_iqr_upper": threshold,
            "users_above_iqr": (
                int(activity["events"].gt(threshold).sum()) if rows else 0
            ),
            "outlier_action": "retain counts; statistical threshold only",
            "deduplication_action": "no further deletion after hourly isolation",
        }
    save_json(settings.outputs / "quality.json", summary)
    return summary
