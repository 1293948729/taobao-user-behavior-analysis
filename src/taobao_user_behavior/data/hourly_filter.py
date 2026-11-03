"""按全量小时频次隔离重复行为，保留隔离记录供复查。

主要接口：HourlySummary、quarantine_hourly_events。
"""

from dataclasses import dataclass
from pathlib import Path

from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import connect


@dataclass(frozen=True)
class HourlySummary:
    """保存小时重复检查的计数。

    Attributes:
        total_groups: 四字段分组总数。
        maximum_count: 隔离前单组最高频次。
        flagged_groups: 严格超过阈值的组数。
        flagged_rows: 这些组包含的全部行数。

    Examples:
        summary = quarantine_hourly_events(staged_path, settings)
    """

    total_groups: int
    maximum_count: int
    flagged_groups: int
    flagged_rows: int


def quarantine_hourly_events(staged_path: Path, settings: Settings) -> HourlySummary:
    """全局计数后分开保存正常事件和疑似异常事件。

    仅用于接入流程生成的临时标准事件文件。成功后消耗该临时文件，
    替换events.parquet和hourly_rejected.parquet；不修改原始ZIP。

    Args:
        staged_path: 已完成所有CSV块规范化的临时Parquet文件。
        settings: 小时重复阈值、输出目录和查询资源限制。

    Returns:
        隔离前的分组规模及本次隔离数量。

    Raises:
        RuntimeError: 查询未能返回计数结果。
    """
    retained_path = settings.data / "events.filtered.parquet.tmp"
    rejected_path = settings.data / "hourly_rejected.parquet.tmp"
    with connect(settings, staged_path) as connection:
        # 小时精度下先保留重复，再用跨全部数据块的频次判断是否异常。
        connection.execute(
            """
            CREATE TEMP TABLE hourly_counts AS
            SELECT user_id, item_id, behavior_type, event_time,
                   count(*) AS hourly_count
            FROM events
            GROUP BY user_id, item_id, behavior_type, event_time
            """,
        )
        measured = connection.execute(
            """
            SELECT count(*), coalesce(max(hourly_count), 0),
                   count(*) FILTER (WHERE hourly_count > ?),
                   coalesce(sum(hourly_count) FILTER (WHERE hourly_count > ?), 0)
            FROM hourly_counts
            """,
            [settings.hourly_repeat_limit, settings.hourly_repeat_limit],
        ).fetchone()
        if measured is None:
            raise RuntimeError("Hourly frequency query returned no counts.")
        summary = HourlySummary(*(int(value) for value in measured))
        connection.execute(
            """
            CREATE TEMP TABLE hourly_suspects AS
            SELECT * FROM hourly_counts WHERE hourly_count > ?
            """,
            [settings.hourly_repeat_limit],
        )
        connection.sql(
            """
            SELECT events.*, hourly_suspects.hourly_count,
                   'hourly_repeat_limit_exceeded' AS reason
            FROM events INNER JOIN hourly_suspects
            USING (user_id, item_id, behavior_type, event_time)
            """,
        ).to_parquet(str(rejected_path), compression="zstd")
        if summary.flagged_rows:
            connection.sql(
                """
                SELECT events.* FROM events ANTI JOIN hourly_suspects
                USING (user_id, item_id, behavior_type, event_time)
                """,
            ).to_parquet(str(retained_path), compression="zstd")
    if summary.flagged_rows:
        retained_path.replace(settings.data / "events.parquet")
        staged_path.unlink()
    else:
        staged_path.replace(settings.data / "events.parquet")
    rejected_path.replace(settings.data / "hourly_rejected.parquet")
    return summary
