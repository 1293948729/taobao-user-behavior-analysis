"""确定完整日边界以及训练、验证、测试和最新样本窗口。

主要接口：FeatureWindow、plan_windows。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

WEEK = timedelta(days=7)


@dataclass(frozen=True)
class FeatureWindow:
    """保存一套样本的历史与标签半开边界。

    Attributes:
        name: 样本集合名称。
        history_start: 历史包含的第一个时刻。
        cutoff: 历史排除、标签包含的分界时刻。
        label_end: 标签排除的终点；最新样本为None。

    Examples:
        windows = plan_windows(first_event, last_event)
    """

    name: str
    history_start: datetime
    cutoff: datetime
    label_end: datetime | None

    def __post_init__(self) -> None:
        """拒绝非零时边界或偏离七日的窗口。

        Raises:
            ValueError: 边界不是零时，或历史及标签不是七天。
        """
        bounds = [self.history_start, self.cutoff]
        if self.label_end is not None:
            bounds.append(self.label_end)
        if any(value != _midnight(value) for value in bounds):
            raise ValueError("Feature boundaries must be at midnight.")
        if self.cutoff - self.history_start != WEEK:
            raise ValueError("Feature history must span exactly seven days.")
        if self.label_end is not None and self.label_end - self.cutoff != WEEK:
            raise ValueError("Labels must span exactly seven days.")


def _midnight(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def plan_windows(first_event: datetime, last_event: datetime) -> list[FeatureWindow]:
    """从事件范围推导四套七日窗口，排除两端不足完整的日期。

    Args:
        first_event: 已完整接入的最早小时。
        last_event: 已完整接入的最晚小时。

    Returns:
        按train、validation、test、latest排列的窗口。

    Raises:
        ValueError: 没有至少二十八个完整日，或首尾顺序错误。
    """
    start = _midnight(first_event)
    if first_event != start:
        start += timedelta(days=1)
    end = _midnight(last_event + timedelta(hours=1))
    if last_event < first_event or end - start < 4 * WEEK:
        raise ValueError("Features require at least 28 complete days.")
    windows = [
        FeatureWindow(name, end - (offset + 1) * WEEK, end - offset * WEEK, label_end)
        for name, offset, label_end in [
            ("train", 3, end - 2 * WEEK),
            ("validation", 2, end - WEEK),
            ("test", 1, end),
            ("latest", 0, None),
        ]
    ]
    return windows
