"""检查完整日推导、标签相邻性和非法时间边界。

测试入口：本模块test_开头的函数。
"""

from datetime import datetime, timedelta

import pytest

from taobao_user_behavior.features.windows import FeatureWindow, plan_windows


def test_windows_use_complete_days_and_disjoint_labels() -> None:
    """末尾不足一天时排除该日，三套标签窗口保持相邻且不交叠。"""
    windows = plan_windows(datetime(2025, 11, 18), datetime(2025, 12, 19, 12))
    assert [window.name for window in windows] == [
        "train",
        "validation",
        "test",
        "latest",
    ]
    assert windows[0].history_start == datetime(2025, 11, 21)
    assert windows[0].cutoff == datetime(2025, 11, 28)
    assert windows[0].label_end == windows[1].cutoff
    assert windows[1].label_end == windows[2].cutoff
    assert windows[2].label_end == windows[3].cutoff == datetime(2025, 12, 19)
    assert windows[3].label_end is None


def test_partial_first_day_cannot_supply_missing_history() -> None:
    """首日缺少零时记录时不能凑足二十八个完整日。"""
    with pytest.raises(ValueError, match="28 complete"):
        plan_windows(datetime(2025, 12, 1, 1), datetime(2025, 12, 28, 23))


@pytest.mark.parametrize("history_days,label_days", [(6, 7), (7, 6), (7, 8)])
def test_window_rejects_wrong_lengths(history_days: int, label_days: int) -> None:
    """手动窗口也必须符合七日契约。

    Args:
        history_days: 测试历史天数。
        label_days: 测试标签天数。
    """
    cutoff = datetime(2025, 12, 8)
    with pytest.raises(ValueError, match="seven days"):
        FeatureWindow(
            "test",
            cutoff - timedelta(days=history_days),
            cutoff,
            cutoff + timedelta(days=label_days),
        )


def test_window_rejects_non_midnight() -> None:
    """小时截点不能作为正式样本边界。"""
    with pytest.raises(ValueError, match="midnight"):
        FeatureWindow("test", datetime(2025, 12, 1, 1), datetime(2025, 12, 8, 1), None)
