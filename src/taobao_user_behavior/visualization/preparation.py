"""绘制时间样本、特征筛选和训练负采样的描述图。

主要接口：draw_preparation。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from taobao_user_behavior.visualization.style import (
    configure_charts,
    save_figure,
)

LABELS = {"train": "训练", "validation": "验证", "test": "测试", "latest": "最新"}


def _positive_rates(folder: Path, figures: Path) -> None:
    windows = pd.read_csv(folder / "window_summary.csv").iloc[:3]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    rates = windows["positive_rate"].to_numpy() * 100
    bars = axis.bar(windows["name"].map(LABELS), rates, color="#487C9D", width=0.55)
    axis.bar_label(bars, labels=[f"{value:.4f}%" for value in rates], padding=5)
    axis.set_ylim(0, max(rates) * 1.22)
    axis.set_ylabel("候选正例比例（%）")
    axis.set_title("三个完整七日标签窗口的正例比例")
    figure.tight_layout()
    save_figure(figure, figures, "01_positive_rates")
    plt.close(figure)


def _daily_purchase(folder: Path, figures: Path) -> None:
    daily = pd.read_csv(folder / "daily_events.csv", parse_dates=["event_date"])
    windows = pd.read_csv(folder / "window_summary.csv")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        daily["event_date"], daily["purchase_events"], color="#3B7093", marker="."
    )
    for row, color in zip(
        windows.iloc[:3].to_dict("records"),
        [
            "#dce8ef",
            "#e4eee6",
            "#f4e7d8",
        ],
        strict=True,
    ):
        axis.axvspan(
            pd.Timestamp(row["cutoff"]),
            pd.Timestamp(row["label_end"]),
            color=color,
            alpha=0.6,
            label=f"{LABELS[row['name']]}标签窗",
        )
    axis.set_ylabel("购买事件数")
    axis.set_title("每日购买事件与标签窗口")
    axis.legend(frameon=False)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    save_figure(figure, figures, "02_daily_purchase")
    plt.close(figure)


def _selection_counts(folder: Path, figures: Path) -> None:
    decisions = pd.read_csv(folder / "feature_decisions.csv")
    counts = decisions["reason"].value_counts()
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    bars = axis.barh(counts.index, counts.values, color="#5F9EA0")
    axis.bar_label(bars, padding=4)
    axis.set_xlim(0, counts.max() * 1.18)
    axis.set_xlabel("特征数")
    axis.set_title("仅依据完整训练集进行初步筛选")
    axis.invert_yaxis()
    figure.tight_layout()
    save_figure(figure, figures, "03_feature_selection")
    plt.close(figure)


def _sampling_coverage(folder: Path, figures: Path) -> None:
    table = pd.read_csv(folder / "sampling_comparison.csv")
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    positions = np.arange(len(table))
    for offset, column, title, color in [
        (-0.24, "user_id_coverage", "用户覆盖", "#3B7093"),
        (0, "category_id_coverage", "类目覆盖", "#5F9EA0"),
        (0.24, "retained_fraction", "样本保留", "#D4A35A"),
    ]:
        axis.bar(
            positions + offset,
            table[column] * 100,
            width=0.24,
            label=title,
            color=color,
        )
    axis.set_xticks(
        positions, [f"{value}:1" for value in table["negative_ratio_limit"]]
    )
    axis.set_ylim(0, 110)
    axis.set_xlabel("训练负正比上限")
    axis.set_ylabel("相对自然训练集比例（%）")
    axis.set_title("负采样的覆盖与规模取舍")
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, figures, "04_sampling_coverage")
    plt.close(figure)


def draw_preparation(folder: Path, figures: Path) -> None:
    """从正式CSV生成四张中文图，描述结果而不声称模型收益。

    Args:
        folder: 准备阶段汇总CSV目录。
        figures: 图片保存目录。
    """
    plt.switch_backend("Agg")
    configure_charts()
    for function in (
        _positive_rates,
        _daily_purchase,
        _selection_counts,
        _sampling_coverage,
    ):
        function(folder, figures)
