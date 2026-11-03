"""绘制用户分层、购买次数、复访及类目偏好。

主要接口：user_figures。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.figure import Figure

from taobao_user_behavior.visualization.style import (
    COLORS,
    annotate_matrix,
    save_figure,
)


def user_figures(tables: dict[str, pd.DataFrame], folder: Path) -> dict[str, Path]:
    """生成用户维度的静态图。

    Args:
        tables: 用户、留存和偏好等汇总表。
        folder: 图片输出目录。

    Returns:
        图表编号及文件路径。
    """
    return {
        "04_activity": save_figure(_activity(tables), folder, "04_activity"),
        "05_purchases": save_figure(
            _purchases(tables["purchase_frequency"]),
            folder,
            "05_purchases",
        ),
        "06_retention": save_figure(
            _retention(tables["retention"]),
            folder,
            "06_retention",
        ),
        "07_preferences": save_figure(
            _preferences(tables["preferences"]),
            folder,
            "07_preferences",
        ),
        "08_concentration": save_figure(
            _concentration(tables["users"]),
            folder,
            "08_concentration",
        ),
    }


def _activity(tables: dict[str, pd.DataFrame]) -> Figure:
    figure = Figure(figsize=(11, 4.6), layout="constrained")
    axes = figure.subplots(1, 2)
    users, grouped = tables["users"], tables["activity"]
    bins = np.geomspace(1, max(2, users["events"].max()), 25)
    axes[0].hist(users["events"], bins=bins, color=COLORS[0], edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("用户全期事件数（对数）")
    axes[0].set_ylabel("用户数")
    axes[0].set_title("活跃度长尾分布")
    positions = np.arange(len(grouped))
    bars = axes[1].bar(positions, grouped["buyer_coverage"], color=COLORS[3])
    axes[1].set_xticks(positions, grouped["bin"], rotation=35, ha="right")
    axes[1].bar_label(
        bars,
        labels=[f"n={value:,}" for value in grouped["users"]],
        rotation=90,
        padding=4,
        fontsize=8,
    )
    axes[1].set_ylim(0, 1.3)
    axes[1].set_yticks(np.arange(0, 1.01, 0.2))
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1))
    axes[1].set_title("各活跃区间的购买用户覆盖")
    axes[1].set_xlabel("用户全期事件数分箱")
    return figure


def _purchases(grouped: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(10.8, 4.0), layout="constrained")
    axis = figure.add_subplot(111)
    bars = axis.bar(grouped["bin"], grouped["users"], color=COLORS[3])
    axis.bar_label(bars, fmt="%.0f", padding=4)
    axis.set_xlabel("每用户购买事件数")
    axis.set_ylabel("用户数")
    axis.set_title("购买事件次数分布：多次购买不等于可靠订单复购")
    axis.margins(y=0.2)
    return figure


def _retention(retention: pd.DataFrame) -> Figure:
    matrix = retention.pivot(index="first_date", columns="lag", values="retention")
    figure = Figure(
        figsize=(10.8, max(4.4, min(9, len(matrix) * 0.25))), layout="constrained"
    )
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="RdYlBu_r")
    axis.set_xticks(range(len(matrix.columns)), [f"D{lag}" for lag in matrix.columns])
    axis.set_yticks(
        range(len(matrix.index)),
        [pd.Timestamp(day).strftime("%m-%d") for day in matrix.index],
    )
    annotate_matrix(axis, matrix, diverging=True)
    axis.set_ylabel("窗口内首次观察日（不是注册日）")
    axis.set_title("首次观察cohort留存：未完成观察窗口标记为 -")
    figure.colorbar(image, ax=axis, format=ticker.PercentFormatter(1), shrink=0.8)
    return figure


def _preferences(preferences: pd.DataFrame) -> Figure:
    matrix = preferences.pivot(
        index="activity_group",
        columns="category_id",
        values="share",
    ).reindex(["低活跃", "中活跃", "高活跃"])
    figure = Figure(figsize=(11, 4.3), layout="constrained")
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(3), matrix.index)
    annotate_matrix(axis, matrix, decimals=1, missing="")
    axis.set_title("各活跃组Top5类目：占本组非缺失类目事件比例")
    figure.colorbar(image, ax=axis, format=ticker.PercentFormatter(1), shrink=0.7)
    return figure


def _concentration(users: pd.DataFrame) -> Figure:
    ordered = users.sort_values("events", ascending=False)
    x_values = np.arange(1, len(ordered) + 1) / len(ordered)
    figure = Figure(figsize=(10.8, 4.4), layout="constrained")
    axis = figure.add_subplot(111)
    axis.plot(
        x_values, ordered["events"].cumsum() / ordered["events"].sum(), label="行为"
    )
    if ordered["purchases"].sum():
        axis.plot(
            x_values,
            ordered["purchases"].cumsum() / ordered["purchases"].sum(),
            label="购买事件",
            color=COLORS[3],
        )
    axis.plot([0, 1], [0, 1], color="#AAB6C1", linestyle="--")
    axis.axvline(0.2, color="#AAB6C1", linestyle=":")
    axis.set_xlabel("按活跃度降序排列的累计用户占比")
    axis.set_ylabel("累计贡献占比")
    axis.set_title("用户贡献集中度")
    axis.xaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.legend()
    return figure
