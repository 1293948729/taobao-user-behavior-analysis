"""绘制行为构成、日趋势及星期小时热图。

主要接口：temporal_figures。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from taobao_user_behavior.visualization.style import (
    BEHAVIOR_NAMES,
    COLORS,
    WEEKDAYS,
    save_figure,
)


def temporal_figures(tables: dict[str, pd.DataFrame], folder: Path) -> dict[str, Path]:
    """根据汇总表绘制宏观时序图。

    Args:
        tables: 已计算完成的EDA汇总表。
        folder: 图片保存目录。

    Returns:
        图表编号及对应文件。
    """
    return {
        "01_behavior": save_figure(_behavior(tables["daily"]), folder, "01_behavior"),
        "02_daily": save_figure(_daily(tables["daily"]), folder, "02_daily"),
        "03_heatmaps": save_figure(_heatmaps(tables["hourly"]), folder, "03_heatmaps"),
    }


def _behavior(daily: pd.DataFrame) -> Figure:
    totals = daily.groupby("behavior_type")["events"].sum().reindex([1, 2, 3, 4])
    figure = Figure(figsize=(10.8, 4.2), layout="constrained")
    axis = figure.add_subplot(111)
    bars = axis.barh(list(BEHAVIOR_NAMES.values()), totals, color=COLORS)
    axis.bar_label(
        bars,
        labels=[f"{value:,.0f}  ({value / totals.sum():.2%})" for value in totals],
        padding=6,
    )
    axis.set_xscale("log")
    axis.set_xlabel("事件数（对数坐标，重复记录保留）")
    axis.set_title("四类行为构成")
    axis.invert_yaxis()
    axis.margins(x=0.3)
    return figure


def _daily(daily: pd.DataFrame) -> Figure:
    pivot = daily.pivot(index="event_date", columns="behavior_type", values="events")
    pivot.index = pd.to_datetime(pivot.index)
    figure = Figure(figsize=(11, 7.2), layout="constrained")
    axes = figure.subplots(3, 2)
    for code, axis in zip([1, 2, 3, 4], axes.flat, strict=False):
        series = pivot[code]
        axis.plot(series.index, series, color=COLORS[code - 1], linewidth=1.8)
        highest = series.idxmax()
        axis.scatter([highest], [series.max()], color=COLORS[code - 1])
        axis.annotate(
            f"{highest:%m-%d} {series.max():,.0f}",
            (highest, series.max()),
            xytext=(4, -15),
            textcoords="offset points",
        )
        axis.set_title(BEHAVIOR_NAMES[code])
        axis.yaxis.set_major_formatter(ticker.EngFormatter())
        _weekends(axis, series.index)
    ratio = pivot[4] / pivot[1].replace(0, np.nan)
    axes[2, 0].plot(pivot.index, ratio, color=COLORS[3])
    axes[2, 0].set_title("购买/浏览事件比（非路径转化）")
    axes[2, 0].yaxis.set_major_formatter(ticker.PercentFormatter(1))
    active = daily.groupby("event_date")["users"].max()
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.05,
        0.65,
        "四条行为线分面展示，避免浏览量盖住其他行为。\n"
        + "浅色背景为周末；不同面板纵轴量级不同。\n"
        + f"每日单类行为最大用户覆盖峰值：{active.max():,.0f}。",
        va="top",
        transform=axes[2, 1].transAxes,
        linespacing=1.7,
    )
    for axis in axes.flat[:5]:
        axis.xaxis.set_major_locator(dates.DayLocator(interval=7))
        axis.xaxis.set_major_formatter(dates.DateFormatter("%m-%d"))
        axis.grid(axis="y", alpha=0.15)
    return figure


def _weekends(axis: Axes, index: pd.DatetimeIndex) -> None:
    for day in index:
        if day.dayofweek >= 5:
            axis.axvspan(day, day + pd.Timedelta(days=1), color="#DDE7EF", alpha=0.4)


def _heatmaps(hourly: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(11, 8.5), layout="constrained")
    axes = figure.subplots(3, 2)
    for code, axis in zip([0, 1, 2, 3, 4], axes.flat, strict=False):
        selected = hourly if code == 0 else hourly.loc[hourly["behavior_type"].eq(code)]
        matrix = (
            selected.pivot_table(
                index="weekday", columns="hour", values="per_day", aggfunc="sum"
            )
            .reindex(index=range(1, 8), columns=range(24), fill_value=0)
            .fillna(0)
        )
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
        axis.set_title(
            "总体日均事件" if code == 0 else f"{BEHAVIOR_NAMES[code]}日均事件"
        )
        _heat_ticks(axis)
        figure.colorbar(image, ax=axis, shrink=0.7)
    raw = hourly.pivot_table(
        index=["weekday", "hour"],
        columns="behavior_type",
        values="events",
        aggfunc="sum",
    ).reindex(columns=[1, 2, 3, 4], fill_value=0)
    ratios = (raw[4] / raw[1].replace(0, np.nan)).unstack("hour")
    ratios = ratios.reindex(index=range(1, 8), columns=range(24))
    image = axes[2, 1].imshow(ratios, aspect="auto", cmap="RdYlBu_r")
    axes[2, 1].set_title("购买/浏览事件比")
    _heat_ticks(axes[2, 1])
    figure.colorbar(image, ax=axes[2, 1], format=ticker.PercentFormatter(1), shrink=0.7)
    return figure


def _heat_ticks(axis: Axes) -> None:
    axis.set_xticks([0, 6, 12, 18, 23], ["0", "6", "12", "18", "23"])
    axis.set_yticks(range(7), WEEKDAYS)
    axis.set_xlabel("小时")
    axis.set_ylabel("星期")
