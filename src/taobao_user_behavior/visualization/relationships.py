"""绘制共现链路、跨小时关系及历史深度关联。

主要接口：relationship_figures。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.figure import Figure

from taobao_user_behavior.visualization.style import (
    BEHAVIOR_NAMES,
    COLORS,
    annotate_matrix,
    save_figure,
)


def relationship_figures(
    tables: dict[str, pd.DataFrame], folder: Path
) -> dict[str, Path]:
    """生成转化代理及数据质量诊断图。

    Args:
        tables: 共现、转移、分箱与阈值统计表。
        folder: 图表输出目录。

    Returns:
        图表编号及保存位置。
    """
    return {
        "13_funnel": save_figure(_funnel(tables["funnel"]), folder, "13_funnel"),
        "14_transitions": save_figure(
            _transitions(tables["transitions"]),
            folder,
            "14_transitions",
        ),
        "15_depth": save_figure(
            _depth(tables["depth_purchase"]),
            folder,
            "15_depth",
        ),
        "16_repeat_scope": save_figure(
            _repeat(tables["hourly_diagnostic"]),
            folder,
            "16_repeat_scope",
        ),
    }


def _funnel(funnel: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(10.8, 4.8), layout="constrained")
    axis = figure.add_subplot(111)
    base = max(1, int(funnel["pairs"].iloc[0]))
    bars = axis.barh(funnel["behavior"], funnel["pairs"] / base, color=COLORS)
    previous = funnel["pairs"].shift(1)
    retention = funnel["pairs"] / previous.replace(0, np.nan)
    labels = []
    for index, value in enumerate(funnel["pairs"]):
        suffix = (
            f" | 相对上级{retention.iloc[index]:.2%}"
            + f"，未覆盖{1 - retention.iloc[index]:.2%}"
            if index and pd.notna(retention.iloc[index])
            else ""
        )
        labels.append(f"{value:,.0f}" + suffix)
    axis.bar_label(bars, labels=labels, padding=5, fontsize=9)
    axis.set_xlim(0, 1.65)
    axis.set_xticks(np.arange(0, 1.01, 0.2))
    axis.xaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.invert_yaxis()
    axis.set_xlabel("逐级交集相对于有浏览用户商品对的占比（线性）")
    axis.set_title("共现链路：按大纲顺序取交集，不代表真实先后漏斗")
    return figure


def _transitions(transitions: pd.DataFrame) -> Figure:
    matrix = transitions.pivot(
        index="current_code",
        columns="next_code",
        values="probability",
    ).reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    figure = Figure(figsize=(8.8, 5.0), layout="constrained")
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="RdYlBu_r")
    axis.set_xticks(range(4), list(BEHAVIOR_NAMES.values()))
    axis.set_yticks(range(4), list(BEHAVIOR_NAMES.values()))
    annotate_matrix(axis, matrix, decimals=2, diverging=True)
    axis.set_xlabel("下一个有行为小时中的行为")
    axis.set_ylabel("当前小时中的行为")
    axis.set_title("跨小时行为集合关系（非逐条事件跳转）")
    figure.colorbar(image, ax=axis, format=ticker.PercentFormatter(1))
    return figure


def _depth(depth: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(11, 4.7), layout="constrained")
    axes = figure.subplots(1, 3)
    labels = ["0", "1", "2-3", "4-7", "8-15", "16+"]
    for axis, behavior, color in zip(
        axes,
        ["浏览", "加购", "收藏"],
        [COLORS[0], COLORS[2], COLORS[1]],
        strict=True,
    ):
        frame = (
            depth.loc[depth["behavior"].eq(behavior)].set_index("bin").reindex(range(6))
        )
        bars = axis.bar(np.arange(6), frame["purchase_rate"], color=color)
        axis.bar_label(
            bars,
            labels=[
                f"n={value:,.0f}" if pd.notna(value) else ""
                for value in frame["candidates"]
            ],
            rotation=90,
            padding=4,
            fontsize=7,
        )
        axis.set_xticks(range(6), labels, rotation=30)
        axis.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        maximum = frame["purchase_rate"].max()
        axis.set_ylim(0, max(0.01, maximum * 1.8) if pd.notna(maximum) else 0.01)
        axis.set_title(f"历史{behavior}次数")
        axis.set_xlabel("历史行为分箱")
    axes[0].set_ylabel("后续7日候选购买率")
    figure.suptitle("历史已交互用户商品对：行为深度与后续购买的关联")
    return figure


def _repeat(diagnostic: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(10.8, 4.4), layout="constrained")
    axes = figure.subplots(1, 2)
    labels = ["用户+商品+行为+小时", "用户+小时（仅对照）"]
    for axis, field, title in zip(
        axes,
        ["maximum", "over100"],
        ["单组最高次数", "超过100次的组数"],
        strict=True,
    ):
        bars = axis.bar(labels, diagnostic[field], color=[COLORS[0], COLORS[3]])
        axis.bar_label(bars, fmt="%.0f", padding=4)
        axis.set_title(title)
        axis.margins(y=0.2)
        axis.tick_params(axis="x", labelsize=9)
    return figure
