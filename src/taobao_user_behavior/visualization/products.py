"""绘制商品和类目的双排名、长尾及四象限。

主要接口：product_figures。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.figure import Figure

from taobao_user_behavior.visualization.style import COLORS, save_figure


def product_figures(tables: dict[str, pd.DataFrame], folder: Path) -> dict[str, Path]:
    """生成商品和类目维度图表。

    Args:
        tables: 商品类目排名与四象限汇总表。
        folder: 图片输出目录。

    Returns:
        图表编号及保存位置。
    """
    return {
        "09_category_flow": save_figure(
            _ranking(tables["category_flow"], "category_id", "events"),
            folder,
            "09_category_flow",
        ),
        "09_category_conversion": save_figure(
            _ranking(tables["category_conversion"], "category_id", "purchase_coverage"),
            folder,
            "09_category_conversion",
        ),
        "10_item_flow": save_figure(
            _ranking(tables["item_flow"], "item_id", "events"),
            folder,
            "10_item_flow",
        ),
        "10_item_conversion": save_figure(
            _ranking(tables["item_conversion"], "item_id", "purchase_coverage"),
            folder,
            "10_item_conversion",
        ),
        "11_category_tail": save_figure(
            _tail(tables["category_tail"]),
            folder,
            "11_category_tail",
        ),
        "12_quadrants": save_figure(_quadrants(tables), folder, "12_quadrants"),
    }


def _ranking(frame: pd.DataFrame, identity: str, field: str) -> Figure:
    figure = Figure(figsize=(9, 7.6), layout="constrained")
    axis = figure.add_subplot(111)
    if frame.empty:
        axis.text(0.1, 0.5, "没有达到最小支持量的项目", transform=axis.transAxes)
        axis.set_axis_off()
        return figure
    selected = frame.iloc[::-1]
    labels = selected[identity].astype(str)
    if identity == "item_id":
        labels = labels + " / 类" + selected["category_id"].astype(str)
    bars = axis.barh(
        np.arange(len(selected)),
        selected[field],
        color=COLORS[0] if field == "events" else COLORS[3],
    )
    axis.set_yticks(np.arange(len(selected)), labels, fontsize=9)
    values = [
        (
            f"{row.events:,.0f} | {row.purchase_coverage:.1%} | n={row.viewed_pairs:,}"
            if pd.notna(row.purchase_coverage)
            else f"{row.events:,.0f} | -"
        )
        for row in selected.itertuples()
    ]
    axis.bar_label(bars, labels=values, padding=3, fontsize=8)
    axis.margins(x=0.65)
    axis.set_xlabel("交互量" if field == "events" else "浏览商品对购买覆盖（n≥20）")
    if field == "purchase_coverage":
        axis.xaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.set_title("条末：交互次数 | 购买覆盖 | 浏览商品对分母", fontsize=10)
    return figure


def _tail(tail: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(10.8, 4.5), layout="constrained")
    axis = figure.add_subplot(111)
    if tail.empty:
        axis.text(0.1, 0.5, "无非缺失类目", transform=axis.transAxes)
        return figure
    ranks = tail["ranking"] / len(tail)
    axis.plot(ranks, tail["cumulative_events"], label="交互事件", color=COLORS[0])
    axis.plot(ranks, tail["cumulative_purchases"], label="购买事件", color=COLORS[3])
    cutoff = max(1, int(np.ceil(len(tail) * 0.2)))
    point = tail.iloc[cutoff - 1]
    axis.axvline(cutoff / len(tail), linestyle=":", color="#AAB6C1")
    axis.annotate(
        f"Top20%类目\n交互{point.cumulative_events:.1%}\n"
        + f"购买{point.cumulative_purchases:.1%}",
        (cutoff / len(tail), point.cumulative_events),
        xytext=(25, -80),
        textcoords="offset points",
    )
    axis.set_xlabel("类目按交互量降序的累计占比")
    axis.set_ylabel("累计事件占比")
    axis.xaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.set_title("类目长尾：检查集中度，不预设符合二八规律")
    axis.legend()
    return figure


def _quadrants(tables: dict[str, pd.DataFrame]) -> Figure:
    figure = Figure(figsize=(10.8, 5.0), layout="constrained")
    axis = figure.add_subplot(111)
    frame = tables["quadrant_sample"]
    limits = tables["quadrant_thresholds"].iloc[0]
    if frame.empty:
        axis.text(0.1, 0.5, "没有达到支持量的商品", transform=axis.transAxes)
        return figure
    for group, color in zip(
        ["明星型", "流量型", "潜力型", "长尾型"],
        [COLORS[3], COLORS[0], "#D4A35A", "#8DA8B8"],
        strict=True,
    ):
        selected = frame.loc[frame["quadrant"].eq(group)]
        axis.scatter(
            selected["views"],
            selected["purchase_coverage"],
            s=12,
            alpha=0.45,
            label=group,
            color=color,
            rasterized=True,
        )
    axis.axvline(limits.view_threshold, color="#334155", linestyle="--")
    axis.axhline(limits.rate_threshold, color="#334155", linestyle="--")
    axis.set_xscale("log")
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    axis.set_xlabel("浏览事件数（曝光代理，对数）")
    axis.set_ylabel("浏览商品对购买覆盖")
    axis.set_title(
        f"商品四象限：全量分层{limits.eligible_items:,.0f}件，"
        + f"绘制{len(frame):,}件确定性样本",
    )
    axis.legend(ncol=2)
    return figure
