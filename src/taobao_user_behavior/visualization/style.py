"""统一中文字体、图表颜色与图片保存方式。

主要接口：configure_charts、save_figure、find_chinese_font。
"""

import os
from pathlib import Path

import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BEHAVIOR_NAMES = {1: "浏览", 2: "收藏", 3: "加购", 4: "购买"}
COLORS = ["#3B7093", "#5F9EA0", "#D4A35A", "#CF6549"]
WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]


def find_chinese_font() -> Path:
    """从显式配置或系统字体库查找中文字体。

    Returns:
        实际存在的字体路径。

    Raises:
        FileNotFoundError: 系统没有可用字体且未指定字体环境变量。
    """
    configured = os.getenv("TAOBAO_CJK_FONT")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    for family in [
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Songti SC",
    ]:
        try:
            candidate = font_manager.findfont(
                font_manager.FontProperties(family=[family]),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Set TAOBAO_CJK_FONT to a Chinese TrueType font file.")


def configure_charts() -> None:
    """设置中文字体与一致的冷暖配色，不修改全局系统字体。"""
    path = find_chinese_font()
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "text.color": "#203449",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: Figure, folder: Path, name: str) -> Path:
    """保存180DPI图片，返回图表路径。

    Args:
        figure: 已绘制完成的图。
        folder: 图片输出目录。
        name: 不含扩展名的内部文件名。

    Returns:
        已保存的PNG文件路径。
    """
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{name}.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    return destination


def annotate_matrix(
    axis: Axes,
    matrix: pd.DataFrame,
    decimals: int = 0,
    missing: str = "-",
    diverging: bool = False,
) -> None:
    """在百分比矩阵各格标注数值，缺失值用明确标记。

    Args:
        axis: 要写入文字的坐标轴。
        matrix: 与热图顺序相同的数值矩阵。
        decimals: 百分比的小数位数。
        missing: 空值标记，空字符串表示不写文字。
        diverging: 双向冷暖色阶的低值端也使用白字。
    """
    lower, upper = matrix.min().min(), matrix.max().max()
    for row, values in enumerate(matrix.to_numpy()):
        for column, value in enumerate(values):
            position = (value - lower) / (upper - lower) if upper > lower else 0.5
            dark_cell = position > 0.78 or (diverging and position < 0.2)
            axis.text(
                column,
                row,
                missing if pd.isna(value) else f"{value:.{decimals}%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if pd.notna(value) and dark_cell else "#203449",
            )
