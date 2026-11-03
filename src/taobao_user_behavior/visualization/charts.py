"""按报告章节生成全部静态图表。

主要接口：render_charts。
"""

from pathlib import Path

import pandas as pd

from taobao_user_behavior.visualization.products import product_figures
from taobao_user_behavior.visualization.relationships import relationship_figures
from taobao_user_behavior.visualization.style import configure_charts
from taobao_user_behavior.visualization.temporal import temporal_figures
from taobao_user_behavior.visualization.users import user_figures


def render_charts(tables: dict[str, pd.DataFrame], folder: Path) -> dict[str, Path]:
    """绘制18张中文图，数据只来自已保存的汇总表。

    Args:
        tables: SQL分析阶段导出的表。
        folder: 本次运行的图片目录。

    Returns:
        已保存的图表名称及对应路径。
    """
    configure_charts()
    return {
        **temporal_figures(tables, folder),
        **user_figures(tables, folder),
        **product_figures(tables, folder),
        **relationship_figures(tables, folder),
    }
