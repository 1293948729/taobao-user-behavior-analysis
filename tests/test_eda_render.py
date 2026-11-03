"""验证图表导出和重绘时的来源检查。

测试入口：本模块test_开头的函数。
"""

from pathlib import Path
from shutil import copytree

import pytest

from taobao_user_behavior.analysis.pipeline import analyze, render_existing
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.storage import read_json
from taobao_user_behavior.visualization.style import find_chinese_font


def test_chart_export_and_table_tampering(eda_settings: Settings) -> None:
    """小数据也能导出完整图表，表被改动后拒绝重绘。

    Args:
        eda_settings: 已生成标准事件的微型运行。
    """
    try:
        find_chinese_font()
    except FileNotFoundError:
        pytest.skip("Rendering requires a local Chinese font.")
    project = eda_settings.root / "project"
    run_root = project / "artifacts/test"
    run_root.mkdir(parents=True)
    copytree(eda_settings.data, run_root / "data")
    copytree(eda_settings.outputs, run_root / "outputs")
    settings = Settings(run_root, threads=1, project_root=project)

    analyze(settings)

    assert len(list((settings.outputs / "eda/figures").glob("*.png"))) == 18
    assert len(read_json(settings.outputs / "eda/chart_manifest.json")["figures"]) == 18
    assert not list(settings.outputs.rglob("*.pdf"))
    summary = read_json(settings.outputs / "eda/summary.json")
    assert summary["aggregate_checks"]["passed"]
    table = settings.outputs / "eda/tables/activity.csv"
    table.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tables or analysis"):
        render_existing(settings)


def test_font_override_needs_real_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """字体环境变量不存在时仍允许回退到可用系统字体。

    Args:
        monkeypatch: pytest临时环境变量工具。
    """
    monkeypatch.setenv("TAOBAO_CJK_FONT", str(Path("nonexistent-font.ttf")))
    try:
        font = find_chinese_font()
    except FileNotFoundError:
        pytest.skip("No local Chinese font installed.")
    assert font.is_file()
