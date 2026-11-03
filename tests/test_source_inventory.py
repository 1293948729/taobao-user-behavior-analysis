"""检查源码拆包后的指纹和依赖识别不会遗漏子包。

测试入口：本模块test_开头的函数。
"""

from pathlib import Path

from taobao_user_behavior.source_inventory import local_imports, source_fingerprints


def test_nested_module_imports_are_not_flattened(tmp_path: Path) -> None:
    """依赖保留data模块的完整名称，不只识别顶层data。

    Args:
        tmp_path: 当前测试独立目录。
    """
    module = tmp_path / "nested.py"
    module.write_text(
        "from taobao_user_behavior.data.normalization import normalize_events\n",
        encoding="utf-8",
    )

    assert local_imports(module) == {"data.normalization"}


def test_nested_source_fingerprint_is_included(tmp_path: Path) -> None:
    """细分的子包文件应进入指纹清单。

    Args:
        tmp_path: 当前测试独立目录。
    """
    module = tmp_path / "src/taobao_user_behavior/analysis/metric.py"
    module.parent.mkdir(parents=True)
    module.write_text('"""分析模块。"""\n', encoding="utf-8")

    fingerprints = source_fingerprints(tmp_path)

    assert str(module.relative_to(tmp_path)) in fingerprints
