"""验证运行目录只写入当前项目。

测试入口：test_runtime_paths_stay_local、test_external_output_is_rejected、
test_parent_traversal_is_rejected。
"""

from pathlib import Path

import pytest

from taobao_user_behavior.settings import load_runtime_settings


def test_runtime_paths_stay_local(tmp_path: Path) -> None:
    """相对运行目录应解析到当前项目中。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    settings = load_runtime_settings(tmp_path, Path("runs/check"))

    assert settings.root == tmp_path / "runs/check"
    assert settings.data.is_dir()
    assert settings.outputs.is_dir()
    assert settings.chunk_size == 250000


def test_external_output_is_rejected(tmp_path: Path) -> None:
    """外部路径应在创建目录前被拒绝。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    outside = tmp_path / "frozen_baseline"

    with pytest.raises(ValueError, match="project/runs"):
        load_runtime_settings(tmp_path / "project", outside)
    assert not outside.exists()


def test_parent_traversal_is_rejected(tmp_path: Path) -> None:
    """相对路径中的上级跳转不能越出项目。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    with pytest.raises(ValueError, match="project/runs"):
        load_runtime_settings(tmp_path, Path("runs/../../outside"))
