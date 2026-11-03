"""验证运行目录只写入当前项目。

测试入口：本模块test_开头的函数。
"""

from pathlib import Path
from typing import cast

import pytest

from taobao_user_behavior.settings import Settings, load_runtime_settings


def _project(directory: Path) -> Path:
    (directory / "configs").mkdir(parents=True)
    (directory / "configs/project.toml").write_text("[data]\n", encoding="utf-8")
    (directory / "configs/runtime.toml").write_text(
        '[artifacts]\nactive_run = "artifacts"\n',
        encoding="utf-8",
    )
    return directory


@pytest.mark.parametrize("relative", ["artifacts", "artifacts/check"])
def test_runtime_paths_stay_local(tmp_path: Path, relative: str) -> None:
    """相对运行目录应解析到当前项目中。

    Args:
        tmp_path: 本测试独立使用的临时目录。
        relative: 项目内的相对输出位置。
    """
    project = _project(tmp_path)
    settings = load_runtime_settings(project, Path(relative))

    assert settings.root == tmp_path / relative
    assert settings.data.is_dir()
    assert settings.outputs.is_dir()
    assert settings.chunk_size == 250000
    assert settings.hourly_repeat_limit == 100
    assert settings.project == tmp_path


def test_external_output_is_rejected(tmp_path: Path) -> None:
    """外部路径应在创建目录前被拒绝。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    outside = tmp_path / "frozen_baseline"

    with pytest.raises(ValueError, match="project/artifacts"):
        load_runtime_settings(tmp_path / "project", outside)
    assert not outside.exists()


@pytest.mark.parametrize(
    "relative",
    ["artifacts/../../outside", "artifacts/../src", "."],
)
def test_parent_traversal_is_rejected(tmp_path: Path, relative: str) -> None:
    """相对路径中的上级跳转不能越出项目。

    Args:
        tmp_path: 本测试独立使用的临时目录。
        relative: 不能作为产物目录的相对位置。
    """
    with pytest.raises(ValueError, match="project/artifacts"):
        load_runtime_settings(tmp_path, Path(relative))


def test_default_paths_ignore_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换工作目录后，默认输出仍位于指定项目内。

    Args:
        tmp_path: 本测试独立使用的临时目录。
        monkeypatch: pytest临时工作目录工具。
    """
    project = _project(tmp_path / "copied project")
    monkeypatch.chdir(tmp_path)

    settings = load_runtime_settings(project)

    assert settings.root == project / "artifacts"
    assert settings.project == project
    assert not (tmp_path / "artifacts").exists()


def test_project_lookup_uses_configuration(tmp_path: Path) -> None:
    """输出层级的名称不影响项目定位。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    project = _project(tmp_path / "analysis")
    settings = Settings(project / "artifacts/experiment")

    assert settings.project == project


def test_configured_output_must_be_relative(tmp_path: Path) -> None:
    """配置不能绑定当前机器的绝对产物路径。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    project = _project(tmp_path)
    absolute = (project / "artifacts").as_posix()
    (project / "configs/runtime.toml").write_text(
        f'[artifacts]\nactive_run = "{absolute}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative to project"):
        load_runtime_settings(project)


def test_directory_link_cannot_redirect_outputs(tmp_path: Path) -> None:
    """已有目录链接不能将产物写到项目之外。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    project = _project(tmp_path / "analysis")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (project / "artifacts").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks require permission on this system.")

    with pytest.raises(ValueError, match="project/artifacts"):
        load_runtime_settings(project)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("value", [0, -1, True, 100.5, "100"])
def test_hourly_limit_requires_positive_integer(tmp_path: Path, value: object) -> None:
    """错误类型或非正阈值应在处理前被拒绝。

    Args:
        tmp_path: 本测试独立使用的临时目录。
        value: 不合法的小时频次阈值。
    """
    with pytest.raises(ValueError, match="hourly_repeat_limit"):
        # cast不转换实际值，这里故意保留错误类型来检查配置校验。
        Settings(tmp_path, hourly_repeat_limit=cast(int, value))
