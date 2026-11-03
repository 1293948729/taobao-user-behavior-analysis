"""读取运行参数，统一处理数据和分析结果的保存位置。

主要接口：Settings、load_settings、load_runtime_settings。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class Settings:
    """保存一次运行使用的路径和资源限制。

    Attributes:
        root: 本次运行的输出根目录。
        chunk_size: 每次读取的最大行数。
        memory_limit: 统计查询可使用的内存。
        threads: 统计查询的线程数。
        hourly_repeat_limit: 同一用户、商品、行为在一小时内允许保留的次数。
        project_root: 可选的项目位置，避免运行目录重名影响定位。

    Examples:
        settings = load_runtime_settings(project)
    """

    root: Path
    chunk_size: int = 250000
    memory_limit: str = "3GB"
    threads: int = 4
    hourly_repeat_limit: int = 100
    project_root: Path | None = None

    def __post_init__(self) -> None:
        """检查计数参数，拒绝布尔值、小数和非正数。

        Raises:
            ValueError: 分块行数、线程数或小时重复阈值不是正整数。
        """
        for name in ("chunk_size", "threads", "hourly_repeat_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def data(self) -> Path:
        """取得处理后数据的保存目录。

        Returns:
            运行根目录下的data目录。
        """
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        """取得检查报告的保存目录。

        Returns:
            运行根目录下的outputs目录。
        """
        return self.root / "outputs"

    @property
    def project(self) -> Path:
        """从配置文件定位项目，不依赖包的安装位置。

        Returns:
            包含configs/project.toml的项目根。

        Raises:
            ValueError: 无法从输出位置找到项目配置。
        """
        if self.project_root is not None:
            return self.project_root.resolve()
        resolved = self.root.resolve()
        for parent in [resolved, *resolved.parents]:
            if (parent / "configs/project.toml").is_file():
                return parent
        raise ValueError("Project root requires configs/project.toml.")


def load_settings(root: Path, project: Path | None = None) -> Settings:
    """读取配置并创建处理数据和分析结果目录。

    Args:
        root: 由运行入口确认过的输出目录。
        project: 配置所在的项目，未指定时从运行根定位。

    Returns:
        已检查参数范围的运行配置。

    Raises:
        ValueError: 输出目录不在artifacts内，或计数参数不是正整数。
    """
    selected_project = project if project is not None else Settings(root).project
    configuration = selected_project / "configs/project.toml"
    with configuration.open("rb") as stream:
        document = tomllib.load(stream)
    settings = Settings(
        root=_artifact_root(selected_project.resolve(), root),
        project_root=selected_project.resolve(),
        **document["data"],
    )
    settings.data.mkdir(parents=True, exist_ok=True)
    settings.outputs.mkdir(parents=True, exist_ok=True)
    return settings


def load_runtime_settings(project: Path, run_root: Path | None = None) -> Settings:
    """确定运行目录，避免把结果写到项目之外。

    Args:
        project: 包含configs目录的项目根目录。
        run_root: 可选的输出目录，可以是绝对路径或项目内相对路径。

    Returns:
        输出位置在本项目artifacts目录内的配置。

    Raises:
        ValueError: 配置使用绝对路径，或输出位置越出本项目artifacts目录。
    """
    project = project.resolve()
    if run_root is None:
        with (project / "configs/runtime.toml").open("rb") as stream:
            run_root = Path(tomllib.load(stream)["artifacts"]["active_run"])
        if run_root.is_absolute():
            raise ValueError("Configured output directory must be relative to project.")
    return load_settings(_artifact_root(project, run_root), project)


def _artifact_root(project: Path, root: Path) -> Path:
    allowed = (project / "artifacts").resolve()
    selected = (project / root).resolve()
    if allowed == project or not allowed.is_relative_to(project):
        raise ValueError("Output directory must stay within project/artifacts.")
    if not selected.is_relative_to(allowed):
        raise ValueError("Output directory must stay within project/artifacts.")
    for directory in ["data", "outputs"]:
        if not (selected / directory).resolve().is_relative_to(selected):
            raise ValueError("Output directory must stay within project/artifacts.")
    return selected
