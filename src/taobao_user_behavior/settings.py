"""读取运行参数，统一数据和报告的保存位置。

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

    Examples:
        settings = load_runtime_settings(project)
    """

    root: Path
    chunk_size: int = 250000
    memory_limit: str = "3GB"
    threads: int = 4

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


def load_settings(root: Path) -> Settings:
    """读取配置并创建数据和报告目录。

    Args:
        root: 由运行入口确认过的输出目录。

    Returns:
        已检查参数范围的运行配置。

    Raises:
        ValueError: 分块行数或线程数不是正数。
    """
    configuration = Path(__file__).resolve().parents[2] / "configs/project.toml"
    with configuration.open("rb") as stream:
        document = tomllib.load(stream)
    settings = Settings(root=root.resolve(), **document["data"])
    if min(settings.chunk_size, settings.threads) <= 0:
        raise ValueError("Chunk and worker limits must be positive.")
    settings.data.mkdir(parents=True, exist_ok=True)
    settings.outputs.mkdir(parents=True, exist_ok=True)
    return settings


def load_runtime_settings(project: Path, run_root: Path | None = None) -> Settings:
    """确定运行目录，避免把结果写到项目之外。

    Args:
        project: 包含configs目录的项目根目录。
        run_root: 可选的输出目录，可以是绝对路径或项目内相对路径。

    Returns:
        输出位置在本项目runs目录内的配置。

    Raises:
        ValueError: 输出目录不在本项目runs目录下。
    """
    project = project.resolve()
    if run_root is None:
        with (project / "configs/runtime.toml").open("rb") as stream:
            run_root = Path(tomllib.load(stream)["artifacts"]["active_run"])
    selected = (project / run_root).resolve()
    allowed = (project / "runs").resolve()
    if selected == allowed or allowed not in selected.parents:
        raise ValueError("Run root must be a child directory of project/runs.")
    return load_settings(selected)
