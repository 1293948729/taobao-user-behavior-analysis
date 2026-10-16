"""统计项目源码指纹和模块依赖，不执行被检查的代码。

主要接口：source_fingerprints、local_imports。
"""

import ast
from pathlib import Path

from taobao_user_behavior.storage import file_hash


def source_fingerprints(project: Path) -> dict[str, str]:
    """计算当前源码、脚本和测试文件的哈希。

    Args:
        project: 待检查的项目根目录。

    Returns:
        相对文件路径及对应SHA-256。
    """
    paths = [
        file
        for directory in ("src", "scripts", "tests")
        for file in (project / directory).rglob("*.py")
    ]
    paths.extend(project.glob("*.py"))
    return {str(file.relative_to(project)): file_hash(file) for file in sorted(paths)}


def local_imports(path: Path) -> set[str]:
    """从语法树中读取本项目的直接导入关系。

    Args:
        path: 要检查的Python文件。

    Returns:
        该文件导入的本地模块名集合。
    """
    dependencies: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        dependencies.update(
            module.split(".")[1]
            for module in modules
            if module.startswith("taobao_user_behavior.")
        )
    return dependencies
