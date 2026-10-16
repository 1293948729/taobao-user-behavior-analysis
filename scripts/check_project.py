"""运行格式、类型和测试检查，保存各工具的实际结果。

主要入口：main。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from taobao_user_behavior.settings import load_runtime_settings


def main() -> None:
    """检查业务代码、脚本和测试，并保存命令输出及源码哈希。

    Raises:
        SystemExit: 任一检查失败时返回非零退出码。
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    targets = ["src", "tests", "scripts", "run_project.py"]
    if (root / "dashboard.py").is_file():
        targets.append("dashboard.py")
    commands = [
        ["black", "--check", *targets],
        ["isort", "--check-only", *targets],
        ["flake8", *targets],
        ["ruff", "check", *targets],
        ["mypy", *targets],
        ["pylint", *targets],
        ["pytest", "-q"],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", *command],
            cwd=root,
            env={**os.environ, "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        result = {
            "command": [sys.executable, "-m", *command],
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    sources = sorted((root / "src").rglob("*.py"))
    sources.extend(sorted((root / "tests").rglob("*.py")))
    sources.extend(sorted((root / "scripts").rglob("*.py")))
    sources.extend(root / target for target in targets if target.endswith(".py"))
    document = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "passed": all(result["exit_code"] == 0 for result in results),
        "coverage": targets,
        "pylint_excludes_tests": False,
        "tool_versions": {
            command[0]: metadata.version(command[0]) for command in commands
        },
        "commands": results,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    destination = arguments.report or (
        load_runtime_settings(root).outputs / "checks" / "quality.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if document["passed"] else 1)


if __name__ == "__main__":
    main()
