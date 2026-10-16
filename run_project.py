"""从命令行或PyCharm启动数据检查。

主要接口：run_stage、main。
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from taobao_user_behavior.inspection import inspect_source
from taobao_user_behavior.settings import Settings, load_runtime_settings
from taobao_user_behavior.source_inventory import source_fingerprints
from taobao_user_behavior.storage import save_json

STAGES = {
    "inspect": ("inspection", "inspect_source"),
    "audit": ("quality", "audit"),
}


def run_stage(name: str, settings: Settings, arguments: argparse.Namespace) -> None:
    """执行样本检查或全量清洗，将结果写入当前运行目录。

    Args:
        name: 阶段名称，取inspect或audit。
        settings: 本次运行的参数与输出目录。
        arguments: 输入文件和样本条数等命令行参数。
    """
    if name == "inspect":
        profile = inspect_source(arguments.input, arguments.sample_rows)
        save_json(settings.outputs / "input_profile.json", asdict(profile))
        logging.info("Inspected %s records; this is a sample.", profile.sampled_rows)
        return
    module_name, function_name = STAGES[name]
    module = importlib.import_module(f"taobao_user_behavior.{module_name}")
    if name == "audit":
        module.ingest(arguments.input, settings)
    getattr(module, function_name)(settings)


def main() -> None:
    """读取命令行参数，执行所选阶段并保存运行记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=os.getenv("TAOBAO_INPUT"))
    parser.add_argument("--stage", choices=list(STAGES), default="inspect")
    parser.add_argument("--sample-rows", type=int, default=1000)
    parser.add_argument("--run-root", type=Path)
    arguments = parser.parse_args()
    if arguments.input is None:
        parser.error("Provide --input or set TAOBAO_INPUT to the local ZIP path.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project = Path(__file__).resolve().parent
    settings = load_runtime_settings(project, arguments.run_root)
    started = time.perf_counter()
    run_stage(arguments.stage, settings, arguments)
    save_json(
        settings.outputs / "last_execution.json",
        {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "stage": arguments.stage,
            "seconds": time.perf_counter() - started,
            "python_executable": sys.executable,
            "source_sha256": source_fingerprints(project),
            "runtime_settings": asdict(settings),
        },
    )


if __name__ == "__main__":
    main()
