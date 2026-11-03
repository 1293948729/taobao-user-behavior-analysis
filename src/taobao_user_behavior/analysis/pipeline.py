"""组织SQL探索分析、汇总导出和图表绘制。

主要接口：analyze、render_existing。
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from taobao_user_behavior.analysis.cohorts import label_tables, user_tables
from taobao_user_behavior.analysis.metrics import (
    metric_dictionary,
    overview_metrics,
    validate_aggregates,
)
from taobao_user_behavior.analysis.products import product_tables
from taobao_user_behavior.analysis.temporal import temporal_tables
from taobao_user_behavior.analysis.warehouse import build_warehouse, open_warehouse
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.source_inventory import source_fingerprints
from taobao_user_behavior.storage import file_hash, read_json, save_json
from taobao_user_behavior.visualization.charts import render_charts


def analyze(settings: Settings) -> dict:
    """构建分析库，导出汇总表和图表，保存来源及耗时。

    Args:
        settings: 已完成接入与质量检查的运行配置。

    Returns:
        总体指标、表索引、聚合核验和来源指纹。
    """
    started = perf_counter()
    build_warehouse(settings)
    with open_warehouse(settings) as connection:
        overview = overview_metrics(connection)
        checks = validate_aggregates(connection)
        tables = {
            **temporal_tables(connection),
            **user_tables(connection),
            **product_tables(connection),
            **label_tables(connection),
            "metric_dictionary": metric_dictionary(),
        }
    folder = settings.outputs / "eda/tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    project = settings.project
    document = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "overview": overview,
        "aggregate_checks": checks,
        "tables": {
            name: {"rows": len(frame), "columns": list(frame.columns)}
            for name, frame in tables.items()
        },
        "input_sha256": read_json(settings.outputs / "ingestion.json")["source_sha256"],
        "events_sha256": file_hash(settings.data / "events.parquet"),
        "database_sha256": file_hash(settings.data / "analysis.duckdb"),
        "table_sha256": {
            path.name: file_hash(path) for path in sorted(folder.glob("*.csv"))
        },
        "analysis_source_sha256": _analysis_sources(project),
        "settings": asdict(settings),
        "analysis_seconds": perf_counter() - started,
    }
    save_json(settings.outputs / "eda/summary.json", document)
    render_existing(settings)
    return document


def render_existing(settings: Settings) -> None:
    """验证已有汇总的来源后重新绘图，不重读原始CSV。

    Args:
        settings: 已有完整EDA结果的运行目录。

    Raises:
        ValueError: 汇总文件、分析源码或标准事件与记录不符。
    """
    project = settings.project
    document = read_json(settings.outputs / "eda/summary.json")
    folder = settings.outputs / "eda/tables"
    expected = document["table_sha256"]
    current = {path.name: file_hash(path) for path in sorted(folder.glob("*.csv"))}
    if expected != current or document["analysis_source_sha256"] != _analysis_sources(
        project
    ):
        raise ValueError("EDA tables or analysis code changed; rerun --stage eda.")
    if document["events_sha256"] != file_hash(settings.data / "events.parquet"):
        raise ValueError("Retained events changed; rerun --stage eda.")
    identifiers = {"user_id": str, "item_id": str, "category_id": str}
    tables = {
        path.stem: pd.read_csv(path, dtype=identifiers)
        for path in sorted(folder.glob("*.csv"))
    }
    charts = render_charts(tables, settings.outputs / "eda/figures")
    save_json(
        settings.outputs / "eda/chart_manifest.json",
        {
            "figures": {key: file_hash(path) for key, path in charts.items()},
            "table_sha256": current,
            "events_sha256": document["events_sha256"],
        },
    )


def _analysis_sources(project: Path) -> dict[str, str]:
    return {
        name: digest
        for name, digest in source_fingerprints(project).items()
        if len(Path(name).parts) >= 3
        and Path(name).parts[2] in {"analysis", "data", "settings.py", "storage.py"}
    }
