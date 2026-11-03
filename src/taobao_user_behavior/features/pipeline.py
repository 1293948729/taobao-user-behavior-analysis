"""按各自历史窗口导出正式特征，保留标签覆盖和来源证据。

主要接口：prepare_window、build_features。
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import duckdb
import pandas as pd

from taobao_user_behavior.features import queries
from taobao_user_behavior.features.dictionary import field_dictionary
from taobao_user_behavior.features.validation import validate_sample
from taobao_user_behavior.features.windows import FeatureWindow, plan_windows
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.source_inventory import source_fingerprints
from taobao_user_behavior.storage import connect, file_hash, save_json


def prepare_window(
    connection: duckdb.DuckDBPyConnection, window: FeatureWindow
) -> None:
    """在独立连接中建立一套历史聚合、未来购买对及样本视图。

    Args:
        connection: 含events视图且尚未建立窗口临时表的连接。
        window: 已验证的七日窗口，latest没有未来标签。
    """
    connection.execute(
        """
        CREATE TEMP TABLE bounds AS SELECT ?::TIMESTAMP AS history_start,
            ?::TIMESTAMP AS cutoff, ?::TIMESTAMP AS label_end
        """,
        [window.history_start, window.cutoff, window.label_end],
    )
    connection.execute(
        """
        CREATE TEMP TABLE history AS
        SELECT events.*, bounds.cutoff FROM events CROSS JOIN bounds
        WHERE event_time >= history_start AND event_time < cutoff
        """,
    )
    connection.execute(
        """
        CREATE TEMP TABLE future AS
        SELECT DISTINCT user_id, item_id FROM events CROSS JOIN bounds
        WHERE event_time >= cutoff AND event_time < label_end AND behavior_type = 4
        """,
    )
    for query in [
        queries.PAIR,
        queries.USER,
        queries.ITEM,
        queries.CATEGORY,
        queries.USER_CATEGORY,
        queries.SAMPLES,
    ]:
        connection.execute(query)


def _export_window(settings: Settings, window: FeatureWindow) -> dict:
    folder = settings.data / "features"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{window.name}.build.parquet"
    target = folder / f"{window.name}.parquet"
    with connect(settings) as connection:
        connection.execute(
            "SET temp_directory = ?", [str(settings.data / "feature_spill")]
        )
        prepare_window(connection, window)
        counts = connection.execute(queries.COVERAGE).fetchone()
        if counts is None or counts[1] == 0:
            raise ValueError(f"No historical candidates in {window.name}.")
        query = queries.LATEST if window.label_end is None else queries.LABELED
        connection.sql(query).write_parquet(str(temporary), compression="zstd")
    checks = validate_sample(temporary, window.label_end is not None, settings)
    temporary.replace(target)
    with duckdb.connect() as connection:
        sample = connection.read_parquet(str(target)).limit(20).df()
    sample.to_csv(
        settings.outputs / f"features/{window.name}_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    history_events, candidates, future_pairs, positives = map(int, counts)
    supervised = window.label_end is not None
    return {
        **asdict(window),
        "history_events": history_events,
        "candidates": candidates,
        "positives": positives if supervised else None,
        "negatives": candidates - positives if supervised else None,
        "positive_rate": positives / candidates if supervised else None,
        "future_purchase_pairs": future_pairs if supervised else None,
        "future_pairs_outside_history": (
            future_pairs - positives if supervised else None
        ),
        "future_pair_coverage": positives / future_pairs if future_pairs else None,
        "file": str(target.relative_to(settings.root)),
        "sha256": file_hash(target),
        "checks": checks,
    }


def build_features(settings: Settings) -> dict:
    """从完整接入的标准事件生成四套正式特征及字段、来源说明。

    Args:
        settings: 标准事件位置、资源限制及当前项目位置。

    Returns:
        四套样本的行数、标签覆盖、全表检查及来源指纹。

    Raises:
        ValueError: 事件为空、完整日不足或任一历史候选为空。
    """
    started = perf_counter()
    with connect(settings) as connection:
        bounds = connection.execute(
            "SELECT min(event_time), max(event_time) FROM events"
        ).fetchone()
    if bounds is None or bounds[0] is None:
        raise ValueError("Features require retained events.")
    windows = plan_windows(bounds[0], bounds[1])
    folder = settings.outputs / "features"
    folder.mkdir(parents=True, exist_ok=True)
    dictionary = field_dictionary()
    pd.DataFrame(dictionary).to_csv(
        folder / "field_dictionary.csv", index=False, encoding="utf-8-sig"
    )
    save_json(folder / "field_dictionary.json", dictionary)
    document = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "history_days": 7,
        "label_days": 7,
        "input_features": sum(row["role"] == "feature" for row in dictionary),
        "metadata_fields": sum(row["role"] == "metadata" for row in dictionary),
        "label_fields": sum(row["role"] == "label" for row in dictionary),
        "events_sha256": file_hash(settings.data / "events.parquet"),
        "source_sha256": feature_sources(settings.project),
        "datasets": [_export_window(settings, window) for window in windows],
        "seconds": perf_counter() - started,
    }
    save_json(folder / "manifest.json", document)
    return document


def feature_sources(project: Path) -> dict[str, str]:
    """记录参与正式特征构建的源码指纹。

    Args:
        project: 当前项目根。

    Returns:
        特征、标准事件、配置及存储模块的相对路径和SHA。
    """
    return {
        name: digest
        for name, digest in source_fingerprints(project).items()
        if len(Path(name).parts) >= 3
        and Path(name).parts[2] in {"features", "data", "settings.py", "storage.py"}
    }
