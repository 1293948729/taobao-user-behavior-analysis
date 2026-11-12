"""组织时间核验、仅训练特征筛选、负采样和固定清单投影。

主要接口：prepare_data。
"""

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from taobao_user_behavior.preparation.common import (
    DATASETS,
    KEYS,
    checked_inputs,
    feature_names,
    load_options,
    output_folder,
    write_table,
)
from taobao_user_behavior.preparation.distribution import (
    compare_distribution,
    distribution_reference,
)
from taobao_user_behavior.preparation.sampling import (
    class_weights,
    sample_training,
    sampling_comparison,
)
from taobao_user_behavior.preparation.selection import select_features
from taobao_user_behavior.preparation.temporal import review_windows
from taobao_user_behavior.settings import Settings
from taobao_user_behavior.source_inventory import source_fingerprints
from taobao_user_behavior.storage import file_hash, save_json
from taobao_user_behavior.visualization.preparation import draw_preparation


def _export_frame(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(".build.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(target)


def _selection_outputs(selection: dict, folder: Path) -> None:
    for name in ("profile", "decisions", "identities"):
        write_table(folder, f"feature_{name}", selection[name])
    for method in ("pearson", "spearman"):
        matrix = selection[method].rename_axis("feature").reset_index()
        write_table(folder, f"correlation_{method}", matrix)


def _sample_outputs(
    train: pd.DataFrame, settings: Settings, selected: list[str]
) -> dict:
    options = load_options(settings.project)
    folder = output_folder(settings, "outputs/preparation")
    destination = output_folder(settings, "data/prepared")
    comparisons = []
    chosen = None
    for ratio in sorted({20, 50, 100, options.negative_ratio}):
        sample, details = sample_training(train, replace(options, negative_ratio=ratio))
        coverage, distribution = sampling_comparison(train, sample, feature_names())
        write_table(folder, f"sampling_distribution_{ratio}", distribution)
        details["maximum_absolute_negative_smd"] = float(
            distribution["absolute_smd"].max()
        )
        for name, values in coverage.items():
            details[f"{name}_coverage"] = values["coverage"]
            details[f"{name}_before"] = values["before"]
            details[f"{name}_after"] = values["after"]
        comparisons.append(details)
        if ratio == options.negative_ratio:
            columns = [
                "user_id",
                "item_id",
                "category_id",
                "cutoff",
                *selected,
                "label",
            ]
            _export_frame(sample[columns], destination / "train_sampled.parquet")
            write_table(folder, "sampled_keys", sample[KEYS + ["label"]])
            chosen = details
    write_table(folder, "sampling_comparison", pd.DataFrame(comparisons))
    return {"chosen": chosen, "comparisons": comparisons}


def _project_datasets(
    settings: Settings, selected: list[str], reference: dict
) -> list[dict]:
    output = output_folder(settings, "data/prepared")
    folder = output_folder(settings, "outputs/preparation")
    distributions = []
    projected = []
    for name in DATASETS:
        frame = pd.read_parquet(settings.data / "features" / f"{name}.parquet")
        columns = ["user_id", "item_id", "category_id", "cutoff", *selected]
        if name != "latest":
            columns.append("label")
        elif "label" in frame:
            raise ValueError("Latest dataset must not have labels.")
        path = output / f"{name}.parquet"
        _export_frame(frame[columns], path)
        frame[columns].head(20).to_csv(
            folder / f"{name}_sample.csv", index=False, encoding="utf-8-sig"
        )
        distributions.append(
            compare_distribution(frame[feature_names()], reference, name)
        )
        projected.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": columns,
                "file": str(path.relative_to(settings.root)),
                "sha256": file_hash(path),
            }
        )
    write_table(
        folder, "feature_distributions", pd.concat(distributions, ignore_index=True)
    )
    return projected


def prepare_data(settings: Settings) -> dict:
    """准备后续建模数据，保留四份原始宽表且不训练模型。

    Args:
        settings: 本次运行目录和项目配置。

    Returns:
        当前来源、筛选规则、采样、权重及输出指纹。
    """
    started = perf_counter()
    manifest = checked_inputs(settings)
    options = load_options(settings.project)
    folder = output_folder(settings, "outputs/preparation")
    completion = folder / "manifest.json"
    completion.unlink(missing_ok=True)
    (folder / "verification.json").unlink(missing_ok=True)
    for name, table in review_windows(settings, manifest).items():
        write_table(folder, name, table)
    train = pd.read_parquet(settings.data / "features/train.parquet")
    names = feature_names()
    selection = select_features(train[names], options)
    _selection_outputs(selection, folder)
    selected = selection["selected"]
    frozen = {
        "fit_dataset": "full_natural_train",
        "fit_rows": len(train),
        "input_features": names,
        "selected_features": selected,
        "parameters": asdict(options),
        "labels_used_for_selection": False,
        "rule": "constant, duplicate, exact totals, same-entity Spearman redundancy",
        "preserved_signals": "time gaps, shares, ratios, active days and cross-entity",
        "priority": "field dictionary order",
        "model_validated": False,
    }
    save_json(folder / "selected_features.json", frozen)
    weights = class_weights(train["label"])
    save_json(folder / "class_weights.json", weights)
    sampling = _sample_outputs(train, settings, selected)
    reference = distribution_reference(train[names])
    del train
    projected = _project_datasets(settings, selected, reference)
    checked_inputs(settings)
    figures = output_folder(settings, "outputs/preparation/figures")
    draw_preparation(folder, figures)
    report = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "seconds": perf_counter() - started,
        "source_sha256": source_fingerprints(settings.project),
        "configuration_sha256": file_hash(settings.project / "configs/project.toml"),
        "feature_manifest_sha256": file_hash(
            settings.outputs / "features/manifest.json"
        ),
        "input_sha256": {row["name"]: row["sha256"] for row in manifest["datasets"]},
        "events_sha256": manifest["events_sha256"],
        "selection": frozen,
        "sampling": sampling,
        "weights": weights,
        "projected": projected,
        "model_training_performed": False,
        "input_files_unchanged": True,
        "output_sha256": {
            str(path.relative_to(settings.root)): file_hash(path)
            for parent in (folder, figures, settings.data / "prepared")
            for path in sorted(parent.glob("*"))
            if path.is_file() and path != completion
        },
    }
    save_json(completion, report)
    return report
