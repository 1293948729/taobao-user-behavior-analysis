"""读取少量原始记录，确认字段和文件格式。

主要接口：SampleProfile、inspect_source。
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from taobao_user_behavior.storage import RAW_COLUMNS, file_hash


@dataclass(frozen=True)
class SampleProfile:
    """保存输入文件信息和样本检查结果。

    Attributes:
        source_name: 压缩包文件名。
        source_sha256: 压缩包的SHA-256。
        csv_member: 实际读取的CSV成员名。
        uncompressed_bytes: CSV解压后的字节数。
        columns: 按原顺序排列的字段名。
        sample_limit: 要读取的最大行数。
        sampled_rows: 实际检查的行数。
        blank_counts: 样本内各字段的空值数。
        behavior_counts: 样本内各行为编码的出现次数。
        scope: 样本结果的适用范围。

    Examples:
        profile = inspect_source(Path("input.zip"), sample_limit=1000)
    """

    source_name: str
    source_sha256: str
    csv_member: str
    uncompressed_bytes: int
    columns: tuple[str, ...]
    sample_limit: int
    sampled_rows: int
    blank_counts: dict[str, int]
    behavior_counts: dict[str, int]
    scope: str = "First N records only; not a full-data quality assessment."


def inspect_source(archive_path: Path, sample_limit: int = 1000) -> SampleProfile:
    """检查指定CSV成员的字段和前若干行数据。

    Args:
        archive_path: 根目录含user_behavior.csv的ZIP文件。
        sample_limit: 要检查的最大行数，必须为正数。

    Returns:
        文件信息和样本计数，不改动原压缩包。

    Raises:
        ValueError: 样本上限、压缩成员、表头或行宽不符合要求。
    """
    if sample_limit < 1:
        raise ValueError("Sample limit must be positive.")
    missing: Counter[str] = Counter()
    behaviors: Counter[str] = Counter()
    sampled_rows = 0
    with ZipFile(archive_path) as archive:
        matches = [
            entry
            for entry in archive.infolist()
            if entry.filename == "user_behavior.csv" and not entry.is_dir()
        ]
        if len(matches) != 1:
            raise ValueError("Archive must contain exactly one root user_behavior.csv.")
        member = matches[0]
        with (
            archive.open(member) as binary,
            io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text,
        ):
            reader = csv.reader(text, strict=True)
            columns = tuple(next(reader, []))
            if columns != RAW_COLUMNS:
                raise ValueError(f"Unexpected columns: {columns}.")
            for record in islice(reader, sample_limit):
                if len(record) != len(RAW_COLUMNS):
                    raise ValueError(
                        f"Invalid CSV width at data row {sampled_rows + 1}.",
                    )
                sampled_rows += 1
                missing.update(
                    field
                    for field, value in zip(RAW_COLUMNS, record, strict=True)
                    if not value.strip()
                )
                behaviors.update([record[-1].strip()])
    return SampleProfile(
        source_name=archive_path.name,
        source_sha256=file_hash(archive_path),
        csv_member=member.filename,
        uncompressed_bytes=member.file_size,
        columns=columns,
        sample_limit=sample_limit,
        sampled_rows=sampled_rows,
        blank_counts={field: missing[field] for field in RAW_COLUMNS},
        behavior_counts=dict(behaviors),
    )
