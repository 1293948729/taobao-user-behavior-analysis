"""验证JSON写入和文件哈希。

测试入口：test_json_roundtrip_and_hash。
"""

import hashlib
from pathlib import Path

from taobao_user_behavior.storage import file_hash, read_json, save_json


def test_json_roundtrip_and_hash(tmp_path: Path) -> None:
    """中文内容保存后应能完整读回。

    Args:
        tmp_path: 本测试独立使用的临时目录。
    """
    output = tmp_path / "outputs/report.json"
    document = {"rows": 3, "说明": "保留重复行为"}

    save_json(output, document)

    assert read_json(output) == document
    assert file_hash(output) == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not output.with_suffix(".json.tmp").exists()
