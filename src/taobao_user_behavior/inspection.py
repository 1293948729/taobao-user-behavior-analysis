"""保留样本检查公共入口，实际读取逻辑位于data包。"""

from taobao_user_behavior.data.inspection import SampleProfile, inspect_source

__all__ = ["SampleProfile", "inspect_source"]
