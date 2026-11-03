"""提供与正式宽表一一对应的字段名称、类型、口径和缺失处理。

主要接口：field_dictionary。
"""

GROUPS = {
    "pair": "用户商品交互",
    "user": "用户活跃与广度",
    "item": "商品热度",
    "category": "类目热度",
}
ENTITIES = {"pair": "用户商品对", "user": "用户", "item": "商品", "category": "类目"}
BEHAVIORS = [
    ("events", "全部行为", "全部事件"),
    ("views", "浏览", "behavior_type=1"),
    ("favorites", "收藏", "behavior_type=2"),
    ("carts", "加购", "behavior_type=3"),
    ("purchases", "购买", "behavior_type=4"),
]
EXTRAS = {
    "pair": [
        ("events_1d", "近1天行为数", "最近1天全部事件数"),
        ("events_3d", "近3天行为数", "最近3天全部事件数"),
        ("active_days", "活跃天数", "有事件的不同自然日数"),
        ("recency_hours", "最近行为距截点小时数", "cutoff减最近事件小时，范围1至168"),
        (
            "purchase_recency_hours",
            "最近购买距截点小时数",
            "cutoff减最近购买小时；没有历史购买返回-1",
        ),
        ("purchase_observed", "历史购买标记", "历史存在购买事件为true，否则false"),
    ],
    "user": [
        ("items", "交互商品数", "不同item_id数量"),
        ("categories", "交互类目数", "不同非缺失category_id数量"),
        ("active_days", "活跃天数", "有事件的不同自然日数"),
        ("events_1d", "近1天行为数", "最近1天全部事件数"),
        ("events_3d", "近3天行为数", "最近3天全部事件数"),
        ("recency_hours", "最近行为距截点小时数", "cutoff减最近事件小时，范围1至168"),
    ],
    "item": [
        ("users", "交互用户数", "不同user_id数量"),
        ("active_days", "活跃天数", "有事件的不同自然日数"),
        ("events_1d", "近1天行为数", "最近1天全部事件数"),
        ("events_3d", "近3天行为数", "最近3天全部事件数"),
    ],
    "category": [
        ("users", "交互用户数", "不同user_id数量"),
        ("items", "交互商品数", "不同item_id数量"),
    ],
}
OTHER = [
    ("user_category_events", "用户在候选类目的行为数", "该用户在候选类目的事件数"),
    (
        "user_category_share",
        "用户类目偏好占比",
        "user_category_events / user_events，范围(0,1]",
    ),
    ("pair_user_share", "商品对占用户行为比", "pair_events / user_events，范围(0,1]"),
    ("pair_item_share", "商品对占商品行为比", "pair_events / item_events，范围(0,1]"),
    ("pair_cart_view_ratio", "加购浏览事件比", "pair_carts / pair_views；零分母返回0"),
    (
        "pair_favorite_view_ratio",
        "收藏浏览事件比",
        "pair_favorites / pair_views；零分母返回0",
    ),
    (
        "pair_purchase_view_ratio",
        "购买浏览事件比",
        "pair_purchases / pair_views；零分母返回0",
    ),
    ("pair_no_views", "无历史浏览标记", "pair_views为0时true，否则false"),
    ("cutoff_weekday", "截点星期", "cutoff星期编码，周一0至周日6"),
    ("category_missing", "候选类目缺失标记", "候选对最近观察类目缺失时true，否则false"),
]


def _feature(name: str, group: str, title: str, calculation: str) -> dict[str, str]:
    is_flag = name in {"pair_purchase_observed", "pair_no_views", "category_missing"}
    is_ratio = name.endswith(("_share", "_ratio"))
    window = "[cutoff-7天, cutoff)"
    if name.endswith("events_1d"):
        window = "[cutoff-1天, cutoff)"
    elif name.endswith("events_3d"):
        window = "[cutoff-3天, cutoff)"
    elif name == "cutoff_weekday":
        window = "cutoff时刻"
    missing = "无缺失；不存在的行为计0"
    if name == "pair_purchase_recency_hours":
        missing = "无历史购买为-1，配合pair_purchase_observed识别"
    elif name.endswith("_ratio"):
        missing = "无浏览时置0，配合pair_no_views识别；不解释为转化概率"
    elif is_ratio:
        missing = "分母来自候选历史，严格大于0"
    elif name == "category_missing":
        missing = "缺失类目归入独立缺失组，不丢弃样本"
    elif name.endswith("_recency_hours"):
        missing = "候选历史内有事件，取值1至168小时"
    elif name.endswith("_active_days"):
        missing = "候选历史内有事件，取值1至7天"
    elif name == "cutoff_weekday":
        missing = "无缺失；周一0至周日6"
    elif is_flag:
        missing = "无缺失；布尔值true或false"
    return {
        "name": name,
        "title": title,
        "role": "feature",
        "group": group,
        "dtype": "BOOLEAN" if is_flag else "DOUBLE" if is_ratio else "BIGINT",
        "window": window,
        "calculation": calculation,
        "missing": missing,
        "description": "按各集合自己的历史窗口计算，事件重复按保留次数累计",
    }


def field_dictionary() -> list[dict[str, str]]:
    """按实际Parquet列顺序返回全部字段元数据。

    Returns:
        四个标识或截点、四十八个特征和一个监督标签的独立说明。
    """
    rows = [
        {
            "name": name,
            "title": title,
            "role": "metadata",
            "group": "样本标识与截点",
            "dtype": dtype,
            "window": "历史内观察" if name == "category_id" else "样本标识",
            "calculation": calculation,
            "missing": missing,
            "description": "不作为本版数值输入；ID始终保留字符串",
        }
        for name, title, dtype, calculation, missing in [
            ("user_id", "用户标识", "VARCHAR", "历史候选用户", "不允许缺失"),
            ("item_id", "商品标识", "VARCHAR", "历史候选商品", "不允许缺失"),
            (
                "category_id",
                "候选对类目",
                "VARCHAR",
                "历史最近事件类目；同小时类目ID升序，仍相同时非缺失优先",
                "缺失保留__missing__并以category_missing区分",
            ),
            ("cutoff", "历史标签分界", "TIMESTAMP", "历史排除、标签包含", "不允许缺失"),
        ]
    ]
    for prefix, group in GROUPS.items():
        rows.extend(
            _feature(
                f"{prefix}_{suffix}",
                group,
                f"{ENTITIES[prefix]}{title}事件数",
                f"同一{ENTITIES[prefix]}在窗口内的{condition}计数",
            )
            for suffix, title, condition in BEHAVIORS
        )
        rows.extend(
            _feature(f"{prefix}_{suffix}", group, title, calculation)
            for suffix, title, calculation in EXTRAS[prefix]
        )
    rows.extend(
        _feature(name, "偏好占比与辅助标记", title, calculation)
        for name, title, calculation in OTHER
    )
    rows.append(
        {
            "name": "label",
            "title": "未来七天购买标签",
            "role": "label",
            "group": "监督标签",
            "dtype": "BIGINT",
            "window": "[cutoff, cutoff+7天)",
            "calculation": "候选用户商品对未来窗口出现购买事件为1，否则0",
            "missing": "latest不包含此列，不能将尚未观察的未来填0",
            "description": "目标变量；未来新出现且历史未交互的商品对不进入候选",
        }
    )
    return rows
