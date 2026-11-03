"""保存特征计算的固定SQL，所有边界值另行绑定。

主要常量：PAIR、USER、ITEM、CATEGORY、USER_CATEGORY、SAMPLES、COVERAGE。
"""

PAIR = """
CREATE TEMP TABLE pair_features AS
SELECT user_id, item_id,
    first(category_id ORDER BY event_time DESC, category_id ASC, category_missing ASC)
        AS category_id,
    first(category_missing ORDER BY event_time DESC,
        category_id ASC, category_missing ASC)
        AS category_missing,
    count(*)::BIGINT AS pair_events,
    count(*) FILTER (WHERE behavior_type = 1)::BIGINT AS pair_views,
    count(*) FILTER (WHERE behavior_type = 2)::BIGINT AS pair_favorites,
    count(*) FILTER (WHERE behavior_type = 3)::BIGINT AS pair_carts,
    count(*) FILTER (WHERE behavior_type = 4)::BIGINT AS pair_purchases,
    count(*) FILTER (WHERE event_time >= cutoff - INTERVAL 1 DAY)::BIGINT
        AS pair_events_1d,
    count(*) FILTER (WHERE event_time >= cutoff - INTERVAL 3 DAY)::BIGINT
        AS pair_events_3d,
    count(DISTINCT CAST(event_time AS DATE))::BIGINT AS pair_active_days,
    date_diff('hour', max(event_time), first(cutoff))::BIGINT AS pair_recency_hours,
    coalesce(date_diff('hour', max(event_time) FILTER (WHERE behavior_type = 4),
        first(cutoff)), -1)::BIGINT AS pair_purchase_recency_hours,
    bool_or(behavior_type = 4) AS pair_purchase_observed
FROM history GROUP BY user_id, item_id
"""

USER = """
CREATE TEMP TABLE user_features AS
SELECT user_id, count(*)::BIGINT AS user_events,
    count(*) FILTER (WHERE behavior_type = 1)::BIGINT AS user_views,
    count(*) FILTER (WHERE behavior_type = 2)::BIGINT AS user_favorites,
    count(*) FILTER (WHERE behavior_type = 3)::BIGINT AS user_carts,
    count(*) FILTER (WHERE behavior_type = 4)::BIGINT AS user_purchases,
    count(DISTINCT item_id)::BIGINT AS user_items,
    count(DISTINCT category_id) FILTER (WHERE NOT category_missing)::BIGINT
        AS user_categories,
    count(DISTINCT CAST(event_time AS DATE))::BIGINT AS user_active_days,
    count(*) FILTER (WHERE event_time >= cutoff - INTERVAL 1 DAY)::BIGINT
        AS user_events_1d,
    count(*) FILTER (WHERE event_time >= cutoff - INTERVAL 3 DAY)::BIGINT
        AS user_events_3d,
    date_diff('hour', max(event_time), first(cutoff))::BIGINT AS user_recency_hours
FROM history GROUP BY user_id
"""

ITEM = """
CREATE TEMP TABLE item_features AS
SELECT item_id, count(*)::BIGINT AS item_events,
    count(*) FILTER (WHERE behavior_type = 1)::BIGINT AS item_views,
    count(*) FILTER (WHERE behavior_type = 2)::BIGINT AS item_favorites,
    count(*) FILTER (WHERE behavior_type = 3)::BIGINT AS item_carts,
    count(*) FILTER (WHERE behavior_type = 4)::BIGINT AS item_purchases,
    count(DISTINCT user_id)::BIGINT AS item_users,
    count(DISTINCT CAST(event_time AS DATE))::BIGINT AS item_active_days,
    count(*) FILTER (WHERE event_time >= cutoff - INTERVAL 1 DAY)::BIGINT
        AS item_events_1d,
    count(*) FILTER (WHERE event_time >= cutoff - INTERVAL 3 DAY)::BIGINT
        AS item_events_3d
FROM history GROUP BY item_id
"""

CATEGORY = """
CREATE TEMP TABLE category_features AS
SELECT category_id, category_missing, count(*)::BIGINT AS category_events,
    count(*) FILTER (WHERE behavior_type = 1)::BIGINT AS category_views,
    count(*) FILTER (WHERE behavior_type = 2)::BIGINT AS category_favorites,
    count(*) FILTER (WHERE behavior_type = 3)::BIGINT AS category_carts,
    count(*) FILTER (WHERE behavior_type = 4)::BIGINT AS category_purchases,
    count(DISTINCT user_id)::BIGINT AS category_users,
    count(DISTINCT item_id)::BIGINT AS category_items
FROM history GROUP BY category_id, category_missing
"""

USER_CATEGORY = """
CREATE TEMP TABLE preference_features AS
SELECT user_id, category_id, category_missing, count(*)::BIGINT AS user_category_events
FROM history GROUP BY user_id, category_id, category_missing
"""

SAMPLES = """
CREATE TEMP VIEW samples AS
SELECT pair.user_id, pair.item_id, pair.category_id, bounds.cutoff,
    pair.pair_events, pair.pair_views, pair.pair_favorites, pair.pair_carts,
    pair.pair_purchases, pair.pair_events_1d, pair.pair_events_3d,
    pair.pair_active_days, pair.pair_recency_hours,
    pair.pair_purchase_recency_hours, pair.pair_purchase_observed,
    users.user_events, users.user_views, users.user_favorites, users.user_carts,
    users.user_purchases, users.user_items, users.user_categories,
    users.user_active_days, users.user_events_1d, users.user_events_3d,
    users.user_recency_hours,
    items.item_events, items.item_views, items.item_favorites, items.item_carts,
    items.item_purchases, items.item_users, items.item_active_days,
    items.item_events_1d, items.item_events_3d,
    categories.category_events, categories.category_views,
    categories.category_favorites, categories.category_carts,
    categories.category_purchases, categories.category_users,
    categories.category_items,
    preferences.user_category_events,
    preferences.user_category_events * 1.0 / users.user_events
        AS user_category_share,
    pair.pair_events * 1.0 / users.user_events AS pair_user_share,
    pair.pair_events * 1.0 / items.item_events AS pair_item_share,
    coalesce(pair.pair_carts * 1.0 / nullif(pair.pair_views, 0), 0)
        AS pair_cart_view_ratio,
    coalesce(pair.pair_favorites * 1.0 / nullif(pair.pair_views, 0), 0)
        AS pair_favorite_view_ratio,
    coalesce(pair.pair_purchases * 1.0 / nullif(pair.pair_views, 0), 0)
        AS pair_purchase_view_ratio,
    pair.pair_views = 0 AS pair_no_views,
    (isodow(bounds.cutoff) - 1)::BIGINT AS cutoff_weekday,
    pair.category_missing
FROM pair_features pair
JOIN user_features users USING (user_id)
JOIN item_features items USING (item_id)
JOIN category_features categories USING (category_id, category_missing)
JOIN preference_features preferences USING (user_id, category_id, category_missing)
CROSS JOIN bounds
"""

LABELED = """
SELECT samples.*, (future.user_id IS NOT NULL)::BIGINT AS label
FROM samples LEFT JOIN future USING (user_id, item_id)
ORDER BY user_id, item_id
"""

LATEST = "SELECT * FROM samples ORDER BY user_id, item_id"

COVERAGE = """
SELECT (SELECT count(*) FROM history) AS history_events,
    (SELECT count(*) FROM pair_features) AS candidates,
    (SELECT count(*) FROM future) AS future_purchase_pairs,
    (SELECT count(*) FROM pair_features JOIN future USING (user_id, item_id))
        AS positives
"""
