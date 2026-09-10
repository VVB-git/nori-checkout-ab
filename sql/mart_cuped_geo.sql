-- Pre-period vs geo-rollout window, user grain, for CUPED / DiD.
-- Pre-period: sessions with experiment_id IS NULL.
-- Does not read experiment_truth.

WITH geo_variant AS (
    SELECT
        u.user_id,
        u.region_id,
        r.name AS region_name,
        r.geo_rollout_flag,
        CASE WHEN r.geo_rollout_flag = 1 THEN 'treatment' ELSE 'control' END AS variant
    FROM users u
    INNER JOIN regions r
        ON r.region_id = u.region_id
),
pre_orders AS (
    SELECT
        o.user_id,
        SUM(o.gmv) AS gmv,
        SUM(o.fee_amount) AS fee_amount,
        COUNT(*) AS n_orders
    FROM orders o
    WHERE o.experiment_id IS NULL
    GROUP BY o.user_id
),
pre_exposed AS (
    SELECT user_id
    FROM exposed_users
    WHERE experiment_id = 'pre'
),
post_orders AS (
    SELECT
        o.user_id,
        SUM(o.gmv) AS gmv,
        SUM(o.fee_amount) AS fee_amount,
        COUNT(*) AS n_orders
    FROM orders o
    WHERE o.experiment_id = 'fee_geo'
    GROUP BY o.user_id
),
post_exposed AS (
    SELECT user_id
    FROM exposed_users
    WHERE experiment_id = 'fee_geo'
)
SELECT
    v.user_id,
    v.region_id,
    v.region_name,
    v.geo_rollout_flag,
    v.variant,
    CASE WHEN pre.user_id IS NOT NULL THEN 1 ELSE 0 END AS exposed_pre,
    CASE WHEN post.user_id IS NOT NULL THEN 1 ELSE 0 END AS exposed_post,
    CASE WHEN po_pre.user_id IS NOT NULL THEN 1 ELSE 0 END AS converted_pre,
    CASE WHEN po_post.user_id IS NOT NULL THEN 1 ELSE 0 END AS converted_post,
    COALESCE(po_pre.gmv, 0.0) AS gmv_pre,
    COALESCE(po_post.gmv, 0.0) AS gmv_post,
    COALESCE(po_pre.fee_amount, 0.0) AS arpu_pre,
    COALESCE(po_post.fee_amount, 0.0) AS arpu_post
FROM geo_variant v
LEFT JOIN pre_exposed pre
    ON pre.user_id = v.user_id
LEFT JOIN post_exposed post
    ON post.user_id = v.user_id
LEFT JOIN pre_orders po_pre
    ON po_pre.user_id = v.user_id
LEFT JOIN post_orders po_post
    ON po_post.user_id = v.user_id
WHERE pre.user_id IS NOT NULL
   OR post.user_id IS NOT NULL;
