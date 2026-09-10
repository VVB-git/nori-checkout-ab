-- User-level metrics for each experiment and variant.
-- Denominator: users with at least one view_item in the experiment window (exposed).
-- Does not read experiment_truth.

WITH exposed AS (
    SELECT
        experiment_id,
        user_id
    FROM exposed_users
    WHERE experiment_id != 'pre'
),
purchases AS (
    SELECT
        o.experiment_id,
        o.user_id,
        COUNT(*) AS n_orders,
        SUM(o.gmv) AS gmv,
        SUM(o.fee_amount) AS fee_amount
    FROM orders o
    WHERE o.experiment_id IS NOT NULL
    GROUP BY o.experiment_id, o.user_id
)
SELECT
    a.experiment_id,
    a.variant,
    x.user_id,
    CASE WHEN p.user_id IS NOT NULL THEN 1 ELSE 0 END AS converted,
    COALESCE(p.n_orders, 0) AS n_orders,
    COALESCE(p.gmv, 0.0) AS gmv,
    COALESCE(p.fee_amount, 0.0) AS arpu
FROM assignments a
INNER JOIN exposed x
    ON x.experiment_id = a.experiment_id
   AND x.user_id = a.user_id
LEFT JOIN purchases p
    ON p.experiment_id = a.experiment_id
   AND p.user_id = a.user_id;
