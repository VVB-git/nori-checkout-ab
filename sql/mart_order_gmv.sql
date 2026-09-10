-- Order-level GMV for the naive money-metric t-test (wrong analysis unit).
-- Does not read experiment_truth.

SELECT
    o.experiment_id,
    a.variant,
    o.order_id,
    o.user_id,
    o.gmv,
    o.fee_amount
FROM orders o
INNER JOIN assignments a
    ON a.experiment_id = o.experiment_id
   AND a.user_id = o.user_id
WHERE o.experiment_id IS NOT NULL;
