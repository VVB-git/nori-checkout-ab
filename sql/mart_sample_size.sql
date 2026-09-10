-- Sample sizes and traffic for power / MDE (MDE itself is computed in Python).
-- Does not read experiment_truth.

WITH assigned AS (
    SELECT
        experiment_id,
        variant,
        COUNT(*) AS n_assigned
    FROM assignments
    GROUP BY experiment_id, variant
),
exposed AS (
    SELECT
        x.experiment_id,
        a.variant,
        COUNT(*) AS n_exposed
    FROM exposed_users x
    INNER JOIN assignments a
        ON a.experiment_id = x.experiment_id
       AND a.user_id = x.user_id
    WHERE x.experiment_id != 'pre'
    GROUP BY x.experiment_id, a.variant
),
daily_avg AS (
    SELECT
        experiment_id,
        AVG(n_unique) AS unique_users_per_day,
        COUNT(*) AS n_days
    FROM daily_traffic
    WHERE experiment_id != 'pre'
    GROUP BY experiment_id
)
SELECT
    x.experiment_id,
    exp.name,
    exp.start_ts,
    exp.end_ts,
    exp.assignment_mode,
    da.n_days,
    da.unique_users_per_day,
    x.variant,
    a.n_assigned,
    x.n_exposed
FROM exposed x
INNER JOIN assigned a
    ON a.experiment_id = x.experiment_id
   AND a.variant = x.variant
INNER JOIN experiments exp
    ON exp.experiment_id = x.experiment_id
INNER JOIN daily_avg da
    ON da.experiment_id = x.experiment_id;
