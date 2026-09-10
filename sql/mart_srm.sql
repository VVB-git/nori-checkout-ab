-- Sample ratio mismatch vs equal 50/50 split (product assumption for a classic A/B).
-- Does not read experiment_truth.

SELECT
    a.experiment_id,
    exp.assignment_mode,
    a.variant,
    COUNT(*) AS n_assigned,
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY a.experiment_id) AS share
FROM assignments a
INNER JOIN experiments exp
    ON exp.experiment_id = a.experiment_id
GROUP BY a.experiment_id, exp.assignment_mode, a.variant;
