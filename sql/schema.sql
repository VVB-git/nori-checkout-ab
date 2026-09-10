PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS experiment_truth;
DROP TABLE IF EXISTS daily_traffic;
DROP TABLE IF EXISTS exposed_users;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS assignments;
DROP TABLE IF EXISTS experiments;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS regions;

CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    geo_rollout_flag INTEGER NOT NULL,
    cr_mult REAL NOT NULL,
    aov_mult REAL NOT NULL
);

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions (region_id),
    registered_at TEXT NOT NULL,
    segment TEXT NOT NULL
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    assignment_mode TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    guardrail_metric TEXT NOT NULL,
    hypothesis TEXT NOT NULL
);

CREATE TABLE assignments (
    experiment_id TEXT NOT NULL REFERENCES experiments (experiment_id),
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    variant TEXT NOT NULL,
    assigned_at TEXT NOT NULL,
    PRIMARY KEY (experiment_id, user_id)
);

CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    experiment_id TEXT REFERENCES experiments (experiment_id),
    started_at TEXT NOT NULL
);

CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions (session_id),
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    experiment_id TEXT REFERENCES experiments (experiment_id),
    event_name TEXT NOT NULL,
    event_ts TEXT NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    session_id INTEGER NOT NULL REFERENCES sessions (session_id),
    experiment_id TEXT REFERENCES experiments (experiment_id),
    order_ts TEXT NOT NULL,
    gmv REAL NOT NULL,
    fee_amount REAL NOT NULL
);

CREATE TABLE experiment_truth (
    experiment_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    true_ate REAL NOT NULL,
    note TEXT NOT NULL,
    PRIMARY KEY (experiment_id, metric)
);

-- Derived at load time from sessions (equivalent to users with view_item).
CREATE TABLE exposed_users (
    experiment_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, user_id)
);

CREATE TABLE daily_traffic (
    experiment_id TEXT NOT NULL,
    dt TEXT NOT NULL,
    n_unique INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, dt)
);

CREATE INDEX idx_assignments_exp_variant ON assignments (experiment_id, variant);
CREATE INDEX idx_sessions_exp_user ON sessions (experiment_id, user_id);
CREATE INDEX idx_events_exp_user_name ON events (experiment_id, user_id, event_name);
CREATE INDEX idx_orders_exp_user ON orders (experiment_id, user_id);
CREATE INDEX idx_users_region ON users (region_id);
CREATE INDEX idx_exposed_exp ON exposed_users (experiment_id);
