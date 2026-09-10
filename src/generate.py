"""Synthetic marketplace Nori: checkout fee experiments. Seed 42. Estimates do not use this module's truth table."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DB_PATH, ROOT, SQL_DIR

SEED = 42
N_USERS = 220_000
BASE_CR = 0.115
HONEST_CR_LIFT = -0.012
GEO_CR_LIFT = -0.004
TAKE_BASE = 0.12
CHECKOUT_FEE = 0.025
AOV_MEAN = 1850.0
AOV_CV = 0.62
DAILY_P = 0.073
SHORT_ENROLL_RATE = 0.06
PRE_START = date(2026, 1, 1)
PRE_END = date(2026, 1, 14)
EXP_START = date(2026, 1, 15)
EXP_END = date(2026, 1, 28)
SHORT_END = date(2026, 1, 21)

REGIONS = [
    ("Москва", 1, 1.0, 1.10),
    ("Санкт-Петербург", 1, 1.0, 1.09),
    ("Казань", 1, 1.0, 1.08),
    ("Екатеринбург", 1, 1.0, 1.07),
    ("Новосибирск", 1, 1.0, 1.07),
    ("Краснодар", 1, 1.0, 1.09),
    ("Нижний Новгород", 1, 1.0, 1.06),
    ("Сочи", 1, 1.0, 1.10),
    ("Самара", 0, 1.0, 0.94),
    ("Воронеж", 0, 1.0, 0.93),
    ("Пермь", 0, 1.0, 0.93),
    ("Волгоград", 0, 1.0, 0.92),
    ("Красноярск", 0, 1.0, 0.95),
    ("Саратов", 0, 1.0, 0.91),
    ("Тюмень", 0, 1.0, 0.96),
    ("Ижевск", 0, 1.0, 0.90),
    ("Ульяновск", 0, 1.0, 0.90),
    ("Барнаул", 0, 1.0, 0.89),
    ("Иркутск", 0, 1.0, 0.94),
    ("Ярославль", 0, 1.0, 0.92),
]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode = OFF")
    con.execute("PRAGMA synchronous = OFF")
    con.execute("PRAGMA foreign_keys = ON")
    return con


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    sigma = float(np.sqrt(np.log(1 + cv**2)))
    mu = float(np.log(mean) - 0.5 * sigma**2)
    return mu, sigma


def _dates(start: date, end: date) -> int:
    return (end - start).days + 1


def _timestamps(start: date, day_off: np.ndarray, rng: np.random.Generator) -> list[str]:
    seconds = rng.integers(8 * 3600, 22 * 3600, size=day_off.size)
    base = pd.Timestamp(start)
    stamps = base + pd.to_timedelta(day_off, unit="D") + pd.to_timedelta(seconds, unit="s")
    return stamps.strftime("%Y-%m-%d %H:%M:%S").tolist()


def build_users(rng: np.random.Generator) -> pd.DataFrame:
    region_ids = np.arange(1, len(REGIONS) + 1)
    rich = np.array([row[1] for row in REGIONS], dtype=int)
    # More users in large cities; rollout regions are larger on purpose (dirty split).
    weights = np.where(rich == 1, 1.35, 0.85).astype(float)
    weights = weights / weights.sum()
    user_regions = rng.choice(region_ids, size=N_USERS, p=weights)
    segments = rng.choice(
        np.array(["core", "new", "reactivation"]),
        size=N_USERS,
        p=[0.62, 0.25, 0.13],
    )
    offsets = rng.integers(30, 800, size=N_USERS)
    registered = [
        (PRE_START - timedelta(days=int(d))).isoformat() for d in offsets
    ]
    return pd.DataFrame(
        {
            "user_id": np.arange(1, N_USERS + 1),
            "region_id": user_regions,
            "registered_at": registered,
            "segment": segments,
        }
    )


def _assign_honest(
    rng: np.random.Generator, user_ids: np.ndarray, assigned_at: str
) -> pd.DataFrame:
    variant = rng.choice(["control", "treatment"], size=user_ids.size)
    return pd.DataFrame(
        {
            "experiment_id": "fee_ab_14d",
            "user_id": user_ids,
            "variant": variant,
            "assigned_at": assigned_at,
        }
    )


def _assign_short(
    rng: np.random.Generator, user_ids: np.ndarray, assigned_at: str
) -> pd.DataFrame:
    enroll = rng.random(user_ids.size) < SHORT_ENROLL_RATE
    enrolled = user_ids[enroll]
    variant = rng.choice(["control", "treatment"], size=enrolled.size)
    return pd.DataFrame(
        {
            "experiment_id": "fee_ab_7d",
            "user_id": enrolled,
            "variant": variant,
            "assigned_at": assigned_at,
        }
    )


def _assign_geo(users: pd.DataFrame, region_flag: dict[int, int], assigned_at: str) -> pd.DataFrame:
    flags = users["region_id"].map(region_flag)
    variant = np.where(flags == 1, "treatment", "control")
    return pd.DataFrame(
        {
            "experiment_id": "fee_geo",
            "user_id": users["user_id"].to_numpy(),
            "variant": variant,
            "assigned_at": assigned_at,
        }
    )


def simulate_activity(
    rng: np.random.Generator,
    users: pd.DataFrame,
    assignments: pd.DataFrame,
    experiment_id: str | None,
    start: date,
    end: date,
    cr_delta: float,
    apply_checkout_fee: bool,
    extra_order_p_control: float,
    extra_order_p_treatment: float,
    aov_treatment_mult: float,
    session_id_start: int,
    event_id_start: int,
    order_id_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, int]:
    region_cr = {i + 1: row[2] for i, row in enumerate(REGIONS)}
    region_aov = {i + 1: row[3] for i, row in enumerate(REGIONS)}
    n_days = _dates(start, end)
    mu, sigma = _lognormal_params(AOV_MEAN, AOV_CV)

    assigned_users = assignments.merge(users, on="user_id", how="inner")
    n = len(assigned_users)
    n_sessions = rng.binomial(n_days, DAILY_P, size=n)
    user_ids = assigned_users["user_id"].to_numpy()
    variants = assigned_users["variant"].to_numpy()
    region_ids = assigned_users["region_id"].to_numpy()

    session_user = np.repeat(user_ids, n_sessions)
    session_var = np.repeat(variants, n_sessions)
    session_region = np.repeat(region_ids, n_sessions)
    day_off = rng.integers(0, n_days, size=session_user.size)
    started = _timestamps(start, day_off, rng)
    n_sess = session_user.size
    session_ids = np.arange(session_id_start, session_id_start + n_sess)

    sessions = pd.DataFrame(
        {
            "session_id": session_ids,
            "user_id": session_user,
            "experiment_id": experiment_id,
            "started_at": started,
            "variant": session_var,
            "region_id": session_region,
        }
    )

    exposed = n_sessions > 0
    cr = BASE_CR * np.array([region_cr[int(r)] for r in region_ids])
    is_t = variants == "treatment"
    cr = np.clip(cr + np.where(is_t, cr_delta, 0.0), 0.01, 0.6)
    purchased = exposed & (rng.random(n) < cr)
    extra_p = np.where(is_t, extra_order_p_treatment, extra_order_p_control)
    n_orders_user = np.where(purchased, 1 + rng.binomial(2, extra_p), 0).astype(int)

    # Map first session of each purchasing user to carry the order(s).
    first_session = (
        sessions.sort_values("started_at")
        .groupby("user_id", as_index=False)
        .first()[["user_id", "session_id", "started_at", "variant", "region_id"]]
    )
    buy_users = assigned_users.loc[purchased, ["user_id"]].copy()
    buy_users["n_orders"] = n_orders_user[purchased]
    buy = buy_users.merge(first_session, on="user_id", how="left")
    order_user = np.repeat(buy["user_id"].to_numpy(), buy["n_orders"].to_numpy())
    order_session = np.repeat(buy["session_id"].to_numpy(), buy["n_orders"].to_numpy())
    order_ts_base = np.repeat(buy["started_at"].to_numpy(), buy["n_orders"].to_numpy())
    order_var = np.repeat(buy["variant"].to_numpy(), buy["n_orders"].to_numpy())
    order_region = np.repeat(buy["region_id"].to_numpy(), buy["n_orders"].to_numpy())

    aov_mult = np.array([region_aov[int(r)] for r in order_region])
    aov_mult = aov_mult * np.where(order_var == "treatment", aov_treatment_mult, 1.0)
    gmv = rng.lognormal(mu, sigma, size=order_user.size) * aov_mult
    take = TAKE_BASE + np.where(
        (order_var == "treatment") & apply_checkout_fee, CHECKOUT_FEE, 0.0
    )
    # Pre-period: no checkout fee, base take still exists (marketplace commission).
    fee = gmv * take
    n_ord = order_user.size
    order_ids = np.arange(order_id_start, order_id_start + n_ord)
    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "user_id": order_user,
            "session_id": order_session,
            "experiment_id": experiment_id,
            "order_ts": order_ts_base,
            "gmv": gmv,
            "fee_amount": fee,
        }
    )

    # Funnel: every session view_item; checkout if purchase session or random browse.
    view_events = pd.DataFrame(
        {
            "session_id": sessions["session_id"],
            "user_id": sessions["user_id"],
            "experiment_id": experiment_id,
            "event_name": "view_item",
            "event_ts": sessions["started_at"],
        }
    )
    purchase_sessions = np.unique(order_session) if n_ord else np.array([], dtype=int)
    checkout_mask = sessions["session_id"].isin(purchase_sessions) | (
        rng.random(n_sess) < 0.22
    )
    checkout_events = sessions.loc[checkout_mask, ["session_id", "user_id", "started_at"]].copy()
    checkout_events["experiment_id"] = experiment_id
    checkout_events["event_name"] = "begin_checkout"
    checkout_events["event_ts"] = checkout_events["started_at"]
    checkout_events = checkout_events.drop(columns=["started_at"])

    purchase_events = pd.DataFrame(
        {
            "session_id": order_session,
            "user_id": order_user,
            "experiment_id": experiment_id,
            "event_name": "purchase",
            "event_ts": order_ts_base,
        }
    )
    events = pd.concat(
        [
            view_events,
            checkout_events[["session_id", "user_id", "experiment_id", "event_name", "event_ts"]],
            purchase_events,
        ],
        ignore_index=True,
    )
    events.insert(0, "event_id", np.arange(event_id_start, event_id_start + len(events)))

    sessions_out = sessions.drop(columns=["variant", "region_id"])
    return (
        sessions_out,
        events,
        orders,
        session_id_start + n_sess,
        event_id_start + len(events),
        order_id_start + n_ord,
    )


def _write_frame(con: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    if df.empty:
        return
    df.to_sql(table, con, if_exists="append", index=False, chunksize=50_000)


def _write_exposure(con: sqlite3.Connection, sessions: pd.DataFrame, experiment_id: str | None) -> None:
    label = experiment_id if experiment_id is not None else "pre"
    if sessions.empty:
        return
    exposed = sessions[["user_id"]].drop_duplicates()
    exposed.insert(0, "experiment_id", label)
    _write_frame(con, exposed, "exposed_users")
    daily = pd.DataFrame(
        {
            "experiment_id": label,
            "dt": pd.Series(sessions["started_at"]).str.slice(0, 10),
            "user_id": sessions["user_id"].to_numpy(),
        }
    )
    traffic = (
        daily.groupby(["experiment_id", "dt"], as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "n_unique"})
    )
    _write_frame(con, traffic, "daily_traffic")


def generate(db_path: Path = DB_PATH, seed: int = SEED) -> Path:
    rng = np.random.default_rng(seed)
    con = _connect(db_path)
    schema = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    con.executescript(schema)

    regions = pd.DataFrame(
        {
            "region_id": np.arange(1, len(REGIONS) + 1),
            "name": [r[0] for r in REGIONS],
            "geo_rollout_flag": [r[1] for r in REGIONS],
            "cr_mult": [r[2] for r in REGIONS],
            "aov_mult": [r[3] for r in REGIONS],
        }
    )
    _write_frame(con, regions, "regions")

    users = build_users(rng)
    _write_frame(con, users, "users")

    experiments = pd.DataFrame(
        [
            {
                "experiment_id": "fee_ab_14d",
                "name": "Честный A/B: сервисный сбор 2.5%",
                "start_ts": f"{EXP_START.isoformat()} 00:00:00",
                "end_ts": f"{EXP_END.isoformat()} 23:59:59",
                "assignment_mode": "user_random",
                "primary_metric": "conversion_to_payment",
                "guardrail_metric": "arpu_platform_fee",
                "hypothesis": "Сбор 2.5% на checkout не снижает конверсию в оплату больше чем на 0.8 п.п.",
            },
            {
                "experiment_id": "fee_geo",
                "name": "Гео-роллаут без рандома",
                "start_ts": f"{EXP_START.isoformat()} 00:00:00",
                "end_ts": f"{EXP_END.isoformat()} 23:59:59",
                "assignment_mode": "geo",
                "primary_metric": "conversion_to_payment",
                "guardrail_metric": "arpu_platform_fee",
                "hypothesis": "Включить сбор в крупных регионах и считать это тестом.",
            },
            {
                "experiment_id": "fee_ab_7d",
                "name": "Короткий A/B на 7 дней",
                "start_ts": f"{EXP_START.isoformat()} 00:00:00",
                "end_ts": f"{SHORT_END.isoformat()} 23:59:59",
                "assignment_mode": "user_random",
                "primary_metric": "conversion_to_payment",
                "guardrail_metric": "arpu_platform_fee",
                "hypothesis": "За неделю можно принять решение о раскатке на всю базу.",
            },
        ]
    )
    _write_frame(con, experiments, "experiments")

    assigned_at = f"{EXP_START.isoformat()} 00:00:00"
    region_flag = {i + 1: r[1] for i, r in enumerate(REGIONS)}
    a_honest = _assign_honest(rng, users["user_id"].to_numpy(), assigned_at)
    a_short = _assign_short(rng, users["user_id"].to_numpy(), assigned_at)
    a_geo = _assign_geo(users, region_flag, assigned_at)
    _write_frame(con, pd.concat([a_honest, a_short, a_geo], ignore_index=True), "assignments")

    sid, eid, oid = 1, 1, 1

    # Pre-period: no experiment_id, no checkout fee, no CR treatment effect.
    dummy_assign = pd.DataFrame(
        {
            "user_id": users["user_id"],
            "variant": "control",
        }
    )
    s, e, o, sid, eid, oid = simulate_activity(
        rng,
        users,
        dummy_assign,
        None,
        PRE_START,
        PRE_END,
        cr_delta=0.0,
        apply_checkout_fee=False,
        extra_order_p_control=0.10,
        extra_order_p_treatment=0.10,
        aov_treatment_mult=1.0,
        session_id_start=sid,
        event_id_start=eid,
        order_id_start=oid,
    )
    _write_frame(con, s, "sessions")
    _write_frame(con, e, "events")
    _write_frame(con, o, "orders")
    _write_exposure(con, s, None)

    windows = [
        (
            a_honest,
            "fee_ab_14d",
            EXP_START,
            EXP_END,
            HONEST_CR_LIFT,
            True,
            0.10,
            0.10,
            1.08,
        ),
        (
            a_geo,
            "fee_geo",
            EXP_START,
            EXP_END,
            GEO_CR_LIFT,
            True,
            0.10,
            0.10,
            1.00,
        ),
        (
            a_short,
            "fee_ab_7d",
            EXP_START,
            SHORT_END,
            HONEST_CR_LIFT,
            True,
            0.10,
            0.10,
            1.08,
        ),
    ]
    for assign, exp_id, start, end, cr_d, fee, p_c, p_t, aov_m in windows:
        s, e, o, sid, eid, oid = simulate_activity(
            rng,
            users,
            assign,
            exp_id,
            start,
            end,
            cr_delta=cr_d,
            apply_checkout_fee=fee,
            extra_order_p_control=p_c,
            extra_order_p_treatment=p_t,
            aov_treatment_mult=aov_m,
            session_id_start=sid,
            event_id_start=eid,
            order_id_start=oid,
        )
        _write_frame(con, s, "sessions")
        _write_frame(con, e, "events")
        _write_frame(con, o, "orders")
        _write_exposure(con, s, exp_id)

    truth = pd.DataFrame(
        [
            {
                "experiment_id": "fee_ab_14d",
                "metric": "conversion_pp",
                "true_ate": HONEST_CR_LIFT,
                "note": "Абсолютный эффект на конверсию в оплату, п.п./100",
            },
            {
                "experiment_id": "fee_ab_14d",
                "metric": "arpu_rel",
                "true_ate": 0.07,
                "note": "Целевой относительный рост ARPU площадки; факт зависит от отсева дешёвых заказов",
            },
            {
                "experiment_id": "fee_geo",
                "metric": "conversion_pp",
                "true_ate": GEO_CR_LIFT,
                "note": "Истинный эффект сбора на CR; наивное сравнение регионов смещено",
            },
            {
                "experiment_id": "fee_geo",
                "metric": "naive_arpu_rel_bias",
                "true_ate": 0.18,
                "note": "Ожидаемый составной разрыв ARPU из-за выбора богатых регионов",
            },
            {
                "experiment_id": "fee_ab_7d",
                "metric": "conversion_pp",
                "true_ate": HONEST_CR_LIFT,
                "note": "Тот же истинный эффект, что у 14-дневного A/B, но выборки не хватает",
            },
        ]
    )
    _write_frame(con, truth, "experiment_truth")
    con.commit()
    con.close()
    return db_path


def main() -> None:
    path = generate()
    print(f"SQLite: {path}")
    print(f"Project root: {ROOT}")


if __name__ == "__main__":
    main()
