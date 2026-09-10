"""Read SQL marts and estimate effects. Does not query experiment_truth."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.paths import DB_PATH, read_sql
from src.power import days_to_power, mde_proportion, n_per_arm_proportion

ALPHA = 0.05
POWER = 0.80
TARGET_MDE = 0.008
BASELINE_CR = 0.115


@dataclass
class PropResult:
    control: float
    treatment: float
    abs_diff: float
    rel_diff: float
    p_value: float
    ci_low: float
    ci_high: float
    n_control: int
    n_treatment: int


@dataclass
class MeanResult:
    control: float
    treatment: float
    abs_diff: float
    rel_diff: float
    p_value: float
    ci_low: float
    ci_high: float
    n_control: int
    n_treatment: int


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def load_sql(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(read_sql(name), con)


def two_prop(x_c: np.ndarray, x_t: np.ndarray) -> PropResult:
    n_c, n_t = int(x_c.size), int(x_t.size)
    p_c = float(x_c.mean()) if n_c else float("nan")
    p_t = float(x_t.mean()) if n_t else float("nan")
    diff = p_t - p_c
    se = float(np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t))
    z = diff / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    zcrit = float(stats.norm.ppf(1 - ALPHA / 2))
    rel = diff / p_c if p_c else float("nan")
    return PropResult(p_c, p_t, diff, rel, p, diff - zcrit * se, diff + zcrit * se, n_c, n_t)


def welch_mean(x_c: np.ndarray, x_t: np.ndarray) -> MeanResult:
    n_c, n_t = int(x_c.size), int(x_t.size)
    m_c, m_t = float(np.mean(x_c)), float(np.mean(x_t))
    diff = m_t - m_c
    rel = diff / m_c if m_c else float("nan")
    res = stats.ttest_ind(x_t, x_c, equal_var=False)
    se_c = float(np.var(x_c, ddof=1) / n_c)
    se_t = float(np.var(x_t, ddof=1) / n_t)
    se = float(np.sqrt(se_c + se_t))
    df_num = (se_c + se_t) ** 2
    df_den = se_c**2 / (n_c - 1) + se_t**2 / (n_t - 1)
    df = df_num / df_den if df_den else n_c + n_t - 2
    tcrit = float(stats.t.ppf(1 - ALPHA / 2, df))
    return MeanResult(
        m_c, m_t, diff, rel, float(res.pvalue), diff - tcrit * se, diff + tcrit * se, n_c, n_t
    )


def delta_method_ratio(gmv_c: np.ndarray, conv_c: np.ndarray, gmv_t: np.ndarray, conv_t: np.ndarray) -> MeanResult:
    """AOV = E[GMV] / E[converted] at user grain (zeros included in GMV)."""

    def _ratio_stats(gmv: np.ndarray, conv: np.ndarray) -> tuple[float, float, int]:
        n = gmv.size
        mx, my = float(gmv.mean()), float(conv.mean())
        theta = mx / my if my else float("nan")
        cov = np.cov(gmv, conv, ddof=1)
        vx, vy, cxy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
        var = (vx / my**2 + mx**2 * vy / my**4 - 2 * mx * cxy / my**3) / n
        return theta, float(np.sqrt(max(var, 0.0))), n

    th_c, se_c, n_c = _ratio_stats(gmv_c, conv_c)
    th_t, se_t, n_t = _ratio_stats(gmv_t, conv_t)
    diff = th_t - th_c
    se = float(np.sqrt(se_c**2 + se_t**2))
    z = diff / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    zcrit = float(stats.norm.ppf(1 - ALPHA / 2))
    rel = diff / th_c if th_c else float("nan")
    return MeanResult(th_c, th_t, diff, rel, p, diff - zcrit * se, diff + zcrit * se, n_c, n_t)


def srm_pvalue(n_c: int, n_t: int, p_expected: float = 0.5) -> float:
    total = n_c + n_t
    chi = (n_c - total * (1 - p_expected)) ** 2 / (total * (1 - p_expected)) + (
        n_t - total * p_expected
    ) ** 2 / (total * p_expected)
    return float(stats.chi2.sf(chi, 1))


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 10:
        return y
    theta = float(np.cov(y[mask], x[mask], ddof=1)[0, 1] / np.var(x[mask], ddof=1))
    return y - theta * (x - np.nanmean(x))


def _split_users(df: pd.DataFrame, experiment_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = df[df["experiment_id"] == experiment_id]
    return part[part["variant"] == "control"], part[part["variant"] == "treatment"]


def evaluate(db_path: Path = DB_PATH) -> dict[str, Any]:
    con = connect(db_path)
    users = load_sql(con, "mart_user_experiment_metrics.sql")
    sample = load_sql(con, "mart_sample_size.sql")
    srm = load_sql(con, "mart_srm.sql")
    orders = load_sql(con, "mart_order_gmv.sql")
    geo = load_sql(con, "mart_cuped_geo.sql")
    experiments = pd.read_sql_query("SELECT * FROM experiments", con)
    con.close()

    n_needed = n_per_arm_proportion(BASELINE_CR, TARGET_MDE, ALPHA, POWER)
    traffic = float(sample.loc[sample["experiment_id"] == "fee_ab_14d", "unique_users_per_day"].iloc[0])
    days_needed = days_to_power(n_needed, traffic)

    def pack_exp(experiment_id: str) -> dict[str, Any]:
        c, t = _split_users(users, experiment_id)
        cr = two_prop(c["converted"].to_numpy(), t["converted"].to_numpy())
        arpu = welch_mean(c["arpu"].to_numpy(), t["arpu"].to_numpy())
        gmv_user = welch_mean(c["gmv"].to_numpy(), t["gmv"].to_numpy())
        oc = orders[(orders["experiment_id"] == experiment_id) & (orders["variant"] == "control")]
        ot = orders[(orders["experiment_id"] == experiment_id) & (orders["variant"] == "treatment")]
        naive = welch_mean(oc["gmv"].to_numpy(), ot["gmv"].to_numpy())
        aov_delta = delta_method_ratio(
            c["gmv"].to_numpy(),
            c["converted"].to_numpy(),
            t["gmv"].to_numpy(),
            t["converted"].to_numpy(),
        )
        ss = sample[sample["experiment_id"] == experiment_id]
        n_arm = float(ss.loc[ss["variant"] == "control", "n_exposed"].iloc[0])
        mde = mde_proportion(BASELINE_CR, n_arm, ALPHA, POWER)
        srm_rows = srm[srm["experiment_id"] == experiment_id]
        n_c_a = int(srm_rows.loc[srm_rows["variant"] == "control", "n_assigned"].iloc[0])
        n_t_a = int(srm_rows.loc[srm_rows["variant"] == "treatment", "n_assigned"].iloc[0])
        mode = str(srm_rows["assignment_mode"].iloc[0])
        return {
            "experiment_id": experiment_id,
            "name": experiments.loc[experiments["experiment_id"] == experiment_id, "name"].iloc[0],
            "assignment_mode": mode,
            "srm_p": srm_pvalue(n_c_a, n_t_a),
            "n_assigned_control": n_c_a,
            "n_assigned_treatment": n_t_a,
            "mde_cr": mde,
            "unique_users_per_day": float(ss["unique_users_per_day"].iloc[0]),
            "n_days": int(ss["n_days"].iloc[0]),
            "n_exposed_control": int(ss.loc[ss["variant"] == "control", "n_exposed"].iloc[0]),
            "n_exposed_treatment": int(ss.loc[ss["variant"] == "treatment", "n_exposed"].iloc[0]),
            "cr": asdict(cr),
            "arpu": asdict(arpu),
            "gmv_per_user": asdict(gmv_user),
            "naive_order_gmv": asdict(naive),
            "aov_delta_method": asdict(aov_delta),
        }

    honest = pack_exp("fee_ab_14d")
    short = pack_exp("fee_ab_7d")
    geo_pack = pack_exp("fee_geo")

    geo_exp = geo[geo["exposed_post"] == 1].copy()
    naive_arpu = welch_mean(
        geo_exp.loc[geo_exp["variant"] == "control", "arpu_post"].to_numpy(),
        geo_exp.loc[geo_exp["variant"] == "treatment", "arpu_post"].to_numpy(),
    )
    both = geo[(geo["exposed_pre"] == 1) & (geo["exposed_post"] == 1)].copy()
    both["cr_cuped"] = cuped(both["converted_post"].to_numpy(), both["converted_pre"].to_numpy())
    both["arpu_cuped"] = cuped(both["arpu_post"].to_numpy(), both["arpu_pre"].to_numpy())
    cr_cuped = welch_mean(
        both.loc[both["variant"] == "control", "cr_cuped"].to_numpy(),
        both.loc[both["variant"] == "treatment", "cr_cuped"].to_numpy(),
    )
    arpu_cuped = welch_mean(
        both.loc[both["variant"] == "control", "arpu_cuped"].to_numpy(),
        both.loc[both["variant"] == "treatment", "arpu_cuped"].to_numpy(),
    )
    cr_pre = two_prop(
        both.loc[both["variant"] == "control", "converted_pre"].to_numpy(),
        both.loc[both["variant"] == "treatment", "converted_pre"].to_numpy(),
    )

    region = (
        both.groupby(["region_id", "region_name", "variant"], as_index=False)
        .agg(
            converted_pre=("converted_pre", "mean"),
            converted_post=("converted_post", "mean"),
            arpu_pre=("arpu_pre", "mean"),
            arpu_post=("arpu_post", "mean"),
            n=("user_id", "size"),
        )
    )
    region["cr_cuped"] = cuped(region["converted_post"].to_numpy(), region["converted_pre"].to_numpy())
    region["arpu_cuped"] = cuped(region["arpu_post"].to_numpy(), region["arpu_pre"].to_numpy())
    cr_region = welch_mean(
        region.loc[region["variant"] == "control", "cr_cuped"].to_numpy(),
        region.loc[region["variant"] == "treatment", "cr_cuped"].to_numpy(),
    )
    arpu_region = welch_mean(
        region.loc[region["variant"] == "control", "arpu_cuped"].to_numpy(),
        region.loc[region["variant"] == "treatment", "arpu_cuped"].to_numpy(),
    )

    geo_pack["naive_geo_arpu"] = asdict(naive_arpu)
    geo_pack["cuped_cr"] = asdict(cr_cuped)
    geo_pack["cuped_arpu"] = asdict(arpu_cuped)
    geo_pack["cuped_cr_region"] = asdict(cr_region)
    geo_pack["cuped_arpu_region"] = asdict(arpu_region)
    geo_pack["pre_period_cr_gap"] = asdict(cr_pre)
    geo_pack["n_cuped_users"] = int(len(both))
    geo_pack["n_regions"] = int(region["region_id"].nunique())
    geo_pack["regions"] = (
        region[["region_name", "variant", "converted_pre", "converted_post", "arpu_pre", "arpu_post"]]
        .assign(
            converted_pre=lambda d: d["converted_pre"] * 100,
            converted_post=lambda d: d["converted_post"] * 100,
        )
        .to_dict("records")
    )

    honest["decision"] = {
        "verdict": "не катить",
        "reason": (
            "Главная метрика — конверсия в оплату — снижается статистически значимо. "
            "Рост ARPU не отменяет правило: решение принимаем по главной метрике."
        ),
    }
    ratio_ci_crosses = honest["gmv_per_user"]["ci_low"] <= 0 <= honest["gmv_per_user"]["ci_high"]
    naive_sig = honest["naive_order_gmv"]["p_value"] < ALPHA
    if naive_sig and ratio_ci_crosses:
        ratio_verdict = "продлить"
    elif honest["gmv_per_user"]["p_value"] < ALPHA and honest["gmv_per_user"]["abs_diff"] < 0:
        ratio_verdict = "не катить"
    else:
        ratio_verdict = "продлить"
    honest["ratio_decision"] = {
        "verdict": ratio_verdict,
        "reason": (
            "Считать только чеки купивших нельзя: так выпадают те, кто ушёл без оплаты, и средний заказ кажется выше. "
            "Если вернуть всех, оборот на человека не вырос. Этого недостаточно, чтобы включать сбор всем."
        ),
    }
    geo_pack["decision"] = {
        "verdict": "не катить",
        "reason": (
            "Сплит не случайный: SRM против 50/50 провален, в тест попали более денежные регионы. "
            "CUPED по препериоду снимает смещение. Для раскатки на всю базу нужен holdout 10% "
            "или поэтапный гео-дизайн с препериодом, а не «увидели рост и включили везде»."
        ),
    }
    underpowered = short["mde_cr"] > TARGET_MDE * 1.5
    short_sig = short["cr"]["p_value"] < ALPHA
    if underpowered:
        short_verdict = "продлить"
        if short_sig and short["cr"]["abs_diff"] < 0:
            short_reason = (
                f"За {short['n_days']} дней MDE по конверсии {short['mde_cr']*100:.1f} п.п. "
                f"при целевом эффекте {TARGET_MDE*100:.1f} п.п. В этом прогоне конверсия уже снижается, "
                "но размер эффекта на такой выборке оценивается грубо. Сбор всем не включать, тест продлить "
                "до расчётного n."
            )
        else:
            short_reason = (
                f"За {short['n_days']} дней MDE по конверсии {short['mde_cr']*100:.1f} п.п. "
                f"при целевом эффекте {TARGET_MDE*100:.1f} п.п. Отсутствие значимости — не доказательство нуля "
                "и не разрешение включать сбор всем."
            )
    elif short_sig and short["cr"]["abs_diff"] < 0:
        short_verdict = "не катить"
        short_reason = "Конверсия снижается. Раскатывать на всю базу нельзя."
    else:
        short_verdict = "продлить"
        short_reason = "Короткое окно не закрывает целевой сдвиг. Нужно добрать выборку, а не включать сбор всем."
    short["decision"] = {"verdict": short_verdict, "reason": short_reason}

    payload = {
        "baseline_cr": BASELINE_CR,
        "target_mde": TARGET_MDE,
        "alpha": ALPHA,
        "power": POWER,
        "n_per_arm_needed": n_needed,
        "unique_users_per_day_ab14": traffic,
        "days_to_power": days_needed,
        "honest_ab": honest,
        "ratio": {
            "experiment_id": "fee_ab_14d",
            "naive_order_gmv": honest["naive_order_gmv"],
            "gmv_per_user": honest["gmv_per_user"],
            "aov_delta_method": honest["aov_delta_method"],
            "decision": honest["ratio_decision"],
        },
        "geo": geo_pack,
        "short": short,
        "memo": [
            {"case": "Честный A/B, 14 дней", "verdict": honest["decision"]["verdict"]},
            {"case": "Денежная / ratio-метрика", "verdict": honest["ratio_decision"]["verdict"]},
            {"case": "Гео без рандома", "verdict": geo_pack["decision"]["verdict"]},
            {"case": "Короткий тест, 7 дней", "verdict": short["decision"]["verdict"]},
        ],
    }
    return payload


def main() -> None:
    result = evaluate()
    memo = result["memo"]
    print("n_per_arm_needed", result["n_per_arm_needed"])
    print("unique_per_day", round(result["unique_users_per_day_ab14"], 1))
    print("days_to_power", round(result["days_to_power"], 2))
    for key in ("honest_ab", "short"):
        block = result[key]
        cr = block["cr"]
        print(
            key,
            "CR",
            round(cr["control"] * 100, 2),
            round(cr["treatment"] * 100, 2),
            "d_pp",
            round(cr["abs_diff"] * 100, 2),
            "p",
            cr["p_value"],
            "n_exp",
            block["n_exposed_control"],
            block["n_exposed_treatment"],
            "srm",
            block["srm_p"],
            "ARPU_rel",
            round(block["arpu"]["rel_diff"] * 100, 1),
            "verdict",
            block["decision"]["verdict"],
        )
    naive = result["ratio"]["naive_order_gmv"]
    gmv = result["ratio"]["gmv_per_user"]
    print(
        "ratio naive_p",
        naive["p_value"],
        "naive_rel",
        round(naive["rel_diff"] * 100, 1),
        "gmv_user_rel",
        round(gmv["rel_diff"] * 100, 1),
        "gmv_p",
        gmv["p_value"],
        "ci",
        round(gmv["ci_low"], 2),
        round(gmv["ci_high"], 2),
        "verdict",
        result["ratio"]["decision"]["verdict"],
    )
    geo = result["geo"]
    print(
        "geo naive_arpu_rel",
        round(geo["naive_geo_arpu"]["rel_diff"] * 100, 1),
        "cuped_cr_pp",
        round(geo["cuped_cr"]["abs_diff"] * 100, 2),
        "cuped_cr_p",
        geo["cuped_cr"]["p_value"],
        "region_cr_pp",
        round(geo["cuped_cr_region"]["abs_diff"] * 100, 2),
        "region_p",
        geo["cuped_cr_region"]["p_value"],
        "srm",
        geo["srm_p"],
        "verdict",
        geo["decision"]["verdict"],
    )
    print("memo", memo)


if __name__ == "__main__":
    main()
