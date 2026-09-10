"""Отчёт Nori: сервисный сбор на checkout."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze import evaluate
from src.paths import DB_PATH
from src.power import days_to_power, mde_proportion, n_per_arm_proportion

INK = "#1A1F2B"
COPPER = "#C45C26"
SLATE = "#3D4F61"
SAGE = "#5F6F64"
PAPER = "#F4F1EA"

st.set_page_config(page_title="Nori — решение по сбору", layout="wide")
alt.data_transformers.disable_max_rows()


def pct_pp(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f} п.п."


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def pfmt(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def num(n: int | float) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def money(x: float) -> str:
    return f"{x:,.1f} ₽".replace(",", " ")


@st.cache_data(show_spinner=False)
def load_results(db_mtime: float, analyze_mtime: float) -> dict:
    return evaluate(DB_PATH)


def get_results() -> dict:
    db_mtime = DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0
    analyze_mtime = (ROOT / "src" / "analyze.py").stat().st_mtime
    return load_results(db_mtime, analyze_mtime)


def inject_theme() -> None:
    st.markdown(
        f"""
<style>
    .stApp {{ background: {PAPER}; }}
    h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}
    [data-testid="stHeader"] {{ background: {PAPER}; }}
    .exec-brief {{
        background: {INK};
        color: {PAPER};
        padding: 1.45rem 1.7rem 1.3rem;
        border-radius: 6px;
        border-left: 7px solid {COPPER};
        margin: 0 0 1.35rem 0;
    }}
    .exec-brief h2 {{
        color: {PAPER};
        font-size: 1.15rem;
        margin: 0 0 0.65rem 0;
        font-weight: 650;
    }}
    .exec-brief p {{ margin: 0 0 0.7rem 0; line-height: 1.45; }}
    .exec-brief .label {{
        color: #C9B8A4;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }}
    .exec-brief .solution {{ color: #F0D3B8; }}
    [data-testid="stMetricValue"] {{
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        line-height: 1.2;
        word-break: normal;
        overflow-wrap: anywhere;
    }}
    .legend-row {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.25rem 1.5rem;
        margin: 0.35rem 0 1.1rem 0;
    }}
    .legend-row .k {{
        color: #5C564E;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }}
    .legend-row .v {{
        color: {INK};
        font-size: 1.15rem;
        font-weight: 650;
        line-height: 1.3;
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def bar_control_treat(
    df: pd.DataFrame,
    value_title: str,
    fmt: str,
    from_zero: bool = True,
) -> alt.Chart:
    plot = df.copy()
    y_scale = alt.Scale(zero=True) if from_zero else alt.Scale(zero=False)
    x_enc = alt.X(
        "group:N",
        title=None,
        sort=["Контроль", "Пилот"],
        axis=alt.Axis(
            labelAngle=0,
            labelPadding=10,
            labelFontSize=12,
            labelOverlap=False,
            labelLimit=0,
            ticks=False,
            domain=False,
        ),
        scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25),
    )
    bars = (
        alt.Chart(plot)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=x_enc,
            y=alt.Y("value:Q", title=value_title, scale=y_scale),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=["Контроль", "Пилот"], range=[SLATE, COPPER]),
                legend=None,
            ),
            tooltip=["group", alt.Tooltip("value:Q", format=fmt)],
        )
    )
    labels = (
        alt.Chart(plot)
        .mark_text(dy=-10, fontSize=13, fontWeight=600, color=INK)
        .encode(
            x=alt.X(
                "group:N",
                sort=["Контроль", "Пилот"],
                scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25),
            ),
            y=alt.Y("value:Q", scale=y_scale),
            text=alt.Text("value:Q", format=fmt),
        )
    )
    return alt.layer(bars, labels).properties(width="container", height=240).configure_view(stroke=None)


def forest_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        y=alt.Y("case:N", sort=None, title=None, axis=alt.Axis(labelLimit=0, labelFontSize=12))
    )
    rules = base.mark_rule(color=SLATE, strokeWidth=1.6).encode(
        x=alt.X("ci_low:Q", title="Эффект по конверсии в оплату, п.п."),
        x2="ci_high:Q",
    )
    points = base.mark_point(filled=True, size=90, color=COPPER).encode(
        x="effect:Q",
        tooltip=["case", alt.Tooltip("effect:Q", format=".2f"), alt.Tooltip("p:N")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8A8178", strokeDash=[4, 4]).encode(x="x:Q")
    return (zero + rules + points).properties(height=180)


VERDICT_PHRASE = {
    "не катить": "Сбор всем не включать",
    "продлить": "Тест продлить, всем не включать",
    "катить": "Можно включать всем",
}


def verdict_box(verdict: str, reason: str) -> None:
    color = {"не катить": "red", "продлить": "orange", "катить": "green"}.get(verdict, "gray")
    st.markdown(f":{color}[**{VERDICT_PHRASE.get(verdict, verdict)}**]")
    st.write(reason)


def details(res: dict, kind: str, what: str) -> str:
    if kind == "prop":
        delta = pct_pp(res["abs_diff"])
        interval = f"{pct_pp(res['ci_low'])} … {pct_pp(res['ci_high'])}"
        levels = f"{pct(res['control'])} без сбора, {pct(res['treatment'])} со сбором"
    else:
        delta = f"{res['rel_diff']*100:+.1f}%"
        interval = f"{res['ci_low']:.2f} … {res['ci_high']:.2f} ₽"
        levels = f"{money(res['control'])} без сбора, {money(res['treatment'])} со сбором"
    return (
        f"{what}: {levels}, сдвиг {delta}. "
        f"Интервал, в котором с высокой вероятностью лежит сдвиг: {interval}. "
        f"Людей в группах: {num(res['n_control'])} и {num(res['n_treatment'])}."
    )


def main() -> None:
    inject_theme()
    st.title("Nori: решение по сервисному сбору")
    st.caption("Деловой отчёт по эксперименту. Синтетический маркетплейс. Оценки не используют служебную таблицу истинного эффекта.")

    if not DB_PATH.exists():
        st.error(
            "Базы ещё нет. Из корня репозитория выполните `python -m src.generate`, "
            "затем обновите эту страницу."
        )
        st.stop()

    data = get_results()
    honest = data["honest_ab"]
    ratio = data["ratio"]
    geo = data["geo"]
    short = data["short"]
    cr = honest["cr"]

    st.markdown(
        f"""
<div class="exec-brief">
  <div class="label">Кратко</div>
  <h2>Проблема</h2>
  <p>Продукт хочет за 7–14 дней включить сервисный сбор 2.5% на экране оплаты для всей базы.
  Нужно понять: хватает ли людей в тесте, что считать главным результатом и можно ли включать сбор всем.</p>
  <h2>Решение</h2>
  <p class="solution"><strong>Сбор на всю базу не включать.</strong>
  В честном тесте доля оплативших снижается на {pct_pp(abs(cr['abs_diff']))}
  (с {pct(cr['control'])} до {pct(cr['treatment'])}).
  Рост выручки площадки на человека не отменяет это правило.
  «Чеки выросли» — артефакт подсчёта по заказам. Рост в регионах — состав городов, не эффект сбора.
  Недели мало: такой тест ловит только сдвиги около {pct_pp(short['mde_cr'])}, а цель — {pct_pp(data['target_mde'])}.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    forest = forest_chart(
        [
            {
                "case": "Честный тест, 14 дней",
                "effect": honest["cr"]["abs_diff"] * 100,
                "ci_low": honest["cr"]["ci_low"] * 100,
                "ci_high": honest["cr"]["ci_high"] * 100,
                "p": pfmt(honest["cr"]["p_value"]),
            },
            {
                "case": "Неделя, малая выборка",
                "effect": short["cr"]["abs_diff"] * 100,
                "ci_low": short["cr"]["ci_low"] * 100,
                "ci_high": short["cr"]["ci_high"] * 100,
                "p": pfmt(short["cr"]["p_value"]),
            },
            {
                "case": "Регионы, после поправки",
                "effect": geo["cuped_cr"]["abs_diff"] * 100,
                "ci_low": geo["cuped_cr"]["ci_low"] * 100,
                "ci_high": geo["cuped_cr"]["ci_high"] * 100,
                "p": pfmt(geo["cuped_cr"]["p_value"]),
            },
        ]
    )
    st.altair_chart(forest, use_container_width=True)
    st.caption(
        "Разброс эффекта по доле оплативших (пункты) и 95% интервал. "
        "Вертикальная черта — ноль: интервал левее нуля означает снижение."
    )

    st.markdown(
        """
<div class="legend-row">
  <div><div class="k">Главная метрика</div><div class="v">Доля оплативших</div></div>
  <div><div class="k">Страховочная</div><div class="v">Выручка площадки на человека</div></div>
  <div><div class="k">Итог по честному тесту</div><div class="v">всем не включать</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Хватает ли выборки")
    st.write(
        f"Базовая доля оплативших {pct(data['baseline_cr'])}. "
        f"Цель: увидеть сдвиг в {pct_pp(data['target_mde'])} "
        f"(α = {data['alpha']}, мощность {pct(data['power'], 0)})."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Нужно на группу", num(data["n_per_arm_needed"]))
    k2.metric("Уникальных в день", f"{data['unique_users_per_day_ab14']:,.0f}".replace(",", " "))
    k3.metric("Дней до нужной выборки", f"{data['days_to_power']:.1f}")
    k4.metric("Что ловит недельный тест", pct_pp(short["mde_cr"]))

    n_grid = np.linspace(1_500, 80_000, 60)
    power_df = pd.DataFrame(
        {
            "n": n_grid,
            "mde": [mde_proportion(data["baseline_cr"], n) * 100 for n in n_grid],
        }
    )
    line = (
        alt.Chart(power_df)
        .mark_line(color=SLATE, strokeWidth=2)
        .encode(
            x=alt.X("n:Q", title="Людей в одной группе"),
            y=alt.Y("mde:Q", title="Сдвиг, который тест ещё видит, п.п."),
        )
    )
    marks = pd.DataFrame(
        [
            {
                "n": short["n_exposed_control"],
                "mde": short["mde_cr"] * 100,
                "label": "7 дней",
            },
            {
                "n": honest["n_exposed_control"],
                "mde": honest["mde_cr"] * 100,
                "label": "14 дней",
            },
            {
                "n": data["n_per_arm_needed"],
                "mde": data["target_mde"] * 100,
                "label": "цель 0.8 п.п.",
            },
        ]
    )
    dots = (
        alt.Chart(marks)
        .mark_point(filled=True, size=110, color=COPPER)
        .encode(x="n:Q", y="mde:Q", tooltip=["label", "n", alt.Tooltip("mde:Q", format=".2f")])
    )
    labels = (
        alt.Chart(marks)
        .mark_text(align="left", dx=8, dy=-8, color=INK, fontSize=12)
        .encode(x="n:Q", y="mde:Q", text="label:N")
    )
    st.altair_chart((line + dots + labels).properties(height=260), use_container_width=True)
    st.caption(
        "Чем правее точка, тем больше людей и тем более мелкий сдвиг тест способен увидеть. "
        "Неделя стоит высоко: мелкий эффект она пропускает."
    )

    with st.expander("Подобрать размер выборки под свой сдвиг"):
        col_a, col_b, col_c = st.columns(3)
        mde_in = col_a.number_input("Сдвиг, который хотим увидеть, п.п.", min_value=0.1, max_value=5.0, value=0.8, step=0.1) / 100
        cr_in = col_b.number_input("Доля оплативших сейчас, %", min_value=1.0, max_value=40.0, value=11.5, step=0.1) / 100
        traffic_in = col_c.number_input(
            "Уникальных пользователей в день",
            min_value=100,
            max_value=100000,
            value=int(data["unique_users_per_day_ab14"]),
            step=500,
        )
        n_calc = n_per_arm_proportion(cr_in, mde_in)
        st.write(
            f"Нужно **{num(n_calc)}** человек на группу, около **{days_to_power(n_calc, traffic_in):.1f}** дней "
            f"при делении 50/50. На текущей недельной выборке тест видит сдвиги от "
            f"**{pct_pp(mde_proportion(cr_in, short['n_exposed_control']))}**."
        )

    st.header("1. Честный тест, 14 дней")
    st.write(
        f"Людей случайно делят пополам. Проверка равных долей: p = {pfmt(honest['srm_p'])}. "
        f"При этой выборке тест видит сдвиги от {pct_pp(honest['mde_cr'])}."
    )
    g1, g2 = st.columns(2)
    with g1:
        st.altair_chart(
            bar_control_treat(
                pd.DataFrame(
                    {
                        "group": ["Контроль", "Пилот"],
                        "value": [honest["cr"]["control"] * 100, honest["cr"]["treatment"] * 100],
                    }
                ),
                "Доля оплативших, %",
                ".2f",
            ),
            use_container_width=True,
        )
        st.caption("Главная метрика. Пилот — со сбором 2.5%.")
    with g2:
        st.altair_chart(
            bar_control_treat(
                pd.DataFrame(
                    {
                        "group": ["Контроль", "Пилот"],
                        "value": [honest["arpu"]["control"], honest["arpu"]["treatment"]],
                    }
                ),
                "Выручка площадки на человека, ₽",
                ".1f",
                from_zero=False,
            ),
            use_container_width=True,
        )
        st.caption("Страховочная метрика. Деньги площадки выросли — вывод всё равно по оплатам.")
    st.write(
        "Со сбором **оплачивают реже**. Площадка при этом зарабатывает больше с человека: "
        "берёт 2.5% сверху. Для решения важны оплаты, а не эта прибавка."
    )
    with st.expander("Цифры, если нужно проверить расчёт"):
        st.write(details(honest["cr"], "prop", "Доля оплативших"))
        st.write(details(honest["arpu"], "mean", "Выручка площадки на человека"))
    verdict_box(honest["decision"]["verdict"], honest["decision"]["reason"])

    st.header("2. Деньги: заказ против человека")
    naive = ratio["naive_order_gmv"]
    gmv_u = ratio["gmv_per_user"]
    st.write(
        "Продукт может сказать: «средний заказ вырос, значит сбор можно включать всем». "
        "Продукт смотрит только на тех, кто уже заплатил. Со сбором часть людей с меньшим заказом "
        "уходит, и среднее среди оставшихся само становится выше. Это не рост денег."
    )
    wrong, right = st.columns(2)
    with wrong:
        st.markdown("**Неверный срез — только оплатившие**")
        st.write(
            f"Средний заказ {money(naive['control'])} без сбора и {money(naive['treatment'])} со сбором "
            f"({naive['rel_diff']*100:+.1f}%). Кажется, что чеки выросли."
        )
    with right:
        st.markdown("**Верный срез — все, кто дошёл до оплаты**")
        st.write(
            f"Оборот на человека {money(gmv_u['control'])} без сбора и {money(gmv_u['treatment'])} со сбором "
            f"({gmv_u['rel_diff']*100:+.1f}%). Кто не заплатил, здесь ноль. Прибавки нет."
        )
    money_df = pd.DataFrame(
        [
            {
                "metric": "Только оплатившие: средний заказ",
                "effect": naive["rel_diff"] * 100,
            },
            {
                "metric": "Все в тесте: оборот на человека",
                "effect": gmv_u["rel_diff"] * 100,
            },
        ]
    )
    money_chart = (
        alt.Chart(money_df)
        .mark_bar(size=36, cornerRadiusEnd=2)
        .encode(
            y=alt.Y("metric:N", title=None, sort=None, axis=alt.Axis(labelLimit=0, labelFontSize=12)),
            x=alt.X("effect:Q", title="На сколько процентов отличается пилот"),
            color=alt.condition(alt.datum.effect > 0, alt.value(COPPER), alt.value(SLATE)),
            tooltip=["metric", alt.Tooltip("effect:Q", format="+.1f")],
        )
        .properties(height=160)
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8A8178", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(money_chart + zero, use_container_width=True)
    st.write(
        "Один тест, два способа посчитать. «Чеки выросли» — не аргумент включать сбор всем."
    )
    with st.expander("Цифры, если нужно проверить расчёт"):
        st.write(details(naive, "mean", "Средний заказ, только купившие"))
        st.write(details(gmv_u, "mean", "Оборот на человека, все в тесте"))
    verdict_box(ratio["decision"]["verdict"], ratio["decision"]["reason"])

    st.header("3. Сбор включили в регионах, без случайного деления")
    st.write(
        f"Это не честный тест. Проверка «50 на 50»: p = {pfmt(geo['srm_p'])}. "
        "В пилот попали крупные города с более высокой выручкой."
    )
    g3, g4 = st.columns(2)
    with g3:
        st.altair_chart(
            bar_control_treat(
                pd.DataFrame(
                    {
                        "group": ["Контроль", "Пилот"],
                        "value": [geo["naive_geo_arpu"]["control"], geo["naive_geo_arpu"]["treatment"]],
                    }
                ),
                "Выручка площадки на человека, ₽",
                ".1f",
                from_zero=False,
            ),
            use_container_width=True,
        )
        st.caption("Наивно кажется, что пилот сильно богаче. Это состав городов плюс сбор.")
    with g4:
        regions = geo.get("regions") or []
        if regions:
            region_df = pd.DataFrame(regions)
            region_df["group"] = region_df["variant"].map({"control": "Контроль", "treatment": "Пилот"})
            scatter = (
                alt.Chart(region_df)
                .mark_circle(size=90, opacity=0.85)
                .encode(
                    x=alt.X(
                        "converted_pre:Q",
                        title="Доля оплативших до сбора, %",
                        scale=alt.Scale(zero=False, nice=True, padding=28),
                    ),
                    y=alt.Y(
                        "converted_post:Q",
                        title="Доля оплативших в окне сбора, %",
                        scale=alt.Scale(zero=False, nice=True, padding=28),
                    ),
                    color=alt.Color(
                        "group:N",
                        scale=alt.Scale(domain=["Контроль", "Пилот"], range=[SLATE, COPPER]),
                        legend=alt.Legend(title="Группа"),
                    ),
                    tooltip=[
                        alt.Tooltip("region_name:N", title="Регион"),
                        alt.Tooltip("converted_pre:Q", format=".2f"),
                        alt.Tooltip("converted_post:Q", format=".2f"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(scatter, use_container_width=True)
            st.caption("Каждая точка — регион. Если бы сбор сильно помогал оплатам, пилот ушёл бы вверх. Этого нет.")
        else:
            st.caption("Точки по регионам появятся после обновления страницы (R).")
    st.write(
        "Сбор включили в крупных городах, которые и так приносят больше денег. "
        "Поэтому сравнение «пилот vs остальные» выглядит как рост. "
        "Когда сравниваем оплаты с неделями до сбора, прироста нет. "
        "Так нельзя доказать, что сбор можно включать всем."
    )
    with st.expander("Цифры, если нужно проверить расчёт"):
        st.write(details(geo["naive_geo_arpu"], "mean", "Выручка площадки, сравнение регионов"))
        st.write(details(geo["pre_period_cr_gap"], "prop", "Доля оплативших до сбора"))
        st.write(details(geo["cuped_cr"], "prop", "Доля оплативших после поправки, человек"))
        st.write(details(geo["cuped_cr_region"], "prop", "Доля оплативших после поправки, 20 регионов"))
    verdict_box(geo["decision"]["verdict"], geo["decision"]["reason"])

    st.header("4. Неделя на малой доле базы")
    target_pp = data["target_mde"] * 100
    mde_pp = short["mde_cr"] * 100
    st.write(
        f"В тест попали {num(short['n_exposed_control'] + short['n_exposed_treatment'])} человек. "
        f"Такой выборки хватает, чтобы заметить только крупный сдвиг — от {pct_pp(short['mde_cr'])}. "
        f"Нужно было уметь заметить {pct_pp(data['target_mde'])}. Это меньше, чем «разрешение» недельного теста."
    )
    axis_max = max(mde_pp, target_pp) * 1.2
    blind = pd.DataFrame({"start": [0.0], "end": [mde_pp]})
    marks = pd.DataFrame(
        [
            {"x": target_pp, "label": "Какой сдвиг хотели заметить"},
            {"x": mde_pp, "label": "Недельный тест замечает сдвиги только отсюда"},
        ]
    )
    blind_rect = (
        alt.Chart(blind)
        .mark_rect(color=SLATE, opacity=0.18)
        .encode(x="start:Q", x2="end:Q")
    )
    axis_line = (
        alt.Chart(pd.DataFrame({"x": [0], "x2": [axis_max]}))
        .mark_rule(color="#C4B8AA", strokeWidth=2)
        .encode(x="x:Q", x2="x2:Q")
    )
    ticks = (
        alt.Chart(marks)
        .mark_rule(strokeWidth=2.4)
        .encode(
            x=alt.X("x:Q", title="Сдвиг доли оплативших, пункты", scale=alt.Scale(domain=[0, axis_max])),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["Какой сдвиг хотели заметить", "Недельный тест замечает сдвиги только отсюда"],
                    range=[COPPER, SLATE],
                ),
                legend=alt.Legend(title=None, orient="bottom", labelLimit=0),
            ),
        )
    )
    dots = (
        alt.Chart(marks)
        .mark_point(filled=True, size=120)
        .encode(x="x:Q", color=alt.Color("label:N", legend=None))
    )
    st.altair_chart(
        (blind_rect + axis_line + ticks + dots).properties(height=140),
        use_container_width=True,
    )
    st.caption(
        "Серое поле — сдвиги, которые недельная выборка почти наверняка не отличит от шума. "
        "Цель попадает внутрь этого поля: людей мало. Сбор всем не включать, окно продлить."
    )
    with st.expander("Цифры, если нужно проверить расчёт"):
        st.write(details(short["cr"], "prop", "Доля оплативших"))
    verdict_box(short["decision"]["verdict"], short["decision"]["reason"])

    st.header("Сводка решений")
    memo_df = pd.DataFrame(data["memo"])
    memo_df.columns = ["Кейс", "Вердикт"]
    memo_df["Вердикт"] = memo_df["Вердикт"].map(lambda v: VERDICT_PHRASE.get(v, v))
    st.table(memo_df)
    st.write(
        "Порядок чтения: сначала качество деления групп, затем хватает ли людей, "
        "затем главная метрика (оплаты), затем страховка (деньги площадки). "
        "Страховка не подменяет главную."
    )

    st.header("Когда людей не делили случайно")
    st.write(
        "Не включать сбор всем после пилота в крупных городах. Рабочие замены: "
        "**оставить 10% людей без сбора** до конца квартала; "
        "включать регионы по очереди и сравнивать с неделями до включения; "
        "не брать сразу все крупные города в одну «пилотную» кучу. "
        "В отложенных 10% считают так же, как в честном тесте: человек, доля оплативших, выручка площадки."
    )

    with st.expander("Служебная таблица генератора (оценки её не используют)"):
        truth = pd.read_sql_query("SELECT * FROM experiment_truth", sqlite3.connect(DB_PATH))
        st.dataframe(truth, hide_index=True, use_container_width=True)
        st.caption("Таблица experiment_truth. Аналитические SQL-витрины к ней не присоединяются.")


if __name__ == "__main__":
    main()
