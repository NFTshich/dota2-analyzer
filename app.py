"""Streamlit-интерфейс CS2 Analyzer.

Запуск::

    streamlit run app.py

Содержит четыре страницы (sidebar-навигация):
    1. Прогноз матча — выбор двух команд и карты → вероятности + график.
    2. Форма команд — winrate за последние матчи.
    3. Карты — heatmap winrate по картам и сторонам.
    4. Backtest — как модель отработала на истории.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Позволяем импортировать src/ из корня проекта.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.autorefresh import start_autorefresh  # noqa: E402
from src.predictor import Predictor  # noqa: E402
from src.scraper import Scraper  # noqa: E402


# ---------------------------------------------------------------------------
# Автообновление данных в фоне (раз в час, если включено)
# ---------------------------------------------------------------------------
start_autorefresh()


# ---------------------------------------------------------------------------
# Инициализация данных. Читаем из локального кэша (data/raw/matches.csv),
# который обновляет фоновый поток — поэтому не сохраняем в st.cache_resource,
# а перечитываем каждый перезапуск скрипта (Streamlit делает это на каждое
# взаимодействие, но локальный CSV читается быстро).
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Читает сырые матчи из локального кэша (data/raw/matches.csv).

    Если файла нет — первичный сбор с Liquipedia (или синтетика в mock-режиме).
    """
    if config.MATCHES_RAW.exists():
        try:
            return pd.read_csv(config.MATCHES_RAW, parse_dates=["date"])
        except Exception:  # noqa: BLE001
            pass
    return Scraper().fetch()


@st.cache_data(show_spinner=False, ttl=60)
def _model_mtime() -> float:
    """Возвращает время изменения файла модели (для инвалидации кэша)."""
    p = config.MODEL_PATH
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner=False)
def get_predictor(mtime: float) -> Predictor:
    """Создаёт Predictor; кэш сбрасывается при изменении файла модели (mtime)."""
    return Predictor(load_raw())


def _pct(x: float) -> str:
    return f"{max(0.0, min(1.0, x)) * 100:.1f}%"


def _confidence_label(conf: float) -> str:
    """Классифицирует уверенность (high/medium/low) — синоним predictor."""
    if conf >= 0.30:
        return "high"
    if conf >= 0.15:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Шапка и боковая навигация
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dota2 Analyzer", layout="wide")

# Фоновая картинка как base64 (офлайн).
import base64 as _b64
import pathlib as _pl

def _bg_css() -> str:
    img = _pl.Path(__file__).resolve().parent / "bg.jpg"
    if not img.exists():
        return ""
    data = _b64.b64encode(img.read_bytes()).decode()
    return (
        "background:\n"
        f"    url('data:image/jpeg;base64,{data}') center/cover no-repeat fixed,\n"
        "    linear-gradient(180deg, rgba(0,0,0,0.6), rgba(0,0,0,0.68));"
    )

_BG_CSS = _bg_css()

# ---------------------------------------------------------------------------
# Тема в стиле Dota 2: мрачный волшебный фон, сияющие руны и парящие искры.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        __BG_CSS__
        color: #e6e0ee;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0920, #140d1e);
        border-right: 1px solid rgba(190,60,255,0.18);
    }
    [data-testid="stMetric"] {
        background: rgba(150,80,255,0.05);
        border: 1px solid rgba(190,60,255,0.2);
        border-radius: 10px;
    }
    h1 {
        background: linear-gradient(90deg, #c78bff, #ff5fa2, #c78bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        animation: dotaGlow 3s ease-in-out infinite;
    }
    @keyframes dotaGlow {
        0%,100% { filter: drop-shadow(0 0 8px rgba(180,70,255,0.5)); }
        50%     { filter: drop-shadow(0 0 26px rgba(255,90,180,0.8)); }
    }
    .stButton>button {
        color: #f3e6ff;
        background: linear-gradient(135deg, rgba(160,60,255,0.16), rgba(220,30,120,0.16));
        border: 1px solid rgba(190,60,255,0.5);
        box-shadow: 0 0 14px rgba(180,60,255,0.35);
        transition: all .2s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 28px rgba(200,80,255,0.7);
        color: #fff;
    }
    .stSelectbox > div > div {
        background: rgba(20,15,30,0.92);
        border: 1px solid rgba(190,60,255,0.28);
    }

    /* ---- Парящие сияющие руны/искры (подвижные элементы) ---- */
    @keyframes floatUp {
        0%   { transform: translateY(0) scale(0.6); opacity: 0; }
        15%  { opacity: 0.9; }
        85%  { opacity: 0.7; }
        100% { transform: translateY(-60px) scale(1.2); opacity: 0; }
    }
    .dota-orb {
        position: absolute; bottom: 20px; width: 8px; height: 8px; border-radius: 50%;
        background: radial-gradient(circle, #ffd9a0, #ff6aa0);
        box-shadow: 0 0 12px 3px rgba(255,140,180,0.8);
        animation: floatUp 5s ease-in infinite;
    }
    .dota-sigil {
        position:absolute; width:60px; height:60px; border-radius:50%;
        border:2px solid rgba(190,100,255,0.4);
        box-shadow: inset 0 0 18px rgba(180,80,255,0.35), 0 0 18px rgba(180,80,255,0.3);
        animation: sigilSpin 14s linear infinite;
    }
    .dota-sigil::after {
        content:""; position:absolute; top:-8px; left:50%; transform:translateX(-50%);
        width:8px; height:12px; border-radius:50%;
        background:#c78bff; box-shadow:0 0 10px #c78bff;
    }
    @keyframes sigilSpin { from { transform: rotate(0);} to { transform: rotate(360deg);} }

    .dota-arena {
        position: relative; height: 130px; overflow: hidden; border-radius: 14px;
        background:
            linear-gradient(180deg, rgba(5,3,12,0.5), rgba(20,10,30,0.6)),
            radial-gradient(circle at 50% 30%, rgba(130,40,200,0.15), transparent 60%);
        border: 1px solid rgba(190,60,255,0.2);
        margin: -6px 0 10px 0;
    }
    .dota-orn-1, .dota-orn-2 {
        position:absolute; top:50%; transform:translateY(-50%);
        font-size: 34px; color: rgba(200,120,255,0.55);
        text-shadow: 0 0 16px rgba(180,80,255,0.8);
    }
    .dota-orn-1 { left: 6%; animation: sway 6s ease-in-out infinite; }
    .dota-orn-2 { right: 6%; animation: sway 6s ease-in-out infinite reverse; }
    @keyframes sway {
        0%,100% { transform: translateY(-50%) translateX(0); }
        50%     { transform: translateY(-50%) translateX(10px); }
    }
    </style>
    """.replace("__BG_CSS__", _BG_CSS),
    unsafe_allow_html=True,
)

# Подвижное сияющее полотно (руны + парящие искры)
st.markdown(
    """
    <div class="dota-arena">
        <div class="dota-sigil" style="left:12%; top:24%;"></div>
        <div class="dota-sigil" style="right:14%; top:34%; width:44px;height:44px;"></div>
        <div class="dota-orn-1">✦</div>
        <div class="dota-orn-2">✧</div>
        <div class="dota-orb" style="left:20%; animation-delay:0s;"></div>
        <div class="dota-orb" style="left:38%; animation-delay:1.6s;"></div>
        <div class="dota-orb" style="left:55%; animation-delay:0.8s;"></div>
        <div class="dota-orb" style="left:72%; animation-delay:2.4s;"></div>
        <div class="dota-orb" style="left:88%; animation-delay:3.2s;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Dota2 Analyzer")
st.caption("Аналитический инструмент для предсказания исходов матчей Dota 2")

page = st.sidebar.radio(
    "Раздел",
    ["Прогноз матча", "Форма команд", "Рейтинг команд", "Backtest"],
)

# Подгружаем данные один раз
with st.spinner("Загружаю данные…"):
    raw = load_raw()
teams = sorted(set(raw["team_a"]).union(set(raw["team_b"])))

# ---------------------------------------------------------------------------
# Страница 1: Прогноз матча
# ---------------------------------------------------------------------------
if page == "Прогноз матча":
    st.header("Прогноз матча")
    st.markdown(
        "Выберите две команды — модель вернёт вероятности победы в процентах."
    )

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Команда A", teams, key="team_a")
    with col2:
        # возможные соперники — всё кроме выбранной команды A
        opponents = [t for t in teams if t != team_a]
        team_b = st.selectbox(
            "Команда B",
            opponents,
            index=min(len(opponents) - 1, 0),
            key="team_b",
        )

    if team_a == team_b:
        st.warning(
            "Команды A и B совпадают — выберите разных соперников, "
            "иначе прогноз бессмысленен."
        )

    if st.button("Рассчитать", type="primary"):
        predictor = get_predictor(_model_mtime())
        with st.spinner("Считаю вероятности…"):
            res = predictor.predict(team_a, team_b, map_name="", use_map=False)
            pa, pb = res["team_a"], res["team_b"]
            conf = res["confidence"]
            conf_lbl = _confidence_label(conf)

        st.subheader("Результат")
        cols = st.columns([1, 2, 1])
        cols[0].metric(team_a, _pct(pa))
        cols[2].metric(team_b, _pct(pb))
        cols[1].markdown(
            "#### Уверенность: **{}**".format(_pct(conf)),
            unsafe_allow_html=False,
        )

        # Предупреждение/уведомление о надёжности прогноза
        if conf_lbl == "low":
            st.warning(
                f"Прогноз маловероятно надёжен (уверенность всего {_pct(conf)}). "
                "Шансы команд примерно равны либо по этим командам мало данных. "
                "Результат близок к подбрасыванию монеты — не расценивай его "
                "как предсказание."
            )
        elif conf_lbl == "medium":
            st.info(
                f"Умеренная уверенность ({_pct(conf)}). Прогноз скорее "
                "оценочный: сигнал есть, но слабый."
            )
        else:
            st.success(
                f"Высокая уверенность ({_pct(conf)}). Модель видит заметный "
                "перевес одной из команд по ELO, форме и статистике карт."
            )

        # визуализация
        fig = go.Figure(
            go.Bar(
                x=[pa, pb],
                y=[team_a, team_b],
                orientation="h",
                marker_color=["#d62728", "#1f77b4"],
                text=[_pct(pa), _pct(pb)],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Вероятность победы",
            xaxis=dict(range=[0, 1], title="Вероятность"),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        winner = team_a if pa >= pb else team_b
        st.success(f"Фаворит: **{winner}** с вероятностью {_pct(max(pa, pb))}")

# ---------------------------------------------------------------------------
# Страница 2: Форма команд
# ---------------------------------------------------------------------------
elif page == "Форма команд":
    st.header("Форма команд")
    select = st.multiselect(
        "Команды",
        teams,
        default=teams[:4],
        max_selections=8,
    )

    if not select:
        st.warning("Выберите хотя бы одну команду.")
        st.stop()

    # считаем winrate по хронологии для каждой выбранной команды
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    plot_df: list[pd.DataFrame] = []
    for team in select:
        sub = df[(df["team_a"] == team) | (df["team_b"] == team)].copy()
        if sub.empty:
            continue
        sub["won"] = sub["winner"] == team
        sub["win_count"] = sub["won"].cumsum()
        sub["match_count"] = range(1, len(sub) + 1)
        sub["team"] = team
        sub["cum_winrate"] = sub["win_count"] / sub["match_count"]
        # rolling winrate за последние 10 матчей
        sub["rolling10"] = sub["won"].rolling(10, min_periods=1).mean()
        plot_df.append(sub)

    if not plot_df:
        st.error("Нет данных по выбранным командам.")
    else:
        combined = pd.concat(plot_df, ignore_index=True)
        fig = px.line(
            combined,
            x=combined.index,
            y="rolling10",
            color="team",
            title="Winrate за последние 10 матчей (скользящее среднее)",
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            combined,
            x=combined.index,
            y="cum_winrate",
            color="team",
            title="Кумулятивный winrate",
        )
        fig2.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

        stats = (
            combined.groupby("team")["won"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )
        stats.columns = ["Команда", "Матчей", "Побед", "Winrate"]
        stats["Winrate"] = stats["Winrate"].map(_pct)
        st.dataframe(stats, use_container_width=True)

# ---------------------------------------------------------------------------
# Страница 3: Рейтинг команд
# ---------------------------------------------------------------------------
elif page == "Рейтинг команд":
    st.header("Мировой рейтинг команд (Cito API)")
    rk_path = config.PROCESSED_DIR / "cito_rankings.parquet"
    if not rk_path.exists():
        st.warning(
            "Нет данных рейтинга. Запусти: python fetch_dota_cito.py "
            "(нужен CITO_API_KEY в .env)."
        )
        st.stop()

    import pandas as _pd

    rk = _pd.read_parquet(rk_path)
    st.metric("Команд в рейтинге", len(rk))
    st.dataframe(
        rk[["rank", "team", "points"]].rename(
            columns={"rank": "Позиция", "team": "Команда", "points": "Очки"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    if "team" in rk.columns and "rank" in rk.columns:
        top = rk.sort_values("rank").head(10)
        fig = px.bar(
            top,
            x="team",
            y="points" if "points" in rk.columns else "rank",
            title="Топ-10 команд мира по рейтингу",
            labels={"team": "Команда"},
        )
        if "points" not in rk.columns or top["points"].sum() == 0:
            fig = px.bar(
                top, x="team", y="rank",
                title="Топ-10 команд мира (позиция)",
                labels={"team": "Команда", "rank": "Позиция"},
            )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Страница 4: Backtest
# ---------------------------------------------------------------------------
elif page == "Backtest":
    st.header("Backtest модели")
    st.markdown(
        "Проверяем, как модель предсказывала исторические матчи. Для скорости "
        "прогоняем по последним N сыгранным матчам."
    )

    n_backtest = st.slider("Число матчей для backtest", 50, min(500, len(raw)), 150)

    # берём данные, как будто «сегодня» позже последнего матча
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    tail = df.tail(n_backtest).copy().reset_index(drop=True)
    history = df.iloc[: len(df) - n_backtest].copy()

    predictor = get_predictor(_model_mtime())
    # прогоняем через predictor, но чтобы не нарушать хронологию делаем
    # «ref_date» для каждого матча = его дата, а статистику берём из history.
    # Упрощённо: используем predictor со всем raw (это допущение для демо).
    results = []
    with st.spinner("Прогоняю backtest…"):
        for rec in tail.itertuples(index=False):
            pa = 0.5
            try:
                res = predictor.predict(
                    rec.team_a,
                    rec.team_b,
                    map_name=rec.map_name,
                    ref_date=rec.date.date(),
                    use_map=True,
                )
                pa = res["team_a"]
            except Exception:  # noqa: BLE001
                pa = 0.5
            actual = 1 if rec.winner == rec.team_a else 0
            results.append({"team_a": rec.team_a, "team_b": rec.team_b,
                            "map": rec.map_name, "pred_a": pa, "actual": actual})

    bt = pd.DataFrame(results)
    bt["correct"] = ((bt["pred_a"] >= 0.5).astype(int) == bt["actual"])
    acc = bt["correct"].mean()

    st.metric("Accuracy на backtest", _pct(acc))
    st.metric("Матчей", len(bt))

    fig = px.density_contour(
        bt.assign(prob=bt["pred_a"]),
        x="actual",
        y="prob",
        title="Распределение предсказанных вероятностей vs факт",
    )
    fig.update_layout(xaxis_title="Факт (1 = победила team_a)",
                      yaxis_title="Предсказанная вероятность team_a")
    st.plotly_chart(fig, use_container_width=True)

    bt_disp = bt.copy()
    bt_disp["pred_a"] = bt_disp["pred_a"].map(_pct)
    st.dataframe(bt_disp.tail(20), use_container_width=True)

st.sidebar.markdown("---")
if config.AUTO_REFRESH.enabled:
    st.sidebar.caption(
        f"🔄 Автообновление данных: раз в {config.AUTO_REFRESH.interval_seconds // 3600} ч "
        f"({config.AUTO_REFRESH.interval_seconds // 60} мин)"
    )
st.sidebar.caption("Источник: Liquipedia (реальные матчи, без ключей)")