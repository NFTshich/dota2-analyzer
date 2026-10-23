"""Вспомогательные функции проекта CS2 Analyzer.

Тут хранятся маленькие переиспользуемые утилиты: генерация синтетических данных,
отображение вероятностей, безопасные преобразования типов и т.п.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Синтетический датасет
# ---------------------------------------------------------------------------
def _team_pool(n_teams: int = 18) -> list[str]:
    """Возвращает набор правдоподобных названий команд Dota 2."""
    names = [
        "Team Spirit",
        "Team Liquid",
        "PARIVISION",
        "BetBoom Team",
        "Team Yandex",
        "Aurora Gaming",
        "Gaimin Gladiators",
        "XG",
        "Tundra Esports",
        "Falcons",
        "Team Falcons",
        "1WIN",
        "Balu Team",
        "Night Pulse",
        "XctJlPbIu Team",
        "Pipeline",
        "MOUZ",
        "Virtus.pro",
        "9Pandas",
        "L1GA Team",
        "Team Waska",
        "beastcoast",
    ]
    return names[:n_teams]


def _base_strength(name: str) -> float:
    """Детерминированная «базовая сила» команды от её имени (0..1)."""
    h = sum(ord(c) for c in name)
    return (h % 997) / 997.0


def generate_synthetic_matches(
    n_matches: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Генерирует синтетический датасет прошедших матчей.

    Используется как заглушка, когда реальный источник недоступен, чтобы проект
    запускался «из коробки». Логика устроена так, чтобы быть реалистичной, но
    не тривиальной: у каждой команды есть **латентная сила**, которая дрейфует
    во времени (случайное блуждание) и не видна модели напрямую. Исход матча
    зависит от разницы текущих сил плюс существенный шум, поэтому предсказание
    неидеально (accuracy ~0.65–0.75) — как в реальном киберспорте.

    Args:
        n_matches: число матчей. Если None — берётся из конфига.
        seed: сид ГПСЧ для воспроизводимости.

    Returns:
        DataFrame с колонками: match_id, date, team_a, team_b, event_name,
        tier, map_name, score_a, score_b, winner, rank_a, rank_b, best_of.
    """
    n_matches = n_matches or config.SCRAPER.synthetic_matches
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    teams = _team_pool()
    # текущая латентная сила каждой команды (0..1), будет дрейфовать
    strength = {t: _base_strength(t) for t in teams}
    tiers = ["S", "A", "B"]
    maps = list(config.MAPS) or [""]  # в Dota2 карт нет — пустая карта
    events_by_tier = {
        "S": ["The International", "ESL One", "DreamLeague"],
        "A": ["PGL Wallachia", "BetBoom Dacha"],
        "B": ["CCT Series", "RES Regional Finals"],
    }

    start = datetime(2024, 6, 1)
    end = datetime(2026, 10, 1)
    span_days = (end - start).days

    # генерируем матчи вместе с их датами, затем сортируем хронологически,
    # чтобы сила дрейфовала согласованно во времени
    raw_matches: list[dict] = []
    for _ in range(n_matches):
        raw_matches.append(
            {
                "pair": tuple(rng.sample(teams, 2)),
                "tier": rng.choices(tiers, weights=[0.25, 0.45, 0.30])[0],
                "map": rng.choice(maps),
                "best_of": rng.choices([1, 3, 5], weights=[0.4, 0.5, 0.1])[0],
                "day_offset": rng.randint(0, span_days),
            }
        )
    # проставляем event по тиру
    for m in raw_matches:
        m["event"] = rng.choice(events_by_tier[m["tier"]])
    raw_matches.sort(key=lambda m: m["day_offset"])

    rows: list[dict] = []
    for i, m in enumerate(raw_matches):
        a, b = m["pair"]
        map_name = m["map"]
        best_of = m["best_of"]
        mdate = (start + timedelta(days=m["day_offset"])).date()

        # дрейф силы: после каждого матча обе команды слегка меняют форму
        # (сильная команда может «просесть», слабая — неожиданно подняться)
        sa = strength[a]
        sb = strength[b]
        diff = sa - sb

        # вероятность победы A на карте: чем больше перевес, тем выше шанс,
        # но всегда есть заметная доля ансета (мин/макс подрезаем)
        p_a = 0.5 + 0.4 * np.tanh(1.5 * diff)

        # разыгрываем серию
        wins_a = wins_b = 0
        target = max(1, best_of // 2 + 1)
        while wins_a < target and wins_b < target:
            if rng.random() < p_a:
                wins_a += 1
            else:
                wins_b += 1

        winner = a if wins_a > wins_b else b

        # после матча обе команды меняются: победитель слегка усиливается
        # (в среднем), проигравший слабеет; плюс случайный дрейф формы
        if winner == a:
            strength[a] = np.clip(sa + np_rng.normal(0.015, 0.02), 0.05, 0.95)
            strength[b] = np.clip(sb + np_rng.normal(-0.01, 0.02), 0.05, 0.95)
        else:
            strength[b] = np.clip(sb + np_rng.normal(0.015, 0.02), 0.05, 0.95)
            strength[a] = np.clip(sa + np_rng.normal(-0.01, 0.02), 0.05, 0.95)

        # ранг: обратно текущей силе с заметным шумом (рейтинг всегда не точен
        # и отстаёт от реальной формы)
        rank_a = max(1, int(round((1 - sa) * 29 + 7 + np_rng.normal(0, 5))))
        rank_b = max(1, int(round((1 - sb) * 29 + 7 + np_rng.normal(0, 5))))

        rows.append(
            {
                "match_id": f"synth_{i:06d}",
                "date": mdate,
                "team_a": a,
                "team_b": b,
                "event_name": m["event"],
                "tier": m["tier"],
                "map_name": map_name,
                "score_a": wins_a,
                "score_b": wins_b,
                "winner": winner,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "best_of": best_of,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # пересоберём match_id строго по порядку даты
    df["match_id"] = [f"synth_{i:06d}" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Прочее
# ---------------------------------------------------------------------------
def probability_text(p: float) -> str:
    """Форматирует долю как процентную строку ('73.4%')."""
    return f"{max(0.0, min(1.0, p)) * 100:.1f}%"


def safe_bool(x: object, default: bool = False) -> bool:
    """Безопасно приводит объект к bool."""
    if isinstance(x, bool):
        return x
    return default


def today_isoweek_ago(days: int) -> date:
    """Дата, отстоящая на ``days`` назад от сегодня."""
    return date.today() - timedelta(days=days)


def team_list_from_frame(df: pd.DataFrame) -> list[str]:
    """Уникальный отсортированный список команд из датафрейма матчей."""
    teams: set[str] = set(df["team_a"]) | set(df["team_b"])
    return sorted(teams)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Ограничивает число диапазоном."""
    return max(lo, min(hi, x))