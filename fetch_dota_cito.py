"""Сбор данных Dota 2 для модели.

Основной источник — Cito API (свежие сыгранные матчи + рейтинг команд).
Дополнительно подмешиваются старые матчи The International из Liquipedia,
чтобы дать больше истории.

Результаты:
  data/raw/matches.csv   — матчи (наш формат: team_a, team_b, winner, дата...)
  data/processed/cito_rankings.parquet — рейтинг команд Dota

Запуск: python fetch_dota_cito.py
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.cito import CitoClient  # noqa: E402

#: пауза между Cito-запросами (бесплатный план 10/мин)
RATE_LIMIT_WAIT = 6.5


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fetch_cito_matches(client: CitoClient) -> pd.DataFrame:
    """Тянет свежие завершённые матчи Cito Dota в наш формат."""
    matches = client.recent_matches(game="dota2", limit=500)
    rows = []
    for m in matches:
        status = m.get("status")
        winner_id = m.get("winnerTeamId")
        if status not in ("completed", "finished", "ended") and not winner_id:
            continue
        t1, t2 = m.get("team1Name"), m.get("team2Name")
        s1, s2 = m.get("team1Score"), m.get("team2Score")
        if not t1 or not t2:
            continue
        winner = t1 if winner_id and winner_id == m.get("team1Id") else (t2 if winner_id else None)
        if winner is None and s1 is not None and s2 is not None:
            winner = t1 if s1 > s2 else t2
        bo = m.get("bestOf") or 3
        rows.append({
            "match_id": f"cito_{m.get('id')}",
            "date": m.get("startsAt"),
            "team_a": t1,
            "team_b": t2,
            "map_name": "",
            "score_a": int(s1 or 0),
            "score_b": int(s2 or 0),
            "winner": winner,
            "event_name": m.get("tournamentName") or "",
            "tier": "A",
            "best_of": bo,
            "roster_a": "[]",
            "roster_b": "[]",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    logger.info("Cito Dota завершённых матчей: {}", len(df))
    return df


def fetch_liquipedia_ti(client_lp) -> pd.DataFrame:
    """Старые матчи The International из Liquipedia (историческая глубина)."""
    from src.liquipedia import LiquipediaLoader

    pages = [
        "The International/2011/Group Stage", "The International/2011/Playoffs",
        "The International/2012/Group Stage", "The International/2012/Playoffs",
        "The International/2013/Group Stage", "The International/2013/Playoffs",
    ]
    loader = LiquipediaLoader()
    ms = []
    for p in pages:
        try:
            ms += loader.scrape_tournament_pages([p], include_subpages=False)
            logger.info("  {} -> {} матчей", p, len(ms) - (len(ms) - 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("{}: {}", p, exc)
    df = loader.to_frame(ms)
    logger.info("Liquipedia TI матчей: {}", len(df))
    return df


def fetch_rankings(client: CitoClient) -> pd.DataFrame:
    rk = client.world_ranking(game="dota2", limit=100)
    df = pd.DataFrame([
        {"team": r.get("teamName"), "team_id": r.get("teamId"),
         "rank": r.get("rank"), "points": r.get("points") or 0}
        for r in rk
    ])
    df.to_parquet(config.PROCESSED_DIR / "cito_rankings.parquet", index=False)
    logger.info("Рейтинг Dota: {} команд", len(df))
    return df


def main() -> None:
    if not config.CITO_API_KEY:
        logger.error("CITO_API_KEY не задан. Добавь его в data/.env")
        sys.exit(1)

    client = CitoClient(game="dota2")
    fetch_rankings(client)
    time.sleep(RATE_LIMIT_WAIT)
    cito_df = fetch_cito_matches(client)

    ti_df = fetch_liquipedia_ti(client)

    dfs = [d for d in (cito_df, ti_df) if d is not None and not d.empty]
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
        all_df = all_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        all_df.to_csv(config.MATCHES_RAW, index=False)
        logger.info("Итог матчей Dota: {} ({} — {})", len(all_df),
                    all_df["date"].min().date(), all_df["date"].max().date())
    else:
        logger.error("Нет данных Dota ни из одного источника.")


if __name__ == "__main__":
    main()