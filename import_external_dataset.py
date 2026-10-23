"""Импорт внешнего открытого датасета матчей в базу проекта.

Принимает CSV в HLTV-зеркальном формате (по одной строке на КАРТУ матча):
  match_id, tournament, match_format, match_date, team_1, team_1_result,
  team_2, team_2_result, winner, map, ...

Группирует карты в матчи, берёт счёт серии из team_*_result, первую карту как
main_map, дедуплицирует против текущих data/raw/matches.csv (по нормализованной
паре команд + дате) и доливает только новые матчи.

Запуск:
  python import_external_dataset.py <путь_к_csv> [tier]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _conv_bo(s: str) -> int:
    s = (s or "").lower()
    if "bo5" in s:
        return 5
    if "bo1" in s:
        return 1
    return 3


def convert_hltv_frame(df: pd.DataFrame, tier: str = "A") -> pd.DataFrame:
    """Преобразует HLTV-зеркальный датафрейм (карты) в наш формат (матчи)."""
    rows: list[dict] = []
    for mid, g in df.groupby("match_id"):
        g = g.sort_values("map_n")
        first = g.iloc[0]
        team_a = str(first.get("team_1", "")).strip()
        team_b = str(first.get("team_2", "")).strip()
        if not team_a or not team_b:
            continue
        # счёт серии (присутствует на каждой карте одинаково)
        score_a = int(first.get("team_1_result", 0) or 0)
        score_b = int(first.get("team_2_result", 0) or 0)
        winner = first.get("winner", "")
        if not winner:
            winner = team_a if score_a > score_b else team_b
        date_str = str(first.get("match_date", "")).strip()
        date_iso = date_str.replace(" ", "T") if date_str else None
        maps = list(g["map"].dropna().astype(str))
        main_map = maps[0] if maps else ""
        event = str(first.get("tournament", "")).strip()
        rows.append({
            "match_id": f"ext_{mid}",
            "date": date_iso,
            "team_a": team_a,
            "team_b": team_b,
            "map_name": main_map,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
            "event_name": event,
            "tier": str(tier),
            "best_of": _conv_bo(str(first.get("match_format", ""))),
            "roster_a": "[]",
            "roster_b": "[]",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _dedup_against_existing(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только те матчи, которых ещё нет в existing_df (по паре+дата)."""
    def _key_from(df, ca, cb, cd):
        def f(r):
            a, b = sorted([str(r[ca]), str(r[cb])])
            return (_norm(a), _norm(b), str(pd.to_datetime(r[cd]).date()))
        return df.apply(f, axis=1)

    ex_keys = set(_key_from(existing_df, "team_a", "team_b", "date")) if not existing_df.empty else set()
    new_keys = _key_from(new_df, "team_a", "team_b", "date")
    mask = ~new_keys.isin(ex_keys)
    return new_df[mask].copy()


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Укажи путь к внешнему CSV: python import_external_dataset.py <file.csv> [tier]")
        sys.exit(1)

    src = sys.argv[1]
    tier = sys.argv[2] if len(sys.argv) > 2 else "A"
    logger.info("Читаю внешний датасет: {}", src)
    ext = pd.read_csv(src)
    new_matches = convert_hltv_frame(ext, tier=tier)
    logger.info("Из внешнего датасета собрано матчей: {}", len(new_matches))

    csv_path = config.MATCHES_RAW
    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    logger.info("В базе сейчас матчей: {}", len(existing))

    to_add = _dedup_against_existing(new_matches, existing)
    logger.info("Новых матчей (не было в базе): {}", len(to_add))
    if to_add.empty:
        logger.info("Нечего добавлять.")
        return

    if existing.empty:
        merged = to_add
    else:
        # приводим колонки existing к единому виду
        need = ["match_id", "date", "team_a", "team_b", "map_name", "score_a",
                "score_b", "winner", "event_name", "tier", "best_of", "roster_a", "roster_b"]
        for c in need:
            if c not in existing.columns:
                existing[c] = "[]" if c in ("roster_a", "roster_b") else ("0" if c.startswith("score") else "")
        existing = existing[need]
        to_add = to_add[need]
        merged = pd.concat([existing, to_add], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = (
            merged.sort_values("date")
            .drop_duplicates(subset=["match_id"], keep="last")
            .reset_index(drop=True)
        )

    merged.to_csv(csv_path, index=False)
    logger.info("База обновлена: теперь {} матчей (добавлено +{}).", len(merged), len(to_add))


if __name__ == "__main__":
    main()