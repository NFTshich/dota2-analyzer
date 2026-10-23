"""Клиент Cito API (CS2 статистика) с локальным кэшем.

Cito API даёт рейтинги команд (world ranking / VRS), профили команд с
составами, игровые статы игроков (K/D, headshot, clutches) и картовые данные.
Бесплатный план — 500 запросов/месяц, поэтому каждый ответ кэшируется на диск
(``data/cito_cache/``), и повторные запросы не тратят лимит.

Ключ читается из окружения ``CITO_API_KEY`` (файл ``.env``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from loguru import logger

from . import config


class CitoClient:
    """Минимальный HTTP-клиент Cito API с дисковым кэшем.

    Args:
        api_key: ключ API. По умолчанию берётся из config.CITO_API_KEY.
        cache_dir: папка кэша. По умолчанию config.CITO_CACHE_DIR.
        ttl_seconds: срок жизни кэша (3600 = 1 час). Запросы старше TTL
            перечитываются.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        ttl_seconds: int = 3600,
        game: str = "dota2",
    ) -> None:
        self.api_key = api_key or config.CITO_API_KEY
        self.cache_dir = cache_dir or config.CITO_CACHE_DIR
        self.ttl = ttl_seconds
        self.game = game  # "dota2" или "cs2"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path:
        name = url.replace("/", "_").replace("?", "_").replace("&", "_")[:120]
        return self.cache_dir / f"{name}.json"

    def get(self, path: str, force: bool = False) -> dict[str, Any]:
        """GET-запрос к Cito API с кэшем.

        Args:
            path: путь эндпоинта (например ``/cs2/rankings``).
            force: игнорировать кэш и запросить заново.

        Returns:
            JSON-тело ответа (dict).
        """
        if not self.api_key:
            logger.warning("CITO_API_KEY не задан — пропускаю запрос {}", path)
            return {}

        url = config.CITO_BASE + path
        cache = self._cache_path(url)

        if not force and cache.exists():
            try:
                age = time.time() - cache.stat().st_mtime
                if age < self.ttl:
                    with open(cache, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:  # noqa: BLE001
                pass

        try:
            req = urllib.request.Request(
                url,
                headers={"x-api-key": self.api_key, "User-Agent": "CS2Analyzer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cito API ошибка {}: {}", path, exc)
            # fallback: вернуть кэш, если есть
            if cache.exists():
                with open(cache, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        # сохраняем кэш (только если ответ не пустой)
        if data.get("data"):
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        return data

    # --- высокоуровневые методы ------------------------------------------
    def world_ranking(self, limit: int = 100, game: str | None = None) -> list[dict]:
        """Мировой рейтинг команд (teamName, rank, points, region)."""
        g = game or self.game
        data = self.get(f"/{g}/rankings")
        rows = data.get("data", []) or []
        return rows[:limit]

    def recent_matches(self, game: str | None = None, limit: int = 200) -> list[dict]:
        """Свежие сыгранные матчи (team1/team2, score, winner, bestOf)."""
        g = game or self.game
        data = self.get(f"/{g}/matches/recent?limit={limit}")
        return data.get("data", []) or []

    def team_profile(self, team_slug: str, game: str | None = None) -> dict:
        """Профиль команды: roster, worldRanking, valveRanking."""
        g = game or self.game
        data = self.get(f"/{g}/teams/{team_slug}")
        return data.get("data", {}) or {}

    def player_stats(self, player_slug: str, game: str | None = None) -> dict:
        """Игровые статы игрока (totals: kills/deaths/assists/headshot)."""
        g = game or self.game
        data = self.get(f"/{g}/players/{player_slug}/stats")
        return data.get("data", {}) or {}


def main() -> None:
    """Точка входа: python -m src.cito — быстрая проверка ключа."""
    if not config.CITO_API_KEY:
        logger.error("CITO_API_KEY не задан. Добавь его в data/.env")
        return
    client = CitoClient()
    ranking = client.world_ranking(limit=5)
    logger.info("Мировой рейтинг (топ-5):")
    for r in ranking:
        logger.info(
            "  #{} {} — {} pts ({})",
            r.get("rank"), r.get("teamName"), r.get("points"), r.get("region"),
        )


if __name__ == "__main__":
    main()