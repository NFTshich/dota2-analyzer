"""Сбор данных о матчах CS2.

Класс :class:`Scraper` умеет собирать матчи из двух источников:

* **Синтетический датасет** (режим ``mock``) — заглушка, которая генерирует
  правдоподобные данные, чтобы проект запускался сразу, без ключей и сети.
* **Liquipedia** (режим ``sync``) — открытый API свободной вики, без ключей.
  Реальный источник данных о турнирах и матчах CS2.

Данные кэшируются в ``data/raw/matches.csv``, чтобы не дёргать источник каждый раз.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from . import config
from .utils import generate_synthetic_matches


class Scraper:
    """Загружает матчи в ``data/raw/matches.csv``.

    Args:
        matches_path: путь к кэшу сырых матчей (CSV).
        mode: режим источника — ``mock`` (синтетика) или ``sync`` (Liquipedia).
    """

    def __init__(
        self,
        matches_path: Path | None = None,
        mode: str | None = None,
    ) -> None:
        self.matches_path = matches_path or config.MATCHES_RAW
        self.mode = mode or config.SCRAPER.mode
        self.session = None  # используются для сетевых запросов в режиме sync

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def fetch(self, force: bool = False) -> pd.DataFrame:
        """Возвращает датафрейм матчей, при необходимости обновляя кэш.

        Args:
            force: если True — пересканировать источник и перезаписать кэш.

        Returns:
            DataFrame прошедших матчей.
        """
        if self.matches_path.exists() and not force:
            logger.info("Загружаю кэш сырых матчей: {}", self.matches_path)
            df = self._read_cache()
            if not df.empty:
                return df

        if self.mode == "mock":
            df = self._fetch_synthetic()
        else:
            df = self._fetch_liquipedia()

        self._save_cache(df)
        return df

    def update(self) -> pd.DataFrame:
        """Инкрементальное обновление кэша свежими матчами.

        Собирает текущие матчи с Liquipedia и **сливает** их с уже сохранёнными
        в ``matches.csv`` (без дублей по ``match_id``). Используется фоновым
        автообновлением — дешевле полного пересканирования.

        Returns:
            Обновлённый DataFrame всех матчей.
        """
        # уже сохранённые матчи
        existing = self._read_cache()

        if self.mode == "mock":
            fresh = self._fetch_synthetic()
        else:
            fresh = self._fetch_liquipedia()

        if fresh.empty:
            return existing

        if existing.empty:
            merged = fresh
        else:
            # union по match_id: оставляем ВСЕ старые записи + новые из fresh.
            # Так кэш никогда не уменьшается из-за временных сбоев Liquipedia.
            by_id = pd.concat([existing, fresh], ignore_index=True)
            merged = (
                by_id.sort_values(["date", "match_id"])
                .drop_duplicates(subset=["match_id"], keep="last")
                .reset_index(drop=True)
            )

        self._save_cache(merged)
        logger.info(
            "Обновление: было {} матчей, стало {} (+{} новых).",
            len(existing), len(merged), max(0, len(merged) - len(existing)),
        )
        return merged

    # ------------------------------------------------------------------
    # Режим mock
    # ------------------------------------------------------------------
    def _fetch_synthetic(self) -> pd.DataFrame:
        """Генерирует синтетический датасет."""
        logger.info("Генерирую синтетический датасет (mode=mock).")
        return generate_synthetic_matches(n_matches=config.SCRAPER.synthetic_matches)

    # ------------------------------------------------------------------
    # Режим sync (Liquipedia)
    # ------------------------------------------------------------------
    def _fetch_liquipedia(self) -> pd.DataFrame:
        """Собирает реальные матчи с открытого API Liquipedia.

        Обходит страницы турниров, указанные в конфиге (или категории
        завершённых турниров). Если парсер ничего не вернул — прозрачный
        fallback на синтетику, чтобы пайплайн не падал.

        Returns:
            DataFrame реальных матчей в формате проекта.
        """
        from .liquipedia import LiquipediaLoader

        loader = LiquipediaLoader()
        pages = list(config.LIQUIPEDIA.tournament_pages)
        logger.info("Собираю реальные матчи с Liquipedia ({} страниц)…", len(pages))
        matches = loader.scrape_tournament_pages(
            pages, include_subpages=config.LIQUIPEDIA.include_subpages
        )
        df = loader.to_frame(matches)
        logger.info("Извлечено реальных матчей: {}", len(df))

        if df.empty:
            logger.warning(
                "Liquipedia не вернула матчей — fallback на синтетику."
            )
            return self._fetch_synthetic()
        return df

    # ------------------------------------------------------------------
    # Кэш
    # ------------------------------------------------------------------
    def _read_cache(self) -> pd.DataFrame:
        """Читает CSV из кэша."""
        try:
            df = pd.read_csv(self.matches_path, parse_dates=["date"])
            return df
        except Exception as exc:  # noqa: BLE001 - поверхностная ошибка кэша
            logger.warning("Не удалось прочитать кэш: {}", exc)
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame) -> None:
        """Пишет датафрейм в CSV."""
        self.matches_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.matches_path, index=False)
        logger.info(
            "Сохранено {} матчей в {}", len(df), self.matches_path
        )


def main() -> None:
    """Точка входа: ``python -m src.scraper``."""
    from .config import MATCHES_RAW
    from .utils import probability_text

    scraper = Scraper(matches_path=MATCHES_RAW)
    df = scraper.fetch(force=True)
    logger.info("Всего матчей в датасете: {}", len(df))
    if not df.empty:
        logger.info("Период: {} — {}", df["date"].min().date(), df["date"].max().date())
        winrate = df["winner"].value_counts(normalize=True).head(5)
        for team, p in winrate.items():
            logger.info("  {}  →  перевес по победам {}", team, probability_text(p))


if __name__ == "__main__":
    main()