"""Конфигурация проекта CS2 Analyzer.

Все пути, настройки и параметры вынесены сюда — никаких хардкодов в остальном коде.
Значения, которые пользователь может переопределить, читаются из переменных окружения
(файл ``.env``) через ``python-dotenv``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Базовые пути. Проект строится вокруг корневой папки, в которой лежит src/ и data/.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Убеждаемся, что папки существуют (idempotent).
for _dir in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Загрузка .env (не ошибка, если файла нет — тогда работают дефолты).
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")


def _env_str(key: str, default: str) -> str:
    """Читает строку из окружения с дефолтом."""
    value = os.getenv(key)
    return value.strip() if value else default


def _env_int(key: str, default: int) -> int:
    """Читает целое число из окружения с дефолтом."""
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Параметры сбора данных.
# ---------------------------------------------------------------------------
class ScraperConfig(BaseModel):
    """Параметры, управляющие сбором матчей."""

    #: число матчей в синтетическом датасете-заглушке
    synthetic_matches: int = Field(default=1200, ge=100, le=10000)
    #: задержка между запросами к HLTV (секунды) — уважаем rate-limit
    request_delay: float = Field(default=2.5, ge=0.5, le=10.0)
    #: таймаут HTTP-запросов (секунды)
    timeout: int = Field(default=15, ge=5, le=60)
    #: User-Agent, чтобы сайт не блокировал по умолчанию
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    )
    #: режим HLTV: "sync" — через свой парсер, "mock" — только синтетика
    mode: str = Field(default="mock")


class LiquipediaConfig(BaseModel):
    """Параметры источника Liquipedia.

    Liquipedia — открытый, без ключей источник; используется в режиме ``sync``.
    """

    #: поддомен Liquipedia: "dota2" — Dota 2, "counterstrike" — CS2
    wiki: str = Field(default="dota2")
    #: категории, из которых берём завершённые турниры
    categories: tuple[str, ...] = ("Finished Tournaments",)
    #: максимум страниц турниров при авто-обходе
    max_pages: int = Field(default=40, ge=1, le=200)
    #: список конкретных страниц турниров (обход по умолчанию)
    tournament_pages: tuple[str, ...] = (
        # --- The International (главный турнир) ---
        "The International/2021",
        "The International/2022",
        "The International/2023",
        "The International/2024",
        # --- DreamLeague (долгая серия турниров) ---
        "DreamLeague/Season 20",
        "DreamLeague/Season 21",
        "DreamLeague/Season 22",
        "DreamLeague/Season 23",
        "DreamLeague/Season 24",
        "DreamLeague/Season 25",
        "DreamLeague/Season 26",
        "DreamLeague/Season 27",
        # --- ESL One ---
        "ESL One/Birmingham/2024",
        "ESL One/Bangkok/2024",
        "ESL One/Riyadh/2024",
        # --- Прочие крупные ---
        "Esports World Cup/2024",
        "BetBoom Dacha/2024/2",
        "PGL/2024/Cluj-Napoca",
        "Riyadh Masters/2024",
        "Blast/Slam/2025/1",
    )
    #: при обходе категорий искать также дочерние страницы турнира
    include_subpages: bool = True


class AutoRefreshConfig(BaseModel):
    """Настройки автообновления данных в фоне.

    Когда Streamlit-приложение запущено постоянic, фоновый поток раз в
    ``interval_seconds`` докачивает новые матчи с Liquipedia и (если появились
    новые) переобучает модель. Управляется переменной окружения.
    """

    #: включить автообновление?
    enabled: bool = True
    #: период между пылесосами (секунды) — по умолчанию 3600 (раз в час)
    interval_seconds: int = Field(default=3600, ge=300, le=86400)
    #: доля запросов, передаваемая в скрапер (задержка между запросами)
    request_delay: float = 1.5


# ---------------------------------------------------------------------------
# Параметры фиче-инжиниринга.
# ---------------------------------------------------------------------------
class FeatureConfig(BaseModel):
    """Окна расчёта признаков."""

    #: окна формы команды (winrate за последние N матчей)
    form_windows: tuple[int, ...] = (5, 10, 30)
    #: окно recent form (последние 30 дней) — в днях
    recent_days: int = 30
    #: штраф за недавнюю замену в составе (доли к вероятности), если данные есть
    roster_penalty: float = 0.03


# ---------------------------------------------------------------------------
# Параметры модели.
# ---------------------------------------------------------------------------
class ModelConfig(BaseModel):
    """Гиперпараметры и прочее для обучения."""

    #: baseline-модель
    logreg_c: float = 0.5
    #: продвинутая модель
    rf_estimators: int = 300
    rf_depth: int | None = 7
    gb_estimators: int = 200
    gb_depth: int = 3
    gb_lr: float = 0.05
    #: валидация
    cv_splits: int = 5
    random_state: int = 42


# ---------------------------------------------------------------------------
# Единый объект-конфиг, к которому обращается весь код.
# ---------------------------------------------------------------------------

#: пути, загружаются из .env (префикс CS2_)
MATCHES_RAW = RAW_DIR / _env_str("CS2_MATCHES_RAW", "matches.csv")
MATCHES_PROCESSED = PROCESSED_DIR / _env_str(
    "CS2_MATCHES_PROCESSED", "features.parquet"
)
MODEL_PATH = MODELS_DIR / _env_str("CS2_MODEL_PATH", "model.pkl")
MODEL_META_PATH = MODELS_DIR / _env_str("CS2_MODEL_META", "model_meta.json")

SCRAPER = ScraperConfig(
    synthetic_matches=_env_int("CS2_SYNTHETIC_MATCHES", 1200),
    request_delay=_env_int("CS2_REQUEST_DELAY", 250) / 100.0,
    mode=_env_str("CS2_SCRAPER_MODE", "sync"),
)
LIQUIPEDIA = LiquipediaConfig(
    wiki=_env_str("CS2_LIQUIPEDIA_WIKI", "dota2"),
    max_pages=_env_int("CS2_LIQUIPEDIA_MAX_PAGES", 40),
)
AUTO_REFRESH = AutoRefreshConfig(
    enabled=_env_str("CS2_AUTO_REFRESH", "1").lower() not in ("0", "false", "no"),
    interval_seconds=_env_int("CS2_AUTO_REFRESH_INTERVAL", 3600),
)
#: ключ Cito API (CS2 статистика) — из .env
CITO_API_KEY = _env_str("CITO_API_KEY", "")
#: базовый URL Cito API
CITO_BASE = "https://api.citoapi.com/api/v1"
#: папка кэша Cito-запросов (чтобы не жечь лимит 500/мес)
CITO_CACHE_DIR = DATA_DIR / "cito_cache"
CITO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = FeatureConfig()
MODEL = ModelConfig()

#: в Dota 2 нет карт (в отличие от CS2) — one-hot по картам не нужен
MAPS: tuple[str, ...] = ()