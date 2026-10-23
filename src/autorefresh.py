"""Фоновое автообновление данных CS2 Analyzer.

Когда Streamlit-приложение запущено постоянно, крайне желательно докачивать
свежие матчи (и переобучать модель), не требуя ручного перезапуска. Этот модуль
предоставляет :func:`start_autorefresh` — поток-демон, который раз в период:

1. Пересканирует Liquipedia (``Scraper.fetch(force=True)``) — дописывает новые
   сырые матчи в ``data/raw/matches.csv``.
2. Если появились новые матчи — пересобирает фичи
   (``data/processed/features.parquet``) и переобучает модель
   (``models/model.pkl``).

Поток логирует прогресс через ``loguru``. Автообновление безопасно запускать
фоново: оно не трогает данные, пока приложение их читает, а читает приложение
из кэша памяти локального сервера Streamlit (не из файлов во время записи).

Управляется настройками ``config.AUTO_REFRESH`` и переменными окружения
(``CS2_AUTO_REFRESH``, ``CS2_AUTO_REFRESH_INTERVAL``).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger

from . import config


def _refresh_once() -> bool:
    """Выполняет один цикл обновления.

    Returns:
        True, если появились новые матчи и модель переобучена; иначе False.
    """
    from .scraper import Scraper

    scraper = Scraper()
    # запоминаем, сколько матчей было до обновления
    old_count = 0
    if config.MATCHES_RAW.exists():
        import pandas as pd

        try:
            old_df = pd.read_csv(config.MATCHES_RAW)
            old_count = len(old_df)
        except Exception:  # noqa: BLE001
            old_count = 0

    logger.info("Автообновление: сканирую Liquipedia…")
    # инкрементальный merge: доливаем только новые матчи
    df = scraper.update()

    if len(df) <= old_count:
        logger.info("Автообновление: новых матчей нет ({} == {}).", old_count, len(df))
        return False

    logger.info("Автообновление: новых матчей {} → {}. Переобучаю…", old_count, len(df))
    # пересчёт фич и модели
    from .features import build_features
    from .model import build_X_y, save_model, train_best_model

    features = build_features(df, str(config.MATCHES_PROCESSED))
    X, y = build_X_y(features)
    model, metrics, _name, feature_names = train_best_model(X, y, return_ensemble=True)
    save_model(model, metrics, feature_names)
    logger.info("Автообновление: модель переобучена. AUC={:.3f}", metrics.get("cv_roc_auc"))
    return True


def _loop(enabled: bool, interval: int) -> None:
    """Основной цикл фонового потока."""
    if not enabled:
        logger.info("Автообновление отключено (CS2_AUTO_REFRESH=0).")
        return
    logger.info("Автообновление запущено: раз в {} сек.", interval)

    # не запускаем обновление мгновенно — даём приложению стартовать
    time.sleep(interval)

    while True:
        try:
            _refresh_once()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Автообновление: ошибка цикла: {}", exc)
        time.sleep(interval)


def start_autorefresh() -> threading.Thread:
    """Запускает фоновый поток-демон автообновления (idempotent).

    Returns:
        Объект потока.
    """
    cfg = config.AUTO_REFRESH
    t = threading.Thread(
        target=_loop,
        args=(cfg.enabled, cfg.interval_seconds),
        name="cs2-autorefresh",
        daemon=True,
    )
    t.start()
    return t