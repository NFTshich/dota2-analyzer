"""Обучение и валидация ML-модели предсказания победителя.

Решает бинарную задачу: ``target == 1`` означает победу ``team_a``. Валидация
выполняется через :class:`TimeSeriesSplit` — порядок матчей критичен, перемешивать
нельзя. Обучаются три кандидата (Logistic, RandomForest, GradientBoosting),
лучший по log-loss выбирается автоматически и сохраняется в ``models/model.pkl``.

Точка входа: ``python -m src.model``.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from . import config

# Колонки-признаки (не метаданные и не target).
META_COLS = [
    "match_id",
    "date",
    "team_a",
    "team_b",
    "map_name",
    "tier",
    "winner",
    "target",
]

CATEGORY_COLS = ["map_name", "team_a", "team_b", "tier"]


# ---------------------------------------------------------------------------
# Подготовка признаков
# ---------------------------------------------------------------------------
def _swap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Зеркалит признаки: меняет местами пары a_<X> / b_<X>.

    Возвращает копию датафрейма, где для каждого префикса ``a_``/``b_`` значения
    переставлены между двумя командами. Разностные признаки вида ``*_diff``
    меняют знак (они равны a-b и при перестановке становятся -(a-b)).
    Немаркированные колонки (map_x, is_bo1 и пр.) не трогаются. Применяется для
    устранения асимметрии модели к порядку команд: если пару (A, B) дополнить
    обратной (B, A), модель обучается инвариантно к перестановке.
    """
    cols = list(df.columns)
    new = df.copy()

    # разностные признаки меняют знак
    for c in cols:
        if c.endswith("_diff") and c in new.columns:
            new[c] = -df[c].to_numpy()

    for c in cols:
        if c.startswith("a_"):
            b = "b_" + c[2:]
            if b in cols:
                new[c] = df[b].to_numpy()
                new[b] = df[c].to_numpy()
    # team_a/team_b / match_id тоже меняем местами для согласованности
    for ident in ("team_a", "team_b"):
        other = "team_b" if ident == "team_a" else "team_a"
        if ident in cols and other in cols:
            new[ident] = df[other].to_numpy()
            new[other] = df[ident].to_numpy()
    if "match_id" in cols:
        new["match_id"] = df["match_id"].astype(str) + "_mir"
    return new


def build_X_y(
    features: pd.DataFrame,
    drop_cats: bool = False,
    mirror: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Разделяет фиче-таблицу на признаковую матрицу и целевой вектор.

    Args:
        features: датафрейм из :func:`src.features.build_features`.
        drop_cats: True — убрать и категориальные, и one-hot карты (для базовой
            интерпретируемой LogisticRegression); False — оставить все числовые.
        mirror: True — продублировать каждую строку зеркальным вариантом
            (A↔B, target 0→1). Устраняет сдвиг, когда в данных команда A
            систематически побеждает чаще (это ломает инвариантность к порядку).

    Returns:
        (X, y) — признаки и массив целевой переменной (0/1).
    """
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "winner" in df.columns:
        df = df.drop(columns=["winner"])

    # зеркалим: добавляем обратную копию каждого матча
    if mirror:
        flipped = _swap_columns(df)
        flipped["target"] = 1 - flipped["target"]
        df = pd.concat([df, flipped], ignore_index=True)
        # перемешиваем, чтобы модель не видела «чистые» и «зеркальные» блоками
        df = df.sample(frac=1.0, random_state=config.MODEL.random_state).reset_index(
            drop=True
        )

    drop = [c for c in META_COLS if c in df.columns]
    # one-hot карт map_<M> и is_bo1
    map_cols = [c for c in df.columns if c.startswith("map_")]
    if drop_cats:
        drop += map_cols + ["is_bo1"]

    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df["target"].to_numpy(dtype=int)
    return X, y


def feature_names_for(
    X: pd.DataFrame, drop_cats: bool = False  # noqa: ARG001
) -> list[str]:
    """Имена признаков, которые получит модель."""
    return list(X.columns)


# ---------------------------------------------------------------------------
# Кандидаты моделей
# ---------------------------------------------------------------------------
def _make_models(random_state: int) -> dict[str, Any]:
    """Возвращает словарь имя → (модель, calibration)."""
    lm = LogisticRegression(
        C=config.MODEL.logreg_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=config.MODEL.rf_estimators,
        max_depth=config.MODEL.rf_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    gb = GradientBoostingClassifier(
        n_estimators=config.MODEL.gb_estimators,
        max_depth=config.MODEL.gb_depth,
        learning_rate=config.MODEL.gb_lr,
        random_state=random_state,
    )
    return {
        "LogisticRegression": lm,
        "RandomForest": rf,
        "GradientBoosting": gb,
    }


def _calibrate(
    model: Any, X_tr: pd.DataFrame, y_tr: np.ndarray
) -> Any:
    """Платоническая калибровка вероятностей (Isotonic) на отложенной выборке.

    Возвращает обёртку :class:`CalibratedClassifierCV`, потому что наши исходные
    вероятности (особенно у RF/GB) не всегда хорошо откалиброваны.
    """
    return CalibratedClassifierCV(model, method="isotonic", cv=3)


# ---------------------------------------------------------------------------
# Кросс-валидация
# ---------------------------------------------------------------------------
def evaluate_models(
    X: pd.DataFrame, y: np.ndarray, n_splits: int | None = None
) -> dict[str, dict[str, float]]:
    """Оценивает кандидатов на TimeSeriesSplit и возвращает метрики.

    Args:
        X: признаки.
        y: целевой вектор.
        n_splits: число разбиений.

    Returns:
        dict: {"<model>": {"accuracy": ..., "log_loss": ..., "roc_auc": ...}}.
    """
    n_splits = n_splits or config.MODEL.cv_splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = _make_models(config.MODEL.random_state)

    results: dict[str, dict[str, float]] = {}
    for name, base in models.items():
        accs: list[float] = []
        losses: list[float] = []
        aucs: list[float] = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            try:
                model = _calibrate(base, X_tr, y_tr)
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_va)[:, 1]
            except Exception as exc:  # noqa: BLE001
                logger.warning("{} failed on fold: {}", name, exc)
                continue

            pred = (proba >= 0.5).astype(int)
            accs.append(accuracy_score(y_va, pred))
            losses.append(log_loss(y_va, proba))
            if len(np.unique(y_va)) == 2:
                aucs.append(roc_auc_score(y_va, proba))

        results[name] = {
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "log_loss": float(np.mean(losses)) if losses else float("inf"),
            "roc_auc": float(np.mean(aucs)) if aucs else 0.0,
        }
        logger.info(
            "{}  →  acc={:.3f}  log_loss={:.3f}  auc={:.3f}",
            name,
            results[name]["accuracy"],
            results[name]["log_loss"],
            results[name]["roc_auc"],
        )
    return results


# ---------------------------------------------------------------------------
# Обучение финальной модели
# ---------------------------------------------------------------------------
def train_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    candidates: dict[str, Any] | None = None,
    return_ensemble: bool = False,
) -> tuple[Any, dict[str, float], str, list[str]]:
    """Обучает лучшего кандидата (или ансамбль) на всех данных.

    Args:
        X: признаки.
        y: целевой вектор.
        candidates: переопределить кандидатов (для тестов).
        return_ensemble: True — вернуть dict {имя: calibrated-model} из всех
            кандидатов (ансамбль, усредняемый при инференсе). False — только
            лучший по логи-лоссу.

    Returns:
        (model_or_dict, metrics, best_name, feature_names)
    """
    candidates = candidates or _make_models(config.MODEL.random_state)
    metrics = evaluate_models(X, y)

    best_name = min(
        candidates.keys(),
        key=lambda k: metrics[k]["log_loss"],
    )
    best_base = _calibrate(candidates[best_name], X, y)
    best_base.fit(X, y)

    # метрики на полных данных для справки
    proba_full = best_base.predict_proba(X)[:, 1]
    full_metrics = {
        "best_model": best_name,
        "cv_accuracy": metrics[best_name]["accuracy"],
        "cv_log_loss": metrics[best_name]["log_loss"],
        "cv_roc_auc": metrics[best_name]["roc_auc"],
        "final_accuracy": float(accuracy_score(y, (proba_full >= 0.5).astype(int))),
        "final_log_loss": float(log_loss(y, proba_full)),
    }

    if not return_ensemble:
        return best_base, full_metrics, best_name, list(X.columns)

    # ансамбль: обучаем всех кандидатов и сохраняем словарь
    ensemble: dict[str, Any] = {}
    for name, base in candidates.items():
        m = _calibrate(base, X, y)
        m.fit(X, y)
        ensemble[name] = m
    return ensemble, full_metrics, best_name, list(X.columns)


def _ensemble_proba(ensemble: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Усредняет вероятности моделей ансамбля (среднее арифметическое)."""
    probs = [m.predict_proba(X)[:, 1] for m in ensemble.values()]
    return np.mean(np.stack(probs, axis=1), axis=1)


def is_ensemble(model: Any) -> bool:
    """True, если сохранённый объект — словарь ансамбля из нескольких моделей."""
    return isinstance(model, dict) and len(model) > 0


def save_model(
    model: Any, metrics: dict[str, float], feature_names: list[str]
) -> None:
    """Сохраняет модель, метаданные и имена признаков в models/."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    meta = {
        "metrics": metrics,
        "feature_names": feature_names,
        "created": pd.Timestamp.now().isoformat(),
    }
    with open(config.MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Сохранено: {} и {}", config.MODEL_PATH, config.MODEL_META_PATH)


def main() -> None:
    """Точка входа: ``python -m src.model`` — обучает и сохраняет модель."""
    feat_path = config.MATCHES_PROCESSED
    if not feat_path.exists():
        logger.error(
            "Фичи не найдены: {}. Сначала выполни: python -m src.features",
            feat_path,
        )
        raise SystemExit(1)

    features = pd.read_parquet(feat_path)
    X, y = build_X_y(features)
    logger.info("Размер выборки: {} строк, {} признаков", X.shape[0], X.shape[1])

    model, metrics, best_name, feature_names = train_best_model(
        X, y, return_ensemble=True
    )
    logger.info("Ансамбль обучен ({}, кандидатов). Лучший сингл: {}", len(model), best_name)
    save_model(model, metrics, feature_names)

    print("\n=== Итоговые метрики (TimeSeriesSplit) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()