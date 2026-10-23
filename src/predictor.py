"""Обёртка для инференса обученной модели.

Позволяет из «сырого» запроса — две команды и карта — получить вероятности
P(team_a win) и P(team_b win) в процентах. Для этого строит новые признаки
строго по принципу «не заглядывать в будущее»: статистика берётся только по
матчам до указанной референс-даты.

Если сохранённой модели нет — автоматически обучает её заново.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from . import config


class Predictor:
    """Предсказание исхода матча.

    Args:
        raw_df: датафрейм прошедших матчей (для построения признаков).
        model_path: путь к сохранённой модели.
        meta_path: путь к метаданным (имена признаков).
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        model_path: Path | None = None,
        meta_path: Path | None = None,
    ) -> None:
        self.raw_df = raw_df
        self.model_path = model_path or config.MODEL_PATH
        self.meta_path = meta_path or config.MODEL_META_PATH
        self.model: Any = None
        self.feature_names: list[str] = []
        self._load_or_train()

    # ------------------------------------------------------------------
    # Загрузка/обучение
    # ------------------------------------------------------------------
    def _load_or_train(self) -> None:
        """Загружает модель; если нет — обучает и сохраняет."""
        if self.model_path.exists() and self.meta_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.feature_names = meta["feature_names"]
                logger.info("Модель загружена: {}", self.model_path)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Не удалось загрузить модель: {}", exc)

        logger.info("Модель не найдена — обучаю заново.")
        self._train_fresh()

    def _train_fresh(self) -> None:
        """Обучает модель с нуля по полному датасету и сохраняет."""
        from .features import build_features
        from .model import build_X_y, save_model, train_best_model

        features = build_features(self.raw_df)
        X, y = build_X_y(features)
        model, metrics, _name, feature_names = train_best_model(
            X, y, return_ensemble=True
        )
        save_model(model, metrics, feature_names)
        self.model = model
        self.feature_names = feature_names
        logger.info("Ансамбль обучен и сохранён.")

    def _model_proba(self, X: pd.DataFrame) -> float:
        """Вероятность победы team_a по текущей модели (сингл или ансамбль)."""
        from .model import _ensemble_proba, is_ensemble

        if is_ensemble(self.model):
            return float(_ensemble_proba(self.model, X)[0])
        return float(self.model.predict_proba(X)[0, 1])

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------
    def predict(
        self,
        team_a: str,
        team_b: str,
        map_name: str,
        ref_date: date | None = None,
        use_map: bool = True,
    ) -> dict[str, float]:
        """Возвращает вероятности победы для пары команд на карте.

        Args:
            team_a: первая команда.
            team_b: вторая команда.
            map_name: карта (Ancient/…).
            ref_date: дата, относительно которой строится статистика
                (по умолчанию сегодня; будущее не используется).
            use_map: учитывать ли карту в признаках.

        Returns:
            dict: {"team_a": <0..1>, "team_b": <0..1>, "confidence": <0..1>,
                   "model_feature_score": ...}
        """
        ref_date = ref_date or date.today()
        row = self._build_feature_row(
            team_a, team_b, map_name, ref_date, use_map=use_map
        )
        # приводим к тому же набору колонок, что и при обучении
        X = self._align_features(pd.DataFrame([row]))
        proba = float(self._model_proba(X))

        # Симметричное усреднение: считаем вероятность в двух порядках команд
        # и усредняем, чтобы прогноз не зависел от того, кто стоит первым.
        row_rev = self._build_feature_row(
            team_b, team_a, map_name, ref_date, use_map=use_map
        )
        X_rev = self._align_features(pd.DataFrame([row_rev]))
        proba_rev = float(self._model_proba(X_rev))
        # proba_rev = P(B побеждает при порядке B,A). Значит P(A побеждает) = 1 - proba_rev.
        p_from_rev = 1.0 - proba_rev
        p_a = 0.5 * (proba + p_from_rev)
        p_a = float(np.clip(p_a, 1e-4, 1 - 1e-4))
        p_b = 1.0 - p_a
        confidence = 2.0 * abs(p_a - 0.5)  # 0..1, чем дальше от 0.5 — тем увереннее
        return {
            "team_a": p_a,
            "team_b": p_b,
            "confidence": confidence,
            "confidence_label": _confidence_label(confidence),
            "map_name": map_name,
        }

    def predict_series(
        self,
        team_a: str,
        team_b: str,
        ref_date: date | None = None,
    ) -> dict:
        """Прогноз BO3-серии через симуляцию вето карт.

        Общий алгоритм (по кейсу map-specific Elo):
        1. Строит трекеры команд на дату, достаёт map-Elo на всех картах.
        2. Симулирует вето: каждая команда банит свою худшую карту, затем
           пикает лучшие; третья — десайдер.
        3. Считает вероятность A на каждой из 3 карт через predict().
        4. Комбинирует в вероятность победы в серии: P2-0 + P2-1.

        Returns:
            dict: {"p_a": <0..1>, "p_b": <0..1>, "maps": [...], "map_probs": [...]}
        """
        ref_date = ref_date or date.today()
        past = self.raw_df[
            pd.to_datetime(self.raw_df["date"]).dt.date < ref_date
        ]
        if past.empty:
            past = self.raw_df
        trackers = _build_trackers(past)
        ta_t = trackers[team_a]
        tb_t = trackers[team_b]

        # вето: баним худшие карты, пикаем лучшие
        maps = list(config.MAPS)
        # команда A банит карту, где её map-Elo наихудший
        ban_a = min(maps, key=lambda m: ta_t.map_rating(m))
        maps.remove(ban_a)
        ban_b = min(maps, key=lambda m: tb_t.map_rating(m))
        maps.remove(ban_b)
        # A пикает лучшую, B пикает лучшую из оставшихся, третья — десайдер
        pick_a = max(maps, key=lambda m: ta_t.map_rating(m))
        maps.remove(pick_a)
        pick_b = max(maps, key=lambda m: tb_t.map_rating(m))
        maps.remove(pick_b)
        # десайдер: карта с максимальной суммарной знакомостью/силой обеих
        decider = max(maps, key=lambda m: ta_t.map_rating(m) + tb_t.map_rating(m))
        chosen = [pick_a, pick_b, decider]

        probs = []
        for m in chosen:
            r = self.predict(team_a, team_b, m, ref_date=ref_date, use_map=True)
            probs.append(r["team_a"])

        p1, p2, p3 = probs
        p_2_0 = p1 * p2
        p_2_1 = p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
        p_a = float(np.clip(p_2_0 + p_2_1, 1e-4, 1 - 1e-4))
        return {
            "p_a": p_a,
            "p_b": 1.0 - p_a,
            "maps": chosen,
            "map_probs": [round(p, 4) for p in probs],
            "confidence": 2.0 * abs(p_a - 0.5),
        }

    # ------------------------------------------------------------------
    # Внутренние помощники
    # ------------------------------------------------------------------
    def team_names(self) -> list[str]:
        """Отсортированный список всех команд в датасете."""
        return sorted(
            set(self.raw_df["team_a"]).union(set(self.raw_df["team_b"]))
        )

    def _last_roster(self, team: str, ref_date: date):
        """Последний известный состав команды до ref_date (tuple никнеймов)."""
        from .features import _parse_roster

        past = self.raw_df[
            pd.to_datetime(self.raw_df["date"]).dt.date <= ref_date
        ]
        sub = past[(past["team_a"] == team) | (past["team_b"] == team)]
        if sub.empty:
            return None
        sub = sub.sort_values("date")
        last = sub.iloc[-1]
        col = "roster_a" if last["team_a"] == team else "roster_b"
        return _parse_roster(last.get(col))

    def _build_feature_row(
        self,
        team_a: str,
        team_b: str,
        map_name: str,
        ref_date: date,
        use_map: bool = True,
    ) -> dict:
        """Строит одну строку признаков для указанного матча.

        Использует :class:`FeatureEngine` по матчам до ``ref_date``, а затем
        добавляет недостающие признаки (разности, one-hot карты) вручную, чтобы
        получить полный набор, совпадающий с обучающим.
        """
        past = self.raw_df[pd.to_datetime(self.raw_df["date"]).dt.date < ref_date]
        if past.empty:
            logger.warning(
                "Нет матчей до {}, возвращаю нейтральный прогноз.", ref_date
            )
            past = self.raw_df  # fallback — заглядываем в прошлое частично

        from .features import TIER_WEIGHT

        # накапливаем статистику до ref_date вручную (дешёвая инкрементальность).
        # ELO и SOS требуют, чтобы обе команды обновлялись одновременно в
        # хронологическом порядке — поэтому строим трекеры всех команд разом.
        trackers = _build_trackers(past)
        ta_t = trackers[team_a]
        tb_t = trackers[team_b]

        row: dict = {}
        row["team_a"] = team_a
        row["team_b"] = team_b
        row["map_name"] = map_name
        row["tier"] = "S"  # нейтральный вес сцены
        row["best_of"] = 1

        for w in config.FEATURES.form_windows:
            row[f"a_form_{w}"] = ta_t.form_winrate(w)
            row[f"b_form_{w}"] = tb_t.form_winrate(w)
        row["a_overall"] = ta_t.overall_winrate()
        row["b_overall"] = tb_t.overall_winrate()
        row["a_map_winrate"] = ta_t.map_winrate(map_name)
        row["b_map_winrate"] = tb_t.map_winrate(map_name)
        row["a_map_pick"] = ta_t.map_pick_rate(map_name)
        row["b_map_pick"] = tb_t.map_pick_rate(map_name)
        row["map_pick_diff"] = row["a_map_pick"] - row["b_map_pick"]
        row["a_h2h"] = ta_t.h2h_winrate(team_b)
        row["b_h2h"] = tb_t.h2h_winrate(team_a)
        row["a_recent_days"] = ta_t.recent_days_winrate(
            config.FEATURES.recent_days, ref_date
        )
        row["b_recent_days"] = tb_t.recent_days_winrate(
            config.FEATURES.recent_days, ref_date
        )

        # рейтинг: средний ранг команд (fallback 50)
        sub = past[(past["team_a"].isin([team_a, team_b])) | (past["team_b"].isin([team_a, team_b]))]
        if not sub.empty:
            row["rank_a"] = _median_rank(sub, team_a)
            row["rank_b"] = _median_rank(sub, team_b)
        else:
            row["rank_a"] = 50.0
            row["rank_b"] = 50.0
        row["rank_diff"] = row["rank_a"] - row["rank_b"]

        # официальный рейтинг команд (Cito)
        from .features import _cito_rank_for, _load_cito_rank_map

        rank_map = _load_cito_rank_map()
        a_cr = _cito_rank_for(team_a, rank_map)
        b_cr = _cito_rank_for(team_b, rank_map)
        row["a_cito_rank"] = a_cr["rank"]
        row["b_cito_rank"] = b_cr["rank"]
        row["cito_rank_diff"] = a_cr["rank"] - b_cr["rank"]
        row["a_cito_points"] = a_cr["points"]
        row["b_cito_points"] = b_cr["points"]
        row["cito_points_diff"] = a_cr["points"] - b_cr["points"]

        # ELO и SOS
        a_elo = ta_t.elo_rating()
        b_elo = tb_t.elo_rating()
        row["a_elo"] = a_elo
        row["b_elo"] = b_elo
        row["elo_diff"] = a_elo - b_elo
        # map-Elo на конкретной карте
        a_map_elo = ta_t.map_rating(map_name)
        b_map_elo = tb_t.map_rating(map_name)
        row["a_map_elo"] = a_map_elo
        row["b_map_elo"] = b_map_elo
        row["map_elo_diff"] = a_map_elo - b_map_elo
        a_sos = ta_t.strength_of_schedule()
        b_sos = tb_t.strength_of_schedule()
        row["a_sos"] = a_sos
        row["b_sos"] = b_sos
        row["sos_diff"] = a_sos - b_sos
        # свежесть состава
        a_fresh = ta_t.roster_freshness()
        b_fresh = tb_t.roster_freshness()
        row["a_roster_fresh"] = a_fresh
        row["b_roster_fresh"] = b_fresh
        row["roster_fresh_diff"] = a_fresh - b_fresh
        # средний персональный ELO составов: в predictor нет глобальной истории
        # игроков, поэтому используем командный ELO как прокси (согласован с фичами)
        row["a_roster_elo"] = a_elo
        row["b_roster_elo"] = b_elo
        row["roster_elo_diff"] = a_elo - b_elo
        # средний K/D состава (индивидуальная сила из Cito), по последним составам
        from .features import _roster_avg_kd

        ra = self._last_roster(team_a, ref_date)
        rb = self._last_roster(team_b, ref_date)
        a_kd = _roster_avg_kd(ra)
        b_kd = _roster_avg_kd(rb)
        row["a_roster_kd"] = a_kd
        row["b_roster_kd"] = b_kd
        row["roster_kd_diff"] = a_kd - b_kd

        row["form5_diff"] = row["a_form_5"] - row["b_form_5"]
        row["map_diff"] = row["a_map_winrate"] - row["b_map_winrate"]
        row["h2h_diff"] = row["a_h2h"] - row["b_h2h"]
        row["overall_diff"] = row["a_overall"] - row["b_overall"]
        row["recent_days_diff"] = row["a_recent_days"] - row["b_recent_days"]

        # one-hot карты + is_bo1
        for m in config.MAPS:
            row[f"map_{m}"] = int(m == map_name) if use_map else 0
        row["is_bo1"] = 1
        row["tier_weight"] = TIER_WEIGHT.get("S", 1.0)
        return row

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Приводит X к обучающему набору колонок (порядок, недостающие дополняем)."""
        missing = [c for c in self.feature_names if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        keep = [c for c in self.feature_names if c in X.columns]
        return X[keep]


def _confidence_label(confidence: float) -> str:
    """Классифицирует уровень уверенности по значению confidence (0..1).

    Args:
        confidence: значение уверенности из predict() (0..1).

    Returns:
        Одна из меток: "high", "medium", "low".
    """
    if confidence >= 0.30:  # вероятность фаворита >= ~65%
        return "high"
    if confidence >= 0.15:  # вероятность фаворита >= ~57.5%
        return "medium"
    return "low"


def _build_trackers(past: pd.DataFrame) -> dict:
    """Строит TeamTracker для всех команд по прошлым матчам хронологически.

    Нужно обновлять обе команды каждого матча одновременно, чтобы ELO
    и strength-of-schedule считались корректно (opponent_elo известен).

    Args:
        past: датафрейм прошлых матчей (сортируется по дате внутри).

    Returns:
        dict {team_name: TeamTracker}.
    """
    from collections import defaultdict
    from .features import TeamTracker, _parse_roster

    ordered = past.sort_values("date").reset_index(drop=True)
    trackers: dict = defaultdict(TeamTracker)
    for rec in ordered.itertuples(index=False):
        a = rec.team_a
        b = rec.team_b
        d = rec.date
        if hasattr(d, "date"):
            d = d.date()
        a_won = rec.winner == a
        ta = trackers[a]
        tb = trackers[b]
        roster_a = _parse_roster(getattr(rec, "roster_a", None))
        roster_b = _parse_roster(getattr(rec, "roster_b", None))
        # ELO-обновление обеих команд симметрично (зависит от порядка записи —
        # поэтому сперва используем текущие рейтинги обоих трекеров)
        a_elo = ta.elo_rating()
        b_elo = tb.elo_rating()
        a_map_elo = ta.map_rating(rec.map_name)
        b_map_elo = tb.map_rating(rec.map_name)
        ta.record(d, a_won, rec.map_name, b,
                  opponent_elo=b_elo, opponent_map_elo=b_map_elo, roster=roster_a)
        tb.record(d, not a_won, rec.map_name, a,
                  opponent_elo=a_elo, opponent_map_elo=a_map_elo, roster=roster_b)
    return trackers


def _median_rank(sub: pd.DataFrame, team: str) -> float:
    """Медианный ранг команды по историям встреч с участием соперников.

    Возвращает 50.0 (нейтральное значение), если ранги в данных отсутствуют
    (real-data режим Liquipedia не даёт рангов).
    """
    if "rank_a" not in sub.columns or "rank_b" not in sub.columns:
        return 50.0
    ranks = []
    for rec in sub.itertuples(index=False):
        if rec.team_a == team:
            ranks.append(float(rec.rank_a))
        elif rec.team_b == team:
            ranks.append(float(rec.rank_b))
    if not ranks:
        return 50.0
    return float(np.median(ranks))


def main() -> None:
    """Точка входа: ``python -m src.predictor`` — пример использования."""
    from .scraper import Scraper
    from .utils import probability_text

    raw = Scraper().fetch()
    predictor = Predictor(raw)
    models = predictor.team_names()
    if len(models) < 2:
        logger.error("Недостаточно команд в данных.")
        return
    a, b = models[0], models[1]
    # в Dota2 нет карт пула (MAPS пуст) — карта не используется
    map_name = config.MAPS[0] if config.MAPS else ""
    result = predictor.predict(a, b, map_name=map_name, use_map=bool(config.MAPS))
    print(
        f"{a} {probability_text(result['team_a'])} — "
        f"{probability_text(result['team_b'])} {b}"
        f"  (conf {probability_text(result['confidence'])})"
        f"  [{result['map_name']}]"
    )


if __name__ == "__main__":
    main()