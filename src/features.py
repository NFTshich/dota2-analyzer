"""Фиче-инжиниринг для матчей CS2.

Ключевой принцип: признаки для конкретного матча строятся **только по матчам,
сыгранным ДО него** (хронологический порядок), чтобы не было утечки данных
(target leakage). Для скорости статистика по командам ведётся инкрементально
через :class:`TeamTracker`, а не пересчётом «в лоб» для каждого матча.

Точка входа: ``python -m src.features`` — генерирует
``data/processed/features.parquet``.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from . import config


# Доли веса тиров: S-tier значимее, её учитываем как весовой коэффициент.
TIER_WEIGHT: dict[str, float] = {"S": 1.0, "A": 0.7, "B": 0.5}

#: начальный ELO всех команд
ELO_START = 1000.0
#: коэффициент обновления ELO общего (K-factor)
ELO_K = 40.0
#: коэффициент обновления ELO на конкретной карте
MAP_ELO_K = 48.0
#: число карт, при котором map-Elo достигает 50% веса в blend-рейтинге
MAP_BLEND_THRESHOLD = 20.0
#: масштаб логистической функции ELO (как сильно разница рейтингов влияет
#: на ожидаемый результат)
ELO_SCALE = 400.0


def _winrate(wins: int, losses: int) -> float:
    """Доля побед; 0.5 при отсутствии данных (нейтральный priors)."""
    total = wins + losses
    if total == 0:
        return 0.5
    return wins / total


def _expected_score(elo_a: float, elo_b: float) -> float:
    """Ожидаемая доля побед команды A против команды B по ELO."""
    diff = elo_b - elo_a
    return 1.0 / (1.0 + 10.0 ** (diff / ELO_SCALE))


def _parse_roster(raw) -> tuple[str, ...] | None:
    """Разбирает колонку состава (JSON или tuple) в кортеж никнеймов.

    Args:
        raw: значение ячейки ``roster_a``/``roster_b`` (JSON-строка, список
            или None).

    Returns:
        tuple никнеймов отсортированный, или None если данных нет.
    """
    if raw is None:
        return None
    if isinstance(raw, str) and raw:
        import json

        try:
            val = json.loads(raw)
        except (ValueError, TypeError):
            val = None
    else:
        val = raw
    if isinstance(val, (list, tuple, set)):
        names = [str(x).strip() for x in val if str(x).strip()]
        if not names:
            return None
        return tuple(sorted(names))
    return None


#: алиасы команд (наши имена → имена в Cito-рейтинге)
_CITO_ALIASES: dict[str, str] = {
    "navi": "Natus Vincere",
    "natusvincere": "Natus Vincere",
    "fazeclan": "FaZe",
    "faze": "FaZe",
    "g2esports": "G2",
    "g2": "G2",
    "heroic": "Heroic",
    "teamliquid": "Liquid",
    "liquid": "Liquid",
    "virtuspro": "Virtus.pro",
    "virtus": "Virtus.pro",
    "mousesports": "MOUZ",
    "mouz": "MOUZ",
    "3dmax": "3DMAX",
    "themongolz": "The MongolZ",
    "mongolz": "The MongolZ",
    "complexitygaming": "Complexity",
    "complexity": "Complexity",
    "eternalfire": "Eternal Fire",
    "ef": "Eternal Fire",
    "betboom": "BETBOOM",
    "b8": "B8",
    "9z": "9z",
}


def _load_cito_rank_map() -> dict[str, dict]:
    """Загружает официальный рейтинг команд из Cito (data/processed/).

    Returns:
        {нормализованное_имя: {"rank": int, "points": int}} — пусто, если файла
        нет (тогда признак рейтинга принимает нейтральные значения).
    """
    import re

    path = config.PROCESSED_DIR / "cito_rankings.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row.get("team", "")).strip()
        if not name:
            continue
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        out[key] = {
            "rank": float(row.get("rank", 50) or 50),
            "points": float(row.get("points", 0) or 0),
        }
    return out


_CITO_RANK_MAP: dict[str, dict] | None = None


def _cito_rank_for(team: str, rank_map: dict[str, dict]) -> dict:
    """Возвращает {"rank", "points"} для команды или нейтральные значения."""
    import re

    if not rank_map:
        return {"rank": 50.0, "points": 0.0}
    key = re.sub(r"[^a-z0-9]", "", team.lower())
    if key in rank_map:
        return rank_map[key]
    alias = _CITO_ALIASES.get(key)
    if alias:
        alias_key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if alias_key in rank_map:
            return rank_map[alias_key]
    return {"rank": 50.0, "points": 0.0}


#: маппинг ников игроков → их K/D (из cito_player_stats.parquet)
_PLAYER_KD_MAP: dict[str, float] | None = None


def _load_player_kd_map() -> dict[str, float]:
    """Загружает {ником: kd} из cito_player_stats.parquet (пусто, если нет файла)."""
    import re

    global _PLAYER_KD_MAP
    if _PLAYER_KD_MAP is not None:
        return _PLAYER_KD_MAP
    path = config.PROCESSED_DIR / "cito_player_stats.parquet"
    _PLAYER_KD_MAP = {}
    if not path.exists():
        return _PLAYER_KD_MAP
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return _PLAYER_KD_MAP
    for _, r in df.iterrows():
        nick = re.sub(r"[^a-z0-9]", "", str(r.get("player", "")).lower())
        kd = r.get("kd")
        if nick and pd.notna(kd):
            _PLAYER_KD_MAP[nick] = float(kd)
    return _PLAYER_KD_MAP


def _roster_avg_kd(roster: tuple[str, ...] | None) -> float:
    """Средний K/D состава команды (по никам), или 1.0 если данных нет."""
    if not roster:
        return 1.0
    kd_map = _load_player_kd_map()
    if not kd_map:
        return 1.0
    import re

    vals = []
    for nick in roster:
        key = re.sub(r"[^a-z0-9]", "", nick.lower())
        if key in kd_map:
            vals.append(kd_map[key])
    if not vals:
        return 1.0
    return round(sum(vals) / len(vals), 3)


class TeamTracker:
    """Инкрементальный накопитель статистики по одной команде.

    Позволяет за O(1) получать форму, winrate по картам, head-to-head,
    ELO-рейтинг и strength-of-schedule (средний ELO соперников), не храня и
    не пересчитывая весь список.
    """

    def __init__(self) -> None:
        #: окно последних матчей для формы (date, team_won, map_name, opponent)
        self.recent: deque[tuple[date, bool, str, str]] = deque(maxlen=60)
        self.total_wins = 0
        self.total_losses = 0
        #: wins/losses по картам
        self.map_wins: defaultdict[str, int] = defaultdict(int)
        self.map_losses: defaultdict[str, int] = defaultdict(int)
        #: head-to-head: {other_team: (wins, losses)}
        self.h2h: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        #: последние даты матчей (date, won) для recent-form окна
        self.dates: deque[tuple[date, bool]] = deque()
        #: текущий ELO-рейтинг
        self.elo = ELO_START
        #: ELO по каждой карте (map -> рейтинг)
        self.map_elo: defaultdict[str, float] = defaultdict(lambda: ELO_START)
        #: число сыгранных карт по каждому типу (для blend-веса)
        self.map_games: defaultdict[str, int] = defaultdict(int)
        #: сумма ELO соперников и их число (для strength-of-schedule)
        self._opp_elo_sum = 0.0
        self._opp_elo_cnt = 0
        #: последний известный состав команды (tuple из никнеймов) и дата смены
        self.roster: tuple[str, ...] = ()
        self.roster_changed_at: date | None = None
        #: штамп времени последней смены состава в днях (для признака «свежести»)
        self._last_roster_change_days: float | None = None

    def record(
        self, match_date: date, won: bool, map_name: str, opponent: str,
        opponent_elo: float, opponent_map_elo: float | None = None,
        roster: tuple[str, ...] | None = None,
    ) -> None:
        """Фиксирует сыгранный матч в статистике команды.

        Помимо обычного учёта, обновляет ELO (общий и по карте) и копит сумму
        ELO соперников (для strength-of-schedule). Если передан ``roster`` —
        сверяет с последним известным составом и запоминает смену.

        Args:
            match_date: дата матча.
            won: победила ли команда.
            map_name: карта.
            opponent: имя соперника.
            opponent_elo: общий ELO соперника на момент матча.
            opponent_map_elo: ELO соперника на этой карте (если доступен).
            roster: состав команды на этот матч (tuple никнеймов), если известен.
        """
        self.total_wins += int(won)
        self.total_losses += int(not won)
        self.map_wins[map_name] += int(won)
        self.map_losses[map_name] += int(not won)
        self.h2h[opponent][0] += int(won)
        self.h2h[opponent][1] += int(not won)
        self.recent.append((match_date, won, map_name, opponent))
        self.dates.append((match_date, won))

        # ELO-обновление
        expected = _expected_score(self.elo, opponent_elo)
        self.elo += ELO_K * (int(won) - expected)

        # ELO на конкретной карте: обновляется по результату на этой карте,
        # используя map-Elo соперника (если есть). Число игр на карте растёт.
        if opponent_map_elo is not None:
            opp_map = opponent_map_elo
        else:
            opp_map = ELO_START
        expected_map = _expected_score(self.map_elo[map_name], opp_map)
        self.map_elo[map_name] += MAP_ELO_K * (int(won) - expected_map)
        self.map_games[map_name] += 1

        # strength of schedule
        self._opp_elo_sum += opponent_elo
        self._opp_elo_cnt += 1

        # Состав команды: фиксируем смену, если пришёл новый состав и он
        # отличается от последнего известного.
        if roster:
            if self.roster and self.roster != roster:
                self.roster_changed_at = match_date
                self._last_roster_change_days = 0.0
            self.roster = tuple(sorted(roster))
        # если состав не изменился в этом матче — накапливаем «возраст» неизменности
        if self.roster and self._last_roster_change_days is not None:
            self._last_roster_change_days += 1.0

    # --- признаки ---------------------------------------------------------
    def form_winrate(self, n: int) -> float:
        """Взвешенный по давности winrate: недавние матчи значат больше.

Каждому из последних матчей даётся вес exp(-(n-1-i)/decay), где i — индекс
считая от самого свежего (0 = последний). Победа вкладивает +вес, поражение 0.
Возвращает 0.5 при отсутствии матчей (нейтральный priors).
"""
        recent = list(self.recent)[-n:]
        if not recent:
            return 0.5
        decay = max(3.0, n / 2.0)
        total_w = 0.0
        wins_w = 0.0
        m = len(recent)
        for i in range(m):
            wgt = math.exp(-(m - 1 - i) / decay)
            total_w += wgt
            if recent[i][1]:  # won
                wins_w += wgt
        return wins_w / total_w if total_w > 0 else 0.5

    def overall_winrate(self) -> float:
        """Общий winrate за всю историю."""
        return _winrate(self.total_wins, self.total_losses)

    def map_winrate(self, map_name: str) -> float:
        """Winrate команды на конкретной карте."""
        return _winrate(self.map_wins[map_name], self.map_losses[map_name])

    def h2h_winrate(self, opponent: str) -> float:
        """Winrate в личных встречах с соперником."""
        wins, losses = self.h2h[opponent]
        return _winrate(wins, losses)

    def recent_days_winrate(self, days: int, ref_date: date) -> float:
        """Winrate за последние ``days`` дней до ``ref_date`` включительно."""
        cutoff = ref_date - timedelta(days=days)
        wins = losses = 0
        for d, won in self.dates:
            if d >= cutoff:
                wins += int(won)
                losses += int(not won)
        return _winrate(wins, losses)

    def map_pick_rate(self, map_name: str) -> float:
        """Долю матчей команды, сыгранных на этой карте (pick rate 0..1).

        Показывает, насколько часто команда выбирает/оказывается на карте.
        Возвращает медианную 1/len-карт при отсутствии данных.
        """
        n_match = self._opp_elo_cnt
        played_on_map = self.map_wins[map_name] + self.map_losses[map_name]
        if n_match == 0:
            return 0.0
        return played_on_map / n_match

    def elo_rating(self) -> float:
        """Текущий ELO команды (стартовое 1000, если ещё ни одной записи)."""
        return self.elo

    def map_rating(self, map_name: str) -> float:
        """Blend-рейтинг команды на конкретной карте (0..~2000).

        Смешивает общий Elo и map-Elo с весом, растущим по мере сыгранности
        карты (по достижении MAP_BLEND_THRESHOLD карт — 50% веса на map-Elo).
        """
        raw_map = self.map_elo[map_name]
        n = self.map_games[map_name]
        w = n / (n + MAP_BLEND_THRESHOLD)
        return w * raw_map + (1.0 - w) * self.elo

    def roster_freshness(self) -> float:
        """Признак «свежести состава» 0..1.

        Возвращает:
            1.0 — состав давно не менялся (стабилен);
            0.0 — состав только что сменился (свежая ротация).
        Для команд без данных о составе — нейтральное значение 0.5.
        """
        if not self.roster or self._last_roster_change_days is None:
            return 0.5
        # плавно «стареет»: за ~30 матчей от 0 до 1
        return max(0.0, min(1.0, self._last_roster_change_days / 30.0))

    def strength_of_schedule(self) -> float:
        """Средний ELO соперников, с которыми команда уже играла.

        Returns:
            Средний ELO соперников, или ELO_START (1000), если данных нет.
        """
        if self._opp_elo_cnt == 0:
            return ELO_START
        return self._opp_elo_sum / self._opp_elo_cnt


class FeatureEngine:
    """Строит фиче-таблицу из хронологически упорядоченного датафрейма матчей."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        #: статистика команд, накопленная на текущем шаге итерации
        self.trackers: dict[str, TeamTracker] = defaultdict(TeamTracker)
        #: персональный ELO игроков (nick -> рейтинг), обновляется после матча
        self.player_elo: dict[str, float] = {}

    def _roster_avg_elo(self, roster: tuple[str, ...] | None) -> float:
        """Средний персональный ELO состава (по игрокам), или ELO_START."""
        if not roster:
            return ELO_START
        vals = [self.player_elo.get(p, ELO_START) for p in roster]
        return sum(vals) / len(vals)

    def _update_player_elo(
        self, roster: tuple[str, ...] | None, opp_roster: tuple[str, ...] | None,
        won: bool,
    ) -> None:
        """Дрейфует персональный ELO игроков состава по исходу матча."""
        if not roster:
            return
        opp_avg = self._roster_avg_elo(opp_roster)
        for player in roster:
            elo = self.player_elo.get(player, ELO_START)
            expected = _expected_score(elo, opp_avg)
            self.player_elo[player] = elo + ELO_K * (int(won) - expected)

    # ------------------------------------------------------------------
    def build(self) -> pd.DataFrame:
        """Возвращает датафрейм признаков + целевой переменной."""
        df = self.df.sort_values("date").reset_index(drop=True)
        rows: list[dict] = []
        windows = config.FEATURES.form_windows
        # официальный рейтинг команд (Cito) — загружаем один раз
        global _CITO_RANK_MAP
        if _CITO_RANK_MAP is None:
            _CITO_RANK_MAP = _load_cito_rank_map()

        for rec in df.itertuples(index=False):
            ta = rec.team_a
            tb = rec.team_b
            map_name = rec.map_name
            mdate = rec.date  # Timestamp

            if hasattr(mdate, "date"):
                mdate_d = mdate.date()
            else:
                mdate_d = mdate

            ta_t = self.trackers[ta]
            tb_t = self.trackers[tb]

            tier_w = TIER_WEIGHT.get(rec.tier, 0.5)
            rank_a = float(getattr(rec, "rank_a", 50))
            rank_b = float(getattr(rec, "rank_b", 50))

            # составы команд на этот матч (если есть в данных)
            roster_a = _parse_roster(getattr(rec, "roster_a", None))
            roster_b = _parse_roster(getattr(rec, "roster_b", None))

            # ELO и SOS на момент матча (до обновления после него)
            a_elo = ta_t.elo_rating()
            b_elo = tb_t.elo_rating()
            a_sos = ta_t.strength_of_schedule()
            b_sos = tb_t.strength_of_schedule()
            a_fresh = ta_t.roster_freshness()
            b_fresh = tb_t.roster_freshness()
            # map-Elo (blend-рейтинг на конкретной карте) — сильный сигнал
            a_map_elo = ta_t.map_rating(map_name)
            b_map_elo = tb_t.map_rating(map_name)
            # средний персональный ELO составов (сила игроков)
            a_roster_elo = self._roster_avg_elo(roster_a)
            b_roster_elo = self._roster_avg_elo(roster_b)

            feat: dict = {}
            feat["match_id"] = rec.match_id
            feat["date"] = mdate
            feat["team_a"] = ta
            feat["team_b"] = tb
            feat["map_name"] = map_name
            feat["tier"] = rec.tier
            feat["best_of"] = int(getattr(rec, "best_of", 1))
            feat["tier_weight"] = tier_w

            # форма
            for w in windows:
                feat[f"a_form_{w}"] = ta_t.form_winrate(w)
                feat[f"b_form_{w}"] = tb_t.form_winrate(w)
            # общий winrate
            feat["a_overall"] = ta_t.overall_winrate()
            feat["b_overall"] = tb_t.overall_winrate()
            # карта
            feat["a_map_winrate"] = ta_t.map_winrate(map_name)
            feat["b_map_winrate"] = tb_t.map_winrate(map_name)
            # map-Elo (сила команды на конкретной карте)
            feat["a_map_elo"] = a_map_elo
            feat["b_map_elo"] = b_map_elo
            feat["map_elo_diff"] = a_map_elo - b_map_elo
            # пик-рейт карты (насколько часто команда бывает на этой карте)
            feat["a_map_pick"] = ta_t.map_pick_rate(map_name)
            feat["b_map_pick"] = tb_t.map_pick_rate(map_name)
            feat["map_pick_diff"] = feat["a_map_pick"] - feat["b_map_pick"]
            # head-to-head
            feat["a_h2h"] = ta_t.h2h_winrate(tb)
            feat["b_h2h"] = tb_t.h2h_winrate(ta)
            # recent form (последние N дней)
            feat["a_recent_days"] = ta_t.recent_days_winrate(
                config.FEATURES.recent_days, mdate_d
            )
            feat["b_recent_days"] = tb_t.recent_days_winrate(
                config.FEATURES.recent_days, mdate_d
            )
            # рейтинг
            feat["rank_a"] = rank_a
            feat["rank_b"] = rank_b
            feat["rank_diff"] = rank_a - rank_b
            # ELO и SOS
            feat["a_elo"] = a_elo
            feat["b_elo"] = b_elo
            feat["elo_diff"] = a_elo - b_elo
            feat["a_sos"] = a_sos
            feat["b_sos"] = b_sos
            feat["sos_diff"] = a_sos - b_sos
            # официальный рейтинг команд (Cito): rank и points
            a_cr = _cito_rank_for(ta, _CITO_RANK_MAP or {})
            b_cr = _cito_rank_for(tb, _CITO_RANK_MAP or {})
            feat["a_cito_rank"] = a_cr["rank"]
            feat["b_cito_rank"] = b_cr["rank"]
            feat["cito_rank_diff"] = a_cr["rank"] - b_cr["rank"]
            feat["a_cito_points"] = a_cr["points"]
            feat["b_cito_points"] = b_cr["points"]
            feat["cito_points_diff"] = a_cr["points"] - b_cr["points"]
            # свежесть состава
            feat["a_roster_fresh"] = a_fresh
            feat["b_roster_fresh"] = b_fresh
            feat["roster_fresh_diff"] = a_fresh - b_fresh
            # средний персональный ELO составов
            feat["a_roster_elo"] = a_roster_elo
            feat["b_roster_elo"] = b_roster_elo
            feat["roster_elo_diff"] = a_roster_elo - b_roster_elo
            # средний K/D состава (индивидуальная сила игроков из Cito)
            a_roster_kd = _roster_avg_kd(roster_a)
            b_roster_kd = _roster_avg_kd(roster_b)
            feat["a_roster_kd"] = a_roster_kd
            feat["b_roster_kd"] = b_roster_kd
            feat["roster_kd_diff"] = a_roster_kd - b_roster_kd

            # разностные признаки (упрощают модели)
            feat["form5_diff"] = feat["a_form_5"] - feat["b_form_5"]
            feat["map_diff"] = feat["a_map_winrate"] - feat["b_map_winrate"]
            feat["h2h_diff"] = feat["a_h2h"] - feat["b_h2h"]
            feat["overall_diff"] = feat["a_overall"] - feat["b_overall"]
            feat["recent_days_diff"] = (
                feat["a_recent_days"] - feat["b_recent_days"]
            )

            # целевая переменная: 1, если победила team_a
            feat["target"] = 1 if rec.winner == ta else 0

            rows.append(feat)

            # --- обновляем статистику только ПОСЛЕ построения признаков ---
            a_won = rec.winner == ta
            ta_t.record(
                mdate_d, a_won, map_name, tb,
                opponent_elo=b_elo, opponent_map_elo=b_map_elo, roster=roster_a,
            )
            tb_t.record(
                mdate_d, not a_won, map_name, ta,
                opponent_elo=a_elo, opponent_map_elo=a_map_elo, roster=roster_b,
            )
            # обновляем персональный ELO игроков по исходу матча
            self._update_player_elo(roster_a, roster_b, a_won)
            self._update_player_elo(roster_b, roster_a, not a_won)

        return pd.DataFrame(rows)


def build_features(
    df: pd.DataFrame, out_path: str | None = None
) -> pd.DataFrame:
    """Оборачивает FeatureEngine в удобную функцию с сохранением.

    Args:
        df: сырые матчи (хронологический порядок необязателен — сортируем сами).
        out_path: куда сохранить parquet. None — не сохранять.

    Returns:
        DataFrame признаков.
    """
    fe = FeatureEngine(df)
    out = fe.build()

    # One-hot по картам (для моделей, которым нужны числовые признаки).
    for m in config.MAPS:
        out[f"map_{m}"] = (out["map_name"] == m).astype(int)
    out["is_bo1"] = (out["best_of"] == 1).astype(int)

    if out_path:
        out.to_parquet(out_path, index=False)
        logger.info("Фичи сохранены: {} ({} строк)", out_path, len(out))
    return out


def main() -> None:
    """Точка входа: ``python -m src.features``."""
    from .scraper import Scraper
    from .utils import probability_text

    logger.info("Загрузка сырых матчей…")
    raw = Scraper().fetch()
    logger.info("Сырых матчей: {}", len(raw))

    features = build_features(raw, str(config.MATCHES_PROCESSED))
    logger.info("Признаков в строке: {}", features.shape[1])
    logger.info(
        "Баланс классов: team_a побеждает с долей {}",
        probability_text(features["target"].mean()),
    )
    logger.info(
        "Пример колонок: {}",
        ", ".join(features.columns[:12].tolist()),
    )


if __name__ == "__main__":
    main()