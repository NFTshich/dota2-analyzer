"""Клиент открытого API Liquipedia (Counter-Strike wiki) + парсер wikitext.

Liquipedia — свободный, без ключей источник данных о киберспортивных турнирах.
Здесь реализован аккуратный клиент к MediaWiki API ``api.php`` с обязательным
``Accept-Encoding: gzip`` (иначе сервер возвращает 406), паузами между запросами
(rate-limit) и парсингом вики-разметки через :mod:`mwparserfromhell`.

Извлекаем из страниц турниров все шаблоны ``{{Match}}``, которые содержат:
команды (``opponent1``/``opponent2``), дату, формат (BO1/BO3/BO5), карты
(``map1..mapN``) и счёт серии. Тир берём из ``{{Infobox league|liquipediatier=…}}``.

Имена команд приводятся к каноническому виду через разрешение редиректов
(``action=query&redirects=1``) и кэшируются, чтобы не дёргать API повторно.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import gzip
import json

import mwparserfromhell
from loguru import logger

from . import config

#: поддомен Liquipedia — берём из конфига (dota2 / counterstrike)
_wiki = getattr(config.LIQUIPEDIA, "wiki", "dota2")
API_BASE = f"https://liquipedia.net/{_wiki}/api.php"
WIKI_BASE = f"https://liquipedia.net/{_wiki}/"

#: русло дат в Liquipedia: «November 7, 2019 - 12:00 {{Abbr/CST}}»
_DATE_RE = re.compile(r"(\w+)\s+(\d{1,2}),\s+(\d{4})")
#: шорткод команды в TeamOpponent -> первая буква в upper
_TIMEZONE_ALIASES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------
@dataclass
class LiquipediaMatch:
    """Один извлечённый матч.

    Args:
        match_id: внутренний id (страница + порядковый номер).
        team_a: название первой команды (каноническое).
        team_b: название второй команды.
        date: дата/время матча (UTC), как строка iso, или None.
        map_name: первая сыгранная карта (или самая значимая).
        score_a: счёт серии команды a.
        score_b: счёт серии команды b.
        winner: название победителя.
        event_name: название турнира.
        tier: тир турнира (S/A/B), нормализованный.
        rank_a / rank_b: ранги команд (если известны, иначе None).
        best_of: формат серии (1/3/5).
        maps: список сыгранных карт.
    """

    match_id: str
    team_a: str
    team_b: str
    date: str | None
    map_name: str | None
    score_a: int = 0
    score_b: int = 0
    winner: str | None = None
    event_name: str = ""
    tier: str = "B"
    rank_a: int | None = None
    rank_b: int | None = None
    best_of: int = 1
    maps: list[str] = field(default_factory=list)
    #: составы (списки никнеймов) обеих команд на этот турнир (если доступны)
    roster_a: list[str] = field(default_factory=list)
    roster_b: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Клиент API
# ---------------------------------------------------------------------------
class LiquipediaClient:
    """Низкоуровневый клиент к MediaWiki API Liquipedia.

    Args:
        delay: задержка между запросами в секундах (уважаем rate-limit).
        timeout: таймаут одного запроса.
        user_agent: UA, обязательный для Liquipedia.
    """

    def __init__(
        self,
        delay: float | None = None,
        timeout: int | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.delay = delay if delay is not None else config.SCRAPER.request_delay
        self.timeout = timeout or config.SCRAPER.timeout
        self.user_agent = (
            user_agent
            or "CS2Analyzer/1.0 (research; https://github.com/example/cs2_analyzer)"
        )
        self._last_call = 0.0
        self._title_cache: dict[str, str | None] = {}

    # --- транспорт ---------------------------------------------------
    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Выполняет GET-запрос к api.php и декодирует gzip-ответ."""
        # соблюдаем паузу между запросами
        wait = self._last_call + self.delay - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

        query = urllib.parse.urlencode(params)
        url = f"{API_BASE}?{query}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8"))

    def get_wikitext(self, page: str) -> str:
        """Возвращает wikitext страницы (через action=parse&prop=wikitext).

        Args:
            page: название страницы вики (с пробелами/подчёркиваниями).

        Returns:
            Исходная разметка страницы; пустая строка, если страницы нет.
        """
        data = self._request(
            {
                "action": "parse",
                "page": page,
                "prop": "wikitext",
                "format": "json",
                "origin": "*",
            }
        )
        try:
            return data["parse"]["wikitext"]["*"]
        except (KeyError, TypeError):
            logger.warning("wikitext не найден для страницы: {}", page)
            return ""

    def canonical_team_title(self, shortcode: str) -> str | None:
        """Приводит шорткод команды к каноническому названию.

        Сначала пробует прямой title (already capitalised), затем редирект.
        Возвращает None, если страница не существует — тогда используем fallback.

        Args:
            shortcode: шорткод команды из разметки (например ``astralis``).

        Returns:
            Каноническое название команды или None.
        """
        if shortcode in self._title_cache:
            return self._title_cache[shortcode]

        result: str | None = None
        # пробуем прямой титул капитализорованный
        candidate = shortcode[0].upper() + shortcode[1:]
        data = self._request(
            {
                "action": "query",
                "titles": shortcode,
                "redirects": "1",
                "format": "json",
                "origin": "*",
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for _k, v in pages.items():
            if int(_k) > 0:
                result = v.get("title") or candidate
                break

        # резолв через redirects
        redirects = data.get("query", {}).get("redirects", [])
        if redirects:
            result = redirects[-1].get("to", candidate)

        self._title_cache[shortcode] = result
        return result

    def tournament_page_names(self, max_pages: int = 40) -> list[str]:
        """Возвращает список страниц завершённых турниров.

        Идём по категории :class:`Category:Finished Tournaments`. Категория может
        быть большой, поэтому берём первые ``max_pages`` страниц.

        Args:
            max_pages: максимум страниц для возврата.

        Returns:
            Список названий страниц турниров (без категории).
        """
        names: list[str] = []
        titles_param = ["Category:" + t for t in config.LIQUIPEDIA.categories]
        seen: set[str] = set()
        for cat in titles_param:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmlimit": str(max_pages),
                "format": "json",
                "origin": "*",
            }
            data = self._request(params)
            for m in data.get("query", {}).get("categorymembers", []):
                t = m.get("title", "")
                if t.startswith("Category:") or t in seen:
                    continue
                seen.add(t)
                names.append(t)
            if len(names) >= max_pages:
                break
        return names[:max_pages]


# ---------------------------------------------------------------------------
# Парсер wikitext
# ---------------------------------------------------------------------------
def _parse_date(raw: str) -> str | None:
    """Разбирает дату Liquipedia в ISO-строку (UTC-наивную).

    Args:
        raw: например ``November 7, 2019 - 12:00 {{Abbr/CST}}``.

    Returns:
        ISO дата-время без таймзоны (``2019-11-07T12:00:00``) или None.
    """
    m = _DATE_RE.search(raw)
    if not m:
        return None
    month_s, day_s, year_s = m.groups()
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    month = month_map.get(month_s)
    if month is None:
        return None
    try:
        dt = datetime(int(year_s), month, int(day_s))
        # берём часы, если есть
        time_m = re.search(r"(\d{1,2}):(\d{2})", raw)
        if time_m:
            dt = dt.replace(hour=int(time_m.group(1)), minute=int(time_m.group(2)))
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def _normalize_tier(tier: str) -> str:
    """Приводит тир Liquipedia к простому S/A/B.

    Args:
        tier: напр. ``S-Tier``, ``A-Tier``, ``B-Tier``, ``Qualifier``.

    Returns:
        Одна буква: S/A/B, иначе 'B'.
    """
    t = (tier or "").lower()
    if "s-tier" in t:
        return "S"
    if "a-tier" in t:
        return "A"
    return "B"


def _int_or_none(m: "re.Match | None") -> int | None:
    """Возвращает int из regex-совпадения или None."""
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None


def _best_of_from_maps(maps: list[str], series_score: tuple[int, int]) -> int:
    """Определяет формат серии (1/3/5) по числу карт и счёту."""
    n = len(maps)
    if n <= 1:
        return 1
    if series_score[0] + series_score[1] >= 4:
        return 5
    return 3


class MatchParser:
    """Парсит wikitext страницы турнира в список матчей.

    Args:
        client: экземпляр LiquipediaClient (для резолва имён команд).
    """

    def __init__(self, client: LiquipediaClient) -> None:
        self.client = client
        self._team_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    def parse_page(self, page: str) -> list[LiquipediaMatch]:
        """Парсит страницу турнира и все её видимые матчи.

        Args:
            page: название страницы вики.

        Returns:
            Список извлечённых матчей.
        """
        wikitext = self.client.get_wikitext(page)
        if not wikitext:
            return []

        code = mwparserfromhell.parse(wikitext)
        tier = _normalize_tier(self._extract_tier(code))
        rosters = self._extract_rosters(code)

        matches: list[LiquipediaMatch] = []
        match_templates = [
            t for t in code.filter_templates() if t.name.strip().lower() == "match"
        ]
        for i, mt in enumerate(match_templates):
            m = self._parse_match(mt, page, i, tier, wikitext, rosters)
            if m is not None:
                matches.append(m)
        return matches

    # ------------------------------------------------------------------
    def _extract_rosters(self, code) -> dict[str, list[str]]:
        """Извлекает составы команд из шаблонов {{TeamCard}} на странице.

        Args:
            code: разобранный wikitext страницы (mwparserfromhell code).

        Returns:
            dict: {каноническое имя команды: [nickname1, ..., nickname5]} —
            составы-участники турнира (из TeamCard), скол.
        """
        rosters: dict[str, list[str]] = {}
        for t in code.filter_templates():
            if str(t.name).strip().lower() != "teamcard":
                continue

            def _p(name: str) -> str:
                try:
                    val = str(t.get(name)).strip()
                except (ValueError, KeyError):
                    return ""
                if "=" in val and val.split("=", 1)[0].strip().lower() == name.lower():
                    val = val.split("=", 1)[1].strip()
                return val

            team_raw = _p("team")
            if not team_raw:
                continue
            # каноническое имя команды — по умолчанию как есть (TeamCard
            # уже содержит полное имя). Сетевой резолв здесь дорогой и часто
            # не даёт совпадения со шорткодами матчей, поэтому не зовём его.
            title = self._team_cache.get(team_raw)
            if title is None:
                title = team_raw
                self._team_cache[team_raw] = title

            players = []
            for slot in ("p1", "p2", "p3", "p4", "p5"):
                nick = _p(slot)
                if nick and nick.lower() not in ("tbd", "tba", ""):
                    players.append(nick)
            if players:
                rosters[title] = players
        return rosters

    # ------------------------------------------------------------------
    def _extract_tier(self, code) -> str:
        """Извлекает тир из шаблона Infobox league."""
        for t in code.filter_templates():
            if t.name.strip().lower() in ("infobox league", "infobox_league"):
                for p in t.params:
                    if p.name.strip().lower() == "liquipediatier":
                        return p.value.strip()
        return "B-Tier"

    # ------------------------------------------------------------------
    def _parse_match(
        self,
        mt,
        page: str,
        index: int,
        tier: str,
        wikitext: str,
        rosters: dict[str, list[str]] | None = None,
    ) -> LiquipediaMatch | None:
        """Разбирает один шаблон {{Match}} в объект."""

        def get_param(name: str) -> str:
            """Возвращает чистое значение параметра шаблона.

            mwparserfromhell хранит параметр как ``key=value``; его метод
            ``Template.get()`` может возвращать и заготовленное значение.
            Достаём значение через ``has()``/``get()`` и извлекаем часть после ``=``,
            чтобы отсечь случай, когда вернулся весь ``key=value``.
            """
            try:
                val = str(mt.get(name)).strip()
            except ValueError:
                return ""
            if "=" in val and val.split("=", 1)[0].strip().lower() == name.lower():
                val = val.split("=", 1)[1].strip()
            return val

        opponent1_raw = get_param("opponent1")
        opponent2_raw = get_param("opponent2")
        team_a, score_a = self._parse_team_opponent(opponent1_raw)
        team_b, score_b = self._parse_team_opponent(opponent2_raw)
        if not team_a or not team_b:
            logger.debug("пропуск матча без команд: {}", page)
            return None

        finished_raw = get_param("finished").lower()
        if finished_raw and finished_raw not in ("true", "yes", "1", ""):
            logger.debug("пропуск незавершённого матча: {}", page)
            return None

        date_raw = get_param("date")
        date_iso = _parse_date(date_raw)

        # карты и счёт серии.
        # CS2: Map-блоки имеют map= и t1t/t1ct/t2t/t2ct (счёт сторон).
        # Dota: Map-блоки имеют winner=1/2 и героев (t1h1..), счёта нет.
        maps: list[str] = []
        map_totals: list[tuple[int | None, int | None]] = []  # (t1, t2) на карте
        map_winners: list[int] = []  # 1=team1 выиграл карту, 2=team2
        for key in ["map1", "map2", "map3", "map4", "map5"]:
            map_raw = get_param(key)
            if not map_raw:
                continue
            mm = re.search(r"\bmap\s*=\s*([^|\s}]+)", map_raw)
            if mm:
                maps.append(mm.group(1))
            t1t = _int_or_none(re.search(r"\bt1t\s*=\s*(\d+)", map_raw))
            t1ct = _int_or_none(re.search(r"\bt1ct\s*=\s*(\d+)", map_raw))
            t2t = _int_or_none(re.search(r"\bt2t\s*=\s*(\d+)", map_raw))
            t2ct = _int_or_none(re.search(r"\bt2ct\s*=\s*(\d+)", map_raw))
            if None not in (t1t, t1ct, t2t, t2ct):
                map_totals.append((t1t + t1ct, t2t + t2ct))
            # Dota: победитель карты (winner=1/2)
            w = _int_or_none(re.search(r"\bwinner\s*=\s*(\d+)", map_raw))
            if w in (1, 2):
                map_winners.append(w)

        # счёт серии: приоритет явного score= у TeamOpponent, иначе по картам
        if str(score_a).isdigit():
            series_a = int(score_a)
        elif map_winners and not map_totals:
            # Dota-путь: счёт из победителей карт
            series_a = sum(1 for w in map_winners if w == 1)
            series_b = sum(1 for w in map_winners if w == 2)
        else:
            series_a = sum(1 for t1, t2 in map_totals if t1 > t2)
            series_b = sum(1 for t1, t2 in map_totals if t2 > t1)

        best_of_raw = get_param("bestof")
        if str(best_of_raw).isdigit():
            best_of = int(best_of_raw)
        else:
            best_of = _best_of_from_maps(maps, (series_a, series_b))

        # пропускаем незавершившиеся/ничьи: нужен решающий счёт серии
        need = max(1, best_of // 2 + 1)
        if series_a == 0 and series_b == 0:
            return None  # ещё не сыгран
        if series_a < need and series_b < need:
            return None  # незавершённая серия
        if series_a == series_b:
            return None  # ничья/неопределённость

        if series_a > series_b:
            winner = team_a
        else:
            winner = team_b

        main_map = maps[0] if maps else None

        # event name = читаемое название страницы
        event_name = page.replace("_", " ").replace("/", " | ")

        # составы с нормализованным подбором (шорткод матча может отличаться
        # от ключа TeamCard посимвольно)
        roster_a: list[str] = list(rosters.get(team_a, [])) if rosters else []
        roster_b: list[str] = list(rosters.get(team_b, [])) if rosters else []
        if rosters and (not roster_a or not roster_b):
            def _norm(s: str) -> str:
                return re.sub(r"[^a-z0-9]", "", (s or "").lower())

            rmap: dict[str, list[str]] = {}
            for k, v in rosters.items():
                rmap.setdefault(_norm(k), v)
            if not roster_a:
                roster_a = list(rmap.get(_norm(team_a), []))
            if not roster_b:
                roster_b = list(rmap.get(_norm(team_b), []))

        return LiquipediaMatch(
            match_id=f"{page}|{index}",
            team_a=team_a,
            team_b=team_b,
            date=date_iso,
            map_name=main_map,
            score_a=series_a,
            score_b=series_b,
            winner=winner,
            event_name=event_name,
            tier=tier,
            best_of=best_of,
            maps=maps,
            roster_a=roster_a,
            roster_b=roster_b,
        )
        return LiquipediaMatch(
            match_id=f"{page}|{index}",
            team_a=team_a, team_b=team_b, date=date_iso,
            map_name=main_map, score_a=series_a, score_b=series_b,
            winner=winner, event_name=event_name, tier=tier,
            best_of=best_of, maps=maps,
            roster_a=list(roster_a), roster_b=list(roster_b),
        )

    # ------------------------------------------------------------------
    def _parse_team_opponent(
        self, raw: str
    ) -> tuple[str | None, int | None]:
        """Извлекает (имя_команды, счёт) из ``{{TeamOpponent|short|score=2}}``.

        Args:
            raw: строка значения параметра opponent1/opponent2.

        Returns:
            (каноническое имя команды, счёт серии) или (None, None).
        """
        # шорткод — первый аргумент TeamOpponent
        mm = re.search(r"\{\{TeamOpponent\s*\|\s*([^|}=]+)", raw)
        if not mm:
            # fallback: возможна прямая ссылка [[Team|Name]]
            mm2 = re.search(r"\[\[([^|\]]+)", raw)
            shortcode = mm2.group(1) if mm2 else raw
        else:
            shortcode = mm.group(1).strip()
        if not shortcode:
            return None, None

        # score — параметр score=2 в том же шаблоне
        score = None
        sm = re.search(r"score\s*=\s*(\d+)", raw)
        if sm:
            score = int(sm.group(1))

        # канонический титул
        title = self._team_cache.get(shortcode)
        if title is None:
            title = self.client.canonical_team_title(shortcode)
            if not title:
                title = shortcode[0].upper() + shortcode[1:]
            self._team_cache[shortcode] = title

        return title, score


# ---------------------------------------------------------------------------
# Загрузчик
# ---------------------------------------------------------------------------
class LiquipediaLoader:
    """Высокоуровневый сборщик: страницы/категории -> DataFrame с колонками
    матчей, совместимый с пайплайном проекта (team_a, team_b, date, map_name,
    score_a, score_b, winner, event_name, tier).

    Args:
        client: LiquipediaClient (создаётся, если не передан).
        parser: MatchParser (создаётся, если не передан).
    """

    def __init__(
        self,
        client: LiquipediaClient | None = None,
        parser: MatchParser | None = None,
    ) -> None:
        self.client = client or LiquipediaClient()
        self.parser = parser or MatchParser(self.client)

    # ------------------------------------------------------------------
    def scrape_tournament_pages(
        self, pages: list[str], include_subpages: bool = True
    ) -> list[LiquipediaMatch]:
        """Парсит список страниц и (опц.) их дочерние (subpages).

        Args:
            pages: страницы турниров.
            include_subpages: добавлять ли дочерние страницы (форматы/квалы).

        Returns:
            Все извлечённые матчи.
        """
        all_matches: list[LiquipediaMatch] = []
        for page in pages:
            try:
                ms = self.parser.parse_page(page)
                all_matches.extend(ms)
                logger.info("  {}  →  {} матчей", page, len(ms))

                if include_subpages:
                    sub = 0
                    for sp in self._find_subpages(page):
                        mss = self.parser.parse_page(sp)
                        all_matches.extend(mss)
                        sub += len(mss)
                    if sub:
                        logger.info("    (включая подстраницы: {} матчей)", sub)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ошибка обработки {}: {}", page, exc)

        return all_matches

    # ------------------------------------------------------------------
    def scrape_by_categories(self, max_pages: int = 40) -> list[LiquipediaMatch]:
        """Собирает матчи из категорий завершённых турниров.

        Args:
            max_pages: лимит страниц для обработки.

        Returns:
            Список матчей.
        """
        pages = self.client.tournament_page_names(max_pages)
        logger.info("Найдено страниц турниров: {}", len(pages))
        return self.scrape_tournament_pages(pages)

    # ------------------------------------------------------------------
    def to_frame(self, matches: list[LiquipediaMatch]) -> "pd.DataFrame":
        """Преобразует список матчей в DataFrame с колонками пайплайна."""
        import json
        import pandas as pd

        columns = [
            "match_id", "date", "team_a", "team_b", "map_name",
            "score_a", "score_b", "winner", "event_name", "tier",
            "best_of", "roster_a", "roster_b",
        ]
        if not matches:
            return pd.DataFrame(columns=columns)
        rows = [
            {
                "match_id": m.match_id,
                "date": m.date,
                "team_a": m.team_a,
                "team_b": m.team_b,
                "map_name": m.map_name,
                "score_a": m.score_a,
                "score_b": m.score_b,
                "winner": m.winner,
                "event_name": m.event_name,
                "tier": m.tier,
                "best_of": m.best_of,
                "roster_a": json.dumps(m.roster_a, ensure_ascii=False),
                "roster_b": json.dumps(m.roster_b, ensure_ascii=False),
            }
            for m in matches
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def _find_subpages(self, page: str) -> list[str]:
        """Возвращает дочерние страницы турнира, если они упомянуты в Tabs.

        Args:
            page: базовая страница турнира.

        Returns:
            Список дочерних страниц.
        """
        wikitext = self.client.get_wikitext(page)
        if not wikitext:
            return []
        hrefs = re.findall(r"\blink\d+\s*=\s*([^\n|]+)", wikitext)
        out: list[str] = []
        for h in hrefs:
            h = h.strip().strip('"')
            if h and h != page and not h.lower().startswith("http"):
                out.append(h)
        return out


def main() -> None:
    """Точка входа ``python -m src.liquipedia`` — быстрый тест сбора."""

    logger.info("Тестирование LiquipediaLoader…")
    loader = LiquipediaLoader()
    # paccмотрим один конкретный турнир для проверки
    test_pages = ["Intel Extreme Masters/Season XIV/Beijing"]
    matches = loader.scrape_tournament_pages(test_pages, include_subpages=False)
    df = loader.to_frame(matches)
    logger.info("Извлечено матчей из тестового турнира: {}", len(df))
    if not df.empty:
        cols = ["team_a", "team_b", "tier", "map_name", "date"]
        logger.info("\n{}", df[cols].head(10).to_string())


if __name__ == "__main__":
    main()