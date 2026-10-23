# Тесты для Dota2 Analyzer: фиче-инжиниринг и симметрия.

import pandas as pd
import pytest

from src import config
from src.features import FeatureEngine, _parse_roster
from src.utils import generate_synthetic_matches


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Небольшой детерминированный датасет Dota-матчей."""
    rows = [
        dict(match_id="m1", date="2024-01-01", team_a="Team Spirit", team_b="Team Liquid",
             map_name="", score_a=2, score_b=0, winner="Team Spirit",
             event_name="E1", tier="S", rank_a=1, rank_b=4, best_of=3),
        dict(match_id="m2", date="2024-01-02", team_a="BetBoom Team", team_b="PARIVISION",
             map_name="", score_a=1, score_b=2, winner="PARIVISION",
             event_name="E1", tier="S", rank_a=5, rank_b=1, best_of=3),
        dict(match_id="m3", date="2024-01-03", team_a="Team Spirit", team_b="BetBoom Team",
             map_name="", score_a=2, score_b=1, winner="Team Spirit",
             event_name="E2", tier="A", rank_a=1, rank_b=5, best_of=3),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_generate_synthetic_columns() -> None:
    df = generate_synthetic_matches(n_matches=50, seed=7)
    assert len(df) == 50
    for col in ["match_id", "date", "team_a", "team_b", "score_a", "score_b",
                "winner", "event_name", "tier", "best_of"]:
        assert col in df.columns, f"missing column: {col}"
    assert df["date"].is_monotonic_increasing


def test_engine_builds_features(sample_df: pd.DataFrame) -> None:
    eng = FeatureEngine(sample_df)
    out = eng.build()
    assert len(out) == len(sample_df)
    # у первой команды в первом матче ещё нет истории до него
    assert out.iloc[0]["a_overall"] == 0.5
    assert out["target"].tolist() == [1, 0, 1]


def test_diff_features_present(sample_df: pd.DataFrame) -> None:
    eng = FeatureEngine(sample_df)
    out = eng.build()
    for c in ["form5_diff", "elo_diff", "overall_diff"]:
        assert c in out.columns


def test_swap_columns_preserves_symmetry() -> None:
    """Разностные признаки при зеркалировании меняют знак."""
    from src.model import _swap_columns

    df = pd.DataFrame({
        "team_a": ["Team Spirit", "Team Liquid"],
        "team_b": ["BetBoom Team", "PARIVISION"],
        "a_elo": [1258.0, 1200.0],
        "b_elo": [1329.0, 1150.0],
        "elo_diff": [1258 - 1329, 1200 - 1150],
        "form5_diff": [0.1, -0.2],
        "target": [1, 0],
    })
    sw = _swap_columns(df)
    assert sw["a_elo"].tolist() == df["b_elo"].tolist()
    assert (sw["elo_diff"] == -df["elo_diff"]).all()
    assert (sw["form5_diff"] == -df["form5_diff"]).all()


def test_parse_roster_json() -> None:
    assert _parse_roster(None) is None
    assert _parse_roster('["Yatoro", "Collapse"]') == ("Collapse", "Yatoro")