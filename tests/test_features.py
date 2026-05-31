"""Tests for the feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.engineer import FEATURE_COLUMNS, build_features

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data/processed/features.csv"

KEY_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "home_win_rate_last_10",
    "away_win_rate_last_10",
    "home_win_rate_last_10_vs_top20",
    "away_win_rate_last_10_vs_top20",
    "home_goals_scored_pg_last_10",
    "away_goals_scored_pg_last_10",
    "home_goals_conceded_pg_last_10",
    "away_goals_conceded_pg_last_10",
    "home_clean_sheet_rate_last_10",
    "away_clean_sheet_rate_last_10",
    "home_wc_appearances",
    "away_wc_appearances",
    "home_wc_wins",
    "away_wc_wins",
    "home_h2h_win_rate",
    "away_h2h_win_rate",
    "home_shootout_win_rate",
    "away_shootout_win_rate",
    "neutral",
]


def _require_raw_data() -> None:
    for rel in (
        "data/raw/results.csv",
        "data/raw/fifa_rankings.csv",
        "data/raw/shootouts.csv",
    ):
        if not (REPO_ROOT / rel).is_file():
            pytest.skip(f"missing fixture: {REPO_ROOT / rel}")


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    """Build the full feature matrix once per test module."""
    _require_raw_data()
    return build_features(save=True, output_path=FEATURES_PATH)


def test_build_features_output_columns(features_df: pd.DataFrame) -> None:
    assert list(features_df.columns) == FEATURE_COLUMNS


def test_build_features_writes_csv(features_df: pd.DataFrame) -> None:
    assert FEATURES_PATH.is_file()
    on_disk = pd.read_csv(FEATURES_PATH)
    assert list(on_disk.columns) == FEATURE_COLUMNS
    assert len(on_disk) == len(features_df)


def test_key_features_have_no_nan(features_df: pd.DataFrame) -> None:
    for col in KEY_FEATURE_COLUMNS:
        assert features_df[col].notna().all(), f"NaN found in {col}"


def test_elo_starts_at_1500_and_updates() -> None:
    results = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            },
            {
                "date": "2020-02-01",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": 0,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            },
        ]
    )
    empty_rankings = pd.DataFrame(columns=["date", "team", "rating", "change"])
    empty_shootouts = pd.DataFrame(
        columns=["date", "home_team", "away_team", "winner", "first_shooter"]
    )
    df = build_features(
        results=results,
        rankings=empty_rankings,
        shootouts=empty_shootouts,
        save=False,
    )
    assert df.iloc[0]["home_elo"] == pytest.approx(1500.0)
    assert df.iloc[0]["away_elo"] == pytest.approx(1500.0)
    assert df.iloc[1]["home_elo"] > 1500.0
    assert df.iloc[1]["away_elo"] == pytest.approx(1500.0)


def test_feature_values_within_reasonable_ranges(features_df: pd.DataFrame) -> None:
    rate_cols = [
        c
        for c in KEY_FEATURE_COLUMNS
        if "win_rate" in c or "clean_sheet_rate" in c or "h2h_win_rate" in c or "shootout_win_rate" in c
    ]
    for col in rate_cols:
        assert features_df[col].between(0.0, 1.0).all(), f"{col} out of [0, 1] range"

    for col in ("home_goals_scored_pg_last_10", "away_goals_scored_pg_last_10",
                "home_goals_conceded_pg_last_10", "away_goals_conceded_pg_last_10"):
        assert (features_df[col] >= 0.0).all()
        assert (features_df[col] <= 25.0).all()

    for col in ("home_wc_appearances", "away_wc_appearances", "home_wc_wins", "away_wc_wins"):
        assert (features_df[col] >= 0).all()

    assert features_df["home_elo"].between(800.0, 2600.0).all()
    assert features_df["away_elo"].between(800.0, 2600.0).all()


def test_first_historical_match_elo_is_1500(features_df: pd.DataFrame) -> None:
    first = features_df.sort_values("date").iloc[0]
    assert first["home_elo"] == pytest.approx(1500.0)
    assert first["away_elo"] == pytest.approx(1500.0)


def test_neutral_venue_column_present(features_df: pd.DataFrame) -> None:
    assert features_df["neutral"].dtype == bool
    assert features_df["neutral"].isin([True, False]).all()
