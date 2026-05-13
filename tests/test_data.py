"""Tests for data loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.collect import (
    TEAM_NAME_MAPPING,
    clean_team_names,
    load_goalscorers,
    load_rankings,
    load_results,
    load_shootouts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_raw_csv(relative: str) -> pd.DataFrame:
    path = REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def test_clean_team_names_fifa_vocabulary_matches_results_after_cleaning() -> None:
    """
    Every team label that appears in FIFA rankings maps, after cleaning, to a label
    that appears in match results. Results also contain many CONIFA / regional sides
    that FIFA never ranks; those extra names are expected.
    """
    results = _load_raw_csv("data/raw/results.csv")
    fifa = _load_raw_csv("data/raw/fifa_rankings.csv")

    results_teams = {clean_team_names(x) for x in results["home_team"]} | {
        clean_team_names(x) for x in results["away_team"]
    }
    results_teams.discard("")

    fifa_teams = {clean_team_names(x) for x in fifa["team"]}
    fifa_teams.discard("")

    assert fifa_teams <= results_teams
    assert fifa_teams == results_teams & fifa_teams
    assert fifa_teams.symmetric_difference(results_teams) == results_teams - fifa_teams


def test_clean_team_names_unicode_and_nbsp() -> None:
    assert clean_team_names("Czech\xa0Republic") == "Czech Republic"
    assert clean_team_names("Ivory\xa0Coast") == "Ivory Coast"
    assert clean_team_names("\ufeffUnited States") == "United States"


def test_clean_team_names_idempotent_on_fifa_and_results_samples() -> None:
    results = _load_raw_csv("data/raw/results.csv")
    fifa = _load_raw_csv("data/raw/fifa_rankings.csv")
    samples = (
        list(dict.fromkeys(fifa["team"].astype(str).head(500)))
        + list(dict.fromkeys(results["home_team"].astype(str).head(500)))
        + list(TEAM_NAME_MAPPING.keys())
        + list(dict.fromkeys(TEAM_NAME_MAPPING.values()))
    )
    for raw in samples:
        once = clean_team_names(raw)
        assert clean_team_names(once) == once


def test_load_results() -> None:
    path = REPO_ROOT / "data/raw/results.csv"
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    raw = _load_raw_csv("data/raw/results.csv")
    df = load_results()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(raw)
    assert df["home_team"].equals(raw["home_team"].map(clean_team_names))
    assert df["away_team"].equals(raw["away_team"].map(clean_team_names))


def test_load_shootouts() -> None:
    path = REPO_ROOT / "data/raw/shootouts.csv"
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    raw = _load_raw_csv("data/raw/shootouts.csv")
    df = load_shootouts()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(raw)
    for col in ("home_team", "away_team", "winner"):
        assert df[col].equals(raw[col].map(clean_team_names))


def test_load_goalscorers() -> None:
    path = REPO_ROOT / "data/raw/goalscorers.csv"
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    raw = _load_raw_csv("data/raw/goalscorers.csv")
    df = load_goalscorers()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(raw)
    for col in ("home_team", "away_team", "team"):
        assert df[col].equals(raw[col].map(clean_team_names))


def test_load_rankings() -> None:
    path = REPO_ROOT / "data/raw/fifa_rankings.csv"
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    raw = _load_raw_csv("data/raw/fifa_rankings.csv")
    df = load_rankings()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(raw)
    assert df["team"].equals(raw["team"].map(clean_team_names))
    assert str(df["date"].dtype).startswith("datetime")
    assert df["date"].notna().all()


def test_load_results_file_not_found() -> None:
    missing = REPO_ROOT / "data/raw/__definitely_missing_results__.csv"
    with pytest.raises(FileNotFoundError, match="Required data file not found"):
        load_results(missing)


def test_load_rankings_file_not_found() -> None:
    missing = REPO_ROOT / "data/raw/__definitely_missing_rankings__.csv"
    with pytest.raises(FileNotFoundError, match="Required data file not found"):
        load_rankings(missing)


def test_load_shootouts_file_not_found() -> None:
    missing = REPO_ROOT / "data/raw/__definitely_missing_shootouts__.csv"
    with pytest.raises(FileNotFoundError, match="Required data file not found"):
        load_shootouts(missing)


def test_load_goalscorers_file_not_found() -> None:
    missing = REPO_ROOT / "data/raw/__definitely_missing_goalscorers__.csv"
    with pytest.raises(FileNotFoundError, match="Required data file not found"):
        load_goalscorers(missing)
