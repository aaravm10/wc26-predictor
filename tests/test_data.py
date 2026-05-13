"""Tests for data loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.collect import TEAM_NAME_MAPPING, clean_team_names

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
