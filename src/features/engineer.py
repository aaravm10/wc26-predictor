"""Feature engineering pipeline for international match prediction."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.collect import load_rankings, load_results, load_shootouts

WC_TOURNAMENT = "FIFA World Cup"
DEFAULT_K = 32
DEFAULT_INITIAL_ELO = 1500
ROLLING_WINDOW = 10
TOP_OPPONENTS = 20

FEATURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
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
]


@dataclass
class MatchRecord:
    """Single completed match stored in a team's rolling history."""

    won: bool
    drew: bool
    goals_for: int
    goals_against: int
    clean_sheet: bool


@dataclass
class TeamState:
    """Accumulated pre-match statistics for one team."""

    elo: float = DEFAULT_INITIAL_ELO
    recent_matches: deque[MatchRecord] = field(
        default_factory=lambda: deque(maxlen=ROLLING_WINDOW)
    )
    recent_vs_top20: deque[MatchRecord] = field(
        default_factory=lambda: deque(maxlen=ROLLING_WINDOW)
    )
    h2h_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    h2h_draws: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    h2h_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    wc_appearances: int = 0
    wc_wins: int = 0
    shootout_wins: int = 0
    shootout_total: int = 0


def _default_processed_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"


def _expected_elo_score(team_elo: float, opponent_elo: float) -> float:
    """Return expected score for ``team_elo`` vs ``opponent_elo``."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))


def _update_elo(
    team_elo: float,
    opponent_elo: float,
    score: float,
    k: float = DEFAULT_K,
) -> float:
    """Apply one Elo rating update."""
    expected = _expected_elo_score(team_elo, opponent_elo)
    return team_elo + k * (score - expected)


def _rolling_win_rate(matches: deque[MatchRecord]) -> float:
    if not matches:
        return 0.0
    wins = sum(1 for m in matches if m.won)
    draws = sum(1 for m in matches if m.drew)
    return (wins + 0.5 * draws) / len(matches)


def _rolling_goals_for(matches: deque[MatchRecord]) -> float:
    if not matches:
        return 0.0
    return sum(m.goals_for for m in matches) / len(matches)


def _rolling_goals_against(matches: deque[MatchRecord]) -> float:
    if not matches:
        return 0.0
    return sum(m.goals_against for m in matches) / len(matches)


def _rolling_clean_sheet_rate(matches: deque[MatchRecord]) -> float:
    if not matches:
        return 0.0
    return sum(1 for m in matches if m.clean_sheet) / len(matches)


def _h2h_win_rate(state: TeamState, opponent: str) -> float:
    total = state.h2h_total.get(opponent, 0)
    if total == 0:
        return 0.0
    wins = state.h2h_wins.get(opponent, 0)
    draws = state.h2h_draws.get(opponent, 0)
    return (wins + 0.5 * draws) / total


def _shootout_win_rate(state: TeamState) -> float:
    if state.shootout_total == 0:
        return 0.0
    return state.shootout_wins / state.shootout_total


def _top_teams(ratings: dict[str, float], n: int = TOP_OPPONENTS) -> set[str]:
    if not ratings:
        return set()
    ranked = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
    return {team for team, _ in ranked[:n]}


def _team_features(state: TeamState, opponent: str) -> dict[str, float | int]:
    """Compute pre-match feature values for one side."""
    return {
        "elo": state.elo,
        "win_rate_last_10": _rolling_win_rate(state.recent_matches),
        "win_rate_last_10_vs_top20": _rolling_win_rate(state.recent_vs_top20),
        "goals_scored_pg_last_10": _rolling_goals_for(state.recent_matches),
        "goals_conceded_pg_last_10": _rolling_goals_against(state.recent_matches),
        "clean_sheet_rate_last_10": _rolling_clean_sheet_rate(state.recent_matches),
        "wc_appearances": state.wc_appearances,
        "wc_wins": state.wc_wins,
        "h2h_win_rate": _h2h_win_rate(state, opponent),
        "shootout_win_rate": _shootout_win_rate(state),
    }


def _match_outcome(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def _prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort results chronologically and coerce numeric / boolean columns."""
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.strip().str.upper().eq("TRUE")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values(["date", "home_team", "away_team"], kind="mergesort").reset_index(
        drop=True
    )
    return df


def _prepare_shootouts(shootouts: pd.DataFrame) -> pd.DataFrame:
    """Sort shootout records chronologically."""
    df = shootouts.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values(["date", "home_team", "away_team"], kind="mergesort").reset_index(
        drop=True
    )


def _prepare_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Sort FIFA ranking snapshots chronologically."""
    df = rankings.copy()
    df = df.dropna(subset=["date", "team", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    return df.sort_values(["date", "team"], kind="mergesort").reset_index(drop=True)


def build_features(
    results: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    shootouts: pd.DataFrame | None = None,
    *,
    k: float = DEFAULT_K,
    initial_elo: float = DEFAULT_INITIAL_ELO,
    output_path: str | Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build per-match features for home and away teams from historical data.

    Features are computed using only information available before each match
    (no leakage). Elo ratings use ``k`` and ``initial_elo``; rolling form uses
    the last 10 completed matches.

    Args:
        results: Match results dataframe. Loaded from ``data/raw/results.csv`` if
            ``None``.
        rankings: FIFA rankings dataframe. Loaded from ``data/raw/fifa_rankings.csv``
            if ``None``.
        shootouts: Penalty shootout dataframe. Loaded from ``data/raw/shootouts.csv``
            if ``None``.
        k: Elo K-factor applied after each match.
        initial_elo: Starting Elo rating for teams with no prior matches.
        output_path: Destination CSV path. Defaults to ``data/processed/features.csv``.
        save: When ``True``, write the feature matrix to ``output_path``.

    Returns:
        Dataframe with one row per match and home/away feature columns.
    """
    if results is None:
        results = load_results()
    if rankings is None:
        rankings = load_rankings()
    if shootouts is None:
        shootouts = load_shootouts()

    matches = _prepare_results(results)
    shootout_rows = _prepare_shootouts(shootouts)
    ranking_rows = _prepare_rankings(rankings)

    teams: dict[str, TeamState] = {}
    ratings: dict[str, float] = {}
    feature_rows: list[dict[str, object]] = []

    def get_team(name: str) -> TeamState:
        if name not in teams:
            teams[name] = TeamState(elo=initial_elo)
        return teams[name]

    shootout_idx = 0
    ranking_idx = 0
    n_rankings = len(ranking_rows)
    n_shootouts = len(shootout_rows)

    for _, row in matches.iterrows():
        match_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        tournament = str(row.get("tournament", "")).strip()
        is_wc = tournament == WC_TOURNAMENT

        while ranking_idx < n_rankings and ranking_rows.iloc[ranking_idx]["date"] <= match_date:
            r_row = ranking_rows.iloc[ranking_idx]
            ratings[str(r_row["team"])] = float(r_row["rating"])
            ranking_idx += 1

        while shootout_idx < n_shootouts and shootout_rows.iloc[shootout_idx]["date"] <= match_date:
            s_row = shootout_rows.iloc[shootout_idx]
            winner = str(s_row["winner"])
            for participant in (str(s_row["home_team"]), str(s_row["away_team"])):
                get_team(participant).shootout_total += 1
                if participant == winner:
                    get_team(participant).shootout_wins += 1
            shootout_idx += 1

        home_state = get_team(home)
        away_state = get_team(away)

        home_feats = _team_features(home_state, away)
        away_feats = _team_features(away_state, home)

        feature_rows.append(
            {
                "date": match_date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": tournament,
                "neutral": bool(row["neutral"]),
                "home_elo": home_feats["elo"],
                "away_elo": away_feats["elo"],
                "home_win_rate_last_10": home_feats["win_rate_last_10"],
                "away_win_rate_last_10": away_feats["win_rate_last_10"],
                "home_win_rate_last_10_vs_top20": home_feats["win_rate_last_10_vs_top20"],
                "away_win_rate_last_10_vs_top20": away_feats["win_rate_last_10_vs_top20"],
                "home_goals_scored_pg_last_10": home_feats["goals_scored_pg_last_10"],
                "away_goals_scored_pg_last_10": away_feats["goals_scored_pg_last_10"],
                "home_goals_conceded_pg_last_10": home_feats["goals_conceded_pg_last_10"],
                "away_goals_conceded_pg_last_10": away_feats["goals_conceded_pg_last_10"],
                "home_clean_sheet_rate_last_10": home_feats["clean_sheet_rate_last_10"],
                "away_clean_sheet_rate_last_10": away_feats["clean_sheet_rate_last_10"],
                "home_wc_appearances": home_feats["wc_appearances"],
                "away_wc_appearances": away_feats["wc_appearances"],
                "home_wc_wins": home_feats["wc_wins"],
                "away_wc_wins": away_feats["wc_wins"],
                "home_h2h_win_rate": home_feats["h2h_win_rate"],
                "away_h2h_win_rate": away_feats["h2h_win_rate"],
                "home_shootout_win_rate": home_feats["shootout_win_rate"],
                "away_shootout_win_rate": away_feats["shootout_win_rate"],
            }
        )

        top20 = _top_teams(ratings)
        home_score_pts, away_score_pts = _match_outcome(home_score, away_score)
        home_elo_pre = home_state.elo
        away_elo_pre = away_state.elo
        home_state.elo = _update_elo(home_elo_pre, away_elo_pre, home_score_pts, k=k)
        away_state.elo = _update_elo(away_elo_pre, home_elo_pre, away_score_pts, k=k)

        home_won = home_score > away_score
        away_won = away_score > home_score
        drew = home_score == away_score
        home_record = MatchRecord(
            won=home_won,
            drew=drew,
            goals_for=home_score,
            goals_against=away_score,
            clean_sheet=away_score == 0,
        )
        away_record = MatchRecord(
            won=away_won,
            drew=drew,
            goals_for=away_score,
            goals_against=home_score,
            clean_sheet=home_score == 0,
        )
        home_state.recent_matches.append(home_record)
        away_state.recent_matches.append(away_record)
        if away in top20:
            home_state.recent_vs_top20.append(home_record)
        if home in top20:
            away_state.recent_vs_top20.append(away_record)

        home_state.h2h_total[away] += 1
        away_state.h2h_total[home] += 1
        if home_won:
            home_state.h2h_wins[away] += 1
        elif away_won:
            away_state.h2h_wins[home] += 1
        elif drew:
            home_state.h2h_draws[away] += 1
            away_state.h2h_draws[home] += 1

        if is_wc:
            home_state.wc_appearances += 1
            away_state.wc_appearances += 1
            if home_won:
                home_state.wc_wins += 1
            elif away_won:
                away_state.wc_wins += 1

    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    if save:
        out = Path(output_path) if output_path is not None else _default_processed_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out, index=False)

    return features
