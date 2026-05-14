"""Data collection helpers (raw downloads, normalization)."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Final

import pandas as pd

# Align FIFA ranking CSV labels with `data/raw/results.csv` spellings.
# Keys are compared after `_normalize_unicode` (NFKC, NBSP→space, strip, collapse spaces).
#
# Among 2026 World Cup qualified sides, FIFA `team` text that differs from results.csv
# after normalization is only: ``Czechia`` → Czech Republic, ``Democratic Republic of Congo``
# → DR Congo. Other FIFA rows use NBSP (U+00A0) between words; `_normalize_unicode` fixes that.
TEAM_NAME_MAPPING: Final[dict[str, str]] = {
    "British Guiana": "Guyana",
    "Burma": "Myanmar",
    "Ceylon": "Sri Lanka",
    "China": "China PR",
    # FIFA lists these territories without corresponding rows in results.csv; map to
    # the sovereign side used in this results dataset so cleaned names stay joinable.
    "Christmas Island": "Australia",
    "Cocos Islands": "Australia",
    "Congo-Brazzaville": "Congo",
    "Czechia": "Czech Republic",
    "Dahomey": "Benin",
    "Democratic Republic of Congo": "DR Congo",
    "East Germany": "German DR",
    "East Timor": "Timor-Leste",
    "Eastern Samoa": "Samoa",
    "Federated States of Micronesia": "Micronesia",
    "Ireland": "Republic of Ireland",
    "Khmer Republic": "Cambodia",
    "Kurdistan": "Iraqi Kurdistan",
    "Macao": "Macau",
    "Macedonia": "North Macedonia",
    "Malaya": "Malaysia",
    "Netherlands Antilles": "Curaçao",
    "North Yemen": "Yemen",
    "Northern Rhodesia": "Zambia",
    "Reunion": "Réunion",
    "Saba": "Netherlands",
    "Saint Barthelemy": "Saint Barthélemy",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Serbia and Montenegro": "Serbia",
    "Sint Eustatius": "Netherlands",
    "South Vietnam": "Vietnam Republic",
    "Southern Rhodesia": "Zimbabwe",
    "Soviet Union": "Russia",
    "Surinam": "Suriname",
    "Swaziland": "Eswatini",
    "Tanganyika": "Tanzania",
    "United Arab Republic": "Egypt",
    "Upper Volta": "Burkina Faso",
    "US Virgin Islands": "United States Virgin Islands",
    "Vatican": "Vatican City",
    "Wallis and Futuna": "Wallis Islands and Futuna",
    "West Germany": "Germany",
}


def _normalize_unicode(name: str) -> str:
    """NFKC, strip, NBSP→ASCII space, collapse internal whitespace."""
    s = unicodedata.normalize("NFKC", name).strip().removeprefix("\ufeff")
    s = s.replace("\xa0", " ").replace("\u00a0", " ")
    return " ".join(s.split())


def clean_team_names(name: object) -> str:
    """
    Normalize team labels so FIFA rankings and match results use the same strings.

    - Unicode normalization (compatibility composition)
    - NBSP and odd whitespace from FIFA CSVs → regular spaces
    - ``TEAM_NAME_MAPPING`` for known synonym / historical labels
    """
    if name is None:
        return ""
    if isinstance(name, float) and name != name:  # NaN
        return ""
    s = _normalize_unicode(str(name))
    if not s or s.lower() == "nan":
        return ""
    prev: str | None = None
    while prev != s:
        prev = s
        s = TEAM_NAME_MAPPING.get(s, s)
    return s


def _default_raw_path(filename: str) -> Path:
    """Resolve ``data/raw/<filename>`` under the repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2] / "data" / "raw" / filename


def _resolve_csv_path(path: str | Path | None, default_filename: str) -> Path:
    if path is None:
        return _default_raw_path(default_filename)
    return Path(path).expanduser().resolve()


def _read_raw_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {csv_path}. "
            "Place CSVs under data/raw/ (see data/DATA_SOURCES.md)."
        )
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load international match results and normalize team labels.

    Parameters
    ----------
    path
        Filesystem path to ``results.csv``. If ``None``, loads
        ``<repository root>/data/raw/results.csv``.

    Returns
    -------
    pandas.DataFrame
        Raw results with ``home_team`` and ``away_team`` transformed by
        :func:`clean_team_names`.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the resolved path.
    """
    csv_path = _resolve_csv_path(path, "results.csv")
    df = _read_raw_csv(csv_path)
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(clean_team_names)
    return df


def load_shootouts(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load penalty shootout records and normalize team labels.

    Parameters
    ----------
    path
        Filesystem path to ``shootouts.csv``. If ``None``, loads
        ``<repository root>/data/raw/shootouts.csv``.

    Returns
    -------
    pandas.DataFrame
        Shootouts with ``home_team``, ``away_team``, and ``winner`` passed through
        :func:`clean_team_names`.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the resolved path.
    """
    csv_path = _resolve_csv_path(path, "shootouts.csv")
    df = _read_raw_csv(csv_path)
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].map(clean_team_names)
    return df


def load_goalscorers(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load goalscorer lines and normalize team labels.

    Parameters
    ----------
    path
        Filesystem path to ``goalscorers.csv``. If ``None``, loads
        ``<repository root>/data/raw/goalscorers.csv``.

    Returns
    -------
    pandas.DataFrame
        Goal events with ``home_team``, ``away_team``, and ``team`` passed through
        :func:`clean_team_names`.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the resolved path.
    """
    csv_path = _resolve_csv_path(path, "goalscorers.csv")
    df = _read_raw_csv(csv_path)
    for col in ("home_team", "away_team", "team"):
        df[col] = df[col].map(clean_team_names)
    return df


def load_rankings(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load FIFA ranking history and normalize the team column.

    Parameters
    ----------
    path
        Filesystem path to ``fifa_rankings.csv``. If ``None``, loads
        ``<repository root>/data/raw/fifa_rankings.csv``.

    Returns
    -------
    pandas.DataFrame
        Rankings with ``team`` passed through :func:`clean_team_names` and ``date``
        parsed to pandas datetime. The CSV mixes ISO dates (``YYYY-MM-DD``) and
        US-style dates (``M/D/YYYY``); ``format="mixed"`` is used so both parse
        correctly (invalid values become ``NaT``).

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the resolved path.
    """
    csv_path = _resolve_csv_path(path, "fifa_rankings.csv")
    df = _read_raw_csv(csv_path)
    df["team"] = df["team"].map(clean_team_names)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    return df
