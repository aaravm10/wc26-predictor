"""Data collection helpers (raw downloads, normalization)."""

from __future__ import annotations

import unicodedata
from typing import Final

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
