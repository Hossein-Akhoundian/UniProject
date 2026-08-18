"""Build a leakage-safe, feature-engineered football match dataset.

The output is intended for classification in RapidMiner (or similar tools).
All dynamic team features are calculated from matches strictly before the
current match date.  GeoNames data is downloaded once and cached locally.
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GEONAMES_BASE_URL = "https://download.geonames.org/export/dump"
INITIAL_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
FORM_WINDOW = 5


def normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


@dataclass(frozen=True)
class Place:
    latitude: float
    longitude: float
    country_code: str
    population: int
    feature_code: str


COUNTRY_ALIASES = {
    "bolivia": "BO",
    "bosniaherzegovina": "BA",
    "brunei": "BN",
    "capeverde": "CV",
    "congodr": "CD",
    "democraticrepublicofthecongo": "CD",
    "drcongo": "CD",
    "easttimor": "TL",
    "england": "GB",
    "eswatini": "SZ",
    "greatbritain": "GB",
    "hongkong": "HK",
    "iran": "IR",
    "ivorycoast": "CI",
    "kosovo": "XK",
    "laos": "LA",
    "macau": "MO",
    "macedonia": "MK",
    "micronesia": "FM",
    "moldova": "MD",
    "northkorea": "KP",
    "northmacedonia": "MK",
    "northernireland": "GB",
    "palestine": "PS",
    "republicofireland": "IE",
    "russia": "RU",
    "scotland": "GB",
    "southkorea": "KR",
    "swaziland": "SZ",
    "syria": "SY",
    "taiwan": "TW",
    "tanzania": "TZ",
    "thebahamas": "BS",
    "unitedstates": "US",
    "unitedstatesofamerica": "US",
    "usa": "US",
    "venezuela": "VE",
    "vietnam": "VN",
    "wales": "GB",
}


def read_country_info(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    name_to_code = dict(COUNTRY_ALIASES)
    capital_by_code: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 17:
                continue
            iso, iso3, _, fips, country, capital = parts[:6]
            for alias in (iso, iso3, fips, country):
                if alias:
                    name_to_code[normalized_name(alias)] = iso
            if capital:
                capital_by_code[iso] = capital
    return name_to_code, capital_by_code


def load_relevant_places(
    cities_zip: Path, query_names: set[str]
) -> dict[str, list[Place]]:
    """Load only GeoNames entries matching cities/capitals needed here."""
    candidates: dict[str, list[Place]] = defaultdict(list)
    with zipfile.ZipFile(cities_zip) as archive:
        txt_name = next(name for name in archive.namelist() if name.endswith(".txt"))
        with archive.open(txt_name) as raw:
            for byte_line in raw:
                parts = byte_line.decode("utf-8").rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue
                name, ascii_name, alternate_names = parts[1:4]
                keys = {
                    normalized_name(item)
                    for item in (name, ascii_name, *alternate_names.split(","))
                    if item
                }
                matched_keys = keys.intersection(query_names)
                if not matched_keys:
                    continue
                try:
                    place = Place(
                        latitude=float(parts[4]),
                        longitude=float(parts[5]),
                        country_code=parts[8],
                        population=int(parts[14] or 0),
                        feature_code=parts[7],
                    )
                except ValueError:
                    continue
                for key in matched_keys:
                    candidates[key].append(place)
    return candidates


def place_score(place: Place) -> float:
    feature_bonus = {
        "PPLC": 8.0,
        "PPLA": 5.0,
        "PPLA2": 3.0,
        "PPLA3": 2.0,
        "PPLA4": 1.0,
    }.get(place.feature_code, 0.0)
    return math.log1p(place.population) + feature_bonus


def choose_place(
    city: object,
    country: object,
    candidates: dict[str, list[Place]],
    country_codes: dict[str, str],
    capital_places: dict[str, Place],
) -> tuple[float, float, str]:
    city_key = normalized_name(city)
    country_code = country_codes.get(normalized_name(country))
    matches = candidates.get(city_key, [])
    if country_code:
        local_matches = [p for p in matches if p.country_code == country_code]
        if local_matches:
            best = max(local_matches, key=place_score)
            return best.latitude, best.longitude, "city_country"
    if matches:
        best = max(matches, key=place_score)
        return best.latitude, best.longitude, "city_global"
    if country_code in capital_places:
        best = capital_places[country_code]
        return best.latitude, best.longitude, "country_capital_fallback"
    return np.nan, np.nan, "unresolved"


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def add_geographic_features(
    df: pd.DataFrame, cache_dir: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    cities_zip = cache_dir / "cities500.zip"
    country_info_file = cache_dir / "countryInfo.txt"
    download_if_missing(f"{GEONAMES_BASE_URL}/cities500.zip", cities_zip)
    download_if_missing(f"{GEONAMES_BASE_URL}/countryInfo.txt", country_info_file)

    country_codes, capital_by_code = read_country_info(country_info_file)
    query_names = {normalized_name(city) for city in df["city"].unique()}
    query_names.update(normalized_name(city) for city in capital_by_code.values())
    candidates = load_relevant_places(cities_zip, query_names)

    capital_places: dict[str, Place] = {}
    for code, capital in capital_by_code.items():
        matches = [
            place
            for place in candidates.get(normalized_name(capital), [])
            if place.country_code == code
        ]
        if matches:
            capital_places[code] = max(matches, key=place_score)

    venue_cache: dict[tuple[str, str], tuple[float, float, str]] = {}
    for city, country in df[["city", "country"]].drop_duplicates().itertuples(
        index=False, name=None
    ):
        venue_cache[(city, country)] = choose_place(
            city, country, candidates, country_codes, capital_places
        )

    venue_values = [
        venue_cache[(city, country)]
        for city, country in df[["city", "country"]].itertuples(
            index=False, name=None
        )
    ]
    df[["venue_latitude", "venue_longitude", "_venue_geo_method"]] = pd.DataFrame(
        venue_values, index=df.index
    )

    # A national team's most frequent non-neutral home venue is used as its
    # reproducible home base. This handles teams such as England and Scotland
    # better than forcing every team name into a modern sovereign-country list.
    exact = df[
        (~df["neutral"])
        & df["venue_latitude"].notna()
        & df["_venue_geo_method"].isin(["city_country", "city_global"])
    ]
    base_counts: dict[str, Counter[tuple[float, float]]] = defaultdict(Counter)
    for row in exact.itertuples(index=False):
        base_counts[row.home_team][
            (float(row.venue_latitude), float(row.venue_longitude))
        ] += 1

    team_bases: dict[str, tuple[float, float]] = {}
    for team, counts in base_counts.items():
        team_bases[team] = counts.most_common(1)[0][0]

    # Fallback for teams without a recorded non-neutral home match.
    for team in set(df["home_team"]).union(df["away_team"]):
        if team in team_bases:
            continue
        country_code = country_codes.get(normalized_name(team))
        place = capital_places.get(country_code)
        if place:
            team_bases[team] = (place.latitude, place.longitude)

    travel_distance: list[float] = []
    travel_known: list[int] = []
    for row in df.itertuples(index=False):
        base = team_bases.get(row.away_team)
        if base and pd.notna(row.venue_latitude) and pd.notna(row.venue_longitude):
            distance = haversine_km(
                base[0], base[1], row.venue_latitude, row.venue_longitude
            )
            travel_distance.append(round(distance, 1))
            travel_known.append(1)
        else:
            travel_distance.append(-1.0)
            travel_known.append(0)

    df["away_travel_km"] = travel_distance
    df["away_travel_known"] = travel_known
    geo_counts = df["_venue_geo_method"].value_counts().to_dict()
    geo_counts["travel_known"] = int(sum(travel_known))
    geo_counts["travel_unknown"] = int(len(travel_known) - sum(travel_known))
    return df, geo_counts


def form_means(history: deque[tuple[float, float, float]]) -> tuple[float, ...]:
    if not history:
        return 0.0, 0.0, 0.0, 0
    values = np.asarray(history, dtype=float)
    return (
        float(values[:, 0].mean()),
        float(values[:, 1].mean()),
        float(values[:, 2].mean()),
        len(history),
    )


def add_sequential_features(df: pd.DataFrame) -> pd.DataFrame:
    ratings: defaultdict[str, float] = defaultdict(lambda: INITIAL_ELO)
    recent: defaultdict[str, deque[tuple[float, float, float]]] = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW)
    )
    last_played: dict[str, pd.Timestamp] = {}
    features: dict[str, list[float]] = defaultdict(list)

    # Matches on the same date share the state available at the start of that
    # date. Their results are applied only after every feature row is created.
    for match_date, day in df.groupby("date", sort=True):
        pending_rating_changes: defaultdict[str, float] = defaultdict(float)
        pending_history: list[tuple[str, tuple[float, float, float]]] = []

        for row in day.itertuples(index=False):
            home_rating = ratings[row.home_team]
            away_rating = ratings[row.away_team]
            home_advantage = 0.0 if row.neutral else ELO_HOME_ADVANTAGE
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (
                    (away_rating - (home_rating + home_advantage))
                    / 400.0
                )
            )
            if row.home_score > row.away_score:
                actual_home = 1.0
                home_points, away_points = 3.0, 0.0
            elif row.home_score < row.away_score:
                actual_home = 0.0
                home_points, away_points = 0.0, 3.0
            else:
                actual_home = 0.5
                home_points, away_points = 1.0, 1.0

            home_form = form_means(recent[row.home_team])
            away_form = form_means(recent[row.away_team])
            home_rest = (
                (match_date - last_played[row.home_team]).days
                if row.home_team in last_played
                else -1
            )
            away_rest = (
                (match_date - last_played[row.away_team]).days
                if row.away_team in last_played
                else -1
            )

            features["elo_home_pre"].append(home_rating)
            features["elo_away_pre"].append(away_rating)
            features["elo_diff"].append(home_rating - away_rating)
            features["home_form_points_5"].append(home_form[0])
            features["away_form_points_5"].append(away_form[0])
            features["form_points_diff_5"].append(home_form[0] - away_form[0])
            features["home_form_goals_for_5"].append(home_form[1])
            features["away_form_goals_for_5"].append(away_form[1])
            features["form_goals_for_diff_5"].append(home_form[1] - away_form[1])
            features["home_form_goals_against_5"].append(home_form[2])
            features["away_form_goals_against_5"].append(away_form[2])
            features["form_goals_against_diff_5"].append(
                home_form[2] - away_form[2]
            )
            features["home_form_matches_5"].append(home_form[3])
            features["away_form_matches_5"].append(away_form[3])
            features["home_rest_days"].append(home_rest)
            features["away_rest_days"].append(away_rest)
            features["home_rest_known"].append(int(home_rest >= 0))
            features["away_rest_known"].append(int(away_rest >= 0))
            features["rest_days_diff"].append(
                home_rest - away_rest
                if home_rest >= 0 and away_rest >= 0
                else 0
            )

            rating_change = ELO_K * (actual_home - expected_home)
            pending_rating_changes[row.home_team] += rating_change
            pending_rating_changes[row.away_team] -= rating_change
            pending_history.extend(
                [
                    (
                        row.home_team,
                        (home_points, row.home_score, row.away_score),
                    ),
                    (
                        row.away_team,
                        (away_points, row.away_score, row.home_score),
                    ),
                ]
            )

        for team, change in pending_rating_changes.items():
            ratings[team] += change
        for team, history_item in pending_history:
            recent[team].append(history_item)
            last_played[team] = match_date

    for column, values in features.items():
        df[column] = values

    rating_columns = ["elo_home_pre", "elo_away_pre", "elo_diff"]
    form_columns = [column for column in features if "_form_" in column]
    df[rating_columns] = df[rating_columns].round(2)
    df[form_columns] = df[form_columns].round(3)
    return df


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def build_dataset(input_path: Path, output_path: Path, cache_dir: Path) -> None:
    df = pd.read_csv(input_path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    original_count = len(df)
    df = df.dropna(subset=list(required)).copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["_source_order"] = np.arange(len(df))
    df = df.sort_values(["date", "_source_order"], kind="stable").reset_index(
        drop=True
    )
    df["neutral"] = df["neutral"].astype(bool)
    df = add_sequential_features(df)
    df, geo_counts = add_geographic_features(df, cache_dir)

    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["Season"] = df["Month"].map(season_from_month)
    df["Month_Sin"] = np.sin(2.0 * np.pi * df["Month"] / 12.0).round(6)
    df["Month_Cos"] = np.cos(2.0 * np.pi * df["Month"] / 12.0).round(6)
    bins = [1872, 1892, 1912, 1932, 1952, 1972, 1992, 2012, 2032]
    labels = [
        "1872-1891",
        "1892-1911",
        "1912-1931",
        "1932-1951",
        "1952-1971",
        "1972-1991",
        "1992-2011",
        "2012-2031",
    ]
    df["Era"] = pd.cut(df["Year"], bins=bins, labels=labels, right=False)
    df["Match_Type"] = np.where(
        df["tournament"].eq("Friendly"), "Friendly", "Official"
    )
    df["Neutral"] = df["neutral"].astype(int)
    df["Home_Win"] = np.where(
        df["home_score"] > df["away_score"], "Win", "NotWin"
    )

    # Return to the source-file order so this output remains row-for-row
    # comparable with the project's earlier final_dataset files.
    df = df.sort_values("_source_order", kind="stable").reset_index(drop=True)

    output_columns = [
        "Year",
        "Month",
        "Season",
        "Month_Sin",
        "Month_Cos",
        "Era",
        "Match_Type",
        "Neutral",
        "elo_home_pre",
        "elo_away_pre",
        "elo_diff",
        "home_form_points_5",
        "away_form_points_5",
        "form_points_diff_5",
        "home_form_goals_for_5",
        "away_form_goals_for_5",
        "form_goals_for_diff_5",
        "home_form_goals_against_5",
        "away_form_goals_against_5",
        "form_goals_against_diff_5",
        "home_form_matches_5",
        "away_form_matches_5",
        "home_rest_days",
        "away_rest_days",
        "rest_days_diff",
        "home_rest_known",
        "away_rest_known",
        "away_travel_km",
        "away_travel_known",
        "Home_Win",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(output_path, index=False, encoding="utf-8")

    print(f"Input rows: {original_count:,}")
    print(f"Rows after dropping incomplete scores: {len(df):,}")
    print(f"Output columns: {len(output_columns)}")
    print(f"Geo resolution counts: {geo_counts}")
    print(f"Saved: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("DataSet/results.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Output/final_dataset_engineered.csv"),
    )
    parser.add_argument(
        "--geo-cache", type=Path, default=Path("DataSet/GeoNames")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.input, args.output, args.geo_cache)
