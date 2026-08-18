"""Reusable data preparation and analysis helpers for the UCL project.

The three source CSV files have different schemas and naming conventions.  This
module keeps the cleaning rules in one place so every notebook uses exactly the
same definitions.
"""

from __future__ import annotations

import math
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


TEAM_ALIASES = {
    "AEK Athens": "AEK Athen",
    "APOEL Nicosia": "APOEL Nikosia",
    "APOEL FC": "APOEL Nikosia",
    "AaB": "Aalborg BK",
    "Ajax": "AFC Ajax",
    "Arsenal": "Arsenal FC",
    "Astana": "FK Astana",
    "FC Astana": "FK Astana",
    "Barcelona": "FC Barcelona",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "Leverkusen": "Bayer Leverkusen",
    "Bayern München": "Bayern Munich",
    "Başakşehir FK": "İstanbul Başakşehir",
    "Başakşehir": "İstanbul Başakşehir",
    "Bordeaux": "Girondins Bordeaux",
    "Borussia M'gladbach": "Bor. Mönchengladbach",
    "M'Gladbach": "Bor. Mönchengladbach",
    "CSKA Moscow": "CSKA Moskva",
    "Celtic": "Celtic FC",
    "Chelsea": "Chelsea FC",
    "Club Brugge": "Club Brugge KV",
    "Dynamo Kyiv": "Dinamo Kiev",
    "FC Krasnodar": "FK Krasnodar",
    "Krasnodar": "FK Krasnodar",
    "FC Petrzalka 1898": "FC Petržalka",
    "FC Rostov": "FK Rostov",
    "Rostov": "FK Rostov",
    "FC Thun": "FC Thun Berner Oberland",
    "FK Crvena zvezda": "Crvena Zvezda",
    "Red Star": "Crvena Zvezda",
    "FK Partizan": "Partizan",
    "Fiorentina": "ACF Fiorentina",
    "GNK Dinamo Zagreb": "Dinamo Zagreb",
    "Inter": "Inter Milan",
    "Lazio": "Lazio Roma",
    "Liverpool": "Liverpool FC",
    "Lokomotiv Moscow": "Lokomotiv Moskva",
    "Loko Moscow": "Lokomotiv Moskva",
    "Ludogorets Razgrad": "PFC Ludogorets Razgrad",
    "Ludogorets": "PFC Ludogorets Razgrad",
    "Malmö": "Malmö FF",
    "Milan": "AC Milan",
    "Montpellier": "Montpellier HSC",
    "Málaga": "Málaga CF",
    "Napoli": "SSC Napoli",
    "Olympiacos": "Olympiakos Piraeus",
    "Olympique Lyonnais": "Olympique Lyon",
    "Lyon": "Olympique Lyon",
    "Olympique de Marseille": "Olympique Marseille",
    "Marseille": "Olympique Marseille",
    "Qarabağ Ağdam FK": "Qarabağ FK",
    "Rangers": "Rangers FC",
    "Real Betis Balompié": "Real Betis",
    "Red Bull Salzburg": "RB Salzburg",
    "Rennes": "Stade Rennais",
    "Roma": "AS Roma",
    "SC Oțelul Galați": "Oţelul Galaţi",
    "Oțelul Galați": "Oţelul Galaţi",
    "SK Rapid Wien": "Rapid Wien",
    "Sevilla": "Sevilla FC",
    "Sparta Praha": "AC Sparta Praha",
    "Spartak Moscow": "Spartak Moskva",
    "Tottenham": "Tottenham Hotspur",
    "Udinese": "Udinese Calcio",
    "Unirea Urziceni": "FC Unirea",
    "Valencia": "Valencia CF",
    "Villarreal": "Villarreal CF",
    "Young Boys": "BSC Young Boys",
    "Anderlecht": "RSC Anderlecht",
    "Athletic Club": "Athletic Bilbao",
    "Auxerre": "AJ Auxerre",
    "Basel": "FC Basel",
    "Benfica": "SL Benfica",
    "Braga": "Sporting Braga",
    "Dortmund": "Borussia Dortmund",
    "FC Copenhagen": "FC København",
    "Ferencváros": "Ferencvárosi TC",
    "Genk": "KRC Genk",
    "Gent": "KAA Gent",
    "Hoffenheim": "1899 Hoffenheim",
    "Legia Warsaw": "Legia Warszawa",
    "Lille": "Lille OSC",
    "Manchester Utd": "Manchester United",
    "Midtjylland": "FC Midtjylland",
    "Monaco": "AS Monaco",
    "Nordsjælland": "FC Nordsjælland",
    "Paris S-G": "Paris Saint-Germain",
    "Porto": "FC Porto",
    "Schalke 04": "FC Schalke 04",
    "Shakhtar": "Shakhtar Donetsk",
    "Sheriff Tiraspol": "FC Sheriff",
    "Slavia Prague": "Slavia Praha",
    "Steaua": "FCSB",
    "Twente": "FC Twente",
    "Wolfsburg": "VfL Wolfsburg",
    "Zenit": "Zenit St. Petersburg",
}


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing DataSet2."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "DataSet2").is_dir():
            return candidate
    raise FileNotFoundError("Could not find a parent directory containing DataSet2")


def repair_mojibake(value: object) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake without changing valid Unicode."""
    text = str(value).strip()
    markers = "ÃÂÅÄÈÐ�\x80\x81\x8d\x8f\x90\x9d"

    def badness(candidate: str) -> int:
        return sum(candidate.count(marker) for marker in markers)

    best = text
    for _ in range(2):
        candidates = [best]
        for encoding in ("latin1", "cp1252"):
            try:
                candidates.append(best.encode(encoding).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        candidate = min(candidates, key=lambda item: (badness(item), len(item)))
        if candidate == best:
            break
        best = candidate
    return best


def normalized_name(value: object) -> str:
    text = repair_mojibake(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def canonicalizer(alltime_teams: pd.Series):
    canonical_by_key = {
        normalized_name(team): repair_mojibake(team) for team in alltime_teams
    }
    canonical_by_key.update(
        {normalized_name(alias): target for alias, target in TEAM_ALIASES.items()}
    )

    def canonicalize(value: object) -> str:
        repaired = repair_mojibake(value)
        return canonical_by_key.get(normalized_name(repaired), repaired)

    return canonicalize


def load_alltime(path: Path) -> pd.DataFrame:
    """Clean the all-time table and calculate reproducible strength features."""
    data = pd.read_csv(path)
    data = data.drop(columns=[c for c in data.columns if c.startswith("Unnamed:")])
    data["Team"] = data["Team"].map(repair_mojibake)
    data = data.rename(columns={"Points": "source_points_invalid"})

    data["outcome_count"] = data["Wins"] + data["Draws"] + data["Losses"]
    data["outcome_count_matches"] = data["outcome_count"].eq(data["Matches"])
    data["calculated_points"] = 3 * data["Wins"] + data["Draws"]
    data["points_per_match"] = data["calculated_points"] / data["Matches"]
    data["win_rate"] = data["Wins"] / data["Matches"]
    data["draw_rate"] = data["Draws"] / data["Matches"]
    data["loss_rate"] = data["Losses"] / data["Matches"]
    data["goal_difference_per_match"] = data["Goal_Difference"] / data["Matches"]

    # Elo-like conversion of all-time result rate. A 20-match neutral prior
    # shrinks tiny historical samples toward 1500 and prevents infinite values.
    prior_matches = 20.0
    result_points = data["Wins"] + 0.5 * data["Draws"]
    shrunk_score = (result_points + 0.5 * prior_matches) / (
        data["Matches"] + prior_matches
    )
    data["alltime_result_score_shrunk"] = shrunk_score
    data["alltime_elo_proxy"] = 1500.0 + 400.0 * np.log10(
        shrunk_score / (1.0 - shrunk_score)
    )
    data["alltime_elo_proxy"] = data["alltime_elo_proxy"].round(1)
    return data


def _leading_score(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"^\s*(\d+)", expand=False)
    if extracted.isna().any():
        bad = series[extracted.isna()].unique().tolist()
        raise ValueError(f"Could not parse score values: {bad}")
    return extracted.astype(int)


def load_detailed_matches(path: Path, alltime: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    canonicalize = canonicalizer(alltime["Team"])
    data = pd.DataFrame(
        {
            "source_row": raw.index,
            "date": pd.to_datetime(raw["date"], errors="raise"),
            "home_team_raw": raw["homeTeam"].map(repair_mojibake),
            "away_team_raw": raw["awayteam"].map(repair_mojibake),
            "home_goals_raw": raw["homeScore"].astype(str),
            "away_goals_raw": raw["awayscore"].astype(str),
            "round_raw": raw["round"],
            "group": raw["group"],
        }
    )
    data["home_team"] = data["home_team_raw"].map(canonicalize)
    data["away_team"] = data["away_team_raw"].map(canonicalize)
    data["home_goals"] = _leading_score(data["home_goals_raw"])
    data["away_goals"] = _leading_score(data["away_goals_raw"])
    data["has_score_annotation"] = (
        data["home_goals_raw"].str.contains("(", regex=False)
        | data["away_goals_raw"].str.contains("(", regex=False)
    )
    data["season_start"] = data["date"].dt.year - data["date"].dt.month.lt(7)
    data["round"] = data["round_raw"].str.replace("round : ", "", regex=False)
    data["phase"] = np.where(
        data["round"].isin(["1", "2", "3", "4", "5", "6"]),
        "Group",
        "Knockout",
    )
    data["result"] = np.select(
        [data["home_goals"].gt(data["away_goals"]), data["home_goals"].lt(data["away_goals"])],
        ["H", "A"],
        default="D",
    )
    data["home_win"] = data["result"].eq("H").astype(int)
    data["goal_difference"] = data["home_goals"] - data["away_goals"]
    data["source"] = "detailed_2005_2021"
    return data


def load_ucl_matches(path: Path, alltime: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    canonicalize = canonicalizer(alltime["Team"])
    duplicate_count = int(raw.duplicated().sum())
    data = raw.drop_duplicates().copy().reset_index(drop=True)
    data.insert(0, "source_row", np.arange(len(data)))
    data["home_team_raw"] = data["home_team"].map(repair_mojibake)
    data["away_team_raw"] = data["away_team"].map(repair_mojibake)
    data["home_team"] = data["home_team_raw"].map(canonicalize)
    data["away_team"] = data["away_team_raw"].map(canonicalize)
    calculated = np.select(
        [data["home_goals"].gt(data["away_goals"]), data["home_goals"].lt(data["away_goals"])],
        ["H", "A"],
        default="D",
    )
    inconsistent_results = int((calculated != data["result"]).sum())
    data["result"] = calculated
    data["home_win"] = data["result"].eq("H").astype(int)
    data["goal_difference"] = data["home_goals"] - data["away_goals"]
    data = data.rename(columns={"season": "season_start"})
    data["source"] = "ucl_deduplicated"
    audit = {
        "raw_rows": int(len(raw)),
        "unique_rows": int(len(data)),
        "exact_duplicate_rows_removed": duplicate_count,
        "result_inconsistencies": inconsistent_results,
    }
    return data, audit


MATCH_KEY = ["season_start", "home_team", "away_team", "home_goals", "away_goals"]


def combine_match_sources(detailed: pd.DataFrame, ucl: pd.DataFrame) -> pd.DataFrame:
    """Create a multiset union while preferring dated detailed records.

    The second source has no date or round. Occurrence numbers prevent a real
    repeated fixture in the dated source from being accidentally removed.
    """
    left = detailed.copy()
    right = ucl.copy()
    left["_occurrence"] = left.groupby(MATCH_KEY, dropna=False).cumcount()
    right["_occurrence"] = right.groupby(MATCH_KEY, dropna=False).cumcount()
    left_keys = set(map(tuple, left[MATCH_KEY + ["_occurrence"]].to_numpy()))
    right["_joined_key"] = list(
        map(tuple, right[MATCH_KEY + ["_occurrence"]].to_numpy())
    )
    supplemental = right.loc[~right["_joined_key"].isin(left_keys)].copy()

    shared_columns = [
        "season_start",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "home_win",
        "goal_difference",
        "source",
    ]
    left_out = left[["date", "phase", "round", *shared_columns]].copy()
    supplemental_out = supplemental[shared_columns].copy()
    supplemental_out.insert(0, "round", "Unknown")
    supplemental_out.insert(0, "phase", "Unknown")
    supplemental_out.insert(0, "date", pd.NaT)
    supplemental_out["source"] = "ucl_supplemental"
    combined = pd.concat([left_out, supplemental_out], ignore_index=True)
    combined.insert(0, "match_id", np.arange(1, len(combined) + 1))
    return combined


def add_dynamic_elo(
    detailed: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add leakage-safe pre-match Elo ratings to dated matches.

    No home bonus is inserted in the Elo expectation because home advantage is
    itself the research target. Same-day matches use the state at day start.
    """
    data = detailed.sort_values(["date", "source_row"], kind="stable").copy()
    ratings: defaultdict[str, float] = defaultdict(lambda: initial_rating)
    output: dict[str, list[float]] = defaultdict(list)

    for match_date, day in data.groupby("date", sort=True):
        pending: defaultdict[str, float] = defaultdict(float)
        for row in day.itertuples(index=False):
            home_pre = ratings[row.home_team]
            away_pre = ratings[row.away_team]
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_pre - home_pre) / 400.0))
            actual_home = 1.0 if row.result == "H" else 0.0 if row.result == "A" else 0.5
            change = k_factor * (actual_home - expected_home)
            pending[row.home_team] += change
            pending[row.away_team] -= change
            output["home_elo_pre"].append(home_pre)
            output["away_elo_pre"].append(away_pre)
            output["elo_diff_pre"].append(home_pre - away_pre)
            output["elo_expected_home_score"].append(expected_home)
        for team, change in pending.items():
            ratings[team] += change

    for column, values in output.items():
        data[column] = np.round(values, 3)
    final = pd.DataFrame(
        {"Team": list(ratings.keys()), "dynamic_elo_final": list(ratings.values())}
    ).sort_values("dynamic_elo_final", ascending=False)
    final["dynamic_elo_final"] = final["dynamic_elo_final"].round(1)
    return data, final.reset_index(drop=True)


def enrich_with_alltime(matches: pd.DataFrame, alltime: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "Team",
        "Rank",
        "Matches",
        "win_rate",
        "points_per_match",
        "Goals_per_match",
        "Goals_conceded_per_match",
        "goal_difference_per_match",
        "alltime_elo_proxy",
    ]
    strength = alltime[fields].copy()
    home = strength.add_prefix("home_alltime_").rename(
        columns={"home_alltime_Team": "home_team"}
    )
    away = strength.add_prefix("away_alltime_").rename(
        columns={"away_alltime_Team": "away_team"}
    )
    data = matches.merge(home, on="home_team", how="left", validate="many_to_one")
    data = data.merge(away, on="away_team", how="left", validate="many_to_one")
    difference_pairs = {
        "alltime_elo_proxy_diff": "alltime_elo_proxy",
        "alltime_win_rate_diff": "win_rate",
        "alltime_ppm_diff": "points_per_match",
        "alltime_goals_per_match_diff": "Goals_per_match",
        "alltime_conceded_per_match_diff": "Goals_conceded_per_match",
    }
    for output, suffix in difference_pairs.items():
        data[output] = data[f"home_alltime_{suffix}"] - data[f"away_alltime_{suffix}"]
    return data


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z**2 / total
    centre = (p + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * total)) / total) / denominator
    return centre - margin, centre + margin


def home_advantage_summary(data: pd.DataFrame, dataset_name: str) -> dict:
    counts = data["result"].value_counts()
    n = len(data)
    home_wins = int(counts.get("H", 0))
    away_wins = int(counts.get("A", 0))
    draws = int(counts.get("D", 0))
    low, high = wilson_interval(home_wins, n)
    decisive = home_wins + away_wins
    return {
        "dataset": dataset_name,
        "matches": n,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "home_win_rate": home_wins / n,
        "home_win_ci_low": low,
        "home_win_ci_high": high,
        "draw_rate": draws / n,
        "away_win_rate": away_wins / n,
        "home_share_decisive": home_wins / decisive if decisive else math.nan,
        "mean_home_goal_difference": data["goal_difference"].mean(),
    }


def build_and_save_all(repo_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    alltime = load_alltime(repo_root / "DataSet2" / "UCL_AllTime.csv")
    detailed = load_detailed_matches(
        repo_root / "DataSet2" / "UEFA Champions League 2004-2021.csv", alltime
    )
    ucl, ucl_audit = load_ucl_matches(repo_root / "DataSet2" / "ucl.csv", alltime)
    combined = combine_match_sources(detailed, ucl)
    detailed_elo, final_elo = add_dynamic_elo(detailed)
    enriched = enrich_with_alltime(detailed_elo, alltime)

    alltime.to_csv(output_dir / "alltime_team_strength.csv", index=False)
    detailed.to_csv(output_dir / "detailed_matches_clean.csv", index=False)
    ucl.to_csv(output_dir / "ucl_matches_deduplicated.csv", index=False)
    combined.to_csv(output_dir / "combined_unique_matches.csv", index=False)
    enriched.to_csv(output_dir / "detailed_matches_enriched.csv", index=False)
    final_elo.to_csv(output_dir / "dynamic_elo_final.csv", index=False)

    alltime_names = set(alltime["Team"])
    match_names = set(detailed["home_team"]) | set(detailed["away_team"]) | set(ucl["home_team"]) | set(ucl["away_team"])
    audit = {
        **ucl_audit,
        "detailed_rows": int(len(detailed)),
        "combined_unique_rows": int(len(combined)),
        "supplemental_ucl_rows": int(combined["source"].eq("ucl_supplemental").sum()),
        "canonical_match_teams": int(len(match_names)),
        "teams_unmatched_to_alltime": sorted(match_names - alltime_names),
        "detailed_score_annotations": int(detailed["has_score_annotation"].sum()),
        "alltime_invalid_source_points_rows": int(
            alltime["source_points_invalid"].eq(alltime["Goal_Difference"]).sum()
        ),
        "alltime_outcome_count_mismatch_rows": int(
            (~alltime["outcome_count_matches"]).sum()
        ),
    }
    (output_dir / "data_quality_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit
