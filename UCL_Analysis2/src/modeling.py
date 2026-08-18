"""Predictive and adjusted home-advantage models for the UCL analysis."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "elo_diff_pre_100",
    "alltime_elo_proxy_diff_100",
    "alltime_goals_per_match_diff",
    "alltime_conceded_per_match_diff",
    "alltime_experience_log_diff",
    "season_centered",
]
CATEGORICAL_FEATURES = ["phase"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TEMPORAL_NUMERIC_FEATURES = ["elo_diff_pre_100", "season_centered"]
TEMPORAL_FEATURES = TEMPORAL_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_model_data(enriched: pd.DataFrame) -> pd.DataFrame:
    data = enriched.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["elo_diff_pre_100"] = data["elo_diff_pre"] / 100.0
    data["alltime_elo_proxy_diff_100"] = data["alltime_elo_proxy_diff"] / 100.0
    data["alltime_experience_log_diff"] = np.log1p(
        data["home_alltime_Matches"]
    ) - np.log1p(data["away_alltime_Matches"])
    data["season_centered"] = data["season_start"] - data["season_start"].median()
    if data[MODEL_FEATURES].isna().any().any():
        missing = data[MODEL_FEATURES].isna().sum()
        raise ValueError(f"Missing model features: {missing[missing.gt(0)].to_dict()}")
    return data.sort_values(["date", "source_row"], kind="stable").reset_index(drop=True)


def make_pipeline(estimator, numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def evaluate(
    name: str, model: Pipeline, test: pd.DataFrame, features: list[str]
) -> tuple[dict, np.ndarray]:
    y = test["home_win"]
    probability = model.predict_proba(test[features])[:, 1]
    predicted = probability.ge(0.5) if isinstance(probability, pd.Series) else probability >= 0.5
    return {
        "model": name,
        "n_test": len(test),
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "log_loss": log_loss(y, probability),
        "brier_score": brier_score_loss(y, probability),
    }, probability


def run_predictive_models(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cutoff_index = int(len(data) * 0.8)
    cutoff = data.iloc[cutoff_index]["date"]
    train = data[data["date"].lt(cutoff)].copy()
    test = data[data["date"].ge(cutoff)].copy()

    models = {
        "logistic_temporal_leakage_safe": (
            make_pipeline(
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                TEMPORAL_NUMERIC_FEATURES,
            ),
            TEMPORAL_FEATURES,
        ),
        "logistic_alltime_enriched": (
            make_pipeline(
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                NUMERIC_FEATURES,
            ),
            MODEL_FEATURES,
        ),
        "random_forest_alltime_enriched": (
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
                NUMERIC_FEATURES,
            ),
            MODEL_FEATURES,
        ),
    }
    rows = []
    probabilities = {}
    for name, (model, features) in models.items():
        model.fit(train[features], train["home_win"])
        row, probability = evaluate(name, model, test, features)
        rows.append(row)
        probabilities[name] = probability
        joblib.dump(model, output_dir / f"{name}_pipeline.joblib", compress=3)

    forest, forest_features = models["random_forest_alltime_enriched"]
    permutation = permutation_importance(
        forest,
        test[forest_features],
        test["home_win"],
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean_auc_decrease": permutation.importances_mean,
            "importance_sd": permutation.importances_std,
        }
    ).sort_values("importance_mean_auc_decrease", ascending=False)
    metadata = {
        "rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff_date": cutoff.date().isoformat(),
        "train_date_min": train["date"].min().date().isoformat(),
        "train_date_max": train["date"].max().date().isoformat(),
        "test_date_min": test["date"].min().date().isoformat(),
        "test_date_max": test["date"].max().date().isoformat(),
        "test_home_win_rate": float(test["home_win"].mean()),
        "temporal_leakage_safe_features": TEMPORAL_FEATURES,
        "alltime_enriched_features": MODEL_FEATURES,
        "warning": (
            "All-time aggregate features summarize the full history and therefore "
            "must not be interpreted as leakage-free historical forecasting inputs."
        ),
    }
    return pd.DataFrame(rows), importance, metadata


def fit_adjusted_decisive_model(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Estimate leakage-safe home advantage among matches without a draw.

    The all-time aggregate variables are intentionally excluded here because
    they include future seasons. They remain useful in the separate descriptive
    prediction model, but not in this temporal home-advantage estimate.
    """
    decisive = data[data["result"].isin(["H", "A"])].copy()
    decisive["home_decisive_win"] = decisive["result"].eq("H").astype(int)
    result = smf.glm(
        formula=(
            "home_decisive_win ~ elo_diff_pre_100 + season_centered "
            "+ C(phase, Treatment(reference='Group'))"
        ),
        data=decisive,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3")
    conf = result.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient_log_odds": result.params.values,
            "robust_se": result.bse.values,
            "p_value": result.pvalues.values,
            "odds_ratio": np.exp(result.params.values),
            "odds_ratio_ci_low": np.exp(conf.iloc[:, 0].values),
            "odds_ratio_ci_high": np.exp(conf.iloc[:, 1].values),
        }
    )
    intercept = float(result.params["Intercept"])
    elo_beta = float(result.params["elo_diff_pre_100"])
    reference_probability = 1.0 / (1.0 + np.exp(-intercept))
    summary = {
        "n_decisive_matches": int(len(decisive)),
        "reference_definition": (
            "Group-stage match, median season, and equal leakage-safe pre-match dynamic Elo."
        ),
        "adjusted_home_win_probability_among_decisive_reference": reference_probability,
        "adjusted_home_odds_ratio_reference": float(np.exp(intercept)),
        "intercept_p_value": float(result.pvalues["Intercept"]),
        "elo_equivalent_home_advantage_points": (
            float(100.0 * intercept / elo_beta) if elo_beta != 0 else None
        ),
        "caution": (
            "This is an adjusted association among decisive matches, not a causal effect; "
            "drawn matches are excluded from this specific estimate."
        ),
    }
    coefficients.to_csv(output_dir / "adjusted_decisive_coefficients.csv", index=False)
    (output_dir / "adjusted_decisive_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "adjusted_decisive_model_summary.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )
    return coefficients, summary


def run_and_save_models(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict]:
    enriched = pd.read_csv(output_dir / "detailed_matches_enriched.csv")
    data = prepare_model_data(enriched)
    metrics, importance, metadata = run_predictive_models(data, output_dir)
    coefficients, adjusted = fit_adjusted_decisive_model(data, output_dir)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    importance.to_csv(output_dir / "random_forest_permutation_importance.csv", index=False)
    (output_dir / "model_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics, importance, metadata, coefficients, adjusted
