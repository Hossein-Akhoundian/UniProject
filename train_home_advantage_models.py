"""Train predictive and inferential models for net football home advantage.

The script combines the leakage-safe engineered data with team/date columns
from the original results file, validates the row alignment, performs a
chronological train/test split, and fits:

1. A regularized multivariable logistic-regression prediction pipeline.
2. An unpenalized binomial GLM with robust confidence intervals.
3. A Random Forest with held-out permutation importance.
4. A binomial mixed-effects model with home/away team random intercepts.

Run from the project root:
    python train_home_advantage_models.py

Use ``--skip-mixed`` only when a quick predictive-only run is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


RANDOM_STATE = 42

REQUIRED_RAW_COLUMNS = {
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

NUMERIC_FEATURES = [
    "is_home",
    "elo_diff_100",
    "form_points_diff_5",
    "form_goals_for_diff_5",
    "form_goals_against_diff_5",
    "rest_diff_7",
    "home_rest_known",
    "away_rest_known",
    "travel_1000",
    "away_travel_known",
    "Month_Sin",
    "Month_Cos",
    "year_decade",
]
CATEGORICAL_FEATURES = ["tournament_grouped"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FIXED_EFFECT_FORMULA = (
    "home_win ~ is_home + elo_diff_100 + form_points_diff_5 "
    "+ form_goals_for_diff_5 + form_goals_against_diff_5 "
    "+ rest_diff_7 + home_rest_known + away_rest_known "
    "+ travel_1000 + away_travel_known + Month_Sin + Month_Cos "
    "+ year_decade + C(tournament_grouped, Treatment(reference='Friendly'))"
)

# Crossed random intercepts already absorb a large amount of team-level
# heterogeneity. A deliberately parsimonious fixed-effects part makes the GLMM
# identifiable and numerically stable, while the full tournament controls
# remain in the main inferential GLM above.
MIXED_EFFECT_FORMULA = (
    "home_win ~ is_home + elo_diff_100 + form_points_diff_5 "
    "+ rest_diff_7 + travel_1000 + away_travel_known "
    "+ C(Match_Type, Treatment(reference='Friendly')) + year_decade"
)


def json_ready(value: Any) -> Any:
    """Convert numpy/pandas values into strict JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def load_and_validate_data(
    raw_path: Path,
    engineered_path: Path,
    min_tournament_matches: int,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    missing = REQUIRED_RAW_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")

    # This exactly mirrors build_engineered_dataset.py before it returns rows
    # to source order.
    raw = raw.dropna(subset=list(REQUIRED_RAW_COLUMNS)).copy().reset_index(drop=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    raw["neutral"] = raw["neutral"].astype(bool)

    engineered = pd.read_csv(engineered_path)
    if len(raw) != len(engineered):
        raise ValueError(
            "Raw/engineered row counts differ after filtering: "
            f"{len(raw):,} != {len(engineered):,}. Rebuild the engineered file."
        )

    expected_year = raw["date"].dt.year.to_numpy()
    expected_target = np.where(
        raw["home_score"].to_numpy() > raw["away_score"].to_numpy(),
        "Win",
        "NotWin",
    )
    expected_neutral = raw["neutral"].astype(int).to_numpy()
    if not np.array_equal(engineered["Year"].to_numpy(), expected_year):
        raise ValueError("Year mismatch: engineered and raw rows are not aligned.")
    if not np.array_equal(engineered["Home_Win"].to_numpy(), expected_target):
        raise ValueError("Target mismatch: engineered and raw rows are not aligned.")
    if not np.array_equal(engineered["Neutral"].to_numpy(), expected_neutral):
        raise ValueError("Neutral mismatch: engineered and raw rows are not aligned.")

    data = engineered.copy()
    for column in ["date", "home_team", "away_team", "tournament"]:
        data[column] = raw[column].to_numpy()

    data["home_win"] = (data["Home_Win"] == "Win").astype(int)
    data["is_home"] = 1 - data["Neutral"].astype(int)

    # Fixed, scientifically interpretable units also reduce numerical issues.
    data["elo_diff_100"] = data["elo_diff"] / 100.0
    data["rest_diff_7"] = data["rest_days_diff"].clip(-365, 365) / 7.0
    data["travel_1000"] = (
        data["away_travel_km"].where(data["away_travel_known"].eq(1), 0.0)
        .clip(lower=0.0, upper=20_000.0)
        / 1000.0
    )
    data["year_decade"] = (data["Year"] - 2000.0) / 10.0

    tournament_counts = data["tournament"].value_counts()
    common = set(
        tournament_counts[tournament_counts >= min_tournament_matches].index
    )
    # Friendly is the explicit reference category in the statistical formulas.
    common.add("Friendly")
    data["tournament_grouped"] = data["tournament"].where(
        data["tournament"].isin(common), "Other"
    )
    data["tournament_grouped"] = pd.Categorical(data["tournament_grouped"])

    required_model_columns = MODEL_FEATURES + [
        "home_win",
        "date",
        "home_team",
        "away_team",
    ]
    if data[required_model_columns].isna().any().any():
        counts = data[required_model_columns].isna().sum()
        raise ValueError(
            f"Unexpected missing model values: {counts[counts > 0].to_dict()}"
        )
    return data


def chronological_split(
    data: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("--train-fraction must be in [0.5, 1.0).")
    ordered_dates = data["date"].sort_values(kind="stable").reset_index(drop=True)
    cutoff = ordered_dates.iloc[int(len(data) * train_fraction)]
    train = data[data["date"] < cutoff].copy()
    test = data[data["date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")
    return train, test, cutoff


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prediction_metrics(
    name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    metrics = {
        "model": name,
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
        "log_loss": log_loss(y_test, probability),
        "brier_score": brier_score_loss(y_test, probability),
    }
    return metrics, probability


def fit_predictive_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    rf_trees: int,
    permutation_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    x_train = train[MODEL_FEATURES]
    y_train = train["home_win"]
    x_test = test[MODEL_FEATURES]
    y_test = test["home_win"]

    logistic = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=rf_trees,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models = {"logistic_regression": logistic, "random_forest": forest}
    metric_rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(x_train, y_train)
        metrics, probability = prediction_metrics(name, model, x_test, y_test)
        metric_rows.append(metrics)
        probabilities[name] = probability
        joblib.dump(model, output_dir / f"{name}_pipeline.joblib", compress=3)

    # Permuting original columns measures each feature as a whole, even when
    # its encoded representation contains many dummy columns.
    print("Calculating held-out Random Forest permutation importance...")
    perm = permutation_importance(
        forest,
        x_test,
        y_test,
        n_repeats=permutation_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="roc_auc",
    )
    permutation_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean_auc_decrease": perm.importances_mean,
            "importance_sd": perm.importances_std,
        }
    ).sort_values("importance_mean_auc_decrease", ascending=False)
    permutation_df.to_csv(
        output_dir / "random_forest_permutation_importance.csv", index=False
    )

    transformed_names = forest.named_steps["preprocess"].get_feature_names_out()
    impurity_df = pd.DataFrame(
        {
            "encoded_feature": transformed_names,
            "importance": forest.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    impurity_df.to_csv(
        output_dir / "random_forest_impurity_importance_detailed.csv", index=False
    )

    return pd.DataFrame(metric_rows), permutation_df, probabilities


def fit_inferential_glm(
    data: pd.DataFrame, output_dir: Path
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    print("Fitting inferential logistic GLM with HC3 robust intervals...")
    result = smf.glm(
        formula=FIXED_EFFECT_FORMULA,
        data=data,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3", maxiter=200)

    conf = result.conf_int()
    coefficient_df = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient_log_odds": result.params.to_numpy(),
            "robust_se": result.bse.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "odds_ratio": np.exp(result.params.to_numpy()),
            "odds_ratio_ci_low": np.exp(conf.iloc[:, 0].to_numpy()),
            "odds_ratio_ci_high": np.exp(conf.iloc[:, 1].to_numpy()),
        }
    )
    coefficient_df.to_csv(
        output_dir / "logistic_inference_coefficients.csv", index=False
    )
    (output_dir / "logistic_inference_summary.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )

    home = coefficient_df.loc[coefficient_df["term"].eq("is_home")].iloc[0]
    counterfactual_home = data.copy()
    counterfactual_neutral = data.copy()
    counterfactual_home["is_home"] = 1
    counterfactual_neutral["is_home"] = 0
    prob_home = np.asarray(result.predict(counterfactual_home))
    prob_neutral = np.asarray(result.predict(counterfactual_neutral))
    home_effect = {
        "coefficient_log_odds": home["coefficient_log_odds"],
        "robust_se": home["robust_se"],
        "p_value": home["p_value"],
        "odds_ratio": home["odds_ratio"],
        "odds_ratio_ci_95": [
            home["odds_ratio_ci_low"],
            home["odds_ratio_ci_high"],
        ],
        "adjusted_probability_if_home": prob_home.mean(),
        "adjusted_probability_if_neutral": prob_neutral.mean(),
        "average_adjusted_probability_difference": (
            prob_home - prob_neutral
        ).mean(),
        "note": (
            "Counterfactual probabilities hold every included control fixed; "
            "this is a model-based direct association, not a causal estimate."
        ),
    }
    return result, coefficient_df, home_effect


def fit_mixed_effects(
    data: pd.DataFrame, output_dir: Path, maxiter: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print("Fitting binomial mixed-effects model (variational Bayes)...")
    variance_formulas = {
        "home_team_random_intercept": "0 + C(home_team)",
        "away_team_random_intercept": "0 + C(away_team)",
    }
    model = BinomialBayesMixedGLM.from_formula(
        MIXED_EFFECT_FORMULA,
        variance_formulas,
        data,
        vcp_p=0.5,
        fe_p=2.0,
    )
    # statsmodels clips small starting posterior SD values in fit_vb, so several
    # nominally warm-started calls are not a true optimizer continuation. One
    # uninterrupted, seeded run is both reproducible and numerically sounder.
    np.random.seed(RANDOM_STATE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_vb(
            fit_method="BFGS",
            minim_opts={"maxiter": maxiter, "gtol": 1e-5},
            scale_fe=False,
            verbose=False,
        )
    warning_messages = [str(item.message) for item in caught]

    fixed_df = pd.DataFrame(
        {
            "term": model.exog_names,
            "posterior_mean_log_odds": result.fe_mean,
            "posterior_sd": result.fe_sd,
        }
    )
    fixed_df["odds_ratio"] = np.exp(fixed_df["posterior_mean_log_odds"])
    fixed_df["odds_ratio_ci_low"] = np.exp(
        fixed_df["posterior_mean_log_odds"] - 1.96 * fixed_df["posterior_sd"]
    )
    fixed_df["odds_ratio_ci_high"] = np.exp(
        fixed_df["posterior_mean_log_odds"] + 1.96 * fixed_df["posterior_sd"]
    )
    fixed_df.to_csv(output_dir / "mixed_effects_fixed_effects.csv", index=False)

    variance_df = pd.DataFrame(
        {
            "random_effect": model.vcp_names,
            "posterior_mean_log_sd": result.vcp_mean,
            "posterior_sd_log_sd": result.vcp_sd,
            "estimated_random_intercept_sd": np.exp(result.vcp_mean),
        }
    )
    variance_df.to_csv(
        output_dir / "mixed_effects_random_intercept_sd.csv", index=False
    )

    optim = getattr(result, "optim_retvals", {})
    final_jacobian = np.asarray(optim.get("jac", []), dtype=float)
    gradient_max_abs = (
        float(np.max(np.abs(final_jacobian))) if final_jacobian.size else None
    )
    optimizer_success = bool(optim.get("success", False))
    # BFGS can report precision loss after reaching an essentially stationary
    # point. Preserve its raw flag, but separately expose a predeclared,
    # auditable gradient-based practical criterion.
    practically_converged = optimizer_success or (
        gradient_max_abs is not None and gradient_max_abs < 1e-3
    )
    convergence = {
        "optimizer_success": optimizer_success,
        "practically_converged": practically_converged,
        "practical_convergence_threshold_max_abs_gradient": 1e-3,
        "message": str(optim.get("message", "")),
        "iterations": optim.get("nit"),
        "gradient_l2_norm": (
            float(np.linalg.norm(final_jacobian))
            if final_jacobian.size
            else None
        ),
        "gradient_max_abs": gradient_max_abs,
        "warnings": warning_messages,
        "method": "statsmodels BinomialBayesMixedGLM fit_vb",
        "interval_type": "normal approximation to posterior credible interval",
    }
    (output_dir / "mixed_effects_convergence.json").write_text(
        json.dumps(json_ready(convergence), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # The fitted result can be large; summaries and random effects are the
    # reproducible scientific outputs, while the script recreates the fit.
    home = fixed_df.loc[fixed_df["term"].eq("is_home")].iloc[0]
    home_effect = {
        "coefficient_log_odds": home["posterior_mean_log_odds"],
        "posterior_sd": home["posterior_sd"],
        "odds_ratio": home["odds_ratio"],
        "odds_ratio_credible_interval_95": [
            home["odds_ratio_ci_low"],
            home["odds_ratio_ci_high"],
        ],
        "practically_converged": convergence["practically_converged"],
        "optimizer_success": convergence["optimizer_success"],
        "gradient_max_abs": convergence["gradient_max_abs"],
        "optimizer_message": convergence["message"],
    }
    return fixed_df, variance_df, home_effect


def save_summary_figure(
    output_dir: Path,
    glm_coefficients: pd.DataFrame,
    permutation_df: pd.DataFrame,
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
) -> None:
    home = glm_coefficients.loc[glm_coefficients["term"].eq("is_home")].iloc[0]
    importance = permutation_df.head(10).sort_values(
        "importance_mean_auc_decrease", ascending=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].errorbar(
        home["odds_ratio"],
        0,
        xerr=[
            [home["odds_ratio"] - home["odds_ratio_ci_low"]],
            [home["odds_ratio_ci_high"] - home["odds_ratio"]],
        ],
        fmt="o",
        color="#0B6E4F",
        capsize=5,
    )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks([0], ["is_home"])
    axes[0].set_xlabel("Odds ratio (95% robust CI)")
    axes[0].set_title("Adjusted home effect: logistic GLM")

    axes[1].barh(
        importance["feature"],
        importance["importance_mean_auc_decrease"],
        xerr=importance["importance_sd"],
        color="#4C78A8",
        alpha=0.9,
    )
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Decrease in test ROC-AUC")
    axes[1].set_title("Random Forest permutation importance")

    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        axes[2].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[2].set_xlabel("False-positive rate")
    axes[2].set_ylabel("True-positive rate")
    axes[2].set_title("Chronological holdout ROC")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "home_advantage_model_summary.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    glm_home: dict[str, Any],
    mixed_home: dict[str, Any] | None,
    permutation_df: pd.DataFrame,
) -> None:
    metric_lines = [
        (
            f"- {row.model}: ROC-AUC={row.roc_auc:.3f}, "
            f"accuracy={row.accuracy:.3f}, Brier={row.brier_score:.3f}"
        )
        for row in metrics.itertuples(index=False)
    ]
    top_features = "\n".join(
        f"- {row.feature}: {row.importance_mean_auc_decrease:.4f}"
        for row in permutation_df.head(8).itertuples(index=False)
    )
    mixed_section = "- Not run."
    if mixed_home is not None:
        low, high = mixed_home["odds_ratio_credible_interval_95"]
        mixed_section = (
            f"- is_home odds ratio: {mixed_home['odds_ratio']:.3f} "
            f"(95% posterior interval {low:.3f}–{high:.3f})\n"
            f"- Practical convergence: {mixed_home['practically_converged']} "
            f"(max |gradient|={mixed_home['gradient_max_abs']:.2g})\n"
            f"- Raw optimizer success: {mixed_home['optimizer_success']} "
            f"({mixed_home['optimizer_message']})"
        )

    low, high = glm_home["odds_ratio_ci_95"]
    report = f"""# Net Home-Advantage Modeling Report

## Design

- Rows: {metadata['n_rows']:,}
- Chronological training period: {metadata['train_date_min']} to {metadata['train_date_max']}
- Chronological test period: {metadata['test_date_min']} to {metadata['test_date_max']}
- Test cutoff: {metadata['cutoff_date']}
- `is_home = 1 - Neutral`; outcome is a home-team win (draws are NotWin).
- Tournament categories with fewer than {metadata['min_tournament_matches']} matches are grouped as Other.

## Held-out predictive performance

{chr(10).join(metric_lines)}

## Adjusted logistic home effect

- is_home coefficient (log odds): {glm_home['coefficient_log_odds']:.3f}
- Odds ratio: {glm_home['odds_ratio']:.3f} (95% robust CI {low:.3f}–{high:.3f})
- p-value: {glm_home['p_value']:.4g}
- Adjusted mean home-win probability if non-neutral: {glm_home['adjusted_probability_if_home']:.3f}
- Adjusted mean home-win probability if neutral: {glm_home['adjusted_probability_if_neutral']:.3f}
- Adjusted probability difference: {glm_home['average_adjusted_probability_difference']:.3f}

These are adjusted associations. In particular, travel can be a mechanism of
home advantage, so controlling it changes the estimand from total association
toward a direct association.

## Mixed-effects home effect

{mixed_section}

## Top Random Forest permutation importances

{top_features}
"""
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=Path("DataSet/results.csv")
    )
    parser.add_argument(
        "--engineered",
        type=Path,
        default=Path("Output/final_dataset_engineered.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Output/HomeAdvantageModels")
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-tournament-matches", type=int, default=100)
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--mixed-maxiter", type=int, default=2000)
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument(
        "--mixed-only",
        action="store_true",
        help="Refit only GLMM outputs and refresh an existing report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and validating aligned data...")
    data = load_and_validate_data(
        args.raw, args.engineered, args.min_tournament_matches
    )
    if args.skip_mixed and args.mixed_only:
        raise ValueError("--skip-mixed and --mixed-only cannot be used together.")

    if args.mixed_only:
        _, _, mixed_home = fit_mixed_effects(
            data, args.output_dir, args.mixed_maxiter
        )
        (args.output_dir / "mixed_effects_home_effect.json").write_text(
            json.dumps(json_ready(mixed_home), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_inputs = [
            args.output_dir / "run_metadata.json",
            args.output_dir / "predictive_metrics.csv",
            args.output_dir / "logistic_home_effect.json",
            args.output_dir / "random_forest_permutation_importance.csv",
        ]
        if all(path.exists() for path in report_inputs):
            metadata = json.loads(report_inputs[0].read_text(encoding="utf-8"))
            metrics = pd.read_csv(report_inputs[1])
            glm_home = json.loads(report_inputs[2].read_text(encoding="utf-8"))
            permutation_df = pd.read_csv(report_inputs[3])
            write_report(
                args.output_dir,
                metadata,
                metrics,
                glm_home,
                mixed_home,
                permutation_df,
            )
        print(f"Mixed-effects outputs saved in: {args.output_dir.resolve()}")
        return

    train, test, cutoff = chronological_split(data, args.train_fraction)
    print(
        f"Rows: {len(data):,}; train: {len(train):,}; test: {len(test):,}; "
        f"cutoff: {cutoff.date()}"
    )

    # This makes team identifiers available for reproducibility and mixed-model
    # diagnostics without copying scores or post-match information into X.
    modeling_columns = [
        "date",
        "home_team",
        "away_team",
        "home_win",
        *MODEL_FEATURES,
    ]
    data[modeling_columns].to_csv(
        args.output_dir / "modeling_dataset.csv", index=False
    )

    metrics, permutation_df, probabilities = fit_predictive_models(
        train,
        test,
        args.output_dir,
        args.rf_trees,
        args.permutation_repeats,
    )
    metrics.to_csv(args.output_dir / "predictive_metrics.csv", index=False)

    _, glm_coefficients, glm_home = fit_inferential_glm(data, args.output_dir)
    (args.output_dir / "logistic_home_effect.json").write_text(
        json.dumps(json_ready(glm_home), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mixed_home: dict[str, Any] | None = None
    if not args.skip_mixed:
        _, _, mixed_home = fit_mixed_effects(
            data, args.output_dir, args.mixed_maxiter
        )
        (args.output_dir / "mixed_effects_home_effect.json").write_text(
            json.dumps(json_ready(mixed_home), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    metadata = {
        "n_rows": len(data),
        "n_train": len(train),
        "n_test": len(test),
        "train_fraction_requested": args.train_fraction,
        "cutoff_date": cutoff.date().isoformat(),
        "train_date_min": train["date"].min().date().isoformat(),
        "train_date_max": train["date"].max().date().isoformat(),
        "test_date_min": test["date"].min().date().isoformat(),
        "test_date_max": test["date"].max().date().isoformat(),
        "target_prevalence_all": data["home_win"].mean(),
        "target_prevalence_train": train["home_win"].mean(),
        "target_prevalence_test": test["home_win"].mean(),
        "n_home_teams": data["home_team"].nunique(),
        "n_away_teams": data["away_team"].nunique(),
        "n_tournaments_original": data["tournament"].nunique(),
        "n_tournaments_grouped": data["tournament_grouped"].nunique(),
        "min_tournament_matches": args.min_tournament_matches,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "mixed_effects_run": not args.skip_mixed,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_summary_figure(
        args.output_dir,
        glm_coefficients,
        permutation_df,
        test["home_win"],
        probabilities,
    )
    write_report(
        args.output_dir,
        metadata,
        metrics,
        glm_home,
        mixed_home,
        permutation_df,
    )
    print(f"Done. Outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
