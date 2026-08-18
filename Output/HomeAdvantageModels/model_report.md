# Net Home-Advantage Modeling Report

## Design

- Rows: 49,487
- Chronological training period: 1872-11-30 to 2016-03-28
- Chronological test period: 2016-03-29 to 2026-07-01
- Test cutoff: 2016-03-29
- `is_home = 1 - Neutral`; outcome is a home-team win (draws are NotWin).
- Tournament categories with fewer than 100 matches are grouped as Other.

## Held-out predictive performance

- logistic_regression: ROC-AUC=0.786, accuracy=0.712, Brier=0.188
- random_forest: ROC-AUC=0.782, accuracy=0.711, Brier=0.193

## Adjusted logistic home effect

- is_home coefficient (log odds): 0.473
- Odds ratio: 1.604 (95% robust CI 1.518–1.695)
- p-value: 5.283e-63
- Adjusted mean home-win probability if non-neutral: 0.515
- Adjusted mean home-win probability if neutral: 0.422
- Adjusted probability difference: 0.093

These are adjusted associations. In particular, travel can be a mechanism of
home advantage, so controlling it changes the estimand from total association
toward a direct association.

## Mixed-effects home effect

- is_home odds ratio: 1.729 (95% posterior interval 1.689–1.769)
- Practical convergence: True (max |gradient|=6.4e-05)
- Raw optimizer success: False (Desired error not necessarily achieved due to precision loss.)

## Top Random Forest permutation importances

- elo_diff_100: 0.1929
- form_goals_against_diff_5: 0.0079
- is_home: 0.0073
- tournament_grouped: 0.0031
- form_goals_for_diff_5: 0.0018
- rest_diff_7: 0.0009
- travel_1000: 0.0009
- form_points_diff_5: 0.0005
