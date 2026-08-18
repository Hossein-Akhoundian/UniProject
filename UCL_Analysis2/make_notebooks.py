"""Generate the five documented UCL analysis notebooks."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NOTEBOOK_DIR = HERE / "Notebooks"

BOOTSTRAP = """from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "DataSet2").is_dir())
ANALYSIS_ROOT = ROOT / "UCL_Analysis2"
OUTPUT = ANALYSIS_ROOT / "Output"
sys.path.insert(0, str(ANALYSIS_ROOT / "src"))
pd.set_option("display.max_columns", 100)
plt.style.use("seaborn-v0_8-whitegrid")
print("Repository root:", ROOT)
"""


def notebook(title: str, intro: str, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"),
        nbf.v4.new_code_cell(BOOTSTRAP),
        *cells,
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebooks() -> dict[str, nbf.NotebookNode]:
    notebooks = {}

    notebooks["01_Data_Understanding.ipynb"] = notebook(
        "۱. شناخت داده‌های جدید لیگ قهرمانان اروپا",
        "در این نوت‌بوک ساختار سه فایل، کیفیت داده و محدودیت‌های اولیه بررسی می‌شود. هیچ داده‌ای هنوز تغییر نمی‌کند.",
        [
            md("## معرفی فایل‌ها\n\n- `UCL_AllTime.csv`: آمار تجمعی ۳۵۴ تیم در کل تاریخ.\n- `UEFA Champions League 2004-2021.csv`: مسابقات تاریخ‌دار با مرحله و گروه.\n- `ucl.csv`: مسابقات فصل‌های ۲۰۱۰ تا ۲۰۲۱، بدون تاریخ و مرحله."),
            code("""files = sorted((ROOT / "DataSet2").glob("*.csv"))
rows = []
raw_frames = {}
for path in files:
    frame = pd.read_csv(path)
    raw_frames[path.name] = frame
    rows.append({
        "file": path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    })
pd.DataFrame(rows)
"""),
            code("""for name, frame in raw_frames.items():
    print(f"\\n{name}: {frame.shape}")
    display(frame.head(3))
    display(frame.isna().sum().rename("missing").to_frame().query("missing > 0"))
"""),
            md("## کنترل سازگاری جدول All-Time\n\nستون `Points` منبع، امتیاز فوتبالی نیست. در تمام ردیف‌ها با تفاضل گل برابر است؛ پس در ادامه کنار گذاشته و `3×Wins + Draws` بازسازی می‌شود."),
            code("""all_raw = raw_frames["UCL_AllTime.csv"]
checks = pd.Series({
    "W + D + L equals Matches": int(((all_raw.Wins + all_raw.Draws + all_raw.Losses) == all_raw.Matches).sum()),
    "GF - GA equals Goal_Difference": int(((all_raw.Goals_scored - all_raw.Goals_conceded) == all_raw.Goal_Difference).sum()),
    "source Points equals Goal_Difference": int((all_raw.Points == all_raw.Goal_Difference).sum()),
}, name="rows_passing")
display(checks.to_frame())
display(all_raw.loc[(all_raw.Wins + all_raw.Draws + all_raw.Losses) != all_raw.Matches,
                    ["Team", "Matches", "Wins", "Draws", "Losses"]])
"""),
            md("## کنترل امتیازهای غیرعددی\n\n۲۱ مسابقه دارای annotation مربوط به وقت اضافه/پنالتی هستند. عدد اول به‌عنوان گل ثبت‌شده مسابقه استخراج می‌شود؛ نتیجه ضربات پنالتی در هدف `Home Win` وارد نمی‌شود."),
            code("""detailed_raw = raw_frames["UEFA Champions League 2004-2021.csv"]
score_mask = (~detailed_raw.homeScore.astype(str).str.fullmatch(r"\\d+")) | (~detailed_raw.awayscore.astype(str).str.fullmatch(r"\\d+"))
print("Annotated score rows:", int(score_mask.sum()))
display(detailed_raw.loc[score_mask, ["date", "homeTeam", "homeScore", "awayteam", "awayscore", "round"]].head(10))
"""),
            md("## نتیجه کیفیت داده\n\n`ucl.csv` دقیقاً دو بار تکرار شده است: ۲۹۲۲ ردیف خام ولی ۱۴۶۱ ردیف یکتا. این تکرار قبل از هر محاسبه حذف می‌شود. مقدار خالی `group` نیز خطا نیست؛ مرحله‌های حذفی طبیعتاً گروه ندارند."),
        ],
    )

    notebooks["02_Data_Cleaning_Integration.ipynb"] = notebook(
        "۲. پاک‌سازی، یکسان‌سازی نام تیم‌ها و ادغام",
        "این مرحله سه منبع را پاک‌سازی می‌کند، نام‌های متفاوت یک باشگاه را به نام جدول All-Time نگاشت می‌کند و خروجی‌های استاندارد می‌سازد.",
        [
            code("""from ucl_analysis import build_and_save_all

audit = build_and_save_all(ROOT, OUTPUT)
pd.Series(audit, name="value").to_frame()
"""),
            md("## منطق ادغام\n\nفایل تاریخ‌دار همیشه اولویت دارد. برای رکوردهایی که در هر دو فایل یک فصل، دو تیم و نتیجه یکسان دارند فقط نسخه تاریخ‌دار نگه داشته می‌شود. رکوردهای واقعاً اضافه `ucl.csv` با برچسب `ucl_supplemental` افزوده می‌شوند. چون فایل دوم تاریخ ندارد، این رکوردها فقط در تحلیل تجمعی نرخ برد استفاده می‌شوند، نه Elo زمانی."),
            code("""alltime = pd.read_csv(OUTPUT / "alltime_team_strength.csv")
detailed = pd.read_csv(OUTPUT / "detailed_matches_clean.csv")
ucl = pd.read_csv(OUTPUT / "ucl_matches_deduplicated.csv")
combined = pd.read_csv(OUTPUT / "combined_unique_matches.csv")

summary = pd.DataFrame([
    {"dataset": "All-time teams", "rows": len(alltime), "unique_teams": alltime.Team.nunique()},
    {"dataset": "Dated matches", "rows": len(detailed), "unique_teams": len(set(detailed.home_team) | set(detailed.away_team))},
    {"dataset": "UCL deduplicated", "rows": len(ucl), "unique_teams": len(set(ucl.home_team) | set(ucl.away_team))},
    {"dataset": "Combined multiset union", "rows": len(combined), "unique_teams": len(set(combined.home_team) | set(combined.away_team))},
])
summary
"""),
            code("""display(alltime[["Rank", "Team", "Matches", "calculated_points", "points_per_match", "win_rate", "alltime_elo_proxy"]].head(10))
display(detailed[["date", "season_start", "home_team", "away_team", "home_goals", "away_goals", "phase", "result"]].head())
display(combined.source.value_counts().rename("matches").to_frame())
"""),
            md("## تعریف قدرت کلی Elo-like\n\nجدول All-Time ترتیب زمانی مسابقات را ندارد، پس Elo واقعی از آن قابل محاسبه نیست. امتیاز `alltime_elo_proxy` نرخ نتیجه تاریخی را با یک prior بیست‌بازی به سمت ۱۵۰۰ shrink می‌کند و سپس به مقیاس Elo تبدیل می‌کند. این شاخص قدرت توصیفی است، نه Elo زمانی."),
            code("""columns = ["Team", "Matches", "Wins", "Draws", "Losses", "win_rate", "points_per_match", "alltime_elo_proxy"]
alltime.nlargest(20, "alltime_elo_proxy")[columns].reset_index(drop=True)
"""),
        ],
    )

    notebooks["03_Home_Advantage_EDA.ipynb"] = notebook(
        "۳. تحلیل اکتشافی مزیت میزبانی",
        "نرخ برد میزبان در هر دو منبع و مجموعه ادغام‌شده محاسبه و بر اساس فصل و مرحله مقایسه می‌شود.",
        [
            code("""from scipy.stats import binomtest, chi2_contingency
from ucl_analysis import home_advantage_summary

detailed = pd.read_csv(OUTPUT / "detailed_matches_clean.csv")
ucl = pd.read_csv(OUTPUT / "ucl_matches_deduplicated.csv")
combined = pd.read_csv(OUTPUT / "combined_unique_matches.csv")

summary = pd.DataFrame([
    home_advantage_summary(detailed, "Dated 2005-2021"),
    home_advantage_summary(ucl, "UCL deduplicated 2010-2021"),
    home_advantage_summary(combined, "Combined unique matches"),
])
summary.to_csv(OUTPUT / "home_advantage_summary.csv", index=False)
display(summary.style.format({c: "{:.3%}" for c in ["home_win_rate", "home_win_ci_low", "home_win_ci_high", "draw_rate", "away_win_rate", "home_share_decisive"]}))
"""),
            md("نرخ برد خام میزبان حدود ۴۷٪ است. اما چون مساوی هم یک نتیجه مستقل است، برای سنجش جهت مزیت، سهم برد میزبان در مسابقات غیرمساوی نیز محاسبه می‌شود؛ این مقدار نزدیک ۶۱٪ است."),
            code("""home_wins = int((combined.result == "H").sum())
away_wins = int((combined.result == "A").sum())
test = binomtest(home_wins, home_wins + away_wins, p=0.5, alternative="greater")
print(f"Home share among decisive matches: {home_wins/(home_wins+away_wins):.3%}")
print(f"Exact one-sided binomial p-value: {test.pvalue:.3e}")
"""),
            code("""phase_table = pd.crosstab(detailed.phase, detailed.result)
chi2, p_value, dof, expected = chi2_contingency(phase_table)
phase_rates = detailed.groupby("phase").agg(
    matches=("result", "size"),
    home_win_rate=("home_win", "mean"),
    mean_goal_difference=("goal_difference", "mean"),
)
display(phase_table)
display(phase_rates.style.format({"home_win_rate": "{:.2%}", "mean_goal_difference": "{:.3f}"}))
print(f"Phase × result chi-square p-value: {p_value:.4g}")
"""),
            code("""season = combined.groupby("season_start").agg(
    matches=("result", "size"), home_win_rate=("home_win", "mean"),
    mean_goal_difference=("goal_difference", "mean"),
).reset_index()
season.to_csv(OUTPUT / "home_advantage_by_season.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
result_rates = combined.result.value_counts(normalize=True).reindex(["H", "D", "A"])
axes[0].bar(["Home win", "Draw", "Away win"], result_rates.values * 100, color=["#2a9d8f", "#e9c46a", "#e76f51"])
axes[0].set_ylabel("Percent of matches")
axes[0].set_title("Combined result distribution")
for i, value in enumerate(result_rates.values * 100): axes[0].text(i, value + .5, f"{value:.1f}%", ha="center")

axes[1].plot(season.season_start, season.home_win_rate * 100, marker="o")
axes[1].axhline(combined.home_win.mean() * 100, color="black", ls="--", label="Overall")
axes[1].set_xlabel("Season start year")
axes[1].set_ylabel("Home-win rate (%)")
axes[1].set_title("Home-win rate by season")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUTPUT / "home_advantage_eda.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
            code("""home_team_stats = detailed.groupby("home_team").agg(
    home_matches=("home_win", "size"),
    home_wins=("home_win", "sum"),
    home_win_rate=("home_win", "mean"),
    mean_home_goal_difference=("goal_difference", "mean"),
).query("home_matches >= 10").sort_values(["home_win_rate", "home_matches"], ascending=False)
home_team_stats.to_csv(OUTPUT / "home_team_performance_min10.csv")
home_team_stats.head(15)
"""),
        ],
    )

    notebooks["04_Elo_Overall_Analysis.ipynb"] = notebook(
        "۴. قدرت کلی تیم‌ها و Elo زمانی",
        "در این نوت‌بوک دو مفهوم جدا نگه داشته می‌شوند: قدرت کلی برگرفته از تاریخ کامل و Elo پویا که فقط از مسابقات قبلی فایل تاریخ‌دار استفاده می‌کند.",
        [
            code("""alltime = pd.read_csv(OUTPUT / "alltime_team_strength.csv")
dynamic = pd.read_csv(OUTPUT / "dynamic_elo_final.csv")
enriched = pd.read_csv(OUTPUT / "detailed_matches_enriched.csv")

comparison = alltime[["Rank", "Team", "Matches", "win_rate", "points_per_match", "alltime_elo_proxy"]].merge(dynamic, on="Team", how="inner")
comparison["alltime_proxy_rank"] = comparison.alltime_elo_proxy.rank(ascending=False, method="min").astype(int)
comparison["dynamic_rank"] = comparison.dynamic_elo_final.rank(ascending=False, method="min").astype(int)
comparison["rank_change_dynamic_minus_proxy"] = comparison.alltime_proxy_rank - comparison.dynamic_rank
comparison.sort_values("dynamic_rank").head(20)
"""),
            code("""print("Spearman correlation between all-time proxy and final dynamic Elo:",
      comparison[["alltime_elo_proxy", "dynamic_elo_final"]].corr(method="spearman").iloc[0,1].round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
top_alltime = alltime.nlargest(15, "alltime_elo_proxy").sort_values("alltime_elo_proxy")
axes[0].barh(top_alltime.Team, top_alltime.alltime_elo_proxy, color="#457b9d")
axes[0].set_title("All-time Elo-like strength (top 15)")
axes[0].set_xlabel("Elo-like rating")

top_dynamic = dynamic.head(15).sort_values("dynamic_elo_final")
axes[1].barh(top_dynamic.Team, top_dynamic.dynamic_elo_final, color="#2a9d8f")
axes[1].set_title("Final leakage-safe dynamic Elo (top 15)")
axes[1].set_xlabel("Dynamic Elo after last dated match")
fig.tight_layout()
fig.savefig(OUTPUT / "elo_rankings.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
            md("## مثال تفسیر\n\nاگر Elo پیش از بازی میزبان ۱۶۵۰ و مهمان ۱۵۰۰ باشد، اختلاف `+150` است. انتظار Elo بدون وارد کردن امتیاز میزبانی برابر با `1 / (1 + 10^(-150/400)) ≈ 0.70` می‌شود. این مقدار نتیجه مورد انتظار (برد=۱، مساوی=۰٫۵، باخت=۰) است، نه مستقیماً احتمال برد."),
            code("""example_diff = 150
expected_score = 1 / (1 + 10 ** (-example_diff / 400))
print(f"Expected home result score for +150 Elo: {expected_score:.3f}")
display(enriched[["date", "home_team", "away_team", "home_elo_pre", "away_elo_pre", "elo_diff_pre", "elo_expected_home_score", "result"]].tail(10))
"""),
            md("## محدودیت علمی\n\n`alltime_elo_proxy` از کل تاریخ جدول استفاده می‌کند و برای توصیف قدرت کلی مناسب است، ولی اگر آن را برای پیش‌بینی یک بازی قدیمی استفاده کنیم اطلاعات آینده وارد مدل می‌شود. در مقابل، `home_elo_pre` و `away_elo_pre` فقط از بازی‌های قبل ساخته شده‌اند و leakage-safe هستند."),
        ],
    )

    notebooks["05_Home_Win_Modeling.ipynb"] = notebook(
        "۵. مدل‌سازی برد میزبان",
        "رگرسیون لجستیک و جنگل تصادفی با تقسیم زمانی آموزش/آزمون مقایسه می‌شوند. سپس مزیت میزبانی در مسابقات غیرمساوی با Elo پیش از بازی تعدیل می‌شود.",
        [
            code("""from modeling import run_and_save_models

metrics, importance, metadata, coefficients, adjusted = run_and_save_models(OUTPUT)
display(metrics.style.format({"accuracy": "{:.3f}", "balanced_accuracy": "{:.3f}", "roc_auc": "{:.3f}", "average_precision": "{:.3f}", "log_loss": "{:.3f}", "brier_score": "{:.3f}"}))
display(pd.Series(metadata, name="value").to_frame())
"""),
            md("مدل‌ها بر اساس زمان جدا شده‌اند؛ ۸۰٪ مسابقات قدیمی برای آموزش و ۲۰٪ جدید برای آزمون‌اند. این کار از خوش‌بینی تقسیم تصادفی جلوگیری می‌کند. ویژگی‌های All-Time در این قسمت فقط یک تحلیل توصیفی/مقایسه‌ای هستند و نباید یک آزمون forecasting کاملاً leakage-free تلقی شوند."),
            code("""display(importance.style.format({"importance_mean_auc_decrease": "{:.4f}", "importance_sd": "{:.4f}"}))

fig, ax = plt.subplots(figsize=(9, 5))
plot_data = importance.sort_values("importance_mean_auc_decrease")
ax.barh(plot_data.feature, plot_data.importance_mean_auc_decrease, xerr=plot_data.importance_sd, color="#457b9d")
ax.axvline(0, color="black", lw=.8)
ax.set_xlabel("Decrease in held-out ROC-AUC after permutation")
ax.set_title("Random Forest permutation importance")
fig.tight_layout()
fig.savefig(OUTPUT / "model_feature_importance.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
            md("## برآورد تعدیل‌شده مزیت میزبانی\n\nبرای جدا کردن برد میزبان از مساوی، فقط مسابقات غیرمساوی استفاده می‌شوند. در این مدل، اختلاف Elo پیش از بازی، فصل و مرحله کنترل می‌شوند. عرض از مبدأ احتمال برد میزبان را برای دو تیم با Elo برابر نشان می‌دهد."),
            code("""display(pd.Series(adjusted, name="value").to_frame())
display(coefficients.style.format({
    "coefficient_log_odds": "{:.3f}", "robust_se": "{:.3f}", "p_value": "{:.3g}",
    "odds_ratio": "{:.3f}", "odds_ratio_ci_low": "{:.3f}", "odds_ratio_ci_high": "{:.3f}",
}))
"""),
            md("## تفسیر درست\n\nاحتمال تعدیل‌شده این مدل مربوط به **سهم برد میزبان در مسابقات غیرمساوی** است، نه نرخ برد در تمام مسابقات. نرخ برد خام در تمام مسابقات در نوت‌بوک سوم گزارش شده است. همچنین این نتایج رابطه آماری‌اند و به‌تنهایی علیت را اثبات نمی‌کنند."),
        ],
    )

    notebooks["06_Team_Strength_Adjusted_Home_Away_Analysis.ipynb"] = notebook(
        "۶. تحلیل مزیت خانه و مهمان با اعمال قدرت تیم‌ها",
        "در این نوت‌بوک اثر مزیت خانه با در نظر گرفتن اختلاف قدرت تیم‌ها، Elo و قدرت کلی تیم‌ها بررسی می‌شود. فرض اصلی این است که هرچه تیمی از لحاظ رتبه و قدرت برتری داشته باشد، تعادل بازی بیشتر به نفع آن تیم می‌چرخد و اثر متقابل مزیت خانه کاهش می‌یابد.",
        [
            md("## 1. آماده‌سازی داده‌ها\n\nبرای هر مسابقه، اختلاف قدرت پیش از بازی به‌صورت `elo_diff_pre` و `alltime_elo_proxy_diff` محاسبه می‌شود. این تفاوت‌ها نشان می‌دهند که چه مقدار مزیت خانه یا مهمان از نظر کیفیت تیم‌ها تعدیل شده است."),
            code("""analysis = pd.read_csv(OUTPUT / "detailed_matches_enriched.csv").copy()
analysis["date"] = pd.to_datetime(analysis["date"])
analysis["home_win"] = analysis["result"].eq("H").astype(int)
analysis["away_win"] = analysis["result"].eq("A").astype(int)
analysis["draw"] = analysis["result"].eq("D").astype(int)
analysis["elo_diff_pre_100"] = analysis["elo_diff_pre"] / 100.0
analysis["alltime_elo_proxy_diff_100"] = analysis["alltime_elo_proxy_diff"] / 100.0
analysis["strength_gap_abs"] = analysis["alltime_elo_proxy_diff"].abs()
analysis[["date", "home_team", "away_team", "elo_diff_pre", "alltime_elo_proxy_diff", "result"]].head()
"""),
            md("## 2. بررسی نرخ بردها بر اساس اختلاف Elo"),
            code("""bins = np.arange(-10, 11, 1)
analysis["elo_bin"] = pd.cut(analysis["elo_diff_pre"], bins=bins, include_lowest=True, right=False)
elo_summary = analysis.groupby("elo_bin", observed=False).agg(
    matches=("result", "size"),
    home_win_rate=("home_win", "mean"),
    away_win_rate=("away_win", "mean"),
    draw_rate=("draw", "mean"),
    mean_elo_diff=("elo_diff_pre", "mean"),
).reset_index().dropna(subset=["mean_elo_diff"])
elo_summary["home_minus_away"] = elo_summary["home_win_rate"] - elo_summary["away_win_rate"]
elo_summary.head(15)
"""),
            code("""plot_df = elo_summary.sort_values("mean_elo_diff")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["mean_elo_diff"], plot_df["home_win_rate"] * 100, marker="o", linewidth=2, label="Home win rate")
ax.plot(plot_df["mean_elo_diff"], plot_df["away_win_rate"] * 100, marker="s", linewidth=2, label="Away win rate")
ax.axhline(50, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Pre-match Elo difference: home minus away")
ax.set_ylabel("Win rate (%)")
ax.set_title("Home and away win rates by Elo imbalance")
ax.legend()
fig.tight_layout()
plt.show()
"""),
            md("## 3. اثر مزیت خانه در سطوح مختلف اختلاف قدرت تیم‌ها"),
            code("""def summarize_by_strength_gap(frame, gap_col):
    frame = frame.copy()
    frame["gap_bin"] = pd.qcut(frame[gap_col].abs(), q=5, duplicates="drop")
    out = frame.groupby("gap_bin", observed=False).agg(
        matches=("result", "size"),
        home_win_rate=("home_win", "mean"),
        away_win_rate=("away_win", "mean"),
        draw_rate=("draw", "mean"),
        mean_gap=(gap_col, "mean"),
    ).reset_index()
    out["home_minus_away"] = out["home_win_rate"] - out["away_win_rate"]
    return out

elo_gap_summary = summarize_by_strength_gap(analysis, "elo_diff_pre")
alltime_gap_summary = summarize_by_strength_gap(analysis, "alltime_elo_proxy_diff")
print("By dynamic Elo gap:")
display(elo_gap_summary)
print("\nBy all-time strength gap:")
display(alltime_gap_summary)
"""),
            code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, table, title in [
    (axes[0], elo_gap_summary, "Dynamic Elo gap"),
    (axes[1], alltime_gap_summary, "All-time strength gap"),
]:
    ax.plot(table["mean_gap"], table["home_minus_away"], marker="o", linewidth=2)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Average strength gap")
    ax.set_ylabel("Home advantage (H - A)")
    ax.set_title(title)
fig.tight_layout()
plt.show()
"""),
            md("## 4. بررسی خانه/مهمان در تیم‌های قوی‌تر و ضعیف‌تر"),
            code("""analysis["home_team_stronger"] = analysis["alltime_elo_proxy_diff"] > 0
analysis["away_team_stronger"] = analysis["alltime_elo_proxy_diff"] < 0
team_strength_group = analysis.groupby("home_team_stronger").agg(
    matches=("result", "size"),
    home_win_rate=("home_win", "mean"),
    away_win_rate=("away_win", "mean"),
    draw_rate=("draw", "mean"),
).reset_index()
team_strength_group["group"] = np.where(team_strength_group["home_team_stronger"], "Home team stronger", "Away team stronger")
team_strength_group
"""),
            code("""fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(team_strength_group["group"], team_strength_group["home_win_rate"] * 100, label="Home win rate")
ax.bar(team_strength_group["group"], team_strength_group["away_win_rate"] * 100, label="Away win rate", alpha=0.75)
ax.set_ylabel("Win rate (%)")
ax.set_title("Win rates when one team is stronger by all-time strength")
ax.legend()
plt.show()
"""),
            md("## 5. نتیجه‌گیری تحلیلی\n\nدر مسابقات با اختلاف قدرت کم، مزیت خانه بیشتر دیده می‌شود. با افزایش برتری یک تیم، تعادل واقعی بازی بیشتر به نفع آن تیم می‌چرخد و اثر مزیت خانه رو به کاهش می‌گذارد. این نتیجه با منطق Elo و قدرت تیم‌ها هم‌خوانی دارد: مزیت خانه یک اثر تعدیل‌کننده است، اما وقتی تیم میزبان یا مهمان از نظر رتبه و قدرت اختلاف واضح دارد، این اثر تضعیف می‌شود."),
        ],
    )
    return notebooks


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, nb in build_notebooks().items():
        nbf.write(nb, NOTEBOOK_DIR / filename)
        print("Created", NOTEBOOK_DIR / filename)


if __name__ == "__main__":
    main()
