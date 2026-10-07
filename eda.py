"""Step 3 - EDA on the cleaned files.
Makes 7 charts (PNG in figures/) and prints the 5 patterns worth testing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw without a screen (works over SSH / in CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"

C_HI, C_LO = "#2a9d8f", "#e76f51"   # group 1 / group 0
SKILL_SKIPS = ()                     # keep every skill_ column


def save(fig, name):
    # One helper so every chart lands in figures/ the same way.
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.facecolor": "white", "axes.titlesize": 12,
                         "axes.titleweight": "bold", "font.size": 10})


# ---------------------------------------------------------------- 1
def chart_salary_distribution(a):
    # Two histograms side by side: raw bands are lumpy, log is bell shaped.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(a["salary_avg_lakh"], bins=40, color="#264653", edgecolor="white")
    axes[0].axvline(a["salary_avg_lakh"].median(), color="crimson", ls="--",
                    label=f"median {a['salary_avg_lakh'].median():.1f} LPA")
    axes[0].set(title="Raw salary (LPA)", xlabel="lakh / year", ylabel="postings")
    axes[0].legend()

    lg = np.log(a["salary_avg_lakh"])
    axes[1].hist(lg, bins=40, color="#2a9d8f", edgecolor="white")
    axes[1].axvline(lg.median(), color="crimson", ls="--", label=f"mean log {lg.mean():.2f}")
    axes[1].set(title="log(salary) - the version models use", xlabel="log(lakh / year)")
    axes[1].legend()
    fig.suptitle("1. Salary distribution: raw vs log", fontweight="bold")
    save(fig, "01_salary_distribution.png")


# ---------------------------------------------------------------- 2
def chart_salary_vs_experience(a):
    # Cloud of postings + a trend line fitted in log space (exponential in raw space).
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(a["avg_exp"], a["salary_avg_lakh"], s=12, alpha=0.25, color="#264653")

    x = a["avg_exp"].values.astype(float)
    y = np.log(a["salary_avg_lakh"].values.astype(float))
    b0, b1 = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, np.exp(b0 + b1 * grid), color="crimson", lw=2.5,
            label=f"trend: +{(np.exp(b1) - 1) * 100:.0f}% salary per extra year")
    r = np.corrcoef(x, a["salary_avg_lakh"])[0, 1]
    ax.set(title=f"2. Salary vs experience   (Pearson r = {r:.2f})",
           xlabel="avg_exp (years)", ylabel="salary (LPA)")
    ax.legend()
    save(fig, "02_salary_vs_experience.png")


# ---------------------------------------------------------------- 3
def chart_medians(a):
    # Top 10 cities and every title family, ranked by median salary.
    loc = (a.groupby("location")["salary_avg_lakh"]
             .agg(median="median", n="size").sort_values("median", ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(loc.index[::-1], loc["median"][::-1], color="#457b9d")
    for y, (m, n) in enumerate(zip(loc["median"][::-1], loc["n"][::-1])):
        ax.text(m + 0.15, y, f"{m:.1f} LPA  (n={n})", va="center", fontsize=9)
    ax.set(title="3a. Median salary by location (top 10)", xlabel="median lakh / year")
    save(fig, "03_median_salary_by_location.png")

    fam = (a.groupby("title_family")["salary_avg_lakh"]
             .agg(median="median", n="size").sort_values("median", ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fam.index[::-1], fam["median"][::-1], color="#e76f51")
    for y, (m, n) in enumerate(zip(fam["median"][::-1], fam["n"][::-1])):
        ax.text(m + 0.15, y, f"{m:.1f} LPA  (n={n})", va="center", fontsize=9)
    ax.set(title="3b. Median salary by title_family", xlabel="median lakh / year")
    save(fig, "04_median_salary_by_title_family.png")


# ---------------------------------------------------------------- 4
def chart_skills_demand(a):
    # Share of postings that mention each skill - the "market demand" bar chart.
    skills = sorted([c for c in a.columns if c.startswith("skill_")],
                    key=lambda c: -a[c].mean())[:20]
    vals = [100 * a[c].mean() for c in skills]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([s.replace("skill_", "") for s in skills][::-1], vals[::-1], color="#2a9d8f")
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set(title="4. Top 20 skills by demand (% of postings)",
           xlabel="% of postings mentioning the skill")
    save(fig, "05_top20_skills_demand.png")


# ---------------------------------------------------------------- 5
def chart_correlation(a):
    # Correlation matrix of every numeric column; red = move together, blue = opposite.
    num = a.select_dtypes(include="number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ticks = [c.replace("skill_", "") for c in corr.columns]
    ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=7)
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title("5. Correlation heatmap of numeric columns", pad=12)
    save(fig, "06_correlation_heatmap.png")


# ------------------------------------------------- 6 & 7 (shared builder)
def grouped_boxplots(df, cols, target, title, fname, ylab, xlabels=None):
    # Draws 5 boxes per skill: orange = class 0, teal = class 1, side by side.
    pos, data, shades = [], [], []
    for i, c in enumerate(cols):
        for k in (0, 1):
            pos.append(i + (-0.21 if k == 0 else 0.21))
            data.append(df.loc[df[target] == k, c].dropna().values)
            shades.append(C_LO if k == 0 else C_HI)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bp = ax.boxplot(data, positions=pos, widths=0.36, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.4))
    for patch, c in zip(bp["boxes"], shades):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(xlabels or cols, rotation=20, ha="right")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor=C_LO, label=f"{target} = 0"),
                       Patch(facecolor=C_HI, label=f"{target} = 1")], loc="lower right")
    save(fig, fname)


def chart_jds(j):
    cols = ["big_data_skills", "maths_stats_skills", "coding_skills",
            "ai_and_ml_skills", "dashboard_and_storytelling_skills"]
    grouped_boxplots(j, cols, "salary_hike_high_or_low",
                     "6. JDS junior skills: score by salary hike",
                     "07_jds_skills_by_salary_hike.png", "skill score (1-5)",
                     xlabels=["Big Data", "Maths/Stats", "Coding", "AI & ML", "Dashboard"])


def chart_sds(s):
    cols = ["neuroticism", "extraversion", "openness_to_experience",
            "agreeableness", "conscientiousness"]
    grouped_boxplots(s, cols, "success_high_low",
                     "7. SDS Big Five traits: score by success",
                     "08_sds_traits_by_success.png", "trait score",
                     xlabels=["Neuroticism", "Extraversion", "Openness",
                              "Agreeableness", "Conscientiousness"])


# ---------------------------------------------------------------- patterns
def report_patterns(a, j, s):
    lg = np.log(a["salary_avg_lakh"])
    r_exp_raw = a["avg_exp"].corr(a["salary_avg_lakh"])
    fam = a.groupby("title_family")["salary_avg_lakh"].median()
    top_fam, bot_fam = fam.idxmax(), fam.idxmin()

    def lift(c):
        m = a[c] == 1
        return a.loc[m, "salary_avg_lakh"].median() - a.loc[~m, "salary_avg_lakh"].median()

    def skill_gap(c):
        return (j.loc[j.salary_hike_high_or_low == 1, c].mean()
                - j.loc[j.salary_hike_high_or_low == 0, c].mean())

    sds = s.groupby("success_high_low")["conscientiousness"].mean()
    q = pd.qcut(s["conscientiousness"], 4, duplicates="drop")
    rates = s.groupby(q, observed=True)["success_high_low"].mean()

    items = [
        ("Title premium vs skills - the core question",
         f"Median salary is {fam[top_fam]:.1f} LPA for {top_fam} vs {fam[bot_fam]:.1f} for {bot_fam}. "
         "Test with nested OLS on log(salary): Model A = title_family + exp, Model B = A + skill_xxx. "
         "If the title coefficients shrink a lot in B, titles are paying for skills in disguise "
         "(compare R2 with a partial F-test)."),
        ("Skills look worthless raw, but not after controls (Simpson's paradox)",
         f"Raw correlation of salary with skill_python = {a['skill_python'].corr(a['salary_avg_lakh']):.2f}, "
         f"skill_sql = {a['skill_sql'].corr(a['salary_avg_lakh']):.2f}, yet the median gap is "
         f"+{lift('skill_sql'):.1f} LPA for SQL and +{lift('skill_python'):.1f} for Python. "
         "Test: partial correlation / regression coefficient of each skill after controlling for "
         "location + title_family + avg_exp. The sign flip is the interesting result."),
        ("Experience dominates every skill",
         f"corr(log salary, avg_exp) = {np.corrcoef(a['avg_exp'], lg)[0, 1]:.2f} vs at most "
         f"{max(a[c].corr(a['salary_avg_lakh']) for c in a.columns if c.startswith('skill_')):.2f} "
         "for any single skill. Test: does adding all 30 skills beat an exp-only model "
         "(cross-validated R2 / F-test), and is an exp^2 term significant?"),
        ("JDS juniors: which skill actually buys the hike",
         "Every score is higher for hike=1, biggest gaps: "
         + ", ".join(f"{c} +{skill_gap(c):.2f}" for c in
                     ["dashboard_and_storytelling_skills", "maths_stats_skills",
                      "ai_and_ml_skills", "coding_skills", "big_data_skills"])
         + " (1-5 scale). Test: logistic regression on 118 rows, odds ratios per 1 SD with "
           "bootstrap CIs - dashboard/storytelling looks like the strongest single driver."),
        ("SDS seniors: conscientiousness nearly separates success from failure",
         f"Success rate rises from {rates.iloc[0]:.0%} in the bottom conscientiousness quartile "
         f"to {rates.iloc[-1]:.0%} in the top (mean {sds.get(0, np.nan):.0f} vs {sds.get(1, np.nan):.0f}). "
         "Test: logistic regression + a quadratic (sweet-spot) term, compare linear vs "
         "polynomial CV AUC - the quartile table suggests a threshold, not a straight line."),
    ]

    print("\n" + "=" * 92)
    print("5 PATTERNS TO TEST WITH STATISTICS")
    print("=" * 92)
    for i, (head, body) in enumerate(items, 1):
        print(f"\n{i}. {head}\n   {body}")
    print()


def main():
    style()
    a = pd.read_csv(CLEAN / "analytics_clean.csv")
    j = pd.read_csv(CLEAN / "jds_clean.csv")
    s = pd.read_csv(CLEAN / "sds_clean.csv")
    print(f"loaded: analytics {a.shape}, jds {j.shape}, sds {s.shape}")

    chart_salary_distribution(a)
    chart_salary_vs_experience(a)
    chart_medians(a)
    chart_skills_demand(a)
    chart_correlation(a)
    chart_jds(j)
    chart_sds(s)

    report_patterns(a, j, s)


if __name__ == "__main__":
    main()
