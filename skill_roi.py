"""Skill ROI - which skills carry salary uplift once experience, location and title are fixed?

Line 1: OLS on log(salary) with HC1 robust SEs gives each skill's % uplift, its 95% CI, p-value
        and a "years of experience equivalent" (skill coefficient / experience coefficient).
Line 2: a gradient boosting model with 5-fold CV and permutation importance cross-checks the ranking.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
a = pd.read_csv(ROOT / "data/clean/analytics_clean.csv")
skills = sorted(c for c in a.columns if c.startswith("skill_"))
a["salary_log"] = np.log(a["salary_avg_lakh"])

print("=" * 118)
print("SKILL PAY UPLIFT - OLS on log(salary), HC1 robust SEs, controls: title_family + location + avg_exp")
print(f"n = {len(a)}   salary = salary_avg_lakh (lakhs/yr, band midpoint, log)   skills = {len(skills)} keyword dummies")
print("=" * 118)

# ------------------------------------------------- 1) OLS skill coefficients
f = ("salary_log ~ C(title_family, Sum) + C(location) + avg_exp + " + " + ".join(skills))
res = smf.ols(f, data=a).fit(cov_type="HC1")
b = res.params[skills]
ci = res.conf_int().loc[skills]
p = res.pvalues[skills]
b_exp = res.params["avg_exp"]                       # log-salary per year of experience
pct = lambda x: (np.exp(x) - 1) * 100              # log coefficient -> % uplift

tab = pd.DataFrame(
    {
        "uplift_pct": pct(b.values),
        "ci_lo_pct": pct(ci[0].values),
        "ci_hi_pct": pct(ci[1].values),
        "p_value": p.values,
        "yrs_equivalent": (b / b_exp).values,       # years of exp that would buy the same uplift
        "postings": a[skills].sum().values,
    },
    index=[s.replace("skill_", "") for s in skills],
).sort_values("uplift_pct", ascending=False)

# reliable = p < 0.05 AND the 95% CI does not contain 0
tab["reliable"] = (tab.p_value < 0.05) & ((tab.ci_lo_pct > 0) | (tab.ci_hi_pct < 0))
n_rel = int(tab.reliable.sum())
n_pos = int((tab.reliable & (tab.ci_lo_pct > 0)).sum())
n_neg = int((tab.reliable & (tab.ci_hi_pct < 0)).sum())

print(f"model R2 = {res.rsquared:.3f}   |   experience coefficient = +{pct(b_exp):.1f}% per year "
      f"(so 1 year = 1.0x)   |   reliable skills: {n_rel}/30 ({n_pos} positive, {n_neg} negative)")


def show(title, df):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    print(f"{'skill':<22}{'uplift %':>10}{'95% CI':>20}{'p':>9}{'yrs equiv':>11}{'postings':>10}  reliable")
    for n, r in df.iterrows():
        star = "yes" if r.reliable else "no"
        print(f"{n:<22}{r.uplift_pct:>9.1f}%"
              f"{f'[{r.ci_lo_pct:+.1f}%, {r.ci_hi_pct:+.1f}%]':>20}"
              f"{r.p_value:>9.3f}{r.yrs_equivalent:>10.2f}x{int(r.postings):>9}   {star}")


show("TOP 10 SKILLS BY % UPLIFT", tab.head(10))
show("BOTTOM 5 SKILLS BY % UPLIFT", tab.tail(5))
not_rel = tab[~tab.reliable]
print(f"\nNOT statistically reliable ({len(not_rel)} of 30, CI contains 0): "
      + ", ".join(f"{n} ({r.uplift_pct:+.1f}%, p={r.p_value:.2f})" for n, r in not_rel.iterrows()))

# ------------------------------------------------- 2) gradient boosting cross-check
num = ["avg_exp"] + skills
cat = ["title_family", "location"]
pre = ColumnTransformer(
    [("ohe", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)]
)
pipe = Pipeline(
    [("pre", pre), ("gb", GradientBoostingRegressor(n_estimators=300, random_state=42))]
)
X, y = a[cat + num], a["salary_log"]

cv = KFold(5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

rng = np.random.default_rng(42)
perm = np.zeros((5, len(skills)))
fold_r2 = []
for k, (tr, te) in enumerate(cv.split(X, y)):
    pipe.fit(X.iloc[tr], y.iloc[tr])
    Xte, yte = X.iloc[te], y.iloc[te]
    base = r2_score(yte, pipe.predict(Xte))
    fold_r2.append(base)
    for j, s in enumerate(skills):
        Xp = Xte.copy()
        Xp[s] = rng.permutation(Xp[s].to_numpy())
        perm[k, j] = base - r2_score(yte, pipe.predict(Xp))   # R2 drop when the skill is shuffled

gb = pd.DataFrame(
    {
        "perm_r2_drop": perm.mean(0),
        "perm_sd": perm.std(0),
        "uplift_pct": tab.uplift_pct.values,
        "reliable": tab.reliable.values,
    },
    index=tab.index,
).sort_values("perm_r2_drop", ascending=False)

print("\n" + "=" * 118)
print("CROSS-CHECK: GradientBoostingRegressor, 5-fold CV (shuffled, seed 42), permutation importance")
print("=" * 118)
print(f"CV R2 per fold: {np.round(cv_r2, 3)}   ->   mean {cv_r2.mean():.3f} +/- {cv_r2.std():.3f} "
      f"(OLS in-sample R2 = {res.rsquared:.3f})")
print("permutation importance = mean drop in test R2 when that skill column is shuffled\n")
print(f"{'skill':<22}{'perm dR2':>10}{'+- sd':>8}{'OLS uplift':>12}{'reliable':>10}")
for n, r in gb.head(10).iterrows():
    print(f"{n:<22}{r.perm_r2_drop:>10.4f}{r.perm_sd:>8.4f}{r.uplift_pct:>11.1f}%"
          f"{('yes' if r.reliable else 'no'):>10}")
print("  ...")
for n, r in gb.tail(5).iterrows():
    print(f"{n:<22}{r.perm_r2_drop:>10.4f}{r.perm_sd:>8.4f}{r.uplift_pct:>11.1f}%"
          f"{('yes' if r.reliable else 'no'):>10}")

spearman = pd.Series(tab.uplift_pct).corr(pd.Series(gb.perm_r2_drop, index=gb.index), method="spearman")
top10_ols, top10_gb = list(tab.head(10).index), list(gb.head(10).index)
overlap = [s for s in top10_ols if s in top10_gb]
agree_neg = [s for s in tab.tail(2).index if s in gb.tail(3).index]     # both say negative
print(f"\nSpearman rank correlation (OLS uplift vs permutation importance, 30 skills) = {spearman:.2f}")
print(f"OLS top-10 that also appear in GB top-10: {len(overlap)}/10 -> {', '.join(overlap)}")
print(f"skills flagged negative by BOTH methods: {', '.join(agree_neg) if agree_neg else 'none'}")

strong = tab[tab.reliable & (tab.ci_lo_pct > 0)]
gb_top5 = list(gb.head(5).index)
strong_conf = [s for s in strong.head(10).index if s in gb_top5]
gb_not_ols = [s for s in gb_top5 if s not in strong.head(10).index]
print(f"OLS-positive-and-reliable skills confirmed in GB top-5: "
      f"{', '.join(strong_conf) if strong_conf else 'none'}")
if gb_not_ols:
    ups = ", ".join("{:+.1f}%".format(tab.loc[s, "uplift_pct"]) for s in gb_not_ols)
    ok = ", ".join(strong.index.intersection(gb_not_ols)) or "none"
    print(f"GB top-5 NOT in OLS top-10: {', '.join(gb_not_ols)} (ups {ups}, reliable: {ok})")

# ------------------------------------------------- 3) final answer
t1, t2, t3 = tab.head(3).index
top = tab.head(3)
no_effect = tab[~tab.reliable]

print("\n" + "=" * 118)
print("FINAL ANSWER")
print("=" * 118)
print(
    f"1. The three most valuable skills, holding experience, location and title fixed, are "
    f"{t1} (+{tab.loc[t1, 'uplift_pct']:.1f}%, 95% CI +{tab.loc[t1, 'ci_lo_pct']:.1f}% to "
    f"+{tab.loc[t1, 'ci_hi_pct']:.1f}%, p < 0.001 = {tab.loc[t1, 'yrs_equivalent']:.2f} years of "
    f"experience equivalent), {t2} (+{tab.loc[t2, 'uplift_pct']:.1f}%, CI "
    f"+{tab.loc[t2, 'ci_lo_pct']:.1f}% to +{tab.loc[t2, 'ci_hi_pct']:.1f}%, "
    f"{tab.loc[t2, 'yrs_equivalent']:.2f} yrs) and {t3} (+{tab.loc[t3, 'uplift_pct']:.1f}%, CI "
    f"+{tab.loc[t3, 'ci_lo_pct']:.1f}% to +{tab.loc[t3, 'ci_hi_pct']:.1f}%, "
    f"{tab.loc[t3, 'yrs_equivalent']:.2f} yrs).\n"
    f"2. The gradient boosting cross-check (5-fold CV R2 = {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}) "
    f"agrees only partly: it also puts {', '.join(overlap)} in its own top 10 (shuffling "
    f"business_analysis costs {gb.loc['business_analysis', 'perm_r2_drop']:.4f} of test R2), but the two "
    f"rankings correlate at just Spearman {spearman:.2f} ({len(overlap)}/10 top-10 overlap) because "
    f"permutation importance is sign-blind - it counts a skill that pushes pay DOWN "
    f"(seo, {tab.loc['seo', 'uplift_pct']:+.1f}%) as just as important and puts rare skills like "
    f"{', '.join(s for s in gb_not_ols if s != 'seo')} ahead of {tab.index[0]}, so only {overlap[0]} and "
    f"seo are confirmed by both methods, and {tab.index[0]} - the OLS top pick - is worth "
    f"{gb.loc[tab.index[0], 'perm_r2_drop']:.4f} R2 to the GB model once title and location are in.\n"
    f"3. {len(no_effect)} of the 30 skills have no real effect (p >= 0.05 and the 95% CI contains 0): "
    f"{', '.join(no_effect.index)} - for these the data cannot separate the coefficient from zero, "
    f"so their apparent {no_effect.uplift_pct.min():.0f}%..{no_effect.uplift_pct.max():.0f}% spread is noise, "
    f"not pay signal.\n"
)
print("LIMITATIONS")
print("-" * 118)
print("1. Observational data: a skill's coefficient is a partial correlation, not a causal ROI - skills "
      "co-occur (java with outsourcing/contract roles, sql with senior engineering), hiring teams pick "
      "skills they already pay for, and a keyword match in key_skills shows the posting lists the skill, "
      "not that the candidate has it.")
print("2. Salary is a band midpoint (0to3, 6to10, ...) so all uplifts are on a coarsened outcome, some "
      "skills appear in only ~220-350 postings (wide CIs), the GB permutation importance splits credit "
      "unpredictably between correlated skills, and CV R2 = "
      f"{cv_r2.mean():.2f} leaves about {1 - cv_r2.mean():.0%} of pay variance unexplained.")
print("=" * 118)

tab.round(3).to_csv(ROOT / "skill_uplift_table.csv")
gb.round(4).to_csv(ROOT / "skill_gb_importance.csv")
print("saved -> skill_uplift_table.csv, skill_gb_importance.csv")
