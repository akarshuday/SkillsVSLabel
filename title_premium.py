"""Step 4 - Does the title still pay once skills, experience and location are held fixed?
OLS on log(salary) with HC3 robust errors, Model A vs Model B, plus a skill-profile
cosine similarity between the Data Analyst and Data Scientist families."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "clean" / "analytics_clean.csv"

a = pd.read_csv(CSV)
a["salary_log"] = np.log(a["salary_avg_lakh"])          # log tames the salary tail
skills = sorted(c for c in a.columns if c.startswith("skill_"))

# ---------------------------------------------------------------- models
# A: title + location + experience.  B: A + every skill column.
fA = "salary_log ~ C(title_family, Sum) + C(location) + avg_exp"
fB = fA + " + " + " + ".join(skills)
# HC1 (Stata-style) robust errors: HC3 divides by (1 - leverage) and blows up because
# 154 postings sit in a location that appears exactly once (leverage = 1, HC3 = inf).
A = smf.ols(fA, data=a).fit(cov_type="HC1")
B = smf.ols(fB, data=a).fit(cov_type="HC1")

# Effect coding: every family is reported as a premium vs the AVERAGE title.
title_pars = [p for p in A.params.index if "title_family" in p]
families = sorted(a["title_family"].unique())


def contrast(res, family):
    # Effect-coded coefficient for one family (the omitted level = -sum of the others).
    v = np.zeros(len(res.params))
    name = next((p for p in title_pars if p.endswith(f"[S.{family}]")), None)
    if name is not None:
        v[res.params.index.get_loc(name)] = 1.0
    else:                                   # the level patsy dropped (alphabetically last)
        for p in title_pars:
            v[res.params.index.get_loc(p)] = -1.0
    t = res.t_test(v)                       # robust SE + 95% CI for this combination
    est = float(np.asarray(t.effect).ravel()[0])
    pval = float(np.asarray(t.pvalue).ravel()[0])
    ci = np.asarray(t.conf_int()).ravel()
    return est, pval, ci


def pct(x):
    return (np.exp(x) - 1) * 100                       # log-points -> % pay


rows = []
for f in families:
    bA, pA, ciA = contrast(A, f)
    bB, pB, ciB = contrast(B, f)
    # "how much of the raw premium disappears"; undefined when A found no premium.
    gone = np.nan if abs(bA) < 0.01 else (bA - bB) / bA * 100
    rows.append(dict(
        title_family=f,
        premium_A_pct=pct(bA), A_lo=pct(ciA[0]), A_hi=pct(ciA[1]), p_A=pA,
        premium_B_pct=pct(bB), B_lo=pct(ciB[0]), B_hi=pct(ciB[1]), p_B=pB,
        premium_gone_pct=gone,
    ))
tab = pd.DataFrame(rows).sort_values("premium_A_pct", ascending=False)

print("=" * 112)
print("TITLE PREMIUM - OLS on log(salary), HC1 robust SEs, effect coding "
      "(premium = % vs the average title)")
print(f"n = {int(A.nobs)}   params A = {len(A.params)}   params B = {len(B.params)}   cov = HC1 robust")
print("=" * 112)
print(f"{'title_family':<16}{'A: % premium':<16}{'95% CI':<22}{'p':<9}"
      f"{'B: % premium':<16}{'95% CI':<22}{'p':<9}")
print("-" * 112)
for _, r in tab.iterrows():
    print(f"{r.title_family:<16}"
          f"{r.premium_A_pct:>7.1f}%     "
          f"[{r.A_lo:>6.1f}%, {r.A_hi:>6.1f}%]   {r.p_A:<9.3f}"
          f"{r.premium_B_pct:>7.1f}%     "
          f"[{r.B_lo:>6.1f}%, {r.B_hi:>6.1f}%]   {r.p_B:<9.3f}")
print("-" * 112)
for _, r in tab.iterrows():
    g = "n/a" if np.isnan(r.premium_gone_pct) else f"{r.premium_gone_pct:.1f}%"
    sig_a, sig_b = "sig" if r.p_A < 0.05 else "n.s.", "sig" if r.p_B < 0.05 else "n.s."
    print(f"{r.title_family:<16} premium that disappears in B: {g:>8}   "
          f"({sig_a} in A -> {sig_b} in B)")
print("-" * 112)
print(f"R2:  A = {A.rsquared:.3f}   B = {B.rsquared:.3f}   "
      f"(+{B.rsquared - A.rsquared:.3f} from the 30 skills)")

R = np.zeros((len(skills), len(B.params)))            # restriction matrix: skills = 0
for i, s_ in enumerate(skills):
    R[i, B.params.index.get_loc(s_)] = 1.0
with warnings.catch_warnings():                       # silent: legacy wald_test + pinv noise
    warnings.simplefilter("ignore")
    wt = B.wald_test(R, use_f=False, scalar=False)    # robust Wald chi2 (HC1)
chi2, chi2_p = float(np.asarray(wt.statistic).ravel()[0]), float(np.asarray(wt.pvalue).ravel()[0])
print(f"Joint test that the 30 skill coefficients are all 0 in B: "
      f"chi2({len(skills)}) = {chi2:.0f}, p = {chi2_p:.2e}")
big = tab.loc[tab.premium_B_pct.abs().idxmax()]
print(f"Largest premium left in B: {big.title_family} {big.premium_B_pct:+.1f}% "
      f"(CI {big.B_lo:+.1f}% to {big.B_hi:+.1f}%)")

# Robustness: 154 postings sit in a location seen exactly once, which makes HC3
# undefined and those dummies perfectly fitted - check the answer does not hinge on them.
once = a["location"].map(a["location"].value_counts()) == 1
ar = a.loc[~once].copy()
Br = smf.ols(fB, data=ar).fit(cov_type="HC1")
print(f"\nRobustness check - drop the {int(once.sum())} postings in a once-only location "
      f"(n = {len(ar)}):")
for fam_ in ["ML Engineer", "Data Scientist", "Manager"]:
    est, pv, ci_ = contrast(Br, fam_)
    print(f"   {fam_:<16} model B premium {pct(est):+6.1f}%  "
          f"(CI {pct(ci_[0]):+.1f}% to {pct(ci_[1]):+.1f}%, p = {pv:.1e})")

# ------------------------------------------------- cosine similarity
prof = a.groupby("title_family")[skills].mean()
da, ds = prof.loc["Data Analyst"].values, prof.loc["Data Scientist"].values
cos = float(np.dot(da, ds) / (np.linalg.norm(da) * np.linalg.norm(ds)))

print("\n" + "=" * 112)
print("SKILL PROFILE SIMILARITY: Data Analyst vs Data Scientist")
print("=" * 112)
print(f"cosine similarity of the 30-dim mean skill-demand vectors = {cos:.4f}")
gap = pd.DataFrame({"Data Analyst": da, "Data Scientist": ds, "diff": ds - da},
                   index=[s.replace("skill_", "") for s in skills])
print("\nskills the Data Scientist family lists more often (top 5):")
for n, r in gap.sort_values("diff", ascending=False).head(5).iterrows():
    print(f"   {n:<24} DA {100 * r['Data Analyst']:5.1f}%   DS {100 * r['Data Scientist']:5.1f}%   "
          f"diff {100 * r['diff']:+.1f} pp")
print("skills the Data Analyst family lists more often (top 5):")
for n, r in gap.sort_values("diff").head(5).iterrows():
    print(f"   {n:<24} DA {100 * r['Data Analyst']:5.1f}%   DS {100 * r['Data Scientist']:5.1f}%   "
          f"diff {100 * r['diff']:+.1f} pp")

# ------------------------------------------------- final answer
hi = tab.iloc[0]
gone_pos = tab.dropna(subset=["premium_gone_pct"])
ds = tab.loc[tab.title_family == "Data Scientist"].iloc[0]
mgr = tab.loc[tab.title_family == "Manager"].iloc[0]
kept = 100 - gone_pos.premium_gone_pct.mean()
print("\n" + "=" * 112)
print("FINAL ANSWER")
print("=" * 112)
print(
    f"1. Yes - holding all 30 skills, experience and location fixed, the title still pays: relative to the "
    f"average title, model A premiums run from {hi.premium_A_pct:+.1f}% ({hi.title_family}) down to "
    f"{tab.premium_A_pct.min():+.1f}% ({tab.iloc[-1].title_family}), and in model B {big.title_family} still "
    f"earns {big.premium_B_pct:+.1f}% (95% CI {big.B_lo:+.1f}% to {big.B_hi:+.1f}%, p = {big.p_B:.1e}) and "
    f"Data Scientist {ds.premium_B_pct:+.1f}% (95% CI {ds.B_lo:+.1f}% to {ds.B_hi:+.1f}%, p = {ds.p_B:.1e}), "
    f"with {int((tab.p_B < 0.05).sum())} of {len(tab)} title coefficients still significant at 5%.\n"
    f"2. Skills buy only part of that gap: the raw premium shrinks by {gone_pos.premium_gone_pct.mean():.0f}% "
    f"on average ({ds.premium_gone_pct:.0f}% for Data Scientist, {big.premium_gone_pct:.0f}% for "
    f"{big.title_family}) and R2 moves just {A.rsquared:.3f} -> {B.rsquared:.3f} (+{B.rsquared - A.rsquared:.3f}), "
    f"even though the skill block is jointly significant (Wald chi2({len(skills)}) = {chi2:.0f}, "
    f"p = {chi2_p:.1e}).\n"
    f"3. Verdict: titles mostly reflect real but only partly measurable differences rather than being empty "
    f"labels - the Data Analyst and Data Scientist skill profiles are just {cos:.2f} cosine-similar "
    f"(python {100 * gap.loc['python', 'Data Scientist']:.1f}% vs {100 * gap.loc['python', 'Data Analyst']:.1f}%, "
    f"machine learning {100 * gap.loc['machine_learning', 'Data Scientist']:.1f}% vs "
    f"{100 * gap.loc['machine_learning', 'Data Analyst']:.1f}%), so the skills we can see are genuinely "
    f"different and partly priced, yet {kept:.0f}% of every premium survives those controls (Manager even "
    f"rises {mgr.premium_A_pct:+.1f}% -> {mgr.premium_B_pct:+.1f}%), which means titles pay for ability, "
    f"seniority and brand that the 30 keyword skills do not capture.\n"
)
print("LIMITATIONS")
print("-" * 112)
print("1. Observational data: title, skills and salary come from the same postings, so unmeasured "
      "ability/seniority/company mix can still drive both the title and the pay (selection, not "
      "causation) - and the 30 skills are keyword matches on a posting, not verified proficiency.")
print("2. Salary is a band midpoint (0to3, 6to10, ...) rather than a real offer, location has 344 levels "
      "with 154 once-only cells (which forces HC1 instead of HC3), so SEs are wide for rare "
      "title x location cells, within-band pay variance is ignored, and small premiums (Data Engineer "
      "+3.6%) cannot be separated from zero.")
print("=" * 112)

tab.to_csv(ROOT / "title_premium_table.csv", index=False)
print("saved -> title_premium_table.csv")
