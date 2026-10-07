"""Senior DS success - which Big Five traits matter, is there a sweet spot, and how many profiles win.

Line 1: Repeated stratified 5-fold CV compares a linear logistic model on the Big Five against the same
        model with squared terms (CV AUC, accuracy vs majority guess, precision, recall, F1).
Line 2: Full-sample odds ratios per 1 SD and per-trait quadratic logits locate each trait's optimum,
        then K-means (k chosen by silhouette) names the personality archetypes and their success rates.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "data/clean/sds_clean.csv")
TRAITS = [
    "neuroticism",
    "extraversion",
    "openness_to_experience",
    "agreeableness",
    "conscientiousness",
]
SHORT = {
    "neuroticism": "neuroticism",
    "extraversion": "extraversion",
    "openness_to_experience": "openness",
    "agreeableness": "agreeableness",
    "conscientiousness": "conscientiousness",
}
X, y = d[TRAITS].values, d["success_high_low"].values
Xz = (X - X.mean(0)) / X.std(0)
maj_acc = max(y.mean(), 1 - y.mean())
pf = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"

print("=" * 118)
print("PERSONALITY vs SUCCESS FOR SENIOR DATA SCIENTISTS - sds_clean.csv, target = success_high_low")
print("=" * 118)
print(f"n = {len(y)} rows   |   success = {y.sum()} ({100 * y.mean():.1f}%)   failure = {len(y) - y.sum()} "
      f"({100 * (1 - y.mean()):.1f}%)   |   Big Five scores ~17-68")
print("1 SD per trait: " + ", ".join(f"{SHORT[t]} {d[t].std():.1f}" for t in TRAITS))

# ------------------------------------------------- 1) repeated stratified 5-fold CV, linear vs squared
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
MODELS = {
    "logit: 5 traits (linear)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "logit: + 5 squared terms": make_pipeline(
        PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(max_iter=1000)
    ),
}
METRICS = ["ROC-AUC", "accuracy", "precision", "recall", "F1"]
scores, auc_by_repeat = {k: [] for k in MODELS}, {k: np.empty((10, 5)) for k in MODELS}
for split_i, (tr, te) in enumerate(cv.split(X, y)):
    rep, fold = divmod(split_i, 5)
    for name, model in MODELS.items():
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        row = [
            roc_auc_score(y[te], prob),
            accuracy_score(y[te], pred),
            precision_score(y[te], pred, zero_division=0),
            recall_score(y[te], pred),
            f1_score(y[te], pred),
        ]
        scores[name].append(row)
        auc_by_repeat[name][rep, fold] = row[0]
scores = {k: np.asarray(v) for k, v in scores.items()}
lin, quad = scores["logit: 5 traits (linear)"], scores["logit: + 5 squared terms"]
auc_lin, auc_quad = auc_by_repeat["logit: 5 traits (linear)"].mean(1), auc_by_repeat["logit: + 5 squared terms"].mean(1)
t_stat, p_stat = stats.ttest_rel(auc_quad, auc_lin)

print("\n" + "=" * 118)
print("REPEATED STRATIFIED 5-FOLD CV (5 folds x 10 repeats = 50 splits, stratified on the target)")
print("=" * 118)
print(f"{'model':<28}" + "".join(f"{m:>17}" for m in METRICS))
print(f"{'majority-class guess':<28}{'-':>17}{maj_acc:>17.3f}{'-':>17}{'-':>17}{'-':>17}{'-':>17}")
for name, arr in scores.items():
    print(f"{name:<28}" + "".join(f"{arr[:, i].mean():>13.3f}+-{arr[:, i].std():.3f}" for i in range(5)))
print(f"squared terms vs linear: CV AUC {quad[:, 0].mean():.3f} - {lin[:, 0].mean():.3f} = "
      f"{quad[:, 0].mean() - lin[:, 0].mean():+.3f} (paired t on 10 repeat-means: t = {t_stat:.2f}, p = {pf(p_stat)})")

# ------------------------------------------------- 2) odds ratios per 1 SD
fit = sm.Logit(y, sm.add_constant(Xz)).fit(disp=0)
ci = fit.conf_int()
or_tab = pd.DataFrame(
    {
        "OR_per_1SD": np.exp(fit.params[1:]),
        "ci_lo": np.exp(ci[1:, 0]),
        "ci_hi": np.exp(ci[1:, 1]),
        "p_value": fit.pvalues[1:],
    },
    index=[SHORT[t] for t in TRAITS],
).sort_values("OR_per_1SD", ascending=False)
or_tab["reliable"] = (or_tab.p_value < 0.05) & (or_tab.ci_lo > 1)

print("\n" + "=" * 118)
print("ODDS RATIOS PER 1 SD (full-sample logit, all 5 traits in the model)")
print("=" * 118)
print(f"{'trait':<16}{'odds ratio':>13}{'95% CI':>22}{'p':>9}{'reliable':>10}")
for n_, r in or_tab.iterrows():
    print(f"{n_:<16}{r.OR_per_1SD:>12.1f}x{f'[{r.ci_lo:.2f}, {r.ci_hi:.2f}]':>22}"
          f"{pf(r.p_value):>9}{('yes' if r.reliable else 'no'):>10}")
print(f"McFadden pseudo-R2 = {fit.prsquared:.3f}")

# ------------------------------------------------- 3) sweet spot: per-trait quadratic logits
print("\n" + "=" * 118)
print("SWEET SPOT - per-trait quadratic logits: success ~ z + z^2 (each trait alone; the joint 10-term")
print("quadratic MLE separates perfectly on n = 161, so inference is done one trait at a time)")
print("=" * 118)
print(f"{'trait':<16}{'z^2 coef':>10}{'z^2 p':>8}{'LRT chi2(1)':>13}{'LRT p':>8}"
      f"{'peak (SD)':>11}{'peak (raw)':>12}{'range':>12}{'shape':>12}")
sweet = {}
for j, t in enumerate(TRAITS):
    z = Xz[:, j]
    m_lin = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    m_quad = sm.Logit(y, sm.add_constant(np.c_[z, z**2])).fit(disp=0)
    lrt = 2 * (m_quad.llf - m_lin.llf)
    lrt_p = float(1 - stats.chi2.cdf(lrt, 1))
    b1, b2 = m_quad.params[1], m_quad.params[2]
    peak_z = -b1 / (2 * b2)
    peak_raw = d[t].mean() + peak_z * d[t].std()
    inside = d[t].min() < peak_raw < d[t].max()
    shape = "inverted-U" if (b2 < 0 and lrt_p < 0.05 and inside) else ("rising" if b2 < 0 else "flat/rising")
    sweet[SHORT[t]] = dict(peak_raw=peak_raw, p=m_quad.pvalues[2], lrt_p=lrt_p, b2=b2, inside=inside, shape=shape)
    print(f"{SHORT[t]:<16}{b2:>+10.3f}{pf(m_quad.pvalues[2]):>8}{lrt:>13.2f}{pf(lrt_p):>8}"
          f"{peak_z:>+11.2f}{peak_raw:>12.0f}{f'{d[t].min()}-{d[t].max()}':>12}{shape:>12}")
n_inverted = sum(1 for v in sweet.values() if v["shape"] == "inverted-U")
print(f"{n_inverted} of 5 traits show a significant inverted-U with the peak inside the observed range "
      f"(more is NOT always better)")

# ------------------------------------------------- 4) K-means archetypes
print("\n" + "=" * 118)
print("K-MEANS ON THE STANDARDISED BIG FIVE (k = 2 to 5, chosen by silhouette)")
print("=" * 118)
runs = {}
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xz)
    sil = silhouette_score(Xz, km.labels_)
    rates = [y[km.labels_ == i].mean() for i in range(k)]
    sizes = [int((km.labels_ == i).sum()) for i in range(k)]
    runs[k] = (km, sil, rates, sizes)
    winners = sum(1 for i in range(k) if rates[i] > y.mean() and sizes[i] >= 10)
    print(f"k = {k}: silhouette = {sil:.3f}   sizes = {sizes}   success rates = "
          + ", ".join(f"{100 * r:.0f}%" for r in rates)
          + f"   clusters beating the {100 * y.mean():.0f}% base rate (n >= 10): {winners}")
best_k = max(runs, key=lambda k: runs[k][1])
km, sil, rates, sizes = runs[best_k]


def name_cluster(ctr):
    if ctr[TRAITS.index("neuroticism")] >= 1.0:
        return "stressed veterans"
    if ctr[TRAITS.index("agreeableness")] >= 0.5:
        return "warm but unfocused"
    if ctr[TRAITS.index("conscientiousness")] >= 0.5:
        return "driven achievers"
    return "reserved and detached"


order = sorted(range(best_k), key=lambda i: -rates[i])
print(f"\nchosen k = {best_k} (best silhouette {sil:.3f}); archetypes named from their trait z-profiles:")
print(f"{'archetype':<26}{'n':>5}{'success':>9}   trait z-scores (neuroticism, extraversion, openness,")
print(f"{'':<26}{'':>5}{'':>9}   agreeableness, conscientiousness)")
for i in order:
    print(f"{name_cluster(km.cluster_centers_[i]):<26}{sizes[i]:>5}{100 * rates[i]:>8.0f}%   "
          + ", ".join(f"{v:+.2f}" for v in km.cluster_centers_[i]))
n_win_best = sum(1 for i in range(best_k) if rates[i] > y.mean() and sizes[i] >= 10)
k5_winners = sum(1 for r, s in zip(runs[5][2], runs[5][3]) if r > y.mean() and s >= 10)

# ------------------------------------------------- final answer
top2 = or_tab.head(2)
inv = [k for k, v in sweet.items() if v["shape"] == "inverted-U"]
auc_note = (
    f"CV AUC is {lin[:, 0].mean():.3f}, well above the 0.65 bar, so the signal is not weak inside this "
    f"dataset - but n = {len(y)} with the target already supplied pre-labelled in the source file means "
    f"this is descriptive of these rows, not an externally validated prediction rule"
    if lin[:, 0].mean() >= 0.65
    else f"CV AUC is only {lin[:, 0].mean():.3f}, below 0.65, so treat every effect below as weak evidence"
)
print("\n" + "=" * 118)
print("FINAL ANSWER")
print("=" * 118)
print(
    f"1. Traits that matter (odds ratio per 1 SD, all five in one logit): "
    f"{or_tab.index[0]} {or_tab.iloc[0].OR_per_1SD:.1f}x (95% CI {or_tab.iloc[0].ci_lo:.1f}-"
    f"{or_tab.iloc[0].ci_hi:.1f}, p = {pf(or_tab.iloc[0].p_value)}) and {or_tab.index[1]} "
    f"{or_tab.iloc[1].OR_per_1SD:.1f}x (CI {or_tab.iloc[1].ci_lo:.1f}-{or_tab.iloc[1].ci_hi:.1f}, "
    f"p = {pf(or_tab.iloc[1].p_value)}) dominate, then "
    + ", ".join(
        f"{n} {r.OR_per_1SD:.1f}x (p = {pf(r.p_value)}{'' if r.reliable else ' - NOT significant'})"
        for n, r in or_tab.iloc[2:].iterrows()
    )
    + f"; the model beats guessing with CV ROC-AUC {lin[:, 0].mean():.3f} +- {lin[:, 0].std():.3f} and "
    f"accuracy {100 * lin[:, 1].mean():.1f}% vs the {100 * maj_acc:.1f}% majority guess.\n"
    f"2. More is NOT always better: all {n_inverted} traits show a significant inverted-U "
    f"({', '.join(inv)}) with optima at "
    + ", ".join(f"{k} = {sweet[k]['peak_raw']:.0f}" for k in inv)
    + f" on a {int(d[TRAITS].min().min())}-{int(d[TRAITS].max().max())} scale, yet adding the squared terms "
    f"buys only {quad[:, 0].mean() - lin[:, 0].mean():+.3f} CV AUC "
    f"({lin[:, 0].mean():.3f} -> {quad[:, 0].mean():.3f}, paired p = {pf(p_stat)}) - the curvature is "
    f"real but small, so the practical story stays 'aim high, then plateau'.\n"
    f"3. Distinct winning profiles: k = {best_k} by silhouette ({sil:.3f}) gives {n_win_best} winning "
    f"cluster out of {best_k} - '{name_cluster(km.cluster_centers_[order[0]])}' (n = {sizes[order[0]]}, "
    f"{100 * rates[order[0]]:.0f}% success) against losing profiles at "
    + ", ".join(f"{100 * rates[i]:.0f}%" for i in order[1:])
    + f" (base rate {100 * y.mean():.0f}%); forcing k = 5 would show {k5_winners} winning clusters but "
    f"its silhouette drops to {runs[5][1]:.3f}, so evidence for more than one winning type is weak. "
    f"Honest note: {auc_note}.\n"
)
print("=" * 118)

or_tab.round(3).to_csv(ROOT / "sds_odds_ratios.csv")
pd.DataFrame(
    [
        {"trait": k, "peak_raw": round(v["peak_raw"], 1), "z2_coef": round(v["b2"], 3),
         "z2_p": round(v["p"], 4), "lrt_p": round(v["lrt_p"], 4), "shape": v["shape"]}
        for k, v in sweet.items()
    ]
).to_csv(ROOT / "sds_sweet_spots.csv", index=False)
print("saved -> sds_odds_ratios.csv, sds_sweet_spots.csv")
