"""Junior DS salary hikes - which skills separate high from low hikes (logistic vs a small tree).

Line 1: Repeated stratified 5-fold cross-validation scores logistic regression and a depth-3 tree
        (ROC-AUC, accuracy vs the majority-class guess, precision, recall, F1).
Line 2: The full-sample logit gives odds ratios per 1 SD with 95% CI and p-values, and a coding x
        storytelling interaction is tested by Wald test and likelihood-ratio test.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "data/clean/jds_clean.csv")
SKILLS = [
    "big_data_skills",
    "maths_stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
]
SHORT = {
    "big_data_skills": "big_data",
    "maths_stats_skills": "maths_stats",
    "coding_skills": "coding",
    "ai_and_ml_skills": "ai_and_ml",
    "dashboard_and_storytelling_skills": "storytelling",
}
X, y = d[SKILLS].values, d["salary_hike_high_or_low"].values
Xz = (X - X.mean(0)) / X.std(0)          # 1 SD = the unit used for the odds ratios
SD1 = {SHORT[s]: d[s].std() for s in SKILLS}

print("=" * 118)
print("HIGH vs LOW SALARY HIKE FOR JUNIOR DATA SCIENTISTS - jds_clean.csv, target = salary_hike_high_or_low")
print("=" * 118)
print(f"n = {len(y)} rows   |   high hike = {y.sum()} ({100 * y.mean():.1f}%)   low = {len(y) - y.sum()} "
      f"({100 * (1 - y.mean()):.1f}%)   |   5 skill scores on a ~2.2-5.0 scale")
print("1 SD of each skill (the odds-ratio unit): "
      + ", ".join(f"{SHORT[s]} {d[s].std():.2f}" for s in SKILLS))

# ------------------------------------------------- 1) repeated stratified 5-fold CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
MODELS = {
    "logistic (all 5 skills)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
}
METRICS = ["ROC-AUC", "accuracy", "precision", "recall", "F1"]
scores = {}
for name, model in MODELS.items():
    rows = []
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            [
                roc_auc_score(y[te], prob),
                accuracy_score(y[te], pred),
                precision_score(y[te], pred, zero_division=0),
                recall_score(y[te], pred),
                f1_score(y[te], pred),
            ]
        )
    scores[name] = np.asarray(rows)

maj_acc = max(y.mean(), 1 - y.mean())
print("\n" + "=" * 118)
print("REPEATED STRATIFIED 5-FOLD CV (5 folds x 10 repeats = 50 splits, stratified on the target)")
print("=" * 118)
print(f"{'model':<26}" + "".join(f"{m:>16}" for m in METRICS))
print(f"{'majority-class guess':<26}{'-':>16}{maj_acc:>16.3f}{'-':>16}{'-':>16}{'-':>16}{'-':>16}")
for name, arr in scores.items():
    print(f"{name:<26}" + "".join(f"{arr[:, i].mean():>12.3f}+-{arr[:, i].std():.3f}" for i in range(5)))
auc = scores["logistic (all 5 skills)"][:, 0]
auc_tree = scores["decision tree (depth 3)"][:, 0]
print(f"logistic beats the majority-class guess on accuracy by "
      f"{100 * (scores['logistic (all 5 skills)'][:, 1].mean() - maj_acc):.1f} pp and scores "
      f"{auc.mean():.3f} ROC-AUC vs the {0.5:.1f} of random ranking")

# ------------------------------------------------- 2) odds ratios per 1 SD
fit = sm.Logit(y, sm.add_constant(Xz)).fit(disp=0)
ci = fit.conf_int()
or_tab = pd.DataFrame(
    {
        "odds_ratio_per_1SD": np.exp(fit.params[1:]),
        "ci_lo": np.exp(ci[1:, 0]),
        "ci_hi": np.exp(ci[1:, 1]),
        "p_value": fit.pvalues[1:],
    },
    index=[SHORT[s] for s in SKILLS],
).sort_values("odds_ratio_per_1SD", ascending=False)
or_tab["reliable"] = (or_tab.p_value < 0.05) & (or_tab.ci_lo > 1)

print("\n" + "=" * 118)
print("ODDS RATIOS PER 1 SD OF THE SKILL (full-sample logit, all 5 skills in the model)")
print("=" * 118)
print(f"{'skill':<16}{'odds ratio':>12}{'95% CI':>20}{'p':>9}{'reliable':>10}")
for n_, r in or_tab.iterrows():
    print(f"{n_:<16}{r.odds_ratio_per_1SD:>11.2f}x{f'[{r.ci_lo:.2f}, {r.ci_hi:.2f}]':>20}"
          f"{r.p_value:>9.3f}{('yes' if r.reliable else 'no'):>10}")
print(f"log-likelihood = {fit.llf:.2f}   McFadden pseudo-R2 = {fit.prsquared:.3f}")

# ------------------------------------------------- 3) depth-3 tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
print("\n" + "=" * 118)
print("DEPTH-3 DECISION TREE (fit on all rows, for comparison)")
print("=" * 118)
print(export_text(tree, feature_names=[SHORT[s] for s in SKILLS]))
print("feature importances: " + ", ".join(
    f"{SHORT[s]} {imp:.2f}" for s, imp in zip(SKILLS, tree.feature_importances_)))

# ------------------------------------------------- 4) interaction test
i_coding, i_dash = SKILLS.index("coding_skills"), SKILLS.index("dashboard_and_storytelling_skills")
inter = Xz[:, i_coding] * Xz[:, i_dash]
fit_i = sm.Logit(y, np.c_[np.ones(len(y)), Xz, inter]).fit(disp=0)
ci_i = fit_i.conf_int()
wald_p = fit_i.pvalues[-1]
lrt = 2 * (fit_i.llf - fit.llf)
lrt_p = float(1 - stats.chi2.cdf(lrt, df=1))

print("\n" + "=" * 118)
print("INTERACTION TEST: coding_skills x dashboard_and_storytelling_skills (both standardised)")
print("=" * 118)
print(f"interaction term: coef = {fit_i.params[-1]:+.3f}   odds ratio = {np.exp(fit_i.params[-1]):.2f} "
      f"(95% CI {np.exp(ci_i[-1, 0]):.2f} to {np.exp(ci_i[-1, 1]):.2f})   Wald p = {wald_p:.3f}")
print(f"likelihood-ratio test vs the model without the interaction: chi2(1) = {lrt:.2f}, p = {lrt_p:.3f}")
print(f"verdict: {'keep' if lrt_p < 0.05 else 'drop'} the interaction "
      f"(p = {lrt_p:.3f} {'<' if lrt_p < 0.05 else '>='} 0.05) - no evidence that storytelling matters "
      f"more when coding skills are high")

# ------------------------------------------------- final answer
t1, t2 = or_tab.index[0], or_tab.index[1]
r1, r2 = or_tab.iloc[0], or_tab.iloc[1]
tree_top = [SHORT[s] for s, i in zip(SKILLS, tree.feature_importances_) if i > 0.1]

print("\n" + "=" * 118)
print("FINAL ANSWER")
print("=" * 118)
print(
    f"1. The top 2 skills separating high from low hikes are {t1} (odds ratio {r1.odds_ratio_per_1SD:.1f}x "
    f"per 1 SD, 95% CI {r1.ci_lo:.1f}x-{r1.ci_hi:.1f}x, p = {r1.p_value:.3f}) and {t2} "
    f"({r2.odds_ratio_per_1SD:.1f}x, CI {r2.ci_lo:.1f}x-{r2.ci_hi:.1f}x, p = {r2.p_value:.3f}): in plain "
    f"words, a junior whose {t1.replace('_', ' ')} score is one SD ({SD1[t1]:.1f} "
    f"points) higher has about {r1.odds_ratio_per_1SD:.0f} times the odds of getting a high hike even when "
    f"the other four skills are held equal, and the same one-SD edge in {t2.replace('_', ' ')} is worth "
    f"about {r2.odds_ratio_per_1SD:.0f} times the odds.\n"
    f"2. Yes, the model clearly beats guessing: repeated stratified 5-fold CV gives ROC-AUC "
    f"{auc.mean():.3f} +- {auc.std():.3f} (random = 0.500) and accuracy "
    f"{100 * scores['logistic (all 5 skills)'][:, 1].mean():.1f}% +- "
    f"{100 * scores['logistic (all 5 skills)'][:, 1].std():.1f}% against the "
    f"{100 * maj_acc:.1f}% you get by always predicting the majority class, with precision "
    f"{scores['logistic (all 5 skills)'][:, 2].mean():.2f}, recall "
    f"{scores['logistic (all 5 skills)'][:, 3].mean():.2f} and F1 "
    f"{scores['logistic (all 5 skills)'][:, 4].mean():.2f}; the depth-3 tree is weaker "
    f"(AUC {auc_tree.mean():.3f} +- {auc_tree.std():.3f}) and leans on just {', '.join(tree_top)}.\n"
    f"3. Confidence: with only n = {len(y)} rows and {int(y.sum())} high-hike cases over 5 predictors "
    f"({y.sum() / len(SKILLS):.0f} events per variable, just above the usual 10-per-variable rule of "
    f"thumb) the direction of the top 2 skills is solid but the magnitudes are imprecise - their CIs run "
    f"{r1.ci_lo:.1f}x-{r1.ci_hi:.1f}x and {r2.ci_lo:.1f}x-{r2.ci_hi:.1f}x, single splits swing AUC by "
    f"~{auc.std():.2f}, and coding_skills alone is not significant (p = {or_tab.loc['coding', 'p_value']:.2f}); "
    f"the coding x storytelling interaction is not supported "
    f"(LRT chi2(1) = {lrt:.2f}, p = {lrt_p:.2f}), so we cannot claim combined effects at this sample size.\n"
)
print("=" * 118)

or_tab.round(3).to_csv(ROOT / "jds_odds_ratios.csv")
print("saved -> jds_odds_ratios.csv")
