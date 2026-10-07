"""STEP 3 - Train and SAVE the models (artifacts/*.joblib) plus the result tables the website shows.
Every model is cross-validated, because with 140-160 rows a single train/test split is unreliable."""
import json, warnings
import numpy as np, pandas as pd, joblib
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from config import *

warnings.filterwarnings("ignore")
METRICS = {}

# ======================= A. SALARY MODEL (market + skills layers) =========================
print("\n[A] Salary model on Analytics Jobs")
df = pd.read_csv(CLEAN / "analytics_clean.csv")
skill_cols = [c for c in df.columns if c.startswith("skill_")]
cat_cols, num_cols = ["location", "title_family"], ["avg_exp"] + skill_cols
X, y = df[cat_cols + num_cols], np.log(df["avg_salary"])           # log tames the long salary tail

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
gb = Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=.07, random_state=RANDOM_STATE))])
cv = cross_validate(gb, X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring=("r2", "neg_mean_absolute_error"))
r2, mae = cv["test_r2"].mean(), -cv["test_neg_mean_absolute_error"].mean()
print(f"  Gradient boosting 5-fold CV: R2={r2:.3f}  MAE(log)={mae:.3f}  (~{(np.exp(mae) - 1) * 100:.0f}% typical error)")
gb.fit(X, y)
METRICS["salary_model"] = {"rows": len(df), "cv_r2": round(r2, 3), "typical_error_pct": round((np.exp(mae) - 1) * 100, 1)}
joblib.dump({"model": gb, "skill_cols": skill_cols, "locations": sorted(df.location.unique()), "titles": sorted(df.title_family.unique()),
             "exp_range": [float(df.avg_exp.min()), float(df.avg_exp.max())]}, ART / "salary_model.joblib")

# ---- interpretable OLS: skill ROI + title effect with and without skill controls ----
def ols(frame, extra_cols, baseline_title):
    D = pd.get_dummies(frame[["location", "title_family"]], drop_first=False).astype(float)
    D = D.drop(columns=[f"title_family_{baseline_title}", f"location_{frame.location.value_counts().index[0]}"])
    M = sm.add_constant(pd.concat([frame[["avg_exp"] + extra_cols].astype(float), D], axis=1))
    return sm.OLS(np.log(frame.avg_salary), M).fit(cov_type="HC3")

base_title = "Data Analyst" if "Data Analyst" in df.title_family.values else df.title_family.value_counts().index[0]
full, no_skill = ols(df, skill_cols, base_title), ols(df, [], base_title)
rows = []
for s in skill_cols:
    b, (lo, hi) = full.params[s], full.conf_int().loc[s]
    rows.append(dict(skill=s.replace("skill_", "").replace("_", " "), uplift_pct=(np.exp(b) - 1) * 100, ci_low=(np.exp(lo) - 1) * 100,
                     ci_high=(np.exp(hi) - 1) * 100, p_value=full.pvalues[s], demand_pct=df[s].mean() * 100,
                     years_of_experience_equivalent=b / full.params["avg_exp"] if full.params["avg_exp"] > 0 else np.nan))
roi = pd.DataFrame(rows).sort_values("uplift_pct", ascending=False); roi.to_csv(ANA / "skill_roi.csv", index=False)
tl = []
for t in sorted(df.title_family.unique()):
    k = f"title_family_{t}"
    if k in full.params:
        a, b = (np.exp(no_skill.params[k]) - 1) * 100, (np.exp(full.params[k]) - 1) * 100
        tl.append(dict(title_family=t, premium_without_skills_pct=a, premium_with_skills_pct=b, premium_explained_by_skills_pct=(1 - b / a) * 100 if abs(a) > 1e-9 else np.nan,
                       n_postings=int((df.title_family == t).sum()), median_salary=df.loc[df.title_family == t, "avg_salary"].median()))
    else:
        tl.append(dict(title_family=t, premium_without_skills_pct=0, premium_with_skills_pct=0, premium_explained_by_skills_pct=np.nan,
                       n_postings=int((df.title_family == t).sum()), median_salary=df.loc[df.title_family == t, "avg_salary"].median()))
pd.DataFrame(tl).to_csv(ANA / "title_effect.csv", index=False)
METRICS["salary_ols"] = {"baseline_title": base_title, "r2_with_skills": round(full.rsquared, 3), "r2_without_skills": round(no_skill.rsquared, 3),
                         "pct_salary_per_year_exp": round((np.exp(full.params["avg_exp"]) - 1) * 100, 2)}
print(f"  OLS R2 without skills {no_skill.rsquared:.3f} -> with skills {full.rsquared:.3f}; top skill: {roi.iloc[0].skill} (+{roi.iloc[0].uplift_pct:.1f}%)")

# ======================= B. SMALL-DATA LOGISTIC MODELS (people layer) =====================
def logit_block(name, csv, feats, target, label):
    print(f"\n[{label}] {name}")
    d = pd.read_csv(CLEAN / csv); Xs, ys = d[feats], d[target]
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    res = cross_validate(pipe, Xs, ys, cv=rcv, scoring=("roc_auc", "accuracy", "precision", "recall", "f1"))
    m = {k.replace("test_", ""): round(float(v.mean()), 3) for k, v in res.items() if k.startswith("test_")}
    m.update(rows=len(d), baseline_accuracy=round(float(max(ys.mean(), 1 - ys.mean())), 3)); METRICS[name] = m
    print("  CV:", {k: m[k] for k in ("roc_auc", "accuracy", "f1")}, "| guess-the-majority accuracy:", m["baseline_accuracy"])
    pipe.fit(Xs, ys)
    Z = sm.add_constant((Xs - Xs.mean()) / Xs.std())
    try:
        lg = sm.Logit(ys, Z).fit(disp=0); ci = lg.conf_int(); pv = lg.pvalues
        orr = pd.DataFrame({"feature": feats, "odds_ratio_per_1SD": np.exp(lg.params[feats].values), "ci_low": np.exp(ci.loc[feats, 0].values),
                            "ci_high": np.exp(ci.loc[feats, 1].values), "p_value": pv[feats].values})
    except Exception as e:
        print("  (Logit CI failed:", e, ") using sklearn coefficients only")
        orr = pd.DataFrame({"feature": feats, "odds_ratio_per_1SD": np.exp(pipe.named_steps["lr"].coef_[0]), "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
    orr["stage"] = label; orr.to_csv(ANA / f"odds_{name}.csv", index=False)
    joblib.dump({"model": pipe, "feats": feats, "means": Xs.mean().to_dict(), "stds": Xs.std().to_dict(),
                 "ranges": {f: [float(Xs[f].min()), float(Xs[f].max())] for f in feats}}, ART / f"{name}_model.joblib")
    return d, orr

jds_feats = ["big_data_skills", "maths_stats_skills", "coding_skills", "ai_and_ml_skills", "dashboard_and_storytelling_skills"]
sds_feats = ["neuroticism", "extraversion", "openness_to_experience", "agreeableness", "conscientiousness"]
jds, o1 = logit_block("hike", "jds_clean.csv", jds_feats, "salary_hike_high_or_low", "Junior stage (skills -> salary hike)")
sds, o2 = logit_block("success", "sds_clean.csv", sds_feats, "success_high_low", "Senior stage (personality -> success)")

# ---- Module 7: personality sweet spot (do squared terms help?) ----
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
lin = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
quad = Pipeline([("sc", StandardScaler()), ("poly", PolynomialFeatures(2, include_bias=False)), ("sc2", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=2000, C=0.3))])
a, b = (cross_val_score(m, sds[sds_feats], sds.success_high_low, cv=rcv, scoring="roc_auc").mean() for m in (lin, quad))
pd.DataFrame({"model": ["Linear (more is always better/worse)", "With squared terms (sweet spot allowed)"], "cv_auc": [a, b]}).to_csv(ANA / "sweet_spot.csv", index=False)
METRICS["sweet_spot"] = {"auc_linear": round(a, 3), "auc_quadratic": round(b, 3), "sweet_spot_supported": bool(b > a + 0.02)}
print(f"\n[Module 7] Sweet-spot test: linear AUC {a:.3f} vs squared {b:.3f} ->", "curved effect supported" if b > a + .02 else "no clear evidence of a sweet spot")
curve = []
for f in sds_feats:                                 # success probability across one trait, others held at average
    grid = np.linspace(sds[f].quantile(.02), sds[f].quantile(.98), 25)
    Q = pd.DataFrame({c: [sds[c].mean()] * 25 for c in sds_feats}); Q[f] = grid
    qm = quad.fit(sds[sds_feats], sds.success_high_low); curve.append(pd.DataFrame({"trait": f, "value": grid, "p_success": qm.predict_proba(Q)[:, 1]}))
pd.concat(curve).to_csv(ANA / "sweet_spot_curves.csv", index=False)

# ---- personality archetypes (clusters) ----
Zs = StandardScaler().fit_transform(sds[sds_feats]); best = (-1, 2)
for k in range(2, 6):
    sc = silhouette_score(Zs, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Zs)); best = max(best, (sc, k))
km = KMeans(best[1], n_init=10, random_state=RANDOM_STATE).fit(Zs); sds["cluster"] = km.labels_
prof = sds.groupby("cluster")[sds_feats].mean().round(2); prof["n_people"] = sds.cluster.value_counts().sort_index(); prof["success_rate_pct"] = (sds.groupby("cluster").success_high_low.mean() * 100).round(1)
def name_cluster(r):
    z = (r[sds_feats] - sds[sds_feats].mean()) / sds[sds_feats].std(); top = z.abs().sort_values(ascending=False).index[:2]
    return " + ".join(("High " if z[t] > 0 else "Low ") + t.replace("_to_experience", "").replace("_", " ") for t in top)
prof["archetype"] = prof.apply(name_cluster, axis=1); prof.reset_index().to_csv(ANA / "personality_clusters.csv", index=False)
METRICS["clusters"] = {"k": best[1], "silhouette": round(best[0], 3)}
print(f"  personality archetypes: k={best[1]} (silhouette {best[0]:.2f})"); print(prof[["n_people", "success_rate_pct", "archetype"]].to_string())

json.dump(METRICS, open(ART / "metrics.json", "w"), indent=2)
print("\nSaved models + metrics to artifacts/")
