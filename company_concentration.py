"""Is hiring concentrated in a few companies, and do big hirers pay less?

dsjobs_clean.csv is posting level (company x job_title). num_of_jobs is the number of
openings for THAT posting, so it varies within a company and must be summed per company.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from common import gini
from config import CLEAN, ROOT

RS = 42
warnings.filterwarnings("ignore", category=RuntimeWarning)

d = pd.read_csv(CLEAN / "dsjobs_clean.csv")
print(f"POSTING LEVEL: {len(d)} rows, {d.company_name.nunique()} companies, "
      f"{d.job_title.nunique()} job titles")

# ---- 1. does num_of_jobs repeat per company or differ per row? ----
per_co = d.groupby("company_name")["num_of_jobs"].agg(rows="size", distinct="nunique")
varies = per_co[per_co.distinct > 1]
print("\n1) num_of_jobs WITHIN COMPANY")
print(f"   companies with >1 posting      : {(per_co.rows > 1).sum()} / {len(per_co)}")
print(f"   companies where it VARIES      : {len(varies)} (every multi-posting company)")
ex = d[d.company_name.isin(varies.sort_values("rows", ascending=False).head(3).index)]
print("   examples:")
for co, g in ex.groupby("company_name"):
    print(f"     {co:<14} " + ", ".join(f"{t}={n}" for t, n in zip(g.job_title, g.num_of_jobs)))
print("   -> it is postings per listing, NOT a company-level constant: SUM it per company.")

# ---- 2. company level aggregation (jobs summed, pay weighted by postings) ----
g = d.groupby("company_name").apply(lambda x: pd.Series({
    "jobs": x.num_of_jobs.sum(),
    "avg_salary": np.average(x.avg_salary, weights=x.num_of_jobs),
    "avg_salary_unw": x.avg_salary.mean(),
    "titles": x.job_title.nunique(),
    "exp": (x.min_experience * x.num_of_jobs).sum() / x.num_of_jobs.sum(),
    "postings": len(x),
}), include_groups=False)
tot = g.jobs.sum()
print(f"\n2) COMPANY LEVEL: {len(g)} companies, {tot:,.0f} total openings")

# ---- 3. concentration ----
gi = gini(g.jobs.values)
ktop = int(np.ceil(0.10 * len(g)))
top10 = g.jobs.nlargest(ktop)
share = top10.sum() / tot
print("\n3) CONCENTRATION")
print(f"   Gini of openings per company   : {gi:.3f}")
print(f"   top 10% of companies = {ktop} firms hold {share:.1%} of all openings")
print("   top 10 hirers:")
for co, r in g.jobs.nlargest(10).items():
    print(f"     {co:<26}{int(r):>6}  {r / tot:>6.1%}")

# ---- 4. do big hirers pay less? ----
rho, p = stats.spearmanr(g.jobs, g.avg_salary)
rho_u, p_u = stats.spearmanr(g.jobs, g.avg_salary_unw)
print("\n4) JOBS vs PAY (company level, Spearman)")
print(f"   weighted avg_salary : rho = {rho:+.3f}, p = {p:.2e}, n = {len(g)}")
print(f"   unweighted avg_salary: rho = {rho_u:+.3f}, p = {p_u:.4f}")
print(f"   median pay, top-decile hirers vs rest: "
      f"{g.loc[top10.index, 'avg_salary'].median():.1f}L vs "
      f"{g.drop(top10.index).avg_salary.median():.1f}L")

# ---- 5. K-means company clusters ----
feat_df = pd.DataFrame({"log_jobs": np.log1p(g.jobs), "avg_salary": g.avg_salary,
                        "exp": g.exp, "titles": g.titles})
X = StandardScaler().fit_transform(feat_df)
best = None
for k_ in range(2, 9):
    lab = KMeans(k_, n_init=20, random_state=RS).fit_predict(X)
    s = silhouette_score(X, lab)
    if best is None or s > best[1]:
        best = (k_, s)
kcl, sil = best
km = KMeans(kcl, n_init=20, random_state=RS).fit(X)
g["cluster"] = km.labels_
print(f"\n5) K-MEANS: k = {kcl} (silhouette {sil:.3f}, best of 2..8)")
cen = g.groupby("cluster").agg(jobs=("jobs", "sum"), firms=("jobs", "size"),
                               pay=("avg_salary", "mean"), exp=("exp", "mean"),
                               titles=("titles", "mean")).sort_values("jobs", ascending=False)
cen["share_jobs"] = cen.jobs / tot
med_pay, med_exp = g.avg_salary.median(), g.exp.median()


def cluster_name(r):
    if r.share_jobs >= 0.40:
        return "high-volume multi-title recruiters"
    if r.pay >= med_pay and r.exp >= med_exp:
        return "senior high-pay boutiques"
    if r.pay >= med_pay:
        return "small high-pay employers"
    return "small low-pay short-tenure employers"


for cl, r in cen.iterrows():
    names = ", ".join(g[g.cluster == cl].jobs.nlargest(4).index)
    print(f"   cluster {cl} [{cluster_name(r)}]: {int(r.firms):>3} firms, {int(r.jobs):>6} openings "
          f"({r.share_jobs:>5.1%}), pay {r.pay:>5.1f}L, exp {r.exp:.1f}y, "
          f"{r.titles:.1f} titles | {names}")

print(f"\nCONCLUSION: Gini {gi:.3f}; top 10% of firms ({ktop}) hold {share:.1%} of openings; "
      f"hiring volume vs pay Spearman rho = {rho:+.3f} (p = {p:.2e}).")
g["cluster_name"] = g.cluster.map(lambda c: cluster_name(cen.loc[c]))
g.sort_values("jobs", ascending=False).to_csv(ROOT / "company_clusters.csv", index_label="company_name")
print("saved -> company_clusters.csv")
