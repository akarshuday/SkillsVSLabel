"""Pay transparency: which title families and cities publish the least salary info?

Input : data/clean/analytics_all.csv  (every posting, incl. those with no salary)
        built by 02_clean_data.py -> clean_analytics(); salary_missing = salary could
        not be read from the raw "salary" string.
Output: share of missing salary per title_family and per location, chi-square tests,
        top/bottom 3 groups, saved to artifacts/salary_missing_groups.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from config import CLEAN, ART

d = pd.read_csv(CLEAN / "analytics_all.csv")
print(f"analytics_all.csv: {len(d)} postings, {d.title_family.nunique()} title families, "
      f"{d.location.nunique()} locations")
print(f"salary_missing = 1 in {int(d.salary_missing.sum())} rows "
      f"({d.salary_missing.mean():.1%}); avg_salary NaN in {int(d.avg_salary.isna().sum())}")


def by(col, label):
    t = d.groupby(col).salary_missing.agg(n="size", missing="sum", share="mean")
    t = t.sort_values("share", ascending=False)
    print(f"\nshare of missing salary by {label} ({len(t)} groups):")
    for name, r in t.iterrows():
        print(f"  {str(name):<22}{int(r.n):>6} rows  {int(r.missing):>4} missing  {r.share:>7.1%}")
    return t


fam = by("title_family", "title_family")
loc = by("location", "location")

print("\n3 highest / lowest title_family:")
print("  high:", ", ".join(f"{i} {s:.1%}" for i, s in fam.share.head(3).items()))
print("  low :", ", ".join(f"{i} {s:.1%}" for i, s in fam.share.tail(3).items()))
print("3 highest / lowest location:")
print("  high:", ", ".join(f"{i} {s:.1%}" for i, s in loc.share.head(3).items()))
print("  low :", ", ".join(f"{i} {s:.1%}" for i, s in loc.share.tail(3).items()))

print("\nCHI-SQUARE (missing salary x group)")


def chi(col, label):
    ct = pd.crosstab(d[col], d.salary_missing)
    if ct.shape[1] < 2:
        print(f"  {label:<16} NOT COMPUTABLE - salary_missing has only one level "
              f"(all {ct.shape[0]} groups are 0% missing), so the table has no variance "
              f"and expected counts are all 0.")
        return None
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    print(f"  {label:<16} chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}")
    return chi2, p, dof


chi("title_family", "title_family")
chi("location", "location")
ct = pd.crosstab([d.title_family, d.location], d.salary_missing)
chi2, p, dof, exp = stats.chi2_contingency(ct) if ct.shape[1] > 1 else (np.nan,) * 4
print(f"  {'title_family x loc':<16} "
      + (f"chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}" if ct.shape[1] > 1
         else f"NOT COMPUTABLE - {ct.shape[0]} title x location cells, all 0% missing"))

# --- the only salary opacity this file actually has: a published floor of 0 -------
d["floor_hidden"] = (d.sal_min == 0).astype(int)
print(f"\nFALLBACK (salary band published with a 0 lower bound): "
      f"{int(d.floor_hidden.sum())} of {len(d)} postings ({d.floor_hidden.mean():.1%})")
for col, label in [("title_family", "title_family"), ("location", "location")]:
    t = d.groupby(col).floor_hidden.agg(n="size", share="mean").sort_values("share", ascending=False)
    print(f"  by {label}: 3 highest = " +
          ", ".join(f"{i} {s:.1%} (n={int(t.loc[i, 'n'])})" for i, s in t.share.head(3).items()))
    print(f"  by {label}: 3 lowest  = " +
          ", ".join(f"{i} {s:.1%} (n={int(t.loc[i, 'n'])})" for i, s in t.share.tail(3).items()))
    ct = pd.crosstab(d[col], d.floor_hidden)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"  chi-square {label}: chi2 = {chi2:.1f}, dof = {dof}, p = {p:.2e}")
    t.assign(kind=label, metric="floor_hidden").to_csv(ART / f"floor_hidden_{label}.csv")

out = pd.concat([fam.assign(metric="salary_missing", group="title_family"),
                 loc.assign(metric="salary_missing", group="location")])
out.to_csv(ART / "salary_missing_groups.csv")
print("\nsaved -> artifacts/salary_missing_groups.csv")
