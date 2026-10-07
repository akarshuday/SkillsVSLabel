"""Experience knee - at what experience level does salary growth flatten?

Line 1: Salary percentiles per experience year plus quadratic quantile regressions (q = 0.1, 0.5, 0.9)
        on log(salary) show how the bottom, middle and top of the pay distribution grow with experience.
Line 2: A piecewise (broken-stick) regression picks the breakpoint by grid search, bootstraps a 95% CI
        for it, and reports the yearly growth rate in % before and after the break.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

warnings.filterwarnings("ignore", category=RuntimeWarning)   # stale FP flags from QuantReg internals (fits checked finite)

ROOT = Path(__file__).resolve().parent
a = pd.read_csv(ROOT / "data/clean/analytics_clean.csv")
a["salary_log"] = np.log(a["salary_avg_lakh"])
e = a["avg_exp"].values                       # years (0 to 26.5, half-year steps)
y = a["salary_log"].values
x = e / 10.0                                  # rescaled experience keeps QuantReg well conditioned
GRID = np.arange(1.0, 20.5, 0.5)              # candidate breakpoints, years
B_BOOT = 1000

print("=" * 118)
print("WHEN DOES SALARY GROWTH FLATTEN? - analytics_clean.csv, salary = salary_avg_lakh band midpoint")
print("=" * 118)
bands = sorted(a.salary_avg_lakh.unique())
print(f"n = {len(a)}   experience = avg_exp (min/max mid)   salary has only {len(bands)} distinct values: "
      + ", ".join(f"{v:g}L" for v in bands))

# ------------------------------------------------- 1) percentiles per experience year
a["yr"] = a["avg_exp"].round().astype(int)
a["yr_group"] = np.where(a["yr"] >= 21, "21+", a["yr"].astype(str))
per_year = (
    a.groupby("yr_group")["salary_avg_lakh"]
    .agg(n="count", p25=lambda s: s.quantile(0.25), median="median", p75=lambda s: s.quantile(0.75))
)
order = [str(i) for i in range(21)] + ["21+"]
per_year = per_year.reindex([o for o in order if o in per_year.index])

print("\n" + "=" * 118)
print("SALARY PER EXPERIENCE YEAR (lakhs/year, band midpoints)")
print("=" * 118)
print(f"{'years':>7}{'n':>8}{'p25':>10}{'median':>10}{'p75':>10}")
for idx, r in per_year.iterrows():
    print(f"{idx:>7}{int(r.n):>8}{r.p25:>10.1f}{r['median']:>10.1f}{r.p75:>10.1f}")

# ------------------------------------------------- 2) quadratic quantile regression
def q_slope_pct(params, years):
    """% pay change per year at a given experience level (model is quadratic in years/10)."""
    xx = np.asarray(years) / 10.0
    return (np.exp((params[1] + 2 * params[2] * xx) / 10.0) - 1) * 100


QX = np.c_[np.ones_like(x), x, x**2]
qres = {}
print("\n" + "=" * 118)
print("QUANTILE REGRESSION: log(salary) ~ experience + experience^2   (q = 0.1, 0.5, 0.9)")
print("=" * 118)
print(f"{'quantile':>10}{'turning pt (yr)':>18}{'%/yr @0yr':>11}{'@5yr':>9}{'@10yr':>9}{'@15yr':>9}{'exp^2 p':>10}")
for q in (0.1, 0.5, 0.9):
    m = QuantReg(y, QX).fit(q=q, max_iter=5000)
    assert np.isfinite(m.params).all()
    p2_p = m.pvalues[2]
    vertex = -m.params[1] / (2 * m.params[2]) * 10.0
    s = q_slope_pct(m.params, [0, 5, 10, 15])
    qres[q] = {"vertex": vertex, "slopes": s, "p2": p2_p}
    print(f"{q:>10.1f}{vertex:>18.1f}{s[0]:>10.1f}%{s[1]:>8.1f}%{s[2]:>8.1f}%{s[3]:>8.1f}%{p2_p:>10.3f}")
print("turning point = experience where the fitted quadratic's yearly growth reaches 0%")

# ------------------------------------------------- 3) piecewise regression, grid search
def piece_fit(e_arr, y_arr, bp):
    X = np.c_[np.ones_like(e_arr), e_arr, np.maximum(0.0, e_arr - bp)]
    with np.errstate(all="ignore"):
        beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    assert np.isfinite(beta).all()
    resid = y_arr - X @ beta
    return beta, float(resid @ resid)


def grid_search(e_arr, y_arr, grid):
    fits = [(piece_fit(e_arr, y_arr, bp)[1], bp) for bp in grid]
    assert all(np.isfinite(s) for s, _ in fits)
    sse, bp = min(fits)
    beta, _ = piece_fit(e_arr, y_arr, bp)
    return bp, beta, sse, sorted(fits)


bp_hat, beta_hat, sse_hat, ranked = grid_search(e, y, GRID)
r2 = 1 - sse_hat / float(((y - y.mean()) ** 2).sum())
before = (np.exp(beta_hat[1]) - 1) * 100
after = (np.exp(beta_hat[1] + beta_hat[2]) - 1) * 100

rng = np.random.default_rng(42)
n = len(e)
boot_bp = np.empty(B_BOOT)
for i in range(B_BOOT):
    idx = rng.integers(0, n, n)
    boot_bp[i] = grid_search(e[idx], y[idx], GRID)[0]
lo, hi = np.percentile(boot_bp, [2.5, 97.5])

print("\n" + "=" * 118)
print(f"PIECEWISE REGRESSION - grid search over {GRID[0]:.1f}-{GRID[-1]:.1f} years in 0.5-yr steps "
      f"({len(GRID)} candidates), least squares on log(salary)")
print("=" * 118)
print("best 5 breakpoints by residual sum of squares:")
for sse, bp in ranked[:5]:
    print(f"   breakpoint {bp:>4.1f} yr   SSE {sse:>9.3f}")
print(f"\nBEST BREAKPOINT = {bp_hat:.1f} YEARS   (model R2 = {r2:.3f})")
print(f"growth BEFORE the break: {before:+.1f}% per year")
print(f"growth AFTER  the break: {after:+.1f}% per year   (slope change {after - before:+.1f} pp/yr)")
print(f"bootstrap 95% CI for the breakpoint (B = {B_BOOT} resamples): "
      f"{lo:.1f} - {hi:.1f} years  (median of bootstraps {np.median(boot_bp):.1f})")
print("bootstrap distribution: " + ", ".join(
    f"{val:.1f} yr = {100 * np.mean(boot_bp == val):.0f}%" for val in np.unique(boot_bp)))

# ------------------------------------------------- 4) piecewise per quantile
def pinball_loss(y_arr, resid, q):
    return float(np.mean(np.maximum(q * resid, (q - 1) * resid)))


def q_piecewise(e_arr, y_arr, q, grid):
    best = None
    for bp in grid:
        X = np.c_[np.ones_like(e_arr), e_arr / 10.0, np.maximum(0.0, e_arr / 10.0 - bp / 10.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = QuantReg(y_arr, X).fit(q=q, max_iter=5000)
        if not np.isfinite(m.params).all():
            continue
        loss = pinball_loss(y_arr, y_arr - X @ m.params, q)
        if best is None or loss < best[0]:
            best = (loss, bp, m.params)
    loss, bp, params = best
    b1, b2 = params[1], params[2]
    return {
        "bp": bp,
        "before": (np.exp(b1 / 10.0) - 1) * 100,
        "after": (np.exp((b1 + b2) / 10.0) - 1) * 100,
        "loss": loss,
    }


print("\n" + "=" * 118)
print("PIECEWISE BREAKPOINT BY QUANTILE - same grid search, pinball loss instead of least squares")
print("=" * 118)
print(f"{'quantile':>10}{'breakpoint (yr)':>17}{'%/yr before':>14}{'%/yr after':>13}")
qpiece = {}
for q in (0.1, 0.5, 0.9):
    res = q_piecewise(e, y, q, GRID)
    qpiece[q] = res
    print(f"{q:>10.1f}{res['bp']:>17.1f}{res['before']:>13.1f}%{res['after']:>12.1f}%")
print(f"\nmedian breaks at {qpiece[0.5]['bp']:.1f} yr, the 90th percentile at {qpiece[0.9]['bp']:.1f} yr "
      f"-> difference {qpiece[0.9]['bp'] - qpiece[0.5]['bp']:+.1f} years")
print(f"share of postings sitting in the top band (37.5L, 25-50L raw): "
      f"{100 * (a.salary_avg_lakh == 37.5).mean():.1f}% overall, "
      f"{100 * (a.loc[a.yr >= 10, 'salary_avg_lakh'] == 37.5).mean():.1f}% at 10+ years, "
      f"{100 * (a.loc[a.yr >= 17, 'salary_avg_lakh'] == 37.5).mean():.1f}% at 17+ years")

# ------------------------------------------------- final answer
med = a[a.yr <= 3]["salary_avg_lakh"].median()
med_mid = a[(a.yr >= 4) & (a.yr <= 5)]["salary_avg_lakh"].median()
med_69 = a[(a.yr >= 6) & (a.yr <= 9)]["salary_avg_lakh"].median()
med_10 = a[(a.yr >= 10)]["salary_avg_lakh"].median()

print("\n" + "=" * 118)
print("FINAL ANSWER")
print("=" * 118)
print(
    f"1. For a fresher: your pay climbs fastest in your first few years - about {before:+.0f}% a year "
    f"up to {bp_hat:.1f} years of experience (95% CI {lo:.1f}-{hi:.1f} years), then only about "
    f"{after:+.0f}% a year after that, so the big raises happen early.\n"
    f"2. In rupees the median salary jumps from {med:g}L (0-3 years) to {med_mid:g}L (4-5 years) to "
    f"{med_69:g}L (6-9 years) - about {med_69 / med:.0f}x while you are junior - then only crawls from "
    f"{med_69:g}L to {med_10:g}L across 10-20 years, so a fresher wins most of the pay growth in the "
    f"first ~{bp_hat:.0f} years.\n"
    f"3. Top earners do NOT flatten at the same point as the median: the median's curve breaks at "
    f"{qpiece[0.5]['bp']:.1f} years (all-postings fit: {bp_hat:.1f} years) while the 90th percentile "
    f"keeps rising until {qpiece[0.9]['bp']:.1f} years before flattening near 0% a year - but that "
    f"flattening is partly an artefact, because {100 * (a.loc[a.yr >= 10, 'salary_avg_lakh'] == 37.5).mean():.0f}% "
    f"of 10+ year postings already sit in the top 25-50L band, so pay above it is never recorded.\n"
)
print("=" * 118)

per_year.round(1).to_csv(ROOT / "experience_quantiles.csv")
pd.DataFrame(
    {
        "quantile": ["0.1", "0.5", "0.9"],
        "quadratic_turning_point_yr": [qres[0.1]["vertex"], qres[0.5]["vertex"], qres[0.9]["vertex"]],
        "piecewise_breakpoint_yr": [qpiece[0.1]["bp"], qpiece[0.5]["bp"], qpiece[0.9]["bp"]],
        "pct_per_yr_before": [qpiece[0.1]["before"], qpiece[0.5]["before"], qpiece[0.9]["before"]],
        "pct_per_yr_after": [qpiece[0.1]["after"], qpiece[0.5]["after"], qpiece[0.9]["after"]],
    }
).to_csv(ROOT / "experience_breakpoints.csv", index=False)
print("saved -> experience_quantiles.csv, experience_breakpoints.csv")
