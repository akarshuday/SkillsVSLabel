"""STEP 2 - Clean all 4 files. Writes data/clean/*.csv and artifacts/cleaning_log.csv
Every change is logged (issue, rows affected, action, why) -> paste the log into your report (25 marks)."""
import re, difflib
import numpy as np, pandas as pd
from config import *
from common import load_raw, find_col, first_number

LOG = []
def log(step, column, rows, action, why):
    LOG.append(dict(file=step, column=column, rows_affected=int(rows), action=action, justification=why))
    print(f"  [{step}] {column}: {int(rows)} rows -> {action}")

NUM = re.compile(r"\d+(?:\.\d+)?")

def parse_range(x):
    """'3-5 yrs' -> (3,5)   '5 yrs' -> (5,5)   'NA' -> (nan,nan)"""
    if pd.isna(x): return (np.nan, np.nan)
    n = [float(v) for v in NUM.findall(str(x).replace(",", ""))]
    if not n: return (np.nan, np.nan)
    return (min(n[:2]), max(n[:2]))

def parse_salary_lakhs(x):
    """'6-10 Lacs PA' -> (6,10). 'Not Disclosed' -> nan. Crores x100. Big plain numbers are rupees -> lakhs."""
    if pd.isna(x): return (np.nan, np.nan)
    s = str(x).lower().replace(",", "")
    n = [float(v) for v in NUM.findall(s)]
    if not n: return (np.nan, np.nan)
    lo, hi = min(n[:2]), max(n[:2])
    if "cr" in s: mult = 100.0
    elif "lac" in s or "lakh" in s or "lpa" in s: mult = 1.0
    elif hi >= 1000: mult = 1 / 100000.0
    else: mult = 1.0
    return (lo * mult, hi * mult)

TITLE_RULES = [("Manager / Lead", ["manager", "lead", "head", "director", "principal", "architect"]),
               ("Data Engineer", ["data engineer", "etl", "big data", "hadoop", "spark"]),
               ("ML / AI Engineer", ["machine learning", "ml ", "ai ", "deep learning", "nlp", "computer vision"]),
               ("Data Scientist", ["data scientist", "scientist"]),
               ("BI / Reporting", ["business intelligence", "bi ", "reporting", "tableau", "power bi", "visuali"]),
               ("Data Analyst", ["analyst", "analytics", "analysis"])]
def title_family(t):
    s = " " + str(t).lower() + " "
    for fam, keys in TITLE_RULES:
        if any(k in s for k in keys): return fam
    return "Other"

LOC_ALIAS = {"bengaluru": "Bangalore", "banglore": "Bangalore", "bombay": "Mumbai", "new delhi": "Delhi",
             "delhi ncr": "Delhi", "ncr": "Delhi", "gurgaon": "Gurugram", "madras": "Chennai", "calcutta": "Kolkata"}
SKILL_ALIAS = {"ml": "machine learning", "ai": "artificial intelligence", "powerbi": "power bi", "power-bi": "power bi",
               "py": "python", "tf": "tensorflow", "dl": "deep learning", "viz": "data visualization",
               "data visualisation": "data visualization", "ms excel": "excel", "advanced excel": "excel",
               "scikit-learn": "scikit learn", "sklearn": "scikit learn", "natural language processing": "nlp"}
def split_skills(x):
    if pd.isna(x): return []
    parts = re.split(r"[|,;\n]+", str(x).lower())
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p and p not in ("na", "nan", "none"): out.append(SKILL_ALIAS.get(p, p))
    return list(dict.fromkeys(out))

# --------------------------------------------------------------------------------------
def clean_analytics():
    print("\n[1/4] Analytics Jobs")
    df, path = load_raw("analytics"); F = "Analytics Jobs"
    print(f"  loaded {len(df)} rows from {path.name}")
    exp_c, sal_c, loc_c, sk_c = find_col(df, "exp"), find_col(df, "sal"), find_col(df, "loc"), find_col(df, "skill")
    des_c = find_col(df, "desig", required=False) or find_col(df, "title")

    key = [c for c in df.columns if c not in ("s_no", "id", "index")]
    d = df.duplicated(subset=key).sum(); df = df.drop_duplicates(subset=key).reset_index(drop=True)
    log(F, "all", d, "dropped exact duplicate postings", "same posting repeated would double-count its salary and skills")

    # experience
    r = df[exp_c].apply(parse_range)
    df["exp_min"] = [a for a, b in r]; df["exp_max"] = [b for a, b in r]; df["avg_exp"] = (df.exp_min + df.exp_max) / 2
    bad = df.avg_exp.isna().sum() + (df.avg_exp > MAX_EXPERIENCE_YEARS).sum()
    df.loc[df.avg_exp > MAX_EXPERIENCE_YEARS, ["exp_min", "exp_max", "avg_exp"]] = np.nan
    df["exp_imputed"] = df.avg_exp.isna().astype(int)
    med = df.avg_exp.median(); df["avg_exp"] = df.avg_exp.fillna(med)
    log(F, exp_c, bad, f"text converted to numbers; unreadable/absurd values replaced by median ({med:.1f} yrs) + flag exp_imputed",
        "'3-5 yrs' is text; a model needs numbers; median is robust to outliers")

    # salary (lakhs per annum)
    r = df[sal_c].apply(parse_salary_lakhs)
    df["sal_min"] = [a for a, b in r]; df["sal_max"] = [b for a, b in r]; df["avg_salary"] = (df.sal_min + df.sal_max) / 2
    df.loc[df.avg_salary <= 0, "avg_salary"] = np.nan
    df["salary_missing"] = df.avg_salary.isna().astype(int)
    log(F, sal_c, df.salary_missing.sum(), "no number found ('Not disclosed'); kept in analytics_all.csv with flag salary_missing, excluded from training",
        "cannot guess pay; missingness itself is analysed in the 'pay transparency' module")

    # location
    base = df[loc_c].astype(str).str.split(r"[,/;|(\-]", regex=True).str[0].str.strip().str.lower()
    base = base.replace({"nan": np.nan, "": np.nan}).map(lambda v: LOC_ALIAS.get(v, v.title()) if isinstance(v, str) else np.nan)
    cnt = base.value_counts(); canon = list(cnt[cnt >= max(MIN_LOCATION_COUNT, 5)].index)
    def fix(v):
        if not isinstance(v, str) or v in canon: return v
        m = difflib.get_close_matches(v, canon, n=1, cutoff=0.85)
        return m[0] if m else v
    fixed = base.map(fix); changed = (fixed != base).sum() + (base != df[loc_c]).sum()
    cnt2 = fixed.value_counts(); rare = fixed.isin(cnt2[cnt2 < MIN_LOCATION_COUNT].index)
    df["location"] = fixed.where(~rare, "Other").fillna("Unknown")
    log(F, loc_c, changed, "standardised case/spelling/aliases (Bengaluru->Bangalore), first city kept for multi-city entries",
        "same city spelled many ways would be treated as different cities")
    log(F, loc_c, rare.sum(), f"cities with < {MIN_LOCATION_COUNT} postings grouped as 'Other'", "tiny groups give unstable estimates")

    # title family
    df["title_family"] = df[des_c].map(title_family)
    log(F, des_c, len(df), f"grouped into {df.title_family.nunique()} title families by keyword rules", "hundreds of title spellings -> comparable groups")

    # skills
    lists = df[sk_c].map(split_skills); df["n_skills"] = lists.map(len)
    top = pd.Series([s for l in lists for s in l]).value_counts().head(TOP_N_SKILLS).index.tolist()
    flags = {"skill_" + re.sub(r"\W+", "_", s).strip("_"): lists.map(lambda l, s=s: int(s in l)) for s in top}
    df = pd.concat([df, pd.DataFrame(flags)], axis=1)
    log(F, sk_c, len(df), f"split into lists; top {len(top)} skills became 0/1 columns skill_*", "models need one column per skill")

    keep = ["s_no", des_c, "title_family", "job_type", "location", "avg_exp", "exp_min", "exp_max", "exp_imputed",
            "avg_salary", "sal_min", "sal_max", "salary_missing", "n_skills"] + list(flags)
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    if "job_description" in df.columns: keep.append("job_description")
    allrows = df[keep].copy()
    allrows.to_csv(CLEAN / "analytics_all.csv", index=False)

    ok = allrows[allrows.salary_missing == 0].copy()
    ls = np.log(ok.avg_salary); q1, q3 = ls.quantile([.25, .75]); iqr = q3 - q1
    out = (ls < q1 - IQR_FENCE * iqr) | (ls > q3 + IQR_FENCE * iqr)
    log(F, "avg_salary", out.sum(), f"removed extreme salaries beyond {IQR_FENCE} x IQR on log scale", "a typo like 999 lakhs would distort every average")
    ok = ok[~out]; ok.to_csv(CLEAN / "analytics_clean.csv", index=False)
    print(f"  -> analytics_clean.csv: {len(ok)} rows | analytics_all.csv: {len(allrows)} rows")

# --------------------------------------------------------------------------------------
def clean_dsjobs():
    print("\n[2/4] Data Science Jobs")
    df, path = load_raw("dsjobs"); F = "Data Science Jobs"
    print(f"  loaded {len(df)} rows from {path.name}")
    num_cols = [c for c in ["min_experience", "avg_salary", "min_salary", "max_salary", "num_of_jobs"] if c in df.columns]
    for c in num_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            n = df[c].isna().sum(); df[c] = first_number(df[c])
            log(F, c, df[c].isna().sum() - n, "text converted to number", "stored as text")
    d = df.duplicated(subset=[c for c in df.columns if c != "reference_no"]).sum()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "reference_no"]); log(F, "all", d, "dropped duplicates", "double counting")
    sw = (df.min_salary > df.max_salary).sum()
    df.loc[df.min_salary > df.max_salary, ["min_salary", "max_salary"]] = df.loc[df.min_salary > df.max_salary, ["max_salary", "min_salary"]].values
    log(F, "min/max_salary", sw, "swapped where min > max", "obvious entry error")
    miss = df[["avg_salary", "num_of_jobs", "company_name"]].isna().any(axis=1).sum()
    df = df.dropna(subset=["avg_salary", "num_of_jobs", "company_name"]); log(F, "avg_salary/num_of_jobs/company_name", miss, "dropped rows missing key fields", "cannot analyse without them")
    df["company_name"] = df.company_name.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["company_key"] = df.company_name.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    disp = df.groupby("company_key").company_name.agg(lambda s: s.value_counts().index[0])
    merged = df.company_name.nunique() - df.company_key.nunique()
    df["company_name"] = df.company_key.map(disp)
    log(F, "company_name", merged, "merged spelling/case variants of the same company", "'TCS' and 'tcs' are one company")
    q1, q3 = df.avg_salary.quantile([.25, .75]); f = IQR_FENCE * (q3 - q1)
    out = ((df.avg_salary < q1 - f) | (df.avg_salary > q3 + f)); log(F, "avg_salary", out.sum(), "removed extreme outliers (3 x IQR)", "protects averages")
    df = df[~out]; df.to_csv(CLEAN / "dsjobs_clean.csv", index=False)
    print(f"  -> dsjobs_clean.csv: {len(df)} rows")

# --------------------------------------------------------------------------------------
def to_binary(s, high_words=("high", "1", "yes", "true")):
    if pd.api.types.is_numeric_dtype(s): return (s > 0).astype(float).where(s.notna())
    t = s.astype(str).str.strip().str.lower()
    return t.map(lambda v: 1.0 if v in high_words else (0.0 if v in ("low", "0", "no", "false") else np.nan))

def clean_jds():
    print("\n[3/4] JDS Skill Traits (juniors)")
    df, path = load_raw("jds"); F = "JDS Skill Traits"
    print(f"  loaded {len(df)} rows from {path.name}")
    kws = {"big_data_skills": ("big", "data"), "maths_stats_skills": ("math",), "coding_skills": ("cod",),
           "ai_and_ml_skills": ("ai",), "dashboard_and_storytelling_skills": ("dash",)}
    ren = {find_col(df, *k): v for v, k in kws.items()}
    tgt = find_col(df, "hike"); ren[tgt] = "salary_hike_high_or_low"; df = df.rename(columns=ren)
    feats = list(kws)
    for c in feats: df[c] = pd.to_numeric(df[c], errors="coerce")
    oor = ((df[feats] < 1) | (df[feats] > 5)).any(axis=1).sum()
    df[feats] = df[feats].where((df[feats] >= 1) & (df[feats] <= 5)); log(F, "skill scores", oor, "scores outside 1-5 set to missing", "scale is defined as 1-5, so 9.0 is a typo")
    df["salary_hike_high_or_low"] = to_binary(df.salary_hike_high_or_low)
    idc = [c for c in df.columns if c == "id"]
    d = df.duplicated(subset=feats + ["salary_hike_high_or_low"]).sum(); df = df.drop_duplicates(subset=feats + ["salary_hike_high_or_low"]); log(F, "all", d, "dropped duplicate rows", "double counting")
    miss = df[feats + ["salary_hike_high_or_low"]].isna().any(axis=1).sum()
    df = df.dropna(subset=feats + ["salary_hike_high_or_low"]); log(F, "skills/target", miss, "dropped rows with missing values", "only ~140 rows, so imputing would invent data; dropping is safer")
    bal = df.salary_hike_high_or_low.mean(); log(F, "salary_hike_high_or_low", len(df), f"class balance: {bal:.0%} high / {1 - bal:.0%} low", "if very unbalanced report precision/recall, not only accuracy")
    df["salary_hike_high_or_low"] = df.salary_hike_high_or_low.astype(int)
    df[idc + feats + ["salary_hike_high_or_low"]].to_csv(CLEAN / "jds_clean.csv", index=False)
    print(f"  -> jds_clean.csv: {len(df)} rows")

def clean_sds():
    print("\n[4/4] SDS Personality (seniors)")
    df, path = load_raw("sds"); F = "SDS Personality"
    print(f"  loaded {len(df)} rows from {path.name}")
    kws = {"neuroticism": ("neuro",), "extraversion": ("extra",), "openness_to_experience": ("open",),
           "agreeableness": ("agree",), "conscientiousness": ("conscient",)}
    ren = {find_col(df, *k): v for v, k in kws.items()}
    tgt = find_col(df, "success"); ren[tgt] = "success_high_low"; df = df.rename(columns=ren)
    feats = list(kws)
    for c in feats: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["success_high_low"] = to_binary(df.success_high_low)
    z = (df[feats] - df[feats].mean()) / df[feats].std()
    ext = (z.abs() > 4).any(axis=1)
    df.loc[ext, feats] = df.loc[ext, feats].where(z.abs().loc[ext] <= 4)
    log(F, "Big Five scores", ext.sum(), "values more than 4 SD from the mean set to missing", "scores are normalised, so +11 is a typo; milder outliers (3-4 SD) are kept")
    d = df.duplicated(subset=feats + ["success_high_low"]).sum(); df = df.drop_duplicates(subset=feats + ["success_high_low"]); log(F, "all", d, "dropped duplicates", "double counting")
    miss = df[feats + ["success_high_low"]].isna().any(axis=1).sum()
    df = df.dropna(subset=feats + ["success_high_low"]); log(F, "traits/target", miss, "dropped rows with missing values", "small sample, so no imputation")
    bal = df.success_high_low.mean(); log(F, "success_high_low", len(df), f"class balance: {bal:.0%} high / {1 - bal:.0%} low", "check before choosing metrics")
    df["success_high_low"] = df.success_high_low.astype(int)
    idc = [c for c in df.columns if c == "id"]
    df[idc + feats + ["success_high_low"]].to_csv(CLEAN / "sds_clean.csv", index=False)
    print(f"  -> sds_clean.csv: {len(df)} rows")

if __name__ == "__main__":
    for fn in (clean_analytics, clean_dsjobs, clean_jds, clean_sds):
        try: fn()
        except FileNotFoundError as e: print("  SKIPPED:", e)
    pd.DataFrame(LOG).to_csv(ART / "cleaning_log.csv", index=False)
    print("\nCleaning log saved to artifacts/cleaning_log.csv")
