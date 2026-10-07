"""Step 2 - Clean the other 3 raw files.
Data Science Jobs -> numbers + company merge + outlier drop.
JDS / SDS          -> range rules, duplicates, 0/1 targets.
Writes data/clean/{dsjobs,jds,sds}_clean.csv and prints log + class balance."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw"
CLEAN = Path(__file__).resolve().parent / "data" / "clean"

IQR_FENCE = 3.0   # outlier rule: avg_salary beyond Q1/Q3 +/- 3 * IQR (raw lakhs)
SD_FENCE = 4.0    # Big Five rule: values further than 4 SD from the mean are typos
JDS_SKILLS = ["big_data_skills", "maths_stats_skills", "coding_skills",
              "ai_and_ml_skills", "dashboard_and_storytelling_skills"]
SDS_TRAITS = ["neuroticism", "extraversion", "openness_to_experience",
              "agreeableness", "conscientiousness"]

# One employer showing up under a short name / subsidiary / different casing.
COMPANY_ALIASES = {
    "atos syntel": "Atos", "bnp paribas india solutions": "BNP Paribas",
    "barclays global service centre": "Barclays", "barclays shared services": "Barclays",
    "dunnhumby it services india": "Dunnhumby", "dell emc": "Dell",
    "hashedin by deloitte": "Deloitte", "wipro digital": "Wipro",
    "ge aviation": "GE", "ge digital": "GE", "ge healthcare": "GE",
    "ge power": "GE", "ge transportation": "GE",
    "gartner india research and advisory services": "Gartner",
    "harman": "Harman", "harman connected services": "Harman",
    "hcl technologies bpo services": "HCL Technologies", "hsbc technology": "HSBC",
    "hitachi vantara": "Hitachi", "infosys bpm": "Infosys", "maersk line": "Maersk",
    "nailbiter research": "Nailbiter", "optum global solutions": "Optum",
    "oracle financial services software": "Oracle", "s&p global market intelligence": "S&P Global",
    "samsung research": "Samsung", "citibank": "Citi", "citigroup": "Citi",
    "citicorp": "Citi", "exl india": "EXL Service", "nielseniq": "Nielsen",
    "franklin templeton international services": "Franklin Templeton Investments",
    "colgate global business services": "Colgate Palmolive",
    "standard chartered bank": "Standard Chartered Bank",
    "standard chartered global business services": "Standard Chartered Bank",
    "ecolab digital center": "Ecolab", "ecolab food safety & hygiene solutions": "Ecolab",
    "reliance jio": "Reliance Jio", "whitehat jr": "Whitehat Jr",
}

LOG = []  # (file, issue, rows/cells affected, action)


def log(fname, issue, n, action):
    LOG.append((fname, int(n), issue, action))


def to_number(series):
    # "7.8L" / "₹ 1,60,000" / "12" -> float, anything unreadable becomes NaN.
    return pd.to_numeric(series.astype("string").str.replace(r"[^\d.]", "", regex=True),
                         errors="coerce")


def to_binary(series, high=("high", "1", "yes", "true")):
    # High/1/yes -> 1, Low/0/no -> 0; already-numeric 0/1 columns pass straight through.
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype("Int64").where(series.notna())
    t = series.astype("string").str.strip().str.lower()
    return t.map(lambda v: 1 if v in high else (0 if v in ("low", "0", "no", "false") else pd.NA)).astype("Int64")


# ======================================================================
# Data Science Jobs
# ======================================================================
def clean_dsjobs():
    F = "DataScience Jobs.csv"
    df = pd.read_csv(RAW / F)
    n0 = len(df)

    # 1. salary columns are text like "7.8L" -> plain numbers in lakhs.
    cells, failed = 0, 0
    for c in ["avg_salary", "min_salary", "max_salary"]:
        was_text = not pd.api.types.is_numeric_dtype(df[c])
        conv = to_number(df[c])
        cells += int(conv.notna().sum()) if was_text else 0
        failed += int((df[c].notna() & conv.isna()).sum())
        df[c] = conv
    log(F, "avg/min/max_salary stored as text ('7.8L')", cells,
        "stripped the L, stored as float lakhs/year")
    log(F, "salary text that could not be parsed", failed, "set to NaN")
    for c in ["min_experience", "num_of_jobs"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            n = int(to_number(df[c]).notna().sum())
            df[c] = to_number(df[c])
            log(F, f"{c} stored as text", n, "converted to number")

    # 2. min salary above max salary is a typo -> swap the two values.
    swap = (df["min_salary"] > df["max_salary"]).fillna(False)
    df.loc[swap, ["min_salary", "max_salary"]] = df.loc[swap, ["max_salary", "min_salary"]].values
    log(F, "min_salary > max_salary", swap.sum(),
        f"checked all {n0} rows, swapped the pair where found")

    # 3. one employer, many spellings -> single canonical company_name.
    before = df["company_name"].nunique()
    clean = df["company_name"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    df["company_name"] = clean.str.lower().map(COMPANY_ALIASES).fillna(clean)
    changed = int(df["company_name"].ne(clean).sum())
    log(F, "company_name variants (Barclays Shared Services, GE Healthcare, ...)",
        changed,
        f"merged into parent brand, {before} -> {df['company_name'].nunique()} companies")

    # 4. exact reposts (same posting under a new reference_no) -> keep the first.
    id_cols = ["reference_no"]
    dupe = df.duplicated(subset=[c for c in df.columns if c not in id_cols])
    df = df.loc[~dupe].copy()
    log(F, "duplicate posting (identical except reference_no)", dupe.sum(), "dropped, keep='first'")

    # 5. salary far outside the pack -> drop, so a few 80 LPA rows do not steer the model.
    q1, q3 = df["avg_salary"].quantile([0.25, 0.75])
    step = IQR_FENCE * (q3 - q1)
    lo, hi = q1 - step, q3 + step
    out = (df["avg_salary"] < lo) | (df["avg_salary"] > hi)
    log(F, f"avg_salary outside {IQR_FENCE}xIQR [{lo:.1f}-{hi:.1f} LPA]", out.sum(), "dropped")
    df = df.loc[~out].copy()

    out_path = CLEAN / "dsjobs_clean.csv"
    df.to_csv(out_path, index=False)
    return df, n0, F, out_path


# ======================================================================
# JDS Skill Traits
# ======================================================================
def clean_jds():
    F = "JDS Skill Traits.xlsx"
    df = pd.read_excel(RAW / F)
    n0 = len(df)

    # 1. tidy column names so the model script can use them directly.
    ren = {c: re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns}
    renamed = [f"{a} -> {b}" for a, b in ren.items() if a != b]
    df = df.rename(columns=ren)

    # 2. the scale is defined 1-5, so 0 or 9 is a typo -> blank it out.
    skills = df[JDS_SKILLS]
    oor = ((skills < 1) | (skills > 5)).any(axis=1)
    df[JDS_SKILLS] = skills.where((skills >= 1) & (skills <= 5))
    log(F, "skill score outside 1-5", oor.sum(), "set to missing (scale only allows 1-5)")

    # 3. same person/same scores listed twice (id differs) -> drop, keep the first.
    body = JDS_SKILLS + ["salary_hike_high_or_low"]
    dupe = df.duplicated(subset=body)
    df = df.loc[~dupe].copy()
    log(F, "duplicate row (same 5 scores + target, id ignored)", dupe.sum(), "dropped, keep='first'")

    # 4. rows missing a score or the target -> drop (only ~140 rows, imputation invents data).
    miss = df[body].isna().any(axis=1)
    df = df.loc[~miss].copy()
    log(F, "row missing a skill score or target", miss.sum(), "dropped")

    # 5. target High/Low -> 1/0.
    col = "salary_hike_high_or_low"
    raw = df[col].astype("string").str.strip().str.lower()
    df[col] = to_binary(df[col]).astype(int)
    changed_rows = int(df[col].astype("string").ne(raw).sum())
    log(F, "target High/Low -> 1/0", changed_rows,
        "already 0/1, cast to int" if changed_rows == 0
        else f"raw values {sorted(raw.unique())} mapped to 1/0")

    out_path = CLEAN / "jds_clean.csv"
    df[["id"] + JDS_SKILLS + ["salary_hike_high_or_low"]].to_csv(out_path, index=False)
    return df, n0, F, out_path, renamed


# ======================================================================
# SDS Personality Traits
# ======================================================================
def clean_sds():
    F = "SDS Personality Traits.xlsx"
    df = pd.read_excel(RAW / F)
    n0 = len(df)

    # 1. tidy names (" extraversion", "success_ classification_ high_low") + numeric traits.
    ren = {c: re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns}
    ren = {k: ("success_high_low" if v.startswith("success") else v) for k, v in ren.items()}
    renamed = [f"{a} -> {b}" for a, b in ren.items() if a != b]
    df = df.rename(columns=ren)

    bad_cells = 0
    for c in SDS_TRAITS:
        n = pd.to_numeric(df[c], errors="coerce").isna().sum() - df[c].isna().sum()
        bad_cells += int(n)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log(F, "trait stored as text", bad_cells, "converted to number, unreadable -> NaN")

    # 2. anything beyond 4 SD from the mean is a keying error -> blank it, drop the row.
    z = (df[SDS_TRAITS] - df[SDS_TRAITS].mean()) / df[SDS_TRAITS].std(ddof=0)
    ext = (z.abs() > SD_FENCE).any(axis=1)
    if ext.any():                       # only touch the frame when something is wrong
        df.loc[ext, SDS_TRAITS] = np.nan
    log(F, f"trait value more than {SD_FENCE:g} SD from the mean", ext.sum(),
        "value removed, row dropped (milder 3-4 SD values kept)")

    # 3. rows left without a full trait profile -> drop.
    miss = df[SDS_TRAITS].isna().any(axis=1)
    df = df.loc[~miss].copy()
    log(F, "row missing a trait value", miss.sum(), "dropped (no imputation on 161 rows)")

    # 4. target High/Low -> 1/0.
    col = "success_high_low"
    raw = df[col].astype("string").str.strip().str.lower()
    df[col] = to_binary(df[col]).astype(int)
    changed_rows = int(df[col].astype("string").ne(raw).sum())
    log(F, "target High/Low -> 1/0", changed_rows,
        "already 0/1, cast to int" if changed_rows == 0
        else f"raw values {sorted(raw.unique())} mapped to 1/0")

    out_path = CLEAN / "sds_clean.csv"
    df[["id"] + SDS_TRAITS + ["success_high_low"]].to_csv(out_path, index=False)
    return df, n0, F, out_path, renamed


# ======================================================================
def print_log(results):
    print("=" * 96)
    print("CLEANING LOG")
    print("=" * 96)
    print(f"{'file':<28}{'issue':<50}{'n':>5}   action")
    print("-" * 96)
    for fname, n, issue, action in LOG:
        print(f"{fname:<28}{issue:<50}{n:>5}   {action}")
    print("-" * 96)
    for fname, n0, n1, path in results:
        print(f"{fname:<28}{'ROWS':<50}{n1:>5}   (was {n0}, dropped {n0 - n1}) -> {path.name}")
    print("=" * 96)


def print_balance(dsj, jds, sds):
    print("\nCLASS BALANCE")
    print("-" * 96)
    print("dsjobs_clean.csv  job_title (10 classes):")
    vc = dsj["job_title"].value_counts()
    for k, v in vc.items():
        print(f"    {k:<32}{v:>5}  ({100 * v / len(dsj):.1f}%)")
    for name, df, col in [("jds_clean.csv", jds, "salary_hike_high_or_low"),
                          ("sds_clean.csv", sds, "success_high_low")]:
        vc = df[col].value_counts().sort_index()
        n = len(df)
        print(f"{name:<18}{col} (1 = high/success):")
        for k, v in vc.items():
            print(f"    {k:<32}{v:>5}  ({100 * v / n:.1f}%)")
        print(f"    {'minority class':<32}{vc.min():>5}  "
              f"-> {'balanced' if vc.min() / n > 0.4 else 'IMBALANCED, report recall/AUC not accuracy'}")
    print("-" * 96)


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    dsj, n_ds, f_ds, p_ds = clean_dsjobs()
    jds, n_j, f_j, p_j, ren_j = clean_jds()
    sds, n_s, f_s, p_s, ren_s = clean_sds()

    print_log([(f_ds, n_ds, len(dsj), p_ds),
               (f_j, n_j, len(jds), p_j),
               (f_s, n_s, len(sds), p_s)])
    if ren_j or ren_s:
        print("renamed columns:")
        for r in ren_j + ren_s:
            print("   ", r)
    print_balance(dsj, jds, sds)
    print(f"saved -> {p_ds} | {p_j} | {p_s}")


if __name__ == "__main__":
    main()
