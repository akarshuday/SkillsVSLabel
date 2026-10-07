"""Step 1 - Clean the Analytics Jobs file for modelling.
Turns experience/salary/location/skills into numeric + grouped features and
writes data/clean/analytics_clean.csv with a printed cleaning log."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw" / "Analytics Jobs.csv"
CLEAN = Path(__file__).resolve().parent / "data" / "clean"
OUT = CLEAN / "analytics_clean.csv"

TOP_N_SKILLS = 30    # how many skills become 0/1 columns
MIN_CITY_COUNT = 50  # cities in fewer postings than this become "Other"
IQR_FENCE = 3.0      # extreme salary fence: Q1 - 3*IQR / Q3 + 3*IQR on log salary
JUNK_TOKENS = {"", "...", "..", ".", "-", "--", "n/a", "na", "none", "nil"}

# Spelling variants seen in the raw "location" column -> one canonical name.
CITY_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bengaluru ": "Bengaluru",
    "gurugram": "Gurgaon", "gurgaon": "Gurgaon",
    "bombay": "Mumbai", "mumbai": "Mumbai", "mumbai suburbs": "Mumbai",
    "calcutta": "Kolkata", "kolkata": "Kolkata",
    "madras": "Chennai", "chennai": "Chennai",
    "new delhi": "Delhi NCR", "delhi": "Delhi NCR", "delhi ncr": "Delhi NCR",
    "hyderabd": "Hyderabad", "hyderabad": "Hyderabad",
    "mysuru": "Mysore", "mysore": "Mysore",
    "nasik": "Nashik", "nashik": "Nashik",
    "panaji": "Panjim", "panjim": "Panjim",
    "thiruvananthapuram": "Trivandrum", "trivandrum": "Trivandrum",
    "vishakhapatnam": "Visakhapatnam", "visakhapatnam": "Visakhapatnam",
}

# title_family rules, checked top to bottom: the first match wins.
TITLE_RULES = [
    ("Manager", r"manager|director|\bhead\b|chief|vice president|\bvp\b|president|founder|\bcto\b|\bceo\b|owner"),
    ("ML Engineer", r"machine learning|\bml\b|deep learning|\bnlp\b|computer vision|artificial intelligence|\bai\b|chatbot|cognitive"),
    ("Data Engineer", r"data engineer|big data|data architect|hadoop|spark|\betl\b|data platform|database|data warehouse|data pipeline|data integration|informatica|teradata|airflow|nosql|data modell?er"),
    ("BI", r"business intelligence|\bbi\b|tableau|power bi|dashboard|report|visuali[sz]ation|qlik|microstrategy|cognos|\bmis\b"),
    ("Data Scientist", r"data scientist|scientist|data science|statistician|predictive|modell?er|pattern recognition"),
    ("Data Analyst", r"analyst|analytics|business analysis|data analysis"),
]

LOG = []  # (issue, rows affected, action)


def log(issue, rows, action):
    LOG.append((issue, int(rows), action))


# --- 1. experience ---------------------------------------------------------
def parse_experience(df):
    # Raw looks like "6-10 yrs", so one regex gives the min, max and midpoint.
    ex = df["experience"].astype("string")
    mn = ex.str.extract(r"(\d+)\s*(?:-|to)\s*\d+", expand=False)
    mx = ex.str.extract(r"\d+\s*(?:-|to)\s*(\d+)", expand=False).fillna(mn)
    df["min_exp"] = pd.to_numeric(mn, errors="coerce")
    df["max_exp"] = pd.to_numeric(mx, errors="coerce")
    df["avg_exp"] = ((df["min_exp"] + df["max_exp"]) / 2).round(2)
    bad = df["avg_exp"].isna()
    log("experience not in 'N-M yrs' format", bad.sum(), "min/max/avg_exp set to NaN")
    log("experience above 40 years (implausible)", (df["max_exp"] > 40).sum(),
        "flagged, left unchanged for review")
    return df


# --- 2. salary -------------------------------------------------------------
def parse_salary(df):
    # Raw looks like "10to15" (lakhs/year); split it into min, max and midpoint.
    band = df["salary"].astype("string").str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*$")
    lo = pd.to_numeric(band[0], errors="coerce")
    hi = pd.to_numeric(band[1], errors="coerce")
    df["salary_min_lakh"] = lo
    df["salary_max_lakh"] = hi
    df["salary_avg_lakh"] = ((lo + hi) / 2).round(2)
    # Anything undisclosed or unparseable keeps its numbers blank but gets a flag.
    df["salary_missing"] = (lo.isna() | hi.isna()).astype("uint8")
    log("salary undisclosed or not a 'NtoM' band", df["salary_missing"].sum(),
        "salary_missing=1, salary columns left NaN")
    return df


# --- 3. location -----------------------------------------------------------
def norm_city(tok):
    # Drop the "(Whitefield)" style suffix and fix known spellings.
    tok = re.sub(r"\s*\([^)]*\)", "", str(tok))
    tok = re.sub(r"\s+", " ", tok).strip(" .,-")
    key = tok.lower()
    if key in CITY_ALIASES:
        return CITY_ALIASES[key]
    return tok.title() if tok.isupper() or tok.islower() else tok


def clean_location(df):
    rows_changed = 0
    grouped = 0
    out = []
    for raw in df["location"].fillna(""):
        raw_list = [p.strip() for p in str(raw).split(",") if p.strip()]
        seen = []
        for part in raw_list:
            c = norm_city(part)
            if c not in seen:
                seen.append(c)
        if seen != raw_list:
            rows_changed += 1
        out.append(seen)

    # Count how many postings mention each city (each city counted once per row).
    counts = {}
    for seen in out:
        for c in seen:
            counts[c] = counts.get(c, 0) + 1
    rare = {c for c, n in counts.items() if n < MIN_CITY_COUNT}

    final = []
    for seen in out:
        if any(c not in rare for c in seen):
            keep = [c for c in seen if c not in rare]  # rare city dropped, big city kept
        elif seen:
            keep = ["Other"]                           # nothing but rare cities left
        else:
            keep = ["Other"]
        if keep != seen:
            grouped += 1
        final.append(", ".join(keep))

    df["location"] = final
    log("location spelling variants (Bangalore/Gurugram/…(area))", rows_changed,
        "standardised to one canonical city name")
    log(f"cities mentioned in <{MIN_CITY_COUNT} postings", grouped,
        'replaced with "Other" (rows with a major city keep it)')
    log("distinct locations after cleaning", df["location"].nunique(),
        "kept as-is; use top locations or primary city as a model feature")
    return df


# --- 4. job title family ---------------------------------------------------
def family_of(title):
    # Try the rules in order; an unmatched title lands in "Other".
    t = str(title).lower()
    for name, pattern in TITLE_RULES:
        if re.search(pattern, t):
            return name
    return "Other"


def add_title_family(df):
    df["title_family"] = df["job_desig"].map(family_of)
    other = df["title_family"].eq("Other")
    log("job_desig has no data/analyst/manager keyword", other.sum(),
        "title_family='Other' (titles kept unchanged in job_desig)")
    return df


# --- 5. skills -------------------------------------------------------------
def skill_name(tok):
    # "Business Analysis" -> skill_business_analysis
    name = re.sub(r"[^a-z0-9]+", "_", tok.lower()).strip("_")
    return f"skill_{name}"


def add_skills(df):
    junk = 0
    parsed = []
    for raw in df["key_skills"].fillna(""):
        toks = []
        for part in str(raw).split(","):
            p = part.strip().lower()
            if p in JUNK_TOKENS or p.startswith("."):
                junk += 1
            else:
                toks.append(p)
        parsed.append(toks)
    df["_skills"] = parsed

    counts = {}
    for toks in parsed:
        for t in set(toks):  # a skill counts once per posting
            counts[t] = counts.get(t, 0) + 1
    top = sorted(counts, key=lambda t: (-counts[t], t))[:TOP_N_SKILLS]

    for t in top:
        df[skill_name(t)] = df["_skills"].map(lambda toks, t=t: int(t in toks)).astype("uint8")

    covered = df[[skill_name(t) for t in top]].sum(axis=1).gt(0)
    log("junk/empty skill tokens ('...' etc.)", junk, "dropped before splitting")
    log(f"row has none of the top {TOP_N_SKILLS} skills", (~covered).sum(),
        "all skill_xxx = 0 for that row (kept)")
    df.drop(columns=["_skills"], inplace=True)
    return df, top


# --- 6. duplicates + salary outliers ---------------------------------------
def drop_duplicates_and_outliers(df):
    # Two postings can be identical except for s_no, so ignore the id column.
    id_cols = ["s_no"]
    dupe = df.duplicated(subset=[c for c in df.columns if c not in id_cols])
    df = df.loc[~dupe].copy()
    log("duplicate posting (identical except s_no)", dupe.sum(), "dropped, keep='first'")

    has_salary = df["salary_missing"].eq(0) & df["salary_avg_lakh"].gt(0)
    lg = np.log(df.loc[has_salary, "salary_avg_lakh"])
    q1, q3 = lg.quantile(0.25), lg.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - IQR_FENCE * iqr, q3 + IQR_FENCE * iqr
    out = has_salary & ((np.log(df["salary_avg_lakh"].clip(lower=1e-9)) < lo) |
                        (np.log(df["salary_avg_lakh"].clip(lower=1e-9)) > hi))
    log(f"salary outside {IQR_FENCE}xIQR of log salary "
        f"[{np.exp(lo):.2f}-{np.exp(hi):.2f} LPA]", out.sum(), "dropped")
    df = df.loc[~out].copy()
    return df


def print_log(n_raw, n_out):
    print("=" * 88)
    print(f"CLEANING LOG   rows in: {n_raw}   rows out: {n_out}   dropped: {n_raw - n_out}")
    print("=" * 88)
    print(f"{'issue':<52}{'rows':>8}   action")
    print("-" * 88)
    for issue, rows, action in LOG:
        print(f"{issue:<52}{rows:>8}   {action}")
    print("=" * 88)


def main():
    df = pd.read_csv(RAW)
    n_raw = len(df)

    df = parse_experience(df)
    df = parse_salary(df)
    df = clean_location(df)
    df = add_title_family(df)
    df, top_skills = add_skills(df)
    df = drop_duplicates_and_outliers(df)

    df = df[["s_no", "job_desig", "title_family", "job_type", "job_description",
             "location", "min_exp", "max_exp", "avg_exp",
             "salary_min_lakh", "salary_max_lakh", "salary_avg_lakh", "salary_missing",
             "key_skills"] + [skill_name(t) for t in top_skills]]

    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print_log(n_raw, len(df))
    print(f"top {TOP_N_SKILLS} skills: {', '.join(top_skills)}")
    print(f"title_family: {df['title_family'].value_counts().to_dict()}")
    print(f"saved -> {OUT}  ({df.shape[0]} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
