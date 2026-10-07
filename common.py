"""Small helpers shared by all scripts."""
import re
import numpy as np
import pandas as pd
from config import RAW, FILE_KEYS


def norm_cols(df):
    """'Maths-Stats Skills ' -> 'maths_stats_skills' so column names never cause bugs."""
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def find_file(key):
    kw = FILE_KEYS[key].lower()
    files = [p for p in RAW.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls") and kw in p.stem.lower()]
    if not files:
        raise FileNotFoundError(f"No file with '{kw}' in its name found in {RAW}. "
                                f"Put your files there or edit FILE_KEYS in config.py.")
    return sorted(files)[0]


def load_raw(key):
    p = find_file(key)
    df = pd.read_csv(p, encoding_errors="replace") if p.suffix.lower() == ".csv" else pd.read_excel(p)
    return norm_cols(df), p


def find_col(df, *keywords, required=True):
    """First column whose name contains ALL keywords."""
    for c in df.columns:
        if all(k in c for k in keywords):
            return c
    if required:
        raise KeyError(f"No column containing {keywords}. Columns are: {list(df.columns)}")
    return None


def first_number(series):
    """'3-5 yrs' -> 3.0 ; '12,000' -> 12000.0 ; text without digits -> NaN"""
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v >= 0]
    if len(v) == 0 or v.sum() == 0:
        return np.nan
    n = len(v)
    cum = np.cumsum(v)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n
