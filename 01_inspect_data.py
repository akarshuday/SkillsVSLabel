"""STEP 1 - Look at the raw data BEFORE cleaning. Writes artifacts/data_audit.txt
Paste the printed output to your agent / mentor to adapt the cleaning rules."""
import io, sys
import pandas as pd
from config import ART, RAW, FILE_KEYS
from common import load_raw

buf = io.StringIO()
def out(*a):
    print(*a); print(*a, file=buf)

out(f"Reading files from: {RAW}\n")
for key in FILE_KEYS:
    try:
        df, path = load_raw(key)
    except FileNotFoundError as e:
        out(f"[MISSING] {key}: {e}\n"); continue
    out("=" * 78); out(f"{key.upper()}  <-  {path.name}"); out("=" * 78)
    out(f"shape: {df.shape} | duplicate rows: {df.duplicated().sum()}")
    out("\n-- columns, type, missing, unique --")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                         "missing_%": (df.isna().mean() * 100).round(1), "unique": df.nunique()})
    out(info.to_string())
    out("\n-- first 5 rows --"); out(df.head().to_string(max_colwidth=40))
    num = df.select_dtypes("number")
    if not num.empty:
        out("\n-- numeric summary --"); out(num.describe().T.round(2).to_string())
    for c in df.select_dtypes(exclude="number").columns:
        if df[c].nunique() <= 60 or c in ("location", "key_skills", "salary", "experience"):
            out(f"\n-- top values of '{c}' --"); out(df[c].astype(str).str[:50].value_counts().head(15).to_string())
    out("")
(ART / "data_audit.txt").write_text(buf.getvalue())
print("\nSaved:", ART / "data_audit.txt")
