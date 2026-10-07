"""Step 0 - Raw data audit.
Loads the 4 raw files from data/raw/ and dumps a full profile into audit.txt."""

import importlib.util
import os
import sys
from pathlib import Path

# --- shadow guard ---------------------------------------------------------
# This file is named inspect.py, so numpy's `import inspect` can pick it up
# instead of the stdlib. When that happens we load the real stdlib inspect
# into sys.modules (numpy then gets the correct module) and do nothing else.
if __name__ == "inspect":
    _std = Path(os.__file__).resolve().parent / "inspect.py"
    _spec = importlib.util.spec_from_file_location("inspect", _std)
    _real = importlib.util.module_from_spec(_spec)
    sys.modules["inspect"] = _real
    _spec.loader.exec_module(_real)
else:
    import pandas as pd

    RAW = Path(__file__).resolve().parent / "data" / "raw"
    OUT = Path(__file__).resolve().parent / "audit.txt"
    MAX_CHARS = 120  # keeps long free-text cells readable in the report

    def load(path):
        # Two of the four files are Excel, so pick the reader by extension.
        return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

    def top_values(s, n=15):
        # Show the 15 most frequent values of a text column, shortened for readability.
        vc = s.astype("string").str.slice(0, MAX_CHARS).value_counts(dropna=False).head(n)
        return "\n".join(f"      {v!r}: {c}" for v, c in vc.items())

    def profile(name, df):
        # One block per file: shape -> dtypes -> missing -> duplicates -> head -> top text values.
        lines = [f"{'=' * 70}", f"FILE: {name}", f"{'=' * 70}"]
        lines.append(f"SHAPE: {df.shape[0]} rows x {df.shape[1]} cols\n")

        lines.append("DTYPES:")
        lines += [f"  {c}: {t}" for c, t in df.dtypes.items()]

        miss = df.isna().sum()
        lines.append("\nMISSING VALUES (column: n / %):")
        lines += [f"  {c}: {n} / {round(100 * n / len(df), 2)}%" for c, n in miss.items()]
        lines.append(f"  TOTAL missing cells: {int(miss.sum())}")

        lines.append(f"\nDUPLICATE ROWS: {int(df.duplicated().sum())}")
        lines.append(f"UNIQUE ROWS: {int(df.drop_duplicates().shape[0])}")

        lines.append("\nFIRST 5 ROWS:")
        lines.append(df.head(5).to_string())

        text_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)
        lines.append(f"\nTEXT COLUMNS ({len(text_cols)}): {text_cols}")
        for c in text_cols:
            lines.append(f"\n  TOP 15 VALUES of '{c}':")
            lines.append(top_values(df[c]))

        lines.append("")
        return "\n".join(lines)

    def main():
        files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".xls"))
        assert files, f"No CSV/XLSX files found in {RAW}"

        # Read every raw file once, then profile it and collect the text.
        report = "\n".join(profile(f.name, load(f)) for f in files)

        OUT.write_text(report, encoding="utf-8")
        print(report)                  # same report on screen
        print(f"\nSaved -> {OUT}")     # and on disk for the repo

    if __name__ == "__main__":
        main()
