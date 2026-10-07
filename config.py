"""Central settings. Edit this file ONLY if your real files behave differently."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = Path(os.environ.get("SVL_DATA", ROOT / "data" / "raw"))   # where your 4 files live
CLEAN = ROOT / "data" / "clean"
ART = ROOT / "artifacts"
ANA = ART / "analytics"
for _p in (CLEAN, ART, ANA):
    _p.mkdir(parents=True, exist_ok=True)

# A file is matched if this keyword appears in its name (case-insensitive).
# Example: "Analytics Jobs.csv" matches "analytics".  Rename the files or edit the keywords if needed.
FILE_KEYS = {
    "analytics": "analytics",   # Analytics Jobs      (~15,800 rows)
    "dsjobs":    "science",     # Data Science Jobs   (~1,600 rows)
    "jds":       "jds",         # JDS Skill Traits    (~140 rows)
    "sds":       "sds",         # SDS Personality     (~160 rows)
}

TOP_N_SKILLS = 30          # how many skills become 0/1 columns
MIN_LOCATION_COUNT = 50    # cities with fewer postings are grouped as "Other"
MAX_EXPERIENCE_YEARS = 40  # anything above is treated as a typo
IQR_FENCE = 3.0            # salary outlier fence on log-salary (3 = only extreme values)
RANDOM_STATE = 42
