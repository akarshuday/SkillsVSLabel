# Skills vs Labels - training pipeline
1. Put your 4 raw CSV files in `data/raw/` (names must contain: analytics, science, jds, sds).
2. `pip install -r requirements.txt`
3. `python run_all.py`   (or `python run_all.py --sample` to practice with fake data first)
Outputs: `data/clean/*.csv`, `artifacts/*.joblib`, `artifacts/metrics.json`, `artifacts/cleaning_log.csv`, `artifacts/analytics/*.csv`
