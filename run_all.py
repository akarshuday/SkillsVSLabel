"""Runs inspect -> clean -> train in order.   python run_all.py            (your real files in data/raw)
                                              python run_all.py --sample   (fake practice data)"""
import os, subprocess, sys
if "--sample" in sys.argv:
    subprocess.run([sys.executable, "make_sample_data.py"], check=True)
    os.environ["SVL_DATA"] = "data/sample"
for s in ("01_inspect_data.py", "02_clean_data.py", "03_train_models.py"):
    print(f"\n{'=' * 20} {s} {'=' * 20}")
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"Stopped: {s} failed. Copy the error to your agent (see the guide).")
print("\nDone. Models are in artifacts/, results tables in artifacts/analytics/")
