"""Creates FAKE practice files in data/sample/ with the same columns as the hackathon data dictionary.
Use them to test the whole pipeline before the real data arrives:   python run_all.py --sample
(The numbers are invented. Never report them as findings.)"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)
N = 6000

SKILLS = ["python", "sql", "machine learning", "deep learning", "tableau", "power bi", "excel", "r", "sas",
          "spark", "aws", "azure", "nlp", "statistics", "tensorflow", "hadoop", "data visualization",
          "etl", "java", "scala", "docker", "git", "pytorch", "mongodb", "looker", "airflow", "kafka",
          "computer vision", "forecasting", "a/b testing"]
SKILL_PAY = {s: 0.0 for s in SKILLS}
SKILL_PAY.update({"machine learning": .18, "deep learning": .14, "spark": .12, "aws": .10, "pytorch": .12,
                  "nlp": .09, "python": .05, "sql": .02, "excel": -.02, "kafka": .08})
TITLES = ["Data Analyst", "Senior Data Analyst", "Data Scientist", "Senior Data Scientist", "Machine Learning Engineer",
          "Data Engineer", "Business Intelligence Developer", "Analytics Manager", "Lead Data Scientist", "Research Analyst"]
TITLE_PAY = {"Data Analyst": 0, "Senior Data Analyst": .1, "Data Scientist": .12, "Senior Data Scientist": .25,
             "Machine Learning Engineer": .22, "Data Engineer": .15, "Business Intelligence Developer": .03,
             "Analytics Manager": .35, "Lead Data Scientist": .38, "Research Analyst": -.05}
CITIES = {"Bangalore": .20, "Mumbai": .08, "Delhi": .08, "Hyderabad": .10, "Pune": .05, "Chennai": .03,
          "Gurugram": .10, "Noida": .02, "Kolkata": -.08, "Ahmedabad": -.10, "Jaipur": -.14, "Chandigarh": -.10}
city_names = list(CITIES)
city_p = np.array([.24, .12, .12, .12, .10, .08, .07, .05, .03, .03, .02, .02]); city_p /= city_p.sum()
LOC_VARIANTS = {"Bangalore": ["Bangalore", "Bengaluru", "Bangalore/Bengaluru", "Banglore", "bangalore"],
                "Delhi": ["Delhi", "New Delhi", "Delhi/NCR", "delhi"], "Mumbai": ["Mumbai", "Mumbai (All Areas)", "Bombay"]}

rows = []
for i in range(N):
    t = rng.choice(TITLES); c = rng.choice(city_names, p=city_p)
    e_lo = int(rng.integers(0, 14)); e_hi = e_lo + int(rng.integers(1, 5)); e = (e_lo + e_hi) / 2
    k = int(rng.integers(3, 9)); sk = list(rng.choice(SKILLS, k, replace=False))
    log_sal = 1.1 + 0.9 * np.log1p(e) - 0.015 * e ** 1.2 / 3 + TITLE_PAY[t] + CITIES[c] + sum(SKILL_PAY[s] for s in sk) + rng.normal(0, .18)
    s = float(np.exp(log_sal)); lo = max(1, round(s * .85)); hi = max(lo + 1, round(s * 1.15))
    sal = f"{lo}-{hi} Lacs PA"
    if rng.random() < .22: sal = "Not Disclosed by Recruiter"
    elif rng.random() < .004: sal = "999-1500 Lacs PA"              # absurd outlier
    loc = c if c not in LOC_VARIANTS or rng.random() < .55 else rng.choice(LOC_VARIANTS[c])
    exp = f"{e_lo}-{e_hi} yrs" if rng.random() > .02 else "NA"
    seps = rng.choice(["|", ",", " , "])
    rows.append(dict(s_no=i + 1, experience=exp, job_description=f"We are hiring a {t} to work on data projects.",
                     job_desig=t, job_type=rng.choice(["Analytics", "Data Science", "BI", "Engineering"]),
                     key_skills=seps.join(sk).title() if rng.random() < .5 else seps.join(sk), location=loc, salary=sal))
an = pd.DataFrame(rows)
an = pd.concat([an, an.sample(40, random_state=1)])        # some duplicates
an.to_csv(OUT / "Analytics Jobs.csv", index=False)

# ---- Data Science Jobs (company level) ----
cos = [f"Company {chr(65 + i % 26)}{i // 26}" for i in range(260)]
jobs = np.maximum(1, (rng.pareto(1.2, 260) * 4).astype(int))
rows = []
for co, nj in zip(cos, jobs):
    base = rng.normal(14, 4) - 0.8 * np.log1p(nj) + rng.normal(0, 2)
    for _ in range(int(rng.integers(1, 12))):
        a = max(3, base + rng.normal(0, 2))
        rows.append(dict(reference_no=len(rows) + 1, company_name=co if rng.random() > .03 else co.upper(),
                         job_title=rng.choice(["Data Scientist", "ML Engineer", "Data Analyst"]),
                         min_experience=int(rng.integers(0, 8)), avg_salary=round(a, 1),
                         min_salary=round(a * .8, 1), max_salary=round(a * 1.25, 1), num_of_jobs=int(nj)))
pd.DataFrame(rows).to_csv(OUT / "Data Science Jobs.csv", index=False)

# ---- JDS Skill Traits (juniors) ----
m = 140
sk = np.clip(rng.normal(3, .9, (m, 5)), 1, 5).round(1)
z = -3.2 + .35 * sk[:, 0] + .3 * sk[:, 1] + .55 * sk[:, 2] + .35 * sk[:, 3] + .45 * sk[:, 4] + rng.normal(0, .7, m)
hike = (z > np.median(z)).astype(int)
jds = pd.DataFrame(sk, columns=["big_data_skills", "maths-stats_skills", "coding_skills", "ai_and_ml_skills",
                                "dashboard_and_storytelling_skills"])
jds.insert(0, "id", range(1, m + 1)); jds["salary_hike_high_or_low"] = hike
jds.loc[3, "coding_skills"] = 9.0; jds.loc[10, "ai_and_ml_skills"] = np.nan   # dirty entries
jds.to_csv(OUT / "JDS Skill Traits.csv", index=False)

# ---- SDS Personality (seniors) ----
m = 160
p = rng.normal(0, 1, (m, 5))      # neuroticism, extraversion, openness, agreeableness, conscientiousness
z = -.5 * p[:, 0] + .4 * p[:, 1] + .3 * p[:, 2] - .35 * p[:, 3] ** 2 + .5 * p[:, 4] + rng.normal(0, .8, m)
succ = np.where(z > np.median(z), "High", "Low")
sds = pd.DataFrame(p.round(3), columns=["neuroticism", "extraversion", "openness_to_experience", "agreeableness", "conscientiousness"])
sds.insert(0, "id", range(1, m + 1)); sds["success_classification_high_low"] = succ
sds.loc[5, "extraversion"] = 11.0
sds.to_csv(OUT / "SDS Personality Traits.csv", index=False)
print("Sample files written to", OUT)
