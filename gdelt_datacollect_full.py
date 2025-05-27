# gdelt_datacollect_full.py – 100‑row‑per‑month version
"""
Collect at most **100 matching articles per calendar month** about
self‑driving / autonomous vehicles from GDELT GKG and save them to a single CSV
file.

Key changes
-----------
* Drops the adaptive slicing; iterates month‑by‑month directly.
* Adds `LIMIT 100` to each SQL query so we never pull more than 100 rows for a
  month, then moves on.
* Keeps byte cap (`MAX_BYTES`) to avoid runaway scans.
* Respects existing `gdelt_download_log.json` so reruns skip completed months.
"""

from datetime import date
import calendar, json, re, sys, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from google.cloud import bigquery

# ---------- CONFIG ----------
PROJECT_ID   = "sentiment-nlp-457500"
MAX_BYTES    = 30 * 1024**3   # 30 GB per query (limits cost)
ROWS_PER_MONTH = 100          # new cap
START_YEAR, END_YEAR = 2000, 2024
OUT_FILE  = Path("gdelt_selfdriving_master.csv")
LOG_FILE  = Path("gdelt_download_log.json")
HYDRATE_TEXT = False

print("Per‑query byte cap:", MAX_BYTES/1e9, "GB  |  max rows per month:", ROWS_PER_MONTH)

KEYWORD_REGEX = r"(?i)(self[- ]?driving|driverless|autonomous[- ]?vehicles?|robot[- ]?cars?)"

client = bigquery.Client(project=PROJECT_ID, location="US")

SQL_TMPL = f"""
SELECT
  PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS Date,
  DocumentIdentifier AS URL,
  SourceCommonName   AS Source
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME BETWEEN '{{start}}' AND '{{end}}'
  AND REGEXP_CONTAINS(DocumentIdentifier, r'{KEYWORD_REGEX}')
LIMIT {ROWS_PER_MONTH}
"""

# ---------- hydration (optional) ----------
if HYDRATE_TEXT:
    from newspaper import Article
    def hydrate(url: str):
        try:
            art = Article(url, language="en"); art.download(); art.parse()
            return art.title, " ".join(art.text.split())
        except Exception:
            return "", ""

# ---------- helpers ----------

def load_log():
    return json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else {}

def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))

log = load_log()
header_written = OUT_FILE.exists() and OUT_FILE.stat().st_size > 0

# ---------- month loop ----------
for yr in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
    for m in range(1, 13):
        start = date(yr, m, 1)
        end   = date(yr, m, calendar.monthrange(yr, m)[1])
        key   = f"{start}_{end}"
        if key in log:
            continue  # already processed

        sql = SQL_TMPL.format(start=start.isoformat(), end=end.isoformat())
        job_cfg = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES)
        try:
            job = client.query(sql, job_cfg)
            df  = job.result().to_dataframe()
            billed = job.total_bytes_processed or 0
        except Exception as e:
            print(f"{start:%Y-%m} skipped – {e}", file=sys.stderr)
            log[key] = {"rows": 0, "billed_bytes": 0}; save_log(log); continue

        if df.empty:
            log[key] = {"rows": 0, "billed_bytes": billed}; save_log(log); continue

        if HYDRATE_TEXT:
            titles, texts = [], []
            for url in tqdm(df["URL"], leave=False, desc=f"hydrate {start:%Y-%m}"):
                ttl, txt = hydrate(url); titles.append(ttl); texts.append(txt); time.sleep(0.3)
            df["Title"], df["ArticleText"] = titles, texts

        df.to_csv(OUT_FILE, mode="a", header=not header_written, index=False)
        header_written = True
        log[key] = {"rows": len(df), "billed_bytes": billed}; save_log(log)
        print(f"✔ {len(df):3d} rows {start:%Y-%m}  | billed {billed/1e9:.2f} GB")

print("Finished →", OUT_FILE)