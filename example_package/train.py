"""Rebuild artifacts/ for the example package from the released CSV only.

Usage (from the kit's root folder):
    python example_package/train.py data/nem_weather_halfhourly_2010_2026.csv

The moving average has nothing to fit, so this script only checks that the file can be read
and writes the window length to artifacts/config.json. Your train.py must fit your models
here, using only the released CSV, and save them in artifacts/.
"""
import json
import os
import sys

import pandas as pd

WINDOW_DAYS = 7

csv_path = sys.argv[1]
df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "config.json"), "w") as fh:
    json.dump({"window_days": WINDOW_DAYS}, fh)
print(f"Read {len(df)} half-hours up to {df.index.max()}; saved the configuration to {out}")
