"""Example model package: a 7-day moving average, for both tasks.

It is deliberately simple and learns nothing from data: it only shows the required
interface, how train.py writes artifacts/ and how __init__ reads them back. A real
package saves fitted models in artifacts/ (for example with joblib.dump or torch.save).

  demand:          average NSW demand at the same half-hour over the last 7 days
  price direction: 1 if the average NSW price at the same half-hour over the last
                   7 days is above the mean NSW price of day D, otherwise 0
"""
import json
import os

import pandas as pd

N_PERIODS = 48


class Forecaster:
    def __init__(self, artifact_dir: str):
        with open(os.path.join(artifact_dir, "config.json")) as fh:
            self.window_days = int(json.load(fh)["window_days"])

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        k = self.window_days
        recent = history.iloc[-N_PERIODS * k:]                    # the last k whole days
        demand = recent["nsw_demand"].to_numpy().reshape(k, N_PERIODS).mean(axis=0)
        price = recent["nsw_price"].to_numpy().reshape(k, N_PERIODS).mean(axis=0)
        mean_price_day_d = history["nsw_price"].iloc[-N_PERIODS:].mean()
        return pd.DataFrame({
            "nsw_demand_mw": demand,
            "nsw_price_up": (price > mean_price_day_d).astype(int),
        }, index=weather_next.index)
