"""Template for your model package. Copy this folder, fill in Forecaster, and add
train.py and artifacts/ next to this file. See example_package/ for a working version.
"""
import pandas as pd


class Forecaster:
    def __init__(self, artifact_dir: str):
        """Load your fitted models from artifact_dir (e.g. with joblib.load or torch.load).

        Do not train here. Time limit: 60 seconds.
        """
        raise NotImplementedError

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        """Forecast the 48 half-hours of day D+1.

        history: every half-hour from 2010-01-01 00:30 to the end of day D
                 (the interval stamped D+1 00:00), with the same columns as the
                 released CSV, indexed by timestamp.
        weather_next: the 12 weather columns for the 48 half-hours of day D+1.

        Return a DataFrame with exactly 48 rows, the same index as weather_next, and columns
          nsw_demand_mw : float, predicted NSW demand in MW
          nsw_price_up  : int, 1 if the NSW price is predicted to be above the mean NSW
                          price of day D, otherwise 0
        Time limit: 20 seconds per call. No network access; write files only under /tmp.
        """
        raise NotImplementedError
