"""Shared helpers for the day-ahead task: day boundaries, inputs, targets, baselines, metrics.

Day convention: each timestamp marks the END of a half-hour interval (market time,
AEST, UTC+10, no daylight saving). Day D therefore consists of the 48 intervals
stamped D 00:30, D 01:00, ..., D 23:30 and (D+1) 00:00.
"""
import numpy as np
import pandas as pd

MARKET_COLS = ["nsw_demand", "nsw_price", "qld_demand", "qld_price", "vic_demand", "vic_price"]
WEATHER_COLS = [f"{c}_{v}" for c in ("syd", "mel", "bne") for v in ("temp", "rh", "cloud", "solar")]
N_PERIODS = 48


def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return df.sort_index()


def market_day(index):
    """Calendar day each interval belongs to (interval start date)."""
    return (index - pd.Timedelta("30min")).normalize()


def complete_days(df):
    days = market_day(df.index)
    counts = pd.Series(1, index=df.index).groupby(days).sum()
    return list(counts[counts == N_PERIODS].index)


def make_inputs(df, target_day):
    """Inputs exactly as the marking runner builds them for predicting target_day."""
    target_day = pd.Timestamp(target_day)
    days = market_day(df.index)
    history = df[days < target_day]
    weather_next = df.loc[days == target_day, WEATHER_COLS]
    return history, weather_next


def targets(df, target_day):
    """True demand and price direction for target_day."""
    target_day = pd.Timestamp(target_day)
    days = market_day(df.index)
    today = df[days == target_day]
    prev_mean = df.loc[days == target_day - pd.Timedelta("1D"), "nsw_price"].mean()
    return pd.DataFrame({"nsw_demand_mw": today["nsw_demand"],
                         "nsw_price_up": (today["nsw_price"] > prev_mean).astype(int)})


def baseline(history, weather_next, lag_days=1):
    """Same half-hour lag_days before the target day (day D for lag 1, day D-6 for lag 7).

    Demand: that half-hour's demand. Price direction: 1 if that half-hour's price is above
    the mean price of day D (the same threshold as the target)."""
    days = market_day(history.index)
    last_day = days.max()
    src_day = last_day - pd.Timedelta(f"{lag_days - 1}D")
    src = history[days == src_day]
    threshold = history.loc[days == last_day, "nsw_price"].mean()
    up = (src["nsw_price"] > threshold).astype(int).to_numpy()
    return pd.DataFrame({"nsw_demand_mw": src["nsw_demand"].to_numpy(), "nsw_price_up": up},
                        index=weather_next.index)


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def balanced_accuracy(y, yhat):
    """Mean recall over the classes present in y (if only one class is present,
    this is the recall of that class)."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    recalls = [np.mean(yhat[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(recalls) * 100)


def accuracy(y, yhat):
    return float(np.mean(np.asarray(y, int) == np.asarray(yhat, int)) * 100)


def score(truth, pred):
    return {"MAPE": mape(truth["nsw_demand_mw"], pred["nsw_demand_mw"]),
            "MAE": mae(truth["nsw_demand_mw"], pred["nsw_demand_mw"]),
            "BalAcc": balanced_accuracy(truth["nsw_price_up"], pred["nsw_price_up"]),
            "Acc": accuracy(truth["nsw_price_up"], pred["nsw_price_up"])}
