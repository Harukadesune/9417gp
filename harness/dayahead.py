"""官方日前任务辅助函数：定义市场日、输入、标签、基线与指标。
时间戳表示半小时区间结束时刻；市场日从当天 00:30 至次日 00:00，共四十八个时段。"""
import numpy as np
import pandas as pd

MARKET_COLS = ["nsw_demand", "nsw_price", "qld_demand", "qld_price", "vic_demand", "vic_price"]
WEATHER_COLS = [f"{c}_{v}" for c in ("syd", "mel", "bne") for v in ("temp", "rh", "cloud", "solar")]
N_PERIODS = 48


def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return df.sort_index()


def market_day(index):
    """返回各半小时区间所属的市场日，即该区间开始时刻的日期。"""
    return (index - pd.Timedelta("30min")).normalize()


def complete_days(df):
    days = market_day(df.index)
    counts = pd.Series(1, index=df.index).groupby(days).sum()
    return list(counts[counts == N_PERIODS].index)


def make_inputs(df, target_day):
    """按官方评测方式，为目标日生成历史市场数据和次日天气输入。"""
    target_day = pd.Timestamp(target_day)
    days = market_day(df.index)
    history = df[days < target_day]
    weather_next = df.loc[days == target_day, WEATHER_COLS]
    return history, weather_next


def targets(df, target_day):
    """生成目标日的真实需求，以及价格是否严格高于前一市场日均价的标签。"""
    target_day = pd.Timestamp(target_day)
    days = market_day(df.index)
    today = df[days == target_day]
    prev_mean = df.loc[days == target_day - pd.Timedelta("1D"), "nsw_price"].mean()
    return pd.DataFrame({"nsw_demand_mw": today["nsw_demand"],
                         "nsw_price_up": (today["nsw_price"] > prev_mean).astype(int)})


def baseline(history, weather_next, lag_days=1):
    """取目标日前 lag_days 天的同时段作为基线。
需求直接使用该时段需求；价格方向比较该时段价格与目标日前一天均价，两个基线使用相同阈值。"""
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
    """计算真实标签中各个已出现类别召回率的平均值；只有一个类别时返回该类别召回率。"""
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
