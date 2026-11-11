"""官方示例模型包：需求使用最近七天同时段的移动平均；价格方向根据历史均价与昨日均价比较。
用于演示模型加载、预测接口与配置文件读取，不作为本项目最终模型。"""
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
        recent = history.iloc[-N_PERIODS * k:]                    # 最近 k 个完整市场日
        demand = recent["nsw_demand"].to_numpy().reshape(k, N_PERIODS).mean(axis=0)
        price = recent["nsw_price"].to_numpy().reshape(k, N_PERIODS).mean(axis=0)
        mean_price_day_d = history["nsw_price"].iloc[-N_PERIODS:].mean()
        return pd.DataFrame({
            "nsw_demand_mw": demand,
            "nsw_price_up": (price > mean_price_day_d).astype(int),
        }, index=weather_next.index)
