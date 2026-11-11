"""模型包模板。复制后实现 Forecaster，并添加 train.py 与保存模型的 artifacts 目录。"""
import pandas as pd


class Forecaster:
    def __init__(self, artifact_dir: str):
        """从 artifacts 目录加载已训练模型或配置，不在初始化时训练；官方初始化时限六十秒。"""
        raise NotImplementedError

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        """预测次日四十八个半小时。history 仅含前一日结束前的数据，weather_next 提供目标日天气。
返回与 weather_next 索引相同的四十八行：nsw_demand_mw 为需求数值，nsw_price_up 为价格方向零或一。
官方预测时限二十秒，不访问网络，仅可写入临时目录。"""
        raise NotImplementedError
