"""日前预测的共享特征与无状态预测接口。训练、历史模拟和实际预测使用同一套特征函数。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import joblib
import numpy as np
import pandas as pd
import holidays

MARKET = ['nsw_demand', 'nsw_price', 'qld_demand', 'qld_price', 'vic_demand', 'vic_price']
WEATHER = [f'{c}_{v}' for c in ('syd', 'mel', 'bne') for v in ('temp', 'rh', 'cloud', 'solar')]
COLS = MARKET + WEATHER

#3. 生成日历特征
def calendar(index):
    #把区间结束时间转换为区间开始时间
    #区间结束：10 月 3 日 00:00 区间开始：10 月 2 日 23:30
    start = index - pd.Timedelta(minutes=30)
    # 半小时时段编号，范围是 0–47
    slot = (start.hour * 2 + start.minute // 30).to_numpy()
    # 星期编号 day of week
    dow = start.dayofweek.to_numpy()
    # 把一年中的日期转换为角度，为季节周期特征作准备
    annual = 2 * np.pi * (start.dayofyear.to_numpy() - 1) / 365.25
    # NSW 假日
    hol = holidays.Australia(state='NSW', years=sorted(set(start.year)))
    # 生成假日标记
    h = np.array([int(d.date() in hol) for d in start])
    #生成包含48个"slot_j"名字的字典
    '''
    {
        'slot_0': 是否是第0个时段，是1非0,
        'slot_1': 一列数值,
        ...
    }
    '''
    feats = {f'slot_{j}': (slot == j).astype(float) for j in range(48)}
    # dow_0 是否周1 dow_6 是否周日
    feats.update({f'dow_{j}': (dow == j).astype(float) for j in range(7)})
    # 加入四列特征
    feats.update(annual_sin=np.sin(annual), annual_cos=np.cos(annual), holiday=h,
                 working_day=((dow < 5) & (h == 0)).astype(float))
    return feats


def build_features(history, weather_next):
    """
    为一个或多个完整目标日构造符合时间边界的特征。
    市场数据只读取至少一天以前的同时段值；日统计量只读取目标日以前的完整市场日。
    """
    idx = weather_next.index
    days = (idx - pd.Timedelta(minutes=30)).normalize()
    feats = calendar(idx)
    for c in WEATHER:
        feats[c] = weather_next[c].to_numpy()
    for lag in (1, 2, 7, 14):
        # 目标时段对齐的历史数据表(1，2，7，14天前的对应数据)
        aligned = history.reindex(idx - pd.Timedelta(days=lag))
        for c in MARKET:
            feats[f'{c}_lag{lag}'] = aligned[c].to_numpy()
        for c in ('syd_temp', 'syd_solar'):
            feats[f'{c}_lag{lag}'] = aligned[c].to_numpy()
    # 前一市场日结束后，其完整日统计量已经可用。
    hdays = (history.index - pd.Timedelta(minutes=30)).normalize()
    #列有两层名称 如 nsw_demand,mean 和 nsw_demand,max 一共四种
    daily = history[MARKET + ['syd_temp', 'syd_solar']].groupby(hdays).agg(['mean', 'std', 'min', 'max'])
    for lag in (1, 7):
        values = daily.reindex(days - pd.Timedelta(days=lag))
        for c, stat in daily.columns:
            #遍历刚才的两层列名，并分别赋给 列名和计算值
            '''
            如
            c = 'nsw_demand'
            stat = 'mean'
            得到 nsw_demand_mean_daylag1 前一市场日的 NSW 平均需求
            '''
            feats[f'{c}_{stat}_daylag{lag}'] = values[(c, stat)].to_numpy()
    # 计算前七天同一半小时时段的统计量
    for c in ('nsw_demand', 'nsw_price'):
        arr = np.stack([history[c].reindex(idx - pd.Timedelta(days=d)).to_numpy() for d in range(1, 8)])
        feats[f'{c}_week_mean'] = arr.mean(axis=0)
        feats[f'{c}_week_std'] = arr.std(axis=0)
    #目标时段温度与昨天同时段温度的差
    feats['temp_change'] = feats['syd_temp'] - feats['syd_temp_lag1']
    #构造高低温特征 越过22度或0度，特征按平方增长，非线性
    feats['temp_hot'] = np.maximum(feats['syd_temp'] - 22, 0) ** 2
    feats['temp_cold'] = np.maximum(18 - feats['syd_temp'], 0) ** 2
    # 历史时段价格与昨日均价的差值，可为价格方向分类提供信息。
    for lag in (1, 7):
        feats[f'price_margin_lag{lag}'] = feats[f'nsw_price_lag{lag}'] - feats['nsw_price_mean_daylag1']
    #48 行 × 176 列
    return pd.DataFrame(feats, index=idx, dtype=np.float32)

#1. 读取数据
def load_table(csv):
    #把 timestamp 列解析为时间类型 把时间戳设为行索引 按时间索引排序
    df = pd.read_csv(csv, parse_dates=['timestamp'], index_col='timestamp').sort_index()
    expected = pd.date_range(df.index[0], df.index[-1], freq='30min')
    if not df.index.equals(expected) or df.isna().any().any():
        raise ValueError('Expected complete, unique half-hourly observations without missing values')
    return df

#2. 把观测表转换成特征和两个标签
def training_table(df):
    #跳过最开始十四天的时间索引
    idx = df.index[14 * 48:]
    #行：idx 中的目标时段
    #列：十二个天气变量
    x = build_features(df, df.loc[idx, WEATHER])
    day = (idx - pd.Timedelta(minutes=30)).normalize()
    #包括最早十四天在内的所有观测
    all_days = (df.index - pd.Timedelta(minutes=30)).normalize()
    # 逐日调用 Series.mean
    # 每个完整市场日的平均 NSW 价格
    price_mean = df.nsw_price.groupby(all_days).agg(lambda v: v.mean())
    # 同一个目标日的 48 个时段，共用前一天的同一个均价阈值
    threshold = price_mean.reindex(day - pd.Timedelta(days=1)).to_numpy()
    y = df.loc[idx, 'nsw_demand'].to_numpy()
    c = (df.loc[idx, 'nsw_price'].to_numpy() > threshold).astype(np.int8)
    return x, y, c #输入特征，实际电力需求，当前电价是否高于前一天（二分类）


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.bundle = joblib.load(os.path.join(artifact_dir, 'models.joblib'))

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        # 截取最近约十六天的数据，满足十四天滞后需求，避免处理全部历史。
        recent = history.loc[history.index > weather_next.index[0] - pd.Timedelta(days=16)]
        x = build_features(recent, weather_next)
        demand = self.bundle['demand'].predict(x)
        prob = self.bundle['price'].predict_proba(x)[:, 1]
        return pd.DataFrame({'nsw_demand_mw': np.clip(demand, 0, 20000),
                             'nsw_price_up': (prob >= self.bundle.get('price_threshold', 0.5)).astype(int)}, index=weather_next.index)
