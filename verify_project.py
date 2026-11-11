"""检查特征的时间因果性、离线与在线一致性，以及重训练预算。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'package'))
sys.path.insert(0, str(ROOT/'harness'))
from model import load_table, build_features, training_table, Forecaster, WEATHER
import dayahead as da
from run_day import validate


def main():
    df = load_table(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    x, y, c = training_table(df)
    checks = []
    prices = df.nsw_price.to_numpy().reshape(-1,48)
    exact_targets = (prices[1:]>prices[:-1].mean(axis=1,keepdims=True)).reshape(-1)
    np.testing.assert_array_equal(c,exact_targets[13*48:])
    checks.append('All classification labels exactly match per-day Series.mean threshold semantics')
    forecaster = Forecaster(str(ROOT/'package/artifacts'))
    for date in ['2014-01-01','2020-02-29','2026-10-31']:
        day = pd.Timestamp(date)
        history, weather = da.make_inputs(df, day)
        online = build_features(history, weather)
        np.testing.assert_allclose(online, x.loc[weather.index], rtol=0, atol=0)
        truth = da.targets(df, day)
        loc = x.index.get_indexer(weather.index)
        np.testing.assert_array_equal(y[loc], truth.nsw_demand_mw)
        np.testing.assert_array_equal(c[loc], truth.nsw_price_up)
        # 将目标日及未来市场值改为异常值，检查目标日特征是否保持不变。
        poison = df.copy()
        future = da.market_day(df.index) >= day
        poison.loc[future, ['nsw_demand','nsw_price','qld_demand','qld_price','vic_demand','vic_price']] = 999999.
        changed = build_features(poison, weather)
        np.testing.assert_array_equal(online.to_numpy(), changed.to_numpy())
        pred = forecaster.predict(history, weather)
        _, error = validate(pred, weather.index)
        assert error is None, error
        full_p = forecaster.bundle['demand'].predict(online)
        np.testing.assert_allclose(pred.nsw_demand_mw, np.clip(full_p,0,20000), rtol=0, atol=0)
        checks.append(f'{date}: feature parity, future poisoning, target parity, stateless API OK')
    events = pd.read_csv(ROOT/'results/retraining_log.csv')
    counts = events.groupby(['task','strategy']).size()
    assert (counts <= 6).all()
    end = pd.to_datetime(events.train_end)
    observed = pd.to_datetime(events.observed_day)
    effective = pd.to_datetime(events.effective_day)
    assert (end <= observed).all()
    assert (effective == observed+pd.Timedelta(days=1)).all()
    checks.append('All task/strategy retraining counts <=6; training ends before deployment')
    # 按照官方提供的检查程序核对基线和标签定义。
    for task in ('demand','price'):
        for lag in (1,7):
            pred = pd.read_parquet(ROOT/f'results/pred_{task}_baseline_{lag}.parquet')
            source = df.reindex(pred.index-pd.Timedelta(days=lag))
            daily_mean = pd.Series(prices.mean(axis=1),index=da.market_day(df.index[::48]))
            threshold = daily_mean.reindex(pred.day-pd.Timedelta(days=1)).to_numpy()
            exact = source.nsw_demand.to_numpy() if task=='demand' else (source.nsw_price.to_numpy()>threshold).astype(int)
            np.testing.assert_array_equal(pred.prediction.to_numpy(),exact)
            for date in ['2015-01-01','2020-02-29','2026-10-31']:
                h,w = da.make_inputs(df,pd.Timestamp(date))
                b = da.baseline(h,w,lag)
                col = 'nsw_demand_mw' if task=='demand' else 'nsw_price_up'
                np.testing.assert_allclose(pred.loc[w.index,'prediction'],b[col],rtol=0,atol=0)
    checks.append('Both baselines: original float64 precision across all simulation rows; exact harness parity on three dates')
    month = pd.read_csv(ROOT/'results/monthly_metrics_ci.csv')
    assert month.groupby(['task','model']).size().eq(142).all()
    assert np.isfinite(month[['iid_low','iid_high','block_low','block_high']]).all().all()
    checks.append('Every model/task: 142 simulation months; finite IID and day-bootstrap intervals')
    train_py = (ROOT/'package/train.py').read_text()
    assert 'random_state=9417' in train_py
    metadata = json.loads((ROOT/'package/artifacts/metadata.json').read_text())
    assert metadata['numpy']=='2.4.6' and metadata['pandas']=='3.1.0' and metadata['sklearn']=='1.9.1'
    checks.append('Artifacts trained with required NumPy/pandas/scikit-learn versions')
    result = {'status':'pass', 'checks':checks, 'retraining_counts':{str(k):int(v) for k,v in counts.items()}}
    (ROOT/'results/verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
