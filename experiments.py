"""按时间顺序运行可复现的三阶段实验，使用 Python 3.11 和指定依赖版本。
算法、参数和报警阈值仅使用二〇一〇至二〇一四年确定；最终策略使用二〇一五至二〇二四年选择。
二〇二五至二〇二六年仅用于后续审计，不据此修改已选策略。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/comp9417-mpl')
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'package'))
from model import load_table, training_table
from train import make_estimator

RESULT = HERE / 'results'
SEED = 9417

# 指标计算
def metrics(y, p, task):
    '''
    y    → 真实答案
    p    → 模型预测结果
    task → 当前任务名称
    '''
    if task == 'demand':
        return {'MAPE': float(np.mean(np.abs(y-p)/y)*100), 'MAE': float(np.mean(np.abs(y-p)))}
    return {'Acc': float(np.mean(y == p)*100),
            'BalAcc': float(np.mean([np.mean(p[y == k] == k) for k in (0, 1) if np.any(y == k)])*100)}


def predictions(model, x, task, threshold=0.5):
    return model.predict(x) if task == 'demand' else (model.predict_proba(x)[:, 1] >= threshold).astype(np.int8)


def log(*s):
    print(time.strftime('%H:%M:%S'), *s, flush=True)


def baseline_values(df, index, task, lag):
    # 保留原始 float64 精度，与官方 harness/dayahead.py 的定义一致。
    source = df.reindex(index-pd.Timedelta(days=lag))
    if task == 'demand':
        return source.nsw_demand.to_numpy()
    hdays = (df.index-pd.Timedelta(minutes=30)).normalize()
    target_days = (index-pd.Timedelta(minutes=30)).normalize()
    threshold = df.nsw_price.groupby(hdays).agg(lambda v:v.mean()).reindex(target_days-pd.Timedelta(days=1)).to_numpy()
    return (source.nsw_price.to_numpy() > threshold).astype(np.int8)


def daytypes(df):
    """仅进行事后日类型诊断；目标日的真实聚类标签不会进入预测特征。"""
    values = df.nsw_demand.to_numpy().reshape(-1, 48)
    days = (df.index[::48] - pd.Timedelta(minutes=30)).normalize()
    norm = values / values.mean(axis=1, keepdims=True)
    train = days.year <= 2013
    pca = PCA(n_components=.95, svd_solver='full').fit(norm[train])
    z = pca.transform(norm)
    trials, models = [], {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(z[train])
        s = silhouette_score(z[train], km.labels_, sample_size=min(1000, train.sum()), random_state=SEED)
        trials.append(dict(k=k, silhouette=s, inertia=km.inertia_))
        models[k] = km
    k = int(max(trials, key=lambda r:r['silhouette'])['k'])
    km = models[k]
    lab = km.predict(z)
    dist = km.transform(z).min(axis=1)
    table = pd.DataFrame({'day': days, 'cluster': lab, 'distance': dist})
    table['novel'] = dist > np.quantile(dist[train], .99)
    table.to_csv(RESULT / 'day_types.csv', index=False)
    pd.DataFrame(trials).to_csv(RESULT / 'cluster_selection.csv', index=False)
    pd.DataFrame({f'cluster_{j}': norm[train][km.labels_ == j].mean(axis=0) for j in range(k)}).to_csv(RESULT / 'cluster_profiles.csv', index_label='slot')
    pd.crosstab(days.year, lab, normalize='index').to_csv(RESULT / 'cluster_yearly_shares.csv', index_label='year')
    joblib.dump(dict(pca=pca, clustering=km), RESULT / 'daytype_reference.joblib')
    return {'k':k, 'pca_dimensions':int(pca.n_components_), 'explained_variance':float(pca.explained_variance_ratio_.sum())}


def aggregate(task, truth, pred, index, name):
    days = (index - pd.Timedelta(minutes=30)).normalize()
    frame = pd.DataFrame({'day': days, 'truth':truth, 'prediction':pred}, index=index)
    frame.to_parquet(RESULT / f'pred_{task}_{name}.parquet')
    annual, monthly = [], []
    rng = np.random.default_rng(SEED)
    for period, target in [('Y', annual), ('M', monthly)]:
        for key, sub in frame.groupby(days.to_period(period)):
            row = dict(task=task, model=name, period=str(key), n=len(sub), **metrics(sub.truth.to_numpy(), sub.prediction.to_numpy(), task))
            if period == 'M':
                errors = (100 * np.abs(sub.truth.to_numpy()-sub.prediction.to_numpy())/sub.truth.to_numpy()
                          if task == 'demand' else (sub.truth.to_numpy()==sub.prediction.to_numpy()).astype(float)*100)
                avg = errors.mean()
                half = 1.96 * errors.std(ddof=1)/np.sqrt(len(errors))
                if task == 'price':
                    p = avg/100
                    half = 1.96*100*np.sqrt(p*(1-p)/len(errors))
                row.update(iid_low=avg-half, iid_high=avg+half)
                if task == 'price':
                    hoeffding = 100 * np.sqrt(np.log(40)/(2*len(errors)))
                    row.update(hoeffding_low=max(0, avg-hoeffding), hoeffding_high=min(100, avg+hoeffding))
                    row['iid_low'], row['iid_high'] = max(0, row['iid_low']), min(100, row['iid_high'])
                daily = errors.reshape(-1, 48).mean(axis=1)
                boot = daily[rng.integers(0, len(daily), size=(2000, len(daily)))].mean(axis=1)
                row.update(zip(('block_low', 'block_high'), np.quantile(boot, [.025, .975])))
            target.append(row)
    return annual, monthly


def adaptive(task, strategy, conf, initial, x, y, days, input_signal, calibration):
    """在第 D 日结束后检查报警，只使用截至 D 日的数据拟合，从 D+1 日部署新模型。
批量预测缓存仅用于加速，未来标签不参与拟合、报警或状态更新；模型替换后更新未来缓存。"""
    start = np.flatnonzero(days >= pd.Timestamp('2015-01-01'))[0]
    simx, simy = x.iloc[start:], y[start:]
    simdays = days[start::48]
    output = np.empty(len(simx), dtype=float if task == 'demand' else np.int8)
    model = initial
    threshold = conf.get('threshold', .5)
    cache = predictions(model, simx, task, threshold)
    ref = float(calibration['mean'])
    ratio = float(calibration['ratio'])
    err_history = []
    triggers, signals = [], []
    last_refit = -10000
    scheduled = {pd.Timestamp(f'{y}-01-01') for y in (2016, 2018, 2020, 2022, 2024, 2026)}
    for d, day in enumerate(simdays):
        p = cache[d*48:(d+1)*48]
        actual = simy[d*48:(d+1)*48]
        output[d*48:(d+1)*48] = p
        err = np.mean(np.abs(actual-p)/actual)*100 if task == 'demand' else np.mean(actual != p)
        err_history.append(float(err))
        err_alarm = len(err_history) >= 28 and np.mean(err_history[-28:]) > ref*ratio
        inp_alarm = bool(input_signal.loc[day])
        signals.append(dict(task=task, strategy=strategy, observed_day=str(day.date()), input_alarm=inp_alarm,
                            error_alarm=bool(err_alarm), error_stat=float(np.mean(err_history[-28:])), error_threshold=ref*ratio))
        next_day = day + pd.Timedelta(days=1)
        due = ((strategy.startswith('periodic') and next_day in scheduled) or
               (strategy == 'drift_error' and err_alarm) or (strategy == 'drift_input' and inp_alarm))
        # 每个任务与策略组合最多替换模型六次；漂移策略的冷却期为三百六十五天。
        if not due or len(triggers) >= 6 or (strategy.startswith('drift') and d-last_refit < 365) or d == len(simdays)-1:
            continue
        end = start + (d+1)*48
        begin = 0 if strategy == 'periodic_expanding' else max(0, end-730*48)
        model = make_estimator(task, conf['family'], conf['param'])
        model.fit(x.iloc[begin:end], y[begin:end])
        cache[(d+1)*48:] = predictions(model, simx.iloc[(d+1)*48:], task, threshold)
        triggers.append(dict(task=task, strategy=strategy, observed_day=str(day.date()),
                             effective_day=str(next_day.date()), train_start=str(days[begin].date()),
                             train_end=str(days[end-1].date()), rows=end-begin))
        last_refit = d
        # 误差参考值仅使用本次替换前已经观测到的误差。
        # 清空观测窗口，避免把旧模型的误差混入新模型的监测窗口。
        ref = max(float(np.mean(err_history[-90:])), 1e-5)
        err_history = []
        log('refit', task, strategy, next_day.date(), len(triggers))
    return output, triggers, signals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=str(HERE/'data/nem_weather_halfhourly_2010_2026.csv'))
    ap.add_argument('--reuse-demand', action='store_true', help='Reuse a completed demand simulation; rerun price experiments')
    args = ap.parse_args()
    RESULT.mkdir(exist_ok=True)
    df = load_table(args.data)
    df.describe().T.to_csv(RESULT/'data_summary.csv')
    log('features')
    x, yd, yc = training_table(df)
    assert np.isfinite(x.to_numpy()).all()
    days = (x.index - pd.Timedelta(minutes=30)).normalize()
    train = days < pd.Timestamp('2014-01-01')
    valid = (days >= pd.Timestamp('2014-01-01')) & (days < pd.Timestamp('2015-01-01'))
    frozen_train = days < pd.Timestamp('2015-01-01')
    test = ~frozen_train
    confs, frozen, validation_rows = {}, {}, []
    calibrations = {}
    for task, y in [('demand', yd), ('price', yc)]:
        confs[task], frozen[task] = {}, {}
        for family in ('linear', 'tree'):
            trials = []
            grid = (10., 1000.) if task == 'demand' else (.01, 1.)
            if family == 'tree': grid = (15, 31)
            for param in grid:
                model = make_estimator(task, family, param)
                model.fit(x.loc[train], y[train])
                threshold = .5
                if task == 'price':
                    prob = model.predict_proba(x.loc[valid])[:, 1]
                    threshold = max(np.linspace(.3, .7, 17), key=lambda t:metrics(y[valid], (prob >= t).astype(int), task)['BalAcc'])
                pred = predictions(model, x.loc[valid], task, threshold)
                scores = metrics(y[valid], pred, task)
                row = dict(task=task, family=family, param=param, threshold=float(threshold), **scores)
                validation_rows.append(row)
                trials.append((row, pred))
                log('validation', row)
            best, valpred = min(trials, key=lambda r:r[0]['MAPE'] if task == 'demand' else -r[0]['BalAcc'])
            confs[task][family] = best
            # 误差来自模型未参与拟合的二〇一四年预测，但该年份也参与了
            # 超参数和分类阈值选择，因此校准结果仍受验证集选择影响。
            err = (np.abs(y[valid]-valpred)/y[valid]*100 if task == 'demand' else (y[valid]!=valpred).astype(float)).reshape(-1, 48).mean(axis=1)
            rolling = pd.Series(err).rolling(28).mean().dropna()
            calibrations[(task, family)] = dict(mean=float(err.mean()), ratio=float(max(1.05, rolling.quantile(.99)/err.mean())))
            model = make_estimator(task, family, best['param'])
            model.fit(x.loc[frozen_train], y[frozen_train])
            frozen[task][family] = model
        confs[task]['selected'] = min(('linear', 'tree'), key=lambda f:confs[task][f]['MAPE'] if task == 'demand' else -confs[task][f]['BalAcc'])
    pd.DataFrame(validation_rows).to_csv(RESULT/'validation.csv', index=False)
    (RESULT/'initial_selection.json').write_text(json.dumps(confs, indent=2))
    log('day types')
    cluster_info = daytypes(df)
    # 输入监测使用每日天气和需求的统计量，这些信息均在
    # 观测日结束时可用；使用二〇一〇至二〇一三年标准化，二〇一四年校准阈值。
    hdays = (df.index - pd.Timedelta(minutes=30)).normalize()
    raw = df[['syd_temp', 'syd_solar', 'nsw_demand']].groupby(hdays).agg(['mean', 'std'])
    refraw = raw.loc[raw.index.year <= 2013]
    z = (raw - refraw.mean()) / refraw.std()
    # 计算滚动均值的偏移程度，尺度由训练年份的典型波动确定。
    stat = np.sqrt((z.rolling(28).mean() ** 2).mean(axis=1))
    inp_threshold = float(stat.loc['2014'].quantile(.99))
    inp_alarm = stat > inp_threshold
    pd.DataFrame({'statistic':stat, 'threshold':inp_threshold, 'alarm':inp_alarm}).to_csv(RESULT/'input_detector.csv', index_label='day')
    annual, monthly, retrains, signal_rows = [], [], [], []
    if args.reuse_demand:
        old_retrains = pd.read_csv(RESULT/'retraining_log.csv')
        retrains.extend(old_retrains[old_retrains.task=='demand'].to_dict('records'))
        old_signals = pd.read_csv(RESULT/'online_signals.csv')
        signal_rows.extend(old_signals[old_signals.task=='demand'].to_dict('records'))
    def save(task, y, p, name):
        a, m = aggregate(task, y[test], p, x.index[test], name)
        annual.extend(a); monthly.extend(m)
    for task, y in [('demand', yd), ('price', yc)]:
        for family, model in frozen[task].items():
            save(task, y, predictions(model, x.loc[test], task, confs[task][family].get('threshold', .5)), f'frozen_{family}')
        for lag in (1, 7):
            p = baseline_values(df, x.index[test], task, lag)
            save(task, y, p, f'baseline_{lag}')
        selected = confs[task]['selected']
        for strategy in ('periodic_expanding', 'periodic_window', 'drift_error', 'drift_input'):
            if args.reuse_demand and task == 'demand':
                log('reuse completed demand simulation', strategy)
                cached = pd.read_parquet(RESULT/f'pred_demand_{strategy}.parquet')
                np.testing.assert_array_equal(cached.truth.to_numpy(), y[test])
                save(task, y, cached.prediction.to_numpy(), strategy)
                continue
            log('simulate', task, strategy, selected)
            p, events, sig = adaptive(task, strategy, confs[task][selected], frozen[task][selected], x, y, days, inp_alarm, calibrations[(task, selected)])
            save(task, y, p, strategy)
            retrains.extend(events); signal_rows.extend(sig)
    pd.DataFrame(annual).to_csv(RESULT/'annual_metrics.csv', index=False)
    pd.DataFrame(monthly).to_csv(RESULT/'monthly_metrics_ci.csv', index=False)
    pd.DataFrame(retrains).to_csv(RESULT/'retraining_log.csv', index=False)
    pd.DataFrame(signal_rows).to_csv(RESULT/'online_signals.csv', index=False)
    # 第三阶段仅依据二〇一五至二〇二四年的模拟结果选择最终策略。
    choice, summary = {}, []
    for task in ('demand', 'price'):
        selected = confs[task]['selected']
        candidates = [f'frozen_{selected}', 'periodic_expanding', 'periodic_window', 'drift_error', 'drift_input']
        for name in candidates:
            frame = pd.read_parquet(RESULT/f'pred_{task}_{name}.parquet')
            for label, mask in [('strategy_selection_2015_2024', frame.day < '2025-01-01'), ('audit_2025_2026', frame.day >= '2025-01-01')]:
                summary.append(dict(task=task, model=name, split=label, **metrics(frame.truth.to_numpy()[mask], frame.prediction.to_numpy()[mask], task)))
        rows = [r for r in summary if r['task']==task and r['split']=='strategy_selection_2015_2024']
        best = min(rows, key=lambda r:r['MAPE'] if task=='demand' else -r['BalAcc'])
        c = confs[task][selected]
        choice[task] = dict(family=selected, param=c['param'], threshold=c['threshold'],
                            strategy=best['model'], window_days=730 if best['model'] in ('periodic_window', 'drift_input', 'drift_error') else 0)
    pd.DataFrame(summary).to_csv(RESULT/'strategy_comparison.csv', index=False)
    (HERE/'package/settings.json').write_text(json.dumps(choice, indent=2))
    # 进行两种特征消融，固定已选定的树模型参数，不再调参。
    ablations = []
    for task, y in [('demand', yd), ('price', yc)]:
        c = confs[task]['tree']
        for label in ('no_interstate', 'no_weather'):
            cols = [c for c in x.columns if not (any(s in c for s in ('qld_', 'vic_', 'mel_', 'bne_')) if label=='no_interstate' else any(s in c for s in ('syd_', 'mel_', 'bne_', 'temp_')))]
            model = make_estimator(task, 'tree', c['param'])
            model.fit(x.loc[train, cols], y[train])
            p = predictions(model, x.loc[valid, cols], task, c['threshold'])
            ablations.append(dict(task=task, ablation=label, n_features=len(cols), **metrics(y[valid], p, task)))
    pd.DataFrame(ablations).to_csv(RESULT/'ablations.csv', index=False)
    # 统计历史周表现的经验分布；该分布不是已校准的未来周预测区间。
    # 二〇二〇至二〇二四年的十二月提供季节证据，二〇二五至二〇二六年提供近期证据。
    weekly_rows = []
    for task in ('demand', 'price'):
        f = pd.read_parquet(RESULT/f'pred_{task}_{choice[task]["strategy"]}.parquet')
        for start in pd.date_range('2020-01-06', f.day.max()-pd.Timedelta(days=6), freq='7D'):
            sub = f[(f.day >= start) & (f.day < start+pd.Timedelta(days=7))]
            if len(sub) != 336: continue
            cohort = 'recent_2025_2026' if start.year>=2025 else ('december_2020_2024' if start.month==12 else None)
            if cohort:
                weekly_rows.append(dict(task=task, cohort=cohort, week_start=str(start.date()), **metrics(sub.truth.to_numpy(), sub.prediction.to_numpy(), task)))
    weekly = pd.DataFrame(weekly_rows)
    weekly.to_csv(RESULT/'weekly_forecast_evidence.csv', index=False)
    intervals = []
    for (task, cohort), sub in weekly.groupby(['task','cohort']):
        metric = 'MAPE' if task=='demand' else 'BalAcc'
        intervals.append(dict(task=task, cohort=cohort, metric=metric, weeks=len(sub), median=float(sub[metric].median()),
                              p025=float(sub[metric].quantile(.025)), p975=float(sub[metric].quantile(.975))))
    pd.DataFrame(intervals).to_csv(RESULT/'weekly_empirical_ranges.csv', index=False)
    pd.DataFrame({'day':days[::48], 'positive_fraction':yc.reshape(-1,48).mean(axis=1)}).to_csv(RESULT/'class_balance.csv', index=False)
    (RESULT/'protocol.json').write_text(json.dumps(dict(seed=SEED, features=list(x.columns), n_features=x.shape[1], cluster=cluster_info,
        input_threshold=inp_threshold, error_calibration={str(k):v for k,v in calibrations.items()}, bootstrap_repetitions=2000,
        stage1_train='2010-01-15 to 2013-12-31', validation='2014', frozen_refit_end='2014-12-31',
        strategy_selection='2015-2024', audit='2025-2026-10', final_settings=choice), indent=2))
    log('complete', choice)


if __name__ == '__main__':
    main()
