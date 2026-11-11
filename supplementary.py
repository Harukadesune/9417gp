"""计算验证期基线和分组置换特征重要性，仅使用发布的数据。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
from model import load_table, training_table
from train import make_estimator
from experiments import metrics, predictions, baseline_values, aggregate


def main():
    df = load_table(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    x, y, c = training_table(df)
    days = (x.index-pd.Timedelta(minutes=30)).normalize()
    tr = days < '2014-01-01'
    va = (days >= '2014-01-01') & (days < '2015-01-01')
    rows = []
    for lag in (1,7):
        for task, target in [('demand', y),('price',c)]:
            p = baseline_values(df, x.index[va], task, lag)
            rows.append(dict(task=task, family=f'baseline_{lag}', **metrics(target[va],p,task)))
    pd.DataFrame(rows).to_csv(ROOT/'results/validation_baselines.csv', index=False)
    settings = json.loads((ROOT/'results/initial_selection.json').read_text())
    importance = []
    groups = {
        'calendar':lambda c: c.startswith(('slot_', 'dow_', 'annual_')) or c in ('holiday','working_day'),
        'weather':lambda c:any(s in c for s in ('syd_','mel_','bne_','temp_')),
        'interstate':lambda c:c.startswith(('qld_','vic_')),
        'nsw_demand_history':lambda c:c.startswith('nsw_demand_'),
        'nsw_price_history':lambda c:c.startswith(('nsw_price_', 'price_margin_')),
    }
    for task, target in [('demand',y),('price',c)]:
        conf = settings[task]['tree']
        m = make_estimator(task,'tree',conf['param']).fit(x.loc[tr],target[tr])
        base = metrics(target[va],predictions(m,x.loc[va],task,conf['threshold']),task)
        xv = x.loc[va].copy()
        rng = np.random.default_rng(9417)
        for group, rule in groups.items():
            cols = [c for c in x if rule(c)]
            for repeat in range(3):
                z = xv.copy()
                # 同一特征组采用相同的整天置换顺序，保留各天内部的时段顺序。
                order = rng.permutation(len(z)//48)
                z.loc[:,cols] = xv[cols].to_numpy().reshape(-1,48,len(cols))[order].reshape(-1,len(cols))
                score = metrics(target[va],predictions(m,z,task,conf['threshold']),task)
                delta = score['MAPE']-base['MAPE'] if task=='demand' else base['BalAcc']-score['BalAcc']
                importance.append(dict(task=task, group=group, repeat=repeat, deterioration_pp=delta))
    pd.DataFrame(importance).to_csv(ROOT/'results/group_permutation_importance.csv', index=False)
    print('Validation baselines and grouped day-permutation importance complete.')


if __name__=='__main__': main()
