"""生成数值诊断表；这里的统计关联不等于因果效应。"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT/'results'
sys.path.insert(0,str(ROOT/'package'))
from model import load_table


def main():
    df = load_table(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    day = (df.index-pd.Timedelta(minutes=30)).normalize()
    price = df.nsw_price.groupby(day).agg(['mean','max','min'])
    price.columns = ['price_mean','price_max','price_min']
    daily = price.join(df.nsw_demand.groupby(day).mean().rename('demand_mean'))
    daily['previous_price_mean'] = daily.price_mean.shift(1)
    daily['previous_price_max'] = daily.price_max.shift(1)
    balance = pd.read_csv(R/'class_balance.csv',parse_dates=['day']).set_index('day')
    daily = daily.join(balance)
    daily['previous_day_spike_1000'] = daily.previous_price_max > 1000
    daily['negative_price_fraction'] = (df.nsw_price<0).groupby(day).mean()
    daily.to_csv(R/'daily_market_diagnostics.csv',index_label='day')
    daily.groupby(daily.index.year).agg(mean_demand=('demand_mean','mean'),mean_price=('price_mean','mean'),
        mean_positive_fraction=('positive_fraction','mean'),negative_price_fraction=('negative_price_fraction','mean'),
        days_with_1000_spike=('price_max',lambda v:int((v>1000).sum()))).to_csv(R/'annual_market_diagnostics.csv')
    choice = json.loads((ROOT/'package/settings.json').read_text())
    records = []
    detector = pd.read_csv(R/'input_detector.csv',parse_dates=['day']).set_index('day')
    types = pd.read_csv(R/'day_types.csv',parse_dates=['day']).set_index('day')
    for task in ('demand','price'):
        for name in [f'frozen_{choice[task]["family"]}',choice[task]['strategy']]:
            f = pd.read_parquet(R/f'pred_{task}_{name}.parquet')
            f['loss'] = 100*np.abs(f.truth-f.prediction)/f.truth if task=='demand' else 100*(f.truth!=f.prediction)
            d = f.groupby('day').loss.mean().to_frame().join(daily[['previous_day_spike_1000','positive_fraction']]).join(detector.alarm).join(types.novel)
            for factor in ('previous_day_spike_1000','alarm','novel'):
                for value, sub in d.groupby(factor):
                    records.append(dict(task=task,model=name,factor=factor,value=bool(value),days=len(sub),
                                        mean_loss=float(sub.loss.mean()),positive_fraction=float(sub.positive_fraction.mean())))
    pd.DataFrame(records).to_csv(R/'stratified_error_diagnostics.csv',index=False)
    # 仅描述事件前后窗口，时间重合不能作为因果归因的依据。
    frozen = pd.read_parquet(R/f'pred_demand_frozen_{choice["demand"]["family"]}.parquet')
    frozen['ape'] = 100*np.abs(frozen.truth-frozen.prediction)/frozen.truth
    de = frozen.groupby('day').ape.mean()
    joined = detector.join(de.rename('frozen_demand_mape'))
    eventrows=[]
    for name,date in [('five_minute_settlement','2021-10-01'),('nem_market_suspension','2022-06-15'),('liddell_closure','2023-04-28')]:
        t=pd.Timestamp(date)
        for label,lo,hi in [('before',t-pd.Timedelta(days=90),t-pd.Timedelta(days=1)),('after',t,t+pd.Timedelta(days=89))]:
            s=joined.loc[lo:hi]
            eventrows.append(dict(event=name,event_date=date,window=label,days=len(s),
                                  input_alarm_fraction=float(s.alarm.mean()),frozen_mape=float(s.frozen_demand_mape.mean())))
    pd.DataFrame(eventrows).to_csv(R/'descriptive_event_windows.csv',index=False)
    print('Diagnostic tables created; event windows are descriptive and season-confounded.')


if __name__=='__main__': main()
