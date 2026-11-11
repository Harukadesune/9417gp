"""从保存的实验结果生成高分辨率图表和数值诊断表。"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/comp9417-mpl')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R, F = ROOT/'results', ROOT/'figures'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.dpi':130, 'savefig.dpi':220, 'axes.grid':True, 'grid.alpha':.18})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F/f'{name}.png', bbox_inches='tight')
    fig.savefig(F/f'{name}.svg', bbox_inches='tight')
    # 另存完整坐标标签的独立子图，便于在报告中保持文字清晰。
    if len(fig.axes)>1:
        panels = F/'panels'
        panels.mkdir(exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for j, ax in enumerate(fig.axes, start=1):
            if not ax.get_xlabel() or not any(t.get_visible() for t in ax.get_xticklabels()):
                continue
            bbox = ax.get_tightbbox(renderer).expanded(1.04,1.08).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(panels/f'{name}_panel{j}.png', bbox_inches=bbox)
            fig.savefig(panels/f'{name}_panel{j}.svg', bbox_inches=bbox)
    plt.close(fig)


def main():
    monthly = pd.read_csv(R/'monthly_metrics_ci.csv')
    # 按照附录中的二项分布正态近似公式计算准确率区间宽度。
    # 同时统一早期基于样本标准差计算的近似区间结果。
    cls = monthly.task == 'price'
    p = monthly.loc[cls, 'Acc']/100
    half = 196*np.sqrt(p*(1-p)/monthly.loc[cls, 'n'])
    monthly.loc[cls, 'iid_low'] = np.maximum(0, monthly.loc[cls, 'Acc']-half)
    monthly.loc[cls, 'iid_high'] = np.minimum(100, monthly.loc[cls, 'Acc']+half)
    monthly.to_csv(R/'monthly_metrics_ci.csv', index=False)
    annual = pd.read_csv(R/'annual_metrics.csv')
    selected = json.loads((ROOT/'package/settings.json').read_text())
    # 一、绘制冻结模型月度表现、两类置信区间和两种基线。
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for i, task in enumerate(('demand','price')):
        metric = 'MAPE' if task=='demand' else 'Acc'
        for j, family in enumerate(('linear','tree')):
            ax = axes[i,j]
            s = monthly[(monthly.task==task)&(monthly.model==f'frozen_{family}')]
            t = pd.to_datetime(s.period)
            ax.fill_between(t, s.block_low, s.block_high, alpha=.2, color='#2a72a5', label='95% day bootstrap')
            ax.fill_between(t, s.iid_low, s.iid_high, alpha=.45, color='#ef9f32', label='95% half-hour IID')
            ax.plot(t, s[metric], lw=1.1, color='#14527b', label=f'Frozen {family}')
            for lag, color in [(1,'#555555'),(7,'#a06363')]:
                b = monthly[(monthly.task==task)&(monthly.model==f'baseline_{lag}')]
                ax.plot(pd.to_datetime(b.period), b[metric], color=color, lw=.7, alpha=.65, label=f'Lag {lag} baseline')
            ax.set_title(f'{task.title()}: {family}'); ax.set_ylabel(f'{metric} (%)')
            ax.legend(fontsize=7, ncol=2)
            ax.set_xlabel('Market month (AEST)')
    save(fig, '01_frozen_monthly_ci')
    # 二、比较所有模型的年度结果，展示四项评估指标。
    fig, axes = plt.subplots(2, 2, figsize=(13,7))
    for ax, (task, metric) in zip(axes.flat, [('demand','MAPE'),('demand','MAE'),('price','BalAcc'),('price','Acc')]):
        for name, s in annual[annual.task==task].groupby('model'):
            ax.plot(s.period.astype(int), s[metric], label=name, lw=1.2)
        ax.axvline(2024.5, ls='--', color='black', lw=.8)
        ax.set(xlabel='Market year (2026: January-October)', ylabel=f'{metric} ({"MW" if metric=="MAE" else "%"})', title=f'{task.title()}: {metric}')
    axes[0,0].legend(fontsize=7, ncol=2)
    save(fig, '02_annual_all_models')
    # 三、比较选定模型在不同适应策略下的月度表现。
    fig, axes = plt.subplots(2, 1, figsize=(13,7), sharex=True)
    for ax, task in zip(axes, ('demand','price')):
        metric = 'MAPE' if task=='demand' else 'BalAcc'
        for name in [f'frozen_{selected[task]["family"]}', 'periodic_expanding','periodic_window','drift_error','drift_input']:
            s = monthly[(monthly.task==task)&(monthly.model==name)]
            ax.plot(pd.to_datetime(s.period), s[metric], lw=.9, label=name)
        ax.set(ylabel=f'{metric} (%)', title=task.title())
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel('Market month (AEST)')
    save(fig, '03_adaptation_monthly')
    # 四、展示主成分分析后的聚类选择、参考曲线与年度占比。
    profiles = pd.read_csv(R/'cluster_profiles.csv', index_col=0)
    shares = pd.read_csv(R/'cluster_yearly_shares.csv', index_col=0)
    cluster_select = pd.read_csv(R/'cluster_selection.csv')
    fig, axes = plt.subplots(1,3, figsize=(14,4))
    axes[0].plot(cluster_select.k, cluster_select.silhouette, 'o-')
    axes[0].set(xlabel='Number of clusters k', ylabel='Training silhouette score', title='2010-2013 cluster selection')
    for col in profiles:
        axes[1].plot(np.arange(48)/2, profiles[col], label=col)
    axes[1].set(xlabel='Interval start hour (AEST)', ylabel='Demand / daily mean', title='Training cluster mean profiles')
    axes[1].legend(fontsize=8)
    axes[2].stackplot(shares.index, *shares.to_numpy().T, labels=[f'cluster_{c}' for c in shares.columns], alpha=.85)
    axes[2].set(xlabel='Market year', ylabel='Share of days', title='Fixed-reference yearly shares', ylim=(0,1))
    save(fig, '04_day_types')
    # 五、展示输入偏移、曲线新颖性和预测误差。
    detector = pd.read_csv(R/'input_detector.csv', parse_dates=['day'])
    types = pd.read_csv(R/'day_types.csv', parse_dates=['day'])
    characteristics = []
    for j, column in enumerate(profiles):
        train_days = types[(types.day<'2014-01-01')&(types.cluster==j)].day
        profile = profiles[column].to_numpy()
        characteristics.append(dict(cluster=j, training_days=len(train_days),
            weekend_fraction=float((train_days.dt.dayofweek>=5).mean()),
            summer_fraction=float(train_days.dt.month.isin([12,1,2]).mean()),
            winter_fraction=float(train_days.dt.month.isin([6,7,8]).mean()),
            peak_start_hour=float(np.argmax(profile)/2), peak_to_mean=float(profile.max()),
            morning_mean=float(profile[12:20].mean()), midday_mean=float(profile[20:30].mean()),
            evening_mean=float(profile[32:42].mean())))
    pd.DataFrame(characteristics).to_csv(R/'cluster_characteristics.csv',index=False)
    fig, axes = plt.subplots(3,1, figsize=(13,8), sharex=True)
    axes[0].plot(detector.day, detector.statistic, lw=.8, label='28-day input displacement')
    axes[0].axhline(detector.threshold.iloc[0], ls='--', color='#b6473a', label='2014 99th percentile')
    axes[0].set_ylabel('Standardised distance'); axes[0].legend(fontsize=8)
    novelty = types.set_index('day').novel.resample('MS').mean()
    axes[1].plot(novelty.index, novelty, color='#9b673a')
    axes[1].set_ylabel('Fraction of novel days')
    s = monthly[(monthly.task=='demand')&(monthly.model==f'frozen_{selected["demand"]["family"]}')]
    axes[2].plot(pd.to_datetime(s.period), s.MAPE, label='Frozen demand MAPE')
    axes[2].set(xlabel='Market date (AEST)', ylabel='MAPE (%)')
    axes[2].legend(fontsize=8)
    save(fig, '05_input_drift_novelty')
    # 六、展示两项任务的误差监测状态及符合时间顺序的模型替换日期。
    signals = pd.read_csv(R/'online_signals.csv', parse_dates=['observed_day'])
    events = pd.read_csv(R/'retraining_log.csv', parse_dates=['effective_day'])
    fig, axes = plt.subplots(2,1, figsize=(13,7), sharex=True)
    for ax, task in zip(axes, ('demand','price')):
        s = signals[(signals.task==task)&(signals.strategy=='drift_error')]
        scale = 100 if task=='price' else 1
        ax.plot(s.observed_day, s.error_stat*scale, lw=.7, label='Trailing daily error (up to 28 days)')
        ax.plot(s.observed_day, s.error_threshold*scale, lw=.8, label='Online alarm threshold')
        for day in events[(events.task==task)&(events.strategy=='drift_error')].effective_day:
            ax.axvline(day, color='#c45a38', lw=.8, alpha=.7)
        ax.set(ylabel='MAPE (%)' if task=='demand' else 'Misclassification (%)', title=task.title())
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Observed date; vertical lines = replacement effective dates')
    save(fig, '06_error_detector_retrains')
    # 七、选择四个不同时期的周，比较需求观测值与预测值。
    chosen = pd.read_parquet(R/f'pred_demand_{selected["demand"]["strategy"]}.parquet')
    fig, axes = plt.subplots(2,2, figsize=(14,7))
    for ax, date in zip(axes.flat, ['2015-01-05','2021-07-05','2025-12-01','2026-10-19']):
        start = pd.Timestamp(date)
        sub = chosen[(chosen.day>=start)&(chosen.day<start+pd.Timedelta(days=7))]
        ax.plot(sub.index, sub.truth, lw=1, label='Observed'); ax.plot(sub.index, sub.prediction, lw=.9, label='Adapted forecast')
        ax.set(title=f'Week from {date}', ylabel='NSW demand (MW)', xlabel='Timestamp (AEST)')
        ax.tick_params(axis='x', labelrotation=20); ax.legend(fontsize=8)
    save(fig, '07_observed_forecasts')
    # 八、展示类别比例，以及平衡准确率与普通准确率的区别。
    balance = pd.read_csv(R/'class_balance.csv', parse_dates=['day'])
    fig, axes = plt.subplots(2,1, figsize=(13,6), sharex=True)
    b = balance.set_index('day').positive_fraction.resample('MS').mean()
    axes[0].plot(b.index, b, color='#3d8d75')
    axes[0].set_ylabel('Monthly positive fraction')
    s = monthly[(monthly.task=='price')&(monthly.model==selected['price']['strategy'])]
    axes[1].plot(pd.to_datetime(s.period), s.Acc, label='Accuracy')
    axes[1].plot(pd.to_datetime(s.period), s.BalAcc, label='Balanced accuracy')
    axes[1].set(ylabel='Classification score (%)', xlabel='Market month (AEST)'); axes[1].legend()
    save(fig, '08_class_balance')
    # 计算聚类与误差、报警与误差的描述性关联，不作因果解释。
    daily_error = chosen.assign(ape=100*np.abs(chosen.truth-chosen.prediction)/chosen.truth).groupby('day').ape.mean()
    linked = types.set_index('day').join(daily_error.rename('adapted_mape')).dropna()
    linked.groupby('cluster').agg(days=('adapted_mape','size'), mean_mape=('adapted_mape','mean'), mean_distance=('distance','mean')).to_csv(R/'cluster_error_summary.csv')
    linked.groupby(linked.index.year).agg(novel_share=('novel','mean'), mean_distance=('distance','mean'), adapted_mape=('adapted_mape','mean')).to_csv(R/'novelty_error_yearly.csv')
    widths = []
    for (task, name), s in monthly.groupby(['task','model']):
        widths.append(dict(task=task, model=name, mean_iid_width=float((s.iid_high-s.iid_low).mean()),
                           mean_block_width=float((s.block_high-s.block_low).mean())))
    pd.DataFrame(widths).to_csv(R/'ci_width_comparison.csv', index=False)
    # 提取连续报警区间的起始日，避免重复列出连续报警的每一天。
    alarms = detector.set_index('day').alarm
    starts = alarms & ~alarms.shift(1, fill_value=False)
    detector.loc[detector.day.isin(starts[starts].index)].to_csv(R/'input_alarm_starts.csv', index=False)
    for task in ('demand','price'):
        s = signals[(signals.task==task)&(signals.strategy=='drift_error')].copy()
        start = s.error_alarm & ~s.error_alarm.shift(1, fill_value=False)
        s.loc[start].to_csv(R/f'error_alarm_starts_{task}.csv', index=False)
    print('Created 8 figures (PNG + SVG), individually cropped panels, and diagnostic tables.')


if __name__=='__main__':
    main()
