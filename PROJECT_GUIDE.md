# 项目运行与结果索引

代码使用说明和材料索引,原始项目说明见 `specs/project_spec.pdf`。项目要求允许 AI 辅助代码，但报告必须由小组撰写；因此这里没有生成可提交的报告正文或报告 PDF。

## 环境与复现

使用 Python 3.11，安装原项目指定的全部版本：

完整交付目录保留了 `data/`。实验代码 zip 不重复包含发布 CSV；若只解压该 zip，请先将原始发布文件放到 `data/nem_weather_halfhourly_2010_2026.csv`，不要替换成其他数据。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

也可分步运行：

```bash
python experiments.py
python supplementary.py
python make_figures.py
python diagnostics.py
python package/train.py data/nem_weather_halfhourly_2010_2026.csv
python verify_project.py
python check_submission.py submission.zip
```

最后一步检查的是交付的 zip；若重新训练，请先重新打包：

```bash
python build_deliverables.py
python check_submission.py submission.zip
```

完整历史模拟会运行多次模型拟合，时间取决于机器。最终模型的 `train.py` 只重建最终模型，不会重复历史模拟。训练、模拟均固定随机种子 9417。`results/environment.txt` 保存实际运行环境，`results/reproducibility.json` 保存两次最终训练的模型文件校验结果。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `package/model.py` | 共享特征函数、训练表、无状态 Forecaster 接口 |
| `package/train.py` | 最终模型训练入口与所有算法构造函数 |
| `package/settings.json` | 历史模拟选出的策略及模型参数 |
| `experiments.py` | 三阶段实验、模型选择、在线监测、适应策略、置信区间 |
| `supplementary.py` | 验证期基线与整天置换的分组特征重要性 |
| `make_figures.py` | 从真实结果生成 PNG/SVG 图表和诊断表 |
| `diagnostics.py` | 价格尖峰、负价格、输入报警与误差的分层统计及事件窗口 |
| `verify_project.py` | 时间泄漏、标签定义、API、重训练预算检查 |
| `build_deliverables.py` | 构造最终提交包及实验代码 zip |
| `results_browser.ipynb` | 在 Notebook 中浏览已保存的结果与图表 |

## 实验边界

- 按区间开始日确定市场日：`timestamp - 30 minutes`。日末 00:00 属于前一天。
- 原始前 14 天仅用于构造滞后特征；模型样本从 2010-01-15 开始。
- 初选：2010–2013 拟合；2014 选择每类算法的超参数和分类阈值；2010–2014 重拟合冻结模型。
- 比较 Ridge / Logistic Regression 与直方图梯度提升树。树模型关闭随机内部早停划分；标准化只在各次训练数据上拟合。
- 分类阈值在 2014 上按平衡准确率选择，以后冻结；没有使用测试期标签重新调阈值。
- 模拟：2015-01-01 至 2026-10-31，逐日计算可用输入、预测和监测状态。
- 策略选择只看 2015–2024；2025–2026 作为后续审计，不用于修改已经选定的策略。2026 年结果只有 1–10 月。
- 最终训练按照选定策略使用全部已发布历史或最后 730 天。最终阶段允许利用截至 2026-10-31 的数据。
- 每任务分别比较冻结、每两年扩展窗口、每两年 730 天滑动窗口、误差触发滑动窗口、输入触发滑动窗口。
- 定期替换生效日为 2016/2018/2020/2022/2024/2026 年 1 月 1 日；漂移策略两次替换至少相隔 365 天；各组合最多 6 次。
- 输入监测：2010–2013 标准化的日均/日标准差（悉尼温度、太阳辐射、NSW 需求），28 天均值向量的均方根距离；阈值为 2014 统计量的第 99 百分位。第 D 天需求在 D 日末已知，触发替换从 D+1 开始生效。
- 误差监测：需求日 MAPE / 分类日错误率的 28 日平均。以 2014 验证误差均值为参考，以 2014 28 日均值的 99 百分位与参考之比设定报警倍数（最低 1.05）。替换后参考改为替换前最近最多 90 个已观测误差的均值，清空监测窗口；至少积累 28 个新误差才重新报警。
- 误差校准来自 2014 选型验证集，因此校准本身受验证选择影响；没有使用 2015 以后信息设初始阈值。
- 为加速，某一固定模型的后续预测会批量缓存；未来目标不参与拟合、决策、阈值或状态更新。每次重训练丢弃未来缓存。
- PCA 和 KMeans 只用 2010–2013 的日均归一化 48 点需求曲线拟合一次。按训练期 silhouette 在 k=2…6 中选择。聚类仅作事后分析，不作为次日预测特征。
- 所有月度区间包含 2000 次整天 bootstrap；需求另算 IID 正态区间，分类另算二项正态及 Hoeffding 区间。

## 原始结果索引

| 材料 | 文件 |
| --- | --- |
| 描述统计 | `results/data_summary.csv` |
| 2014 算法参数比较与基线 | `results/validation.csv`, `validation_baselines.csv` |
| 去天气 / 去州际特征消融 | `results/ablations.csv` |
| 整天分组置换重要性 | `results/group_permutation_importance.csv` |
| 全部模型的年度四指标 | `results/annual_metrics.csv` |
| 全部模型的月度四指标与区间 | `results/monthly_metrics_ci.csv` |
| 策略选择和后续审计 | `results/strategy_comparison.csv` |
| 全量逐时预测与真实标签 | `results/pred_*.parquet` |
| 两种区间的平均宽度 | `results/ci_width_comparison.csv` |
| 聚类选择 / 典型曲线 / 逐年占比 | `results/cluster_selection.csv`, `cluster_profiles.csv`, `cluster_yearly_shares.csv` |
| 逐日聚类、距离、新颖性 | `results/day_types.csv` |
| 各类的周末 / 冬夏占比和峰值时段 | `results/cluster_characteristics.csv` |
| 聚类与误差 / 逐年新颖性与误差 | `results/cluster_error_summary.csv`, `novelty_error_yearly.csv` |
| 类别比例 | `results/class_balance.csv` |
| 价格尖峰 / 负价格 / 年度市场变化 | `results/daily_market_diagnostics.csv`, `annual_market_diagnostics.csv` |
| 按尖峰、报警、新颖性分组的误差 | `results/stratified_error_diagnostics.csv` |
| 市场事件前后 90 天描述统计 | `results/descriptive_event_windows.csv`（受季节混杂，不能据此归因） |
| 输入监测统计量与报警起始日期 | `results/input_detector.csv`, `input_alarm_starts.csv` |
| 误差监测状态与报警起始日期 | `results/online_signals.csv`, `error_alarm_starts_*.csv` |
| 实际重训练日与训练范围 | `results/retraining_log.csv` |
| 周表现证据与经验范围 | `results/weekly_forecast_evidence.csv`, `weekly_empirical_ranges.csv` |
| 全部协议参数 / 特征名称 | `results/protocol.json` |
| 官方 zip 检查输出 | `results/submission_check.txt` |

`weekly_empirical_ranges.csv` 是历史周指标的经验分位数，不是未来周的校准预测区间，也不是月度均值置信区间。分别提供历史十二月周和近期周两组证据。报告中的实时周点预测、区间取舍及理由需要小组自行确定。历史适应模型和新拟合的最终模型并不完全相同，应在自行撰写时说明。

## 图表索引

8 张图均提供高分辨率 PNG 和可编辑 SVG。图内使用英文标签以保证跨机器字体兼容。
`figures/panels/` 另提供具有完整坐标标签的部分子图独立版本，可在报告中按适当宽度插入，避免缩小整张多面板图后文字过小。

1. `01_frozen_monthly_ci`：冻结模型的月度 MAPE / accuracy、两种区间和两个基线。
2. `02_annual_all_models`：所有模型年度 MAPE、MAE、balanced accuracy、accuracy。
3. `03_adaptation_monthly`：选定模型的各适应策略月度比较。
4. `04_day_types`：k 的选择、归一化典型曲线、逐年占比。
5. `05_input_drift_novelty`：输入监测、聚类新颖性、冻结需求误差。
6. `06_error_detector_retrains`：误差监测阈值及重训练生效日期。
7. `07_observed_forecasts`：不同时期的观测与预测需求曲线。
8. `08_class_balance`：类别比例、accuracy 与 balanced accuracy。