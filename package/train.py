import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys
try:
    import resource
except ImportError:  # 兼容 Windows；Linux 推理环境不需要此分支。
    resource = None
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from model import load_table, training_table

#按照任务和参数创建模型
def make_estimator(task, family, param):
    if family == 'linear':
        core = Ridge(alpha=float(param)) if task == 'demand'else LogisticRegression(C=float(param), max_iter=1500, random_state=9417)
        return make_pipeline(StandardScaler(), core)
    #创建树模型
    common = dict(max_iter=160, max_leaf_nodes=int(param), learning_rate=0.08,
                  l2_regularization=10., min_samples_leaf=40, random_state=9417,
                  early_stopping=False)
    #根据任务创建回归树模型或分类树模型
    return HistGradientBoostingRegressor(**common) if task == 'demand' else HistGradientBoostingClassifier(**common)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    settings = json.loads((here / 'settings.json').read_text())
    #读取文件
    df = load_table(args.csv)
    x, y, c = training_table(df)
    days = (x.index - pd.Timedelta(minutes=30)).normalize()
    #空字典用于保存训练好的模型
    fitted = {}
    t0 = time.monotonic() # 测量训练用时
    # 分别训练两个任务  176 个特征 → 真实需求/真实价格类别
    for task, target in [('demand', y), ('price', c)]:
        conf = settings[task]
        #window_days: 从最后一个市场日往前数多少天，窗口为 0，就使用最早的可训练日期
        cutoff = days.max() - pd.Timedelta(days=conf['window_days'] - 1) if conf['window_days'] else days.min()
        mask = days >= cutoff # 日期达到训练起点 → True 早于训练起点 → False
        model = make_estimator(task, conf['family'], conf['param'])
        model.fit(x.loc[mask], target[mask])
        fitted[task] = model # fitted['demand'] = 需求模型 / fitted['price'] = 分类模型
        print(task, conf, 'training rows', int(mask.sum()), flush=True)
    #保存两个模型和分类阈值
    art = here / 'artifacts'
    art.mkdir(exist_ok=True)
    fitted['price_threshold'] = settings['price'].get('threshold', 0.5)
    joblib.dump(fitted, art / 'models.joblib', compress=3)
    metadata = dict(seed=9417, numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                    data_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                    last_market_day=str(days.max().date()), settings=settings,
                    training_seconds=round(time.monotonic() - t0, 2),
                    peak_memory_mb=(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform=='darwin' else 1024), 1) if resource else None))
    (art / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Saved', art, flush=True)


if __name__ == '__main__':
    main()
