"""在独立进程中运行某个目标日的 Forecaster，并校验输出。
该包装器与评分使用的接口一致；初始化限时六十秒，预测限时二十秒，失败写入结果 JSON。
用法：python run_day.py --package DIR --input DIR --output FILE
输入目录须包含 history.parquet 和 weather_next.parquet。"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import traceback

INIT_LIMIT_S = 60
PREDICT_LIMIT_S = 20
N_PERIODS = 48
DEMAND_MIN, DEMAND_MAX = 0.0, 20000.0


class TimeLimit(Exception):
    pass


# 在 Unix 和评分服务器上使用 SIGALRM 中断超时代码；Windows 没有该信号，
# 因此在 Windows 上使用工作线程，并在达到时限后停止等待。
USE_SIGNAL = hasattr(signal, "SIGALRM")


def _alarm(signum, frame):
    raise TimeLimit()


def _run_with_limit(seconds, fn, *args):
    if USE_SIGNAL:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(seconds)
        try:
            return fn(*args)
        finally:
            signal.alarm(0)
    box = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeLimit()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def validate(pred, expected_index):
    """校验预测的行数、索引、列、数值范围和类别；返回清洗后的列表或错误信息。"""
    import numpy as np
    import pandas as pd

    if not isinstance(pred, pd.DataFrame):
        return None, f"predict returned {type(pred).__name__}, expected a pandas DataFrame"
    missing = [c for c in ("nsw_demand_mw", "nsw_price_up") if c not in pred.columns]
    if missing:
        return None, f"missing columns: {missing}"
    if len(pred) != N_PERIODS:
        return None, f"expected {N_PERIODS} rows, got {len(pred)}"
    if not pred.index.equals(expected_index):
        return None, "index does not match weather_next.index"
    demand = pd.to_numeric(pred["nsw_demand_mw"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(demand)):
        return None, "nsw_demand_mw contains NaN, infinite or non-numeric values"
    if demand.min() < DEMAND_MIN or demand.max() > DEMAND_MAX:
        return None, f"nsw_demand_mw outside [{DEMAND_MIN:.0f}, {DEMAND_MAX:.0f}] MW"
    up = pd.to_numeric(pred["nsw_price_up"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isin(up, [0.0, 1.0])):
        return None, "nsw_price_up must contain only 0 and 1"
    return {"nsw_demand_mw": [float(x) for x in demand],
            "nsw_price_up": [int(x) for x in up]}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result = {"status": "error", "stage": "setup", "error": None,
              "init_seconds": None, "predict_seconds": None, "predictions": None}
    try:
        import pandas as pd
        history = pd.read_parquet(f"{args.input}/history.parquet")
        weather_next = pd.read_parquet(f"{args.input}/weather_next.parquet")
        result["target_index"] = [t.isoformat() for t in weather_next.index]

        result["stage"] = "import"
        sys.path.insert(0, args.package)
        import model  # 小组提交的模型模块

        result["stage"] = "init"
        t0 = time.time()
        forecaster = _run_with_limit(INIT_LIMIT_S, model.Forecaster, f"{args.package}/artifacts")
        result["init_seconds"] = round(time.time() - t0, 2)

        result["stage"] = "predict"
        t0 = time.time()
        pred = _run_with_limit(PREDICT_LIMIT_S, forecaster.predict,
                               history.copy(), weather_next.copy())
        result["predict_seconds"] = round(time.time() - t0, 2)

        result["stage"] = "validate"
        clean, err = validate(pred, weather_next.index)
        if err:
            result["error"] = err
        else:
            result["predictions"] = clean
            result["status"] = "ok"
            result["stage"] = "done"
    except TimeLimit:
        limit = INIT_LIMIT_S if result["stage"] == "init" else PREDICT_LIMIT_S
        result["error"] = f"time limit of {limit} s exceeded during {result['stage']}"
    except BaseException as exc:  # 同时捕获提交代码引发的退出和键盘中断异常
        tb = traceback.format_exc(limit=5)
        result["error"] = f"{type(exc).__name__}: {exc}\n{tb[-2000:]}"

    with open(args.output, "w") as fh:
        json.dump(result, fh, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)      # 不再等待已经超时的工作线程


if __name__ == "__main__":
    main()
