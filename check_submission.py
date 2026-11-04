"""Check a model package before you submit it.

Runs your package for the last 14 complete days of the dataset, one day at a time, with
the same wrapper, time limits and output checks used in marking, and compares the result
with the two baselines. In marking, at most one failed day is tolerated (with a 2-mark penalty).

Usage (from the kit's folder, with your environment activated):
    python check_submission.py submission.zip
    python check_submission.py example_package       # a folder also works
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "harness"))
import dayahead as da  # noqa: E402
import sandbox as sb   # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "data", "nem_weather_halfhourly_2010_2026.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", help="submission.zip or a package folder")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--docker", action="store_true", help=argparse.SUPPRESS)  # staff only
    args = ap.parse_args()

    import pandas as pd
    work = tempfile.mkdtemp(prefix="c9417check_")
    try:
        if args.package.lower().endswith(".zip"):
            pkg = sb.safe_extract(args.package, os.path.join(work, "pkg"))
        else:
            pkg = sb.find_package_root(os.path.abspath(args.package))
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    problems = sb.structure_problems(pkg)
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        return 1
    print(f"Package found at {pkg}\n")

    df = da.load_dataset(args.data)
    days = da.complete_days(df)[-args.days:]
    ok_pred, ok_truth, ok_b1, ok_b7 = [], [], [], []
    failures = 0
    for d in days:
        history, weather_next = da.make_inputs(df, d)
        in_dir = os.path.join(work, "in_" + d.strftime("%Y%m%d"))
        os.makedirs(in_dir)
        history.to_parquet(os.path.join(in_dir, "history.parquet"))
        weather_next.to_parquet(os.path.join(in_dir, "weather_next.parquet"))

        res = sb.run_one_day(pkg, in_dir, work, use_docker=args.docker)
        if res["status"] == "ok":
            print(f"{d.date()}  ok    init {res['init_seconds']:5.1f}s  predict {res['predict_seconds']:5.1f}s")
            ok_pred.append(pd.DataFrame(res["predictions"], index=weather_next.index))
            ok_truth.append(da.targets(df, d))
            ok_b1.append(da.baseline(history, weather_next, 1))
            ok_b7.append(da.baseline(history, weather_next, 7))
        else:
            failures += 1
            print(f"{d.date()}  FAIL  {res['stage']}: {res['error'].splitlines()[0]}")

    if ok_pred:
        truth = pd.concat(ok_truth)
        note = "" if not failures else f" (successful days only: {len(ok_pred)} of {len(days)})"
        print(f"\nScores{note}:")
        print("(in-sample if your models were trained on these days: they check that the package\n"
              " runs, not how it will perform in the live week)")
        print(f"{'':28s}{'MAPE %':>8s}{'MAE MW':>9s}{'BalAcc %':>10s}{'Acc %':>8s}")
        for name, p in (("Your package", ok_pred), ("Same half-hour yesterday", ok_b1),
                        ("Same half-hour last week", ok_b7)):
            s = da.score(truth, pd.concat(p))
            print(f"{name:28s}{s['MAPE']:8.2f}{s['MAE']:9.1f}{s['BalAcc']:10.1f}{s['Acc']:8.1f}")
    if failures:
        print(f"\n{failures} of {len(days)} days FAILED. In marking, one failed day costs 2 marks and more than one\n"
              "means no live-evaluation marks. Fix every failure before submitting.")
        return 1
    print("\nAll days ran successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
