"""在 Python 3.11 中安装指定依赖后，一键复现实验、图表、模型和提交检查。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
R = ROOT/'results'


def run(args, logname):
    command = [sys.executable, '-u', *args]
    print('Running:', ' '.join(command), flush=True)
    proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (R/logname).write_text(proc.stdout)
    print(proc.stdout[-5000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f'Failed: {args}; see results/{logname}')


def main():
    if sys.version_info[:2] != (3,11):
        raise RuntimeError('Use Python 3.11 as required by the marking environment')
    R.mkdir(exist_ok=True)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (R/'environment.txt').write_text(freeze)
    run(['experiments.py'], 'experiment_run.txt')
    run(['supplementary.py'], 'supplementary_run.txt')
    run(['make_figures.py'], 'figure_run.txt')
    run(['diagnostics.py'], 'diagnostics_run.txt')
    csv = 'data/nem_weather_halfhourly_2010_2026.csv'
    run(['package/train.py', csv], 'final_training.txt')
    art = ROOT/'package/artifacts/models.joblib'
    first = hashlib.sha256(art.read_bytes()).hexdigest()
    run(['package/train.py', csv], 'reproducibility_training.txt')
    second = hashlib.sha256(art.read_bytes()).hexdigest()
    (R/'reproducibility.json').write_text(json.dumps({'first_sha256':first,'second_sha256':second,'identical':first==second}, indent=2))
    if first != second:
        raise RuntimeError('Two fits produced different artifact hashes')
    run(['verify_project.py'], 'verification_run.txt')
    run(['build_deliverables.py'], 'build_run.txt')
    run(['check_submission.py', 'submission.zip'], 'submission_check.txt')
    run(['build_deliverables.py'], 'build_run.txt')
    print('All steps passed; report prose and PDF must be authored by the group.')


if __name__=='__main__': main()
