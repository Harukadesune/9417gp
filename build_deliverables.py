"""构造模型包与实验代码包，排除报告 PDF、运行缓存和不需要重复提交的原始数据。"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def build():
    with zipfile.ZipFile(ROOT/'submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'package').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p, str(p.relative_to(ROOT/'package')))
    with zipfile.ZipFile(ROOT/'experiment_code.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            rel = p.relative_to(ROOT)
            if not p.is_file() or any(part.startswith('.') or part=='__pycache__' for part in rel.parts):
                continue
            if p.suffix in ('.zip','.pdf') or rel.parts[0] in ('data','specs','example_package','template'):
                continue
            z.write(p,str(rel))
    print('Built submission.zip and experiment_code.zip')


if __name__=='__main__': build()
