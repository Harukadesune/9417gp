"""在本地或受限评分沙箱中运行单日预测，并对模型压缩包作安全检查。"""
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = "comp9417-26t3"
HARD_TIMEOUT_S = 120          # 整个容器的时限，包含启动、导入、初始化六十秒和预测二十秒
MAX_PACKAGE_MB = 100
MAX_UNZIPPED_MB = 200


def docker_cmd(package_dir, input_dir, output_dir, name, image=IMAGE):
    """构造评分沙箱命令：禁用网络、根目录只读、仅临时目录可写，并限制用户权限和资源。"""
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=512m,exec",
        "--user", "65534:65534",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "4g", "--memory-swap", "4g", "--cpus", "2", "--pids-limit", "256",
        "-e", "HOME=/tmp",
        "-v", f"{os.path.abspath(package_dir)}:/pkg:ro",
        "-v", f"{os.path.abspath(input_dir)}:/in:ro",
        "-v", f"{os.path.abspath(output_dir)}:/out:rw",
        image, "python", "/harness/run_day.py",
        "--package", "/pkg", "--input", "/in", "--output", "/out/result.json",
    ]


def run_one_day(package_dir, input_dir, work_dir, use_docker, image=IMAGE):
    """运行模型包的一天预测；总是返回结果字典，将失败记录为错误状态。"""
    out_dir = os.path.join(work_dir, "out_" + uuid.uuid4().hex[:8])
    os.makedirs(out_dir)
    os.chmod(out_dir, 0o777)                     # 允许沙箱中的低权限用户写入
    result_path = os.path.join(out_dir, "result.json")
    if use_docker:
        name = "c9417-" + uuid.uuid4().hex[:12]
        cmd = docker_cmd(package_dir, input_dir, out_dir, name, image)
    else:
        name = None
        cmd = [sys.executable, os.path.join(HERE, "run_day.py"), "--package", package_dir,
               "--input", input_dir, "--output", result_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=HARD_TIMEOUT_S)
        code, tail = proc.returncode, (proc.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        if name:
            subprocess.run(["docker", "kill", name], capture_output=True)
        return {"status": "error", "stage": "sandbox",
                "error": f"hard time limit of {HARD_TIMEOUT_S} s exceeded", "predictions": None}
    if os.path.exists(result_path):
        try:
            with open(result_path) as fh:
                res = json.load(fh)
        except Exception as exc:
            res = {"status": "error", "stage": "sandbox",
                   "error": f"unreadable result file: {exc}", "predictions": None}
        shutil.rmtree(out_dir, ignore_errors=True)
        return recheck(res)
    shutil.rmtree(out_dir, ignore_errors=True)
    reason = "killed for exceeding the 4 GB memory limit" if code == 137 else f"exited with code {code}"
    return {"status": "error", "stage": "sandbox", "error": f"process {reason}. {tail}",
            "predictions": None}


def recheck(res):
    """在沙箱外再次校验输出，避免直接信任提交代码写出的结果。"""
    import math
    if not isinstance(res, dict) or res.get("status") != "ok":
        return res if isinstance(res, dict) else {"status": "error", "stage": "sandbox",
                                                  "error": "malformed result", "predictions": None}
    p = res.get("predictions")
    try:
        d = [float(x) for x in p["nsw_demand_mw"]]
        u = [float(x) for x in p["nsw_price_up"]]
        ok = (len(d) == 48 and len(u) == 48 and all(math.isfinite(x) and 0 <= x <= 20000 for x in d)
              and all(x in (0.0, 1.0) for x in u))
    except Exception:
        ok = False
    if not ok:
        return {"status": "error", "stage": "validate", "predictions": None,
                "error": "result failed validation outside the sandbox"}
    res["predictions"] = {"nsw_demand_mw": d, "nsw_price_up": [int(x) for x in u]}
    return res


def safe_extract(zip_path, dest):
    """安全解压模型包。兼容不同系统的路径分隔符，忽略 macOS 元数据。
拒绝绝对路径、父目录跳转、符号链接以及超过解压大小限制的内容。"""
    total = 0
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if name.startswith("/") or (parts and ":" in parts[0]) or ".." in parts:
                raise ValueError(f"unsafe path in zip: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symbolic link in zip: {info.filename}")
            if not parts or parts[0] == "__MACOSX":
                continue
            total += info.file_size
            if total > MAX_UNZIPPED_MB * 1024 * 1024:
                raise ValueError(f"zip expands to more than {MAX_UNZIPPED_MB} MB")
            members.append((info, parts, name.endswith("/")))
        for info, parts, is_dir in members:
            target = os.path.realpath(os.path.join(root, *parts))
            if not target.startswith(root + os.sep):
                raise ValueError(f"unsafe path in zip: {info.filename}")
            if is_dir:
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return find_package_root(dest)


def find_package_root(folder):
    """寻找顶层 model.py，或唯一顶层文件夹中的 model.py。"""
    if os.path.isfile(os.path.join(folder, "model.py")):
        return folder
    entries = [e for e in os.listdir(folder) if not e.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and os.path.isfile(os.path.join(folder, entries[0], "model.py")):
        return os.path.join(folder, entries[0])
    raise ValueError("model.py not found at the top level of the package")


def structure_problems(package_dir):
    problems = []
    for item in ("model.py", "train.py"):
        if not os.path.isfile(os.path.join(package_dir, item)):
            problems.append(f"missing {item}")
    art = os.path.join(package_dir, "artifacts")
    if not os.path.isdir(art):
        problems.append("missing artifacts/ folder")
    else:
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(art) for f in fs)
        if size > MAX_PACKAGE_MB * 1024 * 1024:
            problems.append(f"artifacts/ is {size / 2**20:.0f} MB, above the {MAX_PACKAGE_MB} MB limit")
    return problems
