"""First-run setup used by the Windows launchers (works on any OS).

Run with the interpreter of the environment that should be set up:

    runtime\\Scripts\\python.exe tools\\setup_env.py

It installs the dependency set matching the graphics card, downloads the base
models and, on Windows, ffmpeg. Later runs only verify that nothing is missing
and return immediately, so launchers can call it every time.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.download_models import (  # noqa: E402
    download_assets,
    inference_assets,
    missing_assets,
    windows_assets,
)

# Lives inside the environment, so each environment is set up once.
MARKER = Path(sys.prefix) / ".rvc_installed"
TORCH_VERSION = "2.7.1"
# (PyTorch wheel index, PyPI index) pairs, tried in order.
INDEXES = {
    "cu118": (
        ("https://mirrors.nju.edu.cn/pytorch/whl/cu118", "https://mirrors.pku.edu.cn/pypi/simple"),
        ("https://download.pytorch.org/whl/cu118", "https://pypi.org/simple"),
    ),
    "cu128": (
        ("https://mirrors.nju.edu.cn/pytorch/whl/cu128", "https://mirrors.pku.edu.cn/pypi/simple"),
        ("https://download.pytorch.org/whl/cu128", "https://pypi.org/simple"),
    ),
}


def gpu_names():
    """Names of the NVIDIA cards, empty when there is none (or no driver)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def pick_variant(names):
    """Dependency set for these cards: cu128 (RTX 50), cu118 (other NVIDIA), cpu."""
    if not names:
        return "cpu"
    if any("RTX 50" in name.upper().replace("RTX50", "RTX 50") for name in names):
        return "cu128"
    return "cu118"


def install_commands(variant, python=None):
    """The pip invocations for ``variant``; each entry is a list of fallbacks."""
    pip = [python or sys.executable, "-m", "pip", "install"]
    steps = [[pip + ["--upgrade", "pip", "wheel"]]]
    requirements = "requirments_%s_py312.txt" % variant
    if variant in INDEXES:
        pair = ["torch==%s+%s" % (TORCH_VERSION, variant), "torchaudio==%s+%s" % (TORCH_VERSION, variant)]
        steps.append(
            [
                pip + pair + ["--index-url", torch_index, "--extra-index-url", pypi_index]
                for torch_index, pypi_index in INDEXES[variant]
            ]
        )
    steps.append([pip + ["-r", requirements]])
    return steps


def run_steps(steps):
    for alternatives in steps:
        for command in alternatives:
            print("\n> " + " ".join(command), flush=True)
            if subprocess.call(command, cwd=PROJECT_ROOT) == 0:
                break
        else:
            return False
    return True


def torch_ready(variant, everything=False):
    """True when torch imports and, for CUDA variants, sees the card. With
    ``everything`` the other key packages must import as well."""
    check = "import torch,sys;%ssys.exit(0 if (%s) else 1)" % (
        "import gradio,faiss,FreeSimpleGUI;" if everything else "",
        "True" if variant == "cpu" else "torch.cuda.is_available()",
    )
    return (
        subprocess.call(
            [sys.executable, "-c", check],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variant", choices=("cpu", "cu118", "cu128"))
    parser.add_argument("--dry-run", action="store_true", help="only print the plan")
    parser.add_argument("--force", action="store_true", help="reinstall dependencies")
    args = parser.parse_args()

    names = gpu_names()
    variant = args.variant or pick_variant(names)
    assets = inference_assets() + (windows_assets() if os.name == "nt" else [])
    installed = MARKER.is_file() and MARKER.read_text(encoding="utf8").strip() == variant
    if args.dry_run:
        print("GPU: %s" % (", ".join(names) or "none"))
        print("variant: %s (installed: %s)" % (variant, installed))
        for alternatives in install_commands(variant, "python"):
            print("  " + " ".join(alternatives[0]))
        print("missing files: %d" % len(missing_assets(assets)))
        return 0

    # An environment that already works (e.g. a portable bundle) is adopted
    # as it is instead of being reinstalled.
    if not installed and not args.force and torch_ready(variant, everything=True):
        MARKER.write_text(variant, encoding="utf8")
        installed = True
    if not installed or args.force:
        print("显卡 / GPU: %s" % (", ".join(names) or "未检测到 NVIDIA 显卡，使用 CPU 版本"))
        print("首次运行，正在安装运行环境（%s），需要下载约 3~5GB，请耐心等待…" % variant)
        if not run_steps(install_commands(variant)):
            print("\n安装依赖失败，请检查网络后重新运行。")
            return 1
        if not torch_ready(variant):
            print("\nPyTorch 无法使用显卡，请更新 NVIDIA 显卡驱动后重新运行。")
            return 1
        MARKER.write_text(variant, encoding="utf8")

    if missing_assets(assets):
        print("正在下载基础模型…")
        try:
            for line in download_assets(assets):
                if " MB" in line:
                    print("\r" + line[:110].ljust(110), end="", flush=True)
                else:
                    print("\n" + line, flush=True)
        except Exception as error:
            print("\n%s" % error)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
