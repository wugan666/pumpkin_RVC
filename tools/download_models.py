"""Download the base models RVC needs (HuBERT, RMVPE, pretrained G/D, mute set).

Used by the WebUI "一键克隆" flow so a fresh checkout can train without any
manual model setup, and usable from the command line:

    python -m tools.download_models            # inference + v2 40k training
    python -m tools.download_models --sr 48k   # another sample rate
    python -m tools.download_models --check    # only report what is missing

Files come from the official model repository. ``HF_ENDPOINT`` selects a mirror;
without it the official host is tried first and hf-mirror.com second.
"""

import argparse
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = "lj1995/VoiceConversionWebUI"
DEFAULT_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
CHUNK = 1 << 20


class Asset:
    def __init__(self, remote, local, label, extract_to=None, marker=None):
        self.remote = remote
        self.local = PROJECT_ROOT / local
        self.label = label
        # Archives are unpacked into ``extract_to``; ``marker`` proves it was done.
        self.extract_to = PROJECT_ROOT / extract_to if extract_to else None
        self.marker = PROJECT_ROOT / marker if marker else None

    def present(self):
        target = self.marker or self.local
        if target.is_dir():
            return any(target.iterdir())
        return target.is_file() and target.stat().st_size > 0


def inference_assets():
    return [
        Asset("hubert_base/config.json", "assets/hubert_base/config.json", "HuBERT"),
        Asset(
            "hubert_base/preprocessor_config.json",
            "assets/hubert_base/preprocessor_config.json",
            "HuBERT",
        ),
        Asset(
            "hubert_base/pytorch_model.bin",
            "assets/hubert_base/pytorch_model.bin",
            "HuBERT",
        ),
        Asset("rmvpe.pt", "assets/rmvpe/rmvpe.pt", "RMVPE"),
    ]


def training_assets(version="v2", sr="40k", if_f0=True):
    folder = "pretrained" if version == "v1" else "pretrained_v2"
    prefix = "f0" if if_f0 else ""
    assets = [
        Asset(
            "%s/%s%s%s.pth" % (folder, prefix, kind, sr),
            "assets/%s/%s%s%s.pth" % (folder, prefix, kind, sr),
            "pretrained %s" % kind,
        )
        for kind in ("G", "D")
    ]
    assets.append(
        Asset(
            "mute.zip",
            ".model-downloads/mute.zip",
            "mute",
            extract_to="logs",
            marker="logs/mute/0_gt_wavs",
        )
    )
    return assets


def windows_assets():
    """ffmpeg, which the WebUI needs to read audio files (Windows only; on
    Linux and macOS it comes from the package manager)."""
    return [
        Asset("ffmpeg.exe", "ffmpeg.exe", "ffmpeg"),
        Asset("ffprobe.exe", "ffprobe.exe", "ffmpeg"),
    ]


def missing_assets(assets):
    return [asset for asset in assets if not asset.present()]


def _endpoints():
    custom = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    return (custom,) if custom else DEFAULT_ENDPOINTS


def _fetch(url, destination, on_progress):
    """Download ``url`` to ``destination`` atomically, resuming a partial file."""
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "rvc-webui"})
    if offset:
        request.add_header("Range", "bytes=%d-" % offset)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code != 416 or not offset:
            raise
        # The partial file is not resumable (already complete or stale).
        part.unlink()
        return _fetch(url, destination, on_progress)
    with response:
        if offset and response.status != 206:
            offset = 0
        length = response.headers.get("Content-Length")
        total = offset + int(length) if length else 0
        done = offset
        with open(part, "ab" if offset else "wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                on_progress(done, total)
    if total and done != total:
        raise IOError("incomplete download: %d of %d bytes" % (done, total))
    os.replace(part, destination)


def download_assets(assets, should_stop=None):
    """Yield human readable progress lines while fetching the missing assets.

    Raises RuntimeError when a file cannot be fetched from any endpoint.
    """
    todo = missing_assets(assets)
    for number, asset in enumerate(todo, 1):
        name = "%s (%s)" % (asset.remote, asset.label)
        last_error = None
        for endpoint in _endpoints():
            url = "%s/%s/resolve/main/%s" % (endpoint, REPO, asset.remote)
            progress = {"line": "", "at": 0.0}

            def on_progress(done, total, progress=progress):
                if should_stop is not None and should_stop():
                    raise InterruptedError("stopped")
                now = time.time()
                if now - progress["at"] < 1.0 and done != total:
                    return
                progress["at"] = now
                progress["line"] = "[%d/%d] %s  %.0f / %.0f MB" % (
                    number,
                    len(todo),
                    name,
                    done / 1e6,
                    total / 1e6,
                )

            yield "[%d/%d] %s  <- %s" % (number, len(todo), name, endpoint)
            try:
                for line in _fetch_with_progress(url, asset.local, on_progress, progress):
                    yield line
                last_error = None
                break
            except InterruptedError:
                raise
            except Exception as error:  # try the next endpoint
                last_error = error
                yield "  ! %s: %s" % (endpoint, error)
        if last_error is not None:
            raise RuntimeError(
                "无法下载 %s，请检查网络或设置 HF_ENDPOINT 镜像后重试（%s）"
                % (asset.remote, last_error)
            )
        if asset.extract_to is not None:
            with zipfile.ZipFile(asset.local) as archive:
                root = asset.extract_to.resolve()
                for member in archive.namelist():
                    if not (root / member).resolve().is_relative_to(root):
                        raise RuntimeError("unsafe path in archive: %s" % member)
                archive.extractall(root)
    yield "基础模型已就绪" if todo else "基础模型已存在，无需下载"


def _fetch_with_progress(url, destination, on_progress, progress):
    """Run ``_fetch`` in a thread and yield its progress lines as they change."""
    result = {}

    def run():
        try:
            _fetch(url, destination, on_progress)
        except BaseException as error:
            result["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    shown = ""
    while worker.is_alive():
        worker.join(0.5)
        if progress["line"] and progress["line"] != shown:
            shown = progress["line"]
            yield shown
    if "error" in result:
        raise result["error"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--sr", choices=("32k", "40k", "48k"), default="40k")
    parser.add_argument("--no-f0", action="store_true", help="models without pitch")
    parser.add_argument(
        "--inference-only", action="store_true", help="skip the training models"
    )
    parser.add_argument("--check", action="store_true", help="only list missing files")
    args = parser.parse_args()

    assets = inference_assets()
    if not args.inference_only:
        assets += training_assets(args.version, args.sr, not args.no_f0)
    if args.check:
        missing = missing_assets(assets)
        for asset in missing:
            print("missing: %s" % asset.local.relative_to(PROJECT_ROOT))
        print("%d missing" % len(missing))
        return 1 if missing else 0
    is_tty = sys.stdout.isatty()
    for line in download_assets(assets):
        if is_tty and " MB" in line:
            width = shutil.get_terminal_size().columns - 1
            print("\r" + line[:width].ljust(width), end="", flush=True)
        else:
            print(("\n" if is_tty else "") + line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
