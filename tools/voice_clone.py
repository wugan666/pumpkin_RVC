"""Dataset preparation for the WebUI one-click voice cloning flow."""

import os
import re
import shutil

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".webm",
    ".mp4",
}
# The name becomes a folder, a model file and a faiss index path; faiss cannot
# open non-ASCII paths on Windows, so the name is kept ASCII.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
MIN_SECONDS = 10
RECOMMENDED_SECONDS = 60


def validate_name(name):
    name = str(name or "").strip()
    if not NAME_PATTERN.match(name):
        raise ValueError(
            "音色名称只能使用英文字母、数字、下划线和短横线（最长48个字符），例如 my_voice"
        )
    return name


def _uploaded_path(item):
    if isinstance(item, (str, os.PathLike)):
        return os.fspath(item)
    if isinstance(item, dict):
        return item.get("name") or item.get("path")
    return getattr(item, "name", None)


def _is_audio(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


def audio_seconds(path):
    """Duration in seconds, or 0 when the file cannot be probed."""
    try:
        import av

        with av.open(path) as container:
            if container.duration:
                return container.duration / av.time_base
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception:
        return 0.0


def clean_folder_path(folder):
    return str(folder or "").strip().strip('"').strip("'").strip()


def prepare_dataset(dataset_dir, files=None, recording=None, folder=None):
    """Gather the training audio and return ``(trainset_dir, count, seconds)``.

    A folder on its own is trained in place. Uploads and recordings are copied
    into ``dataset_dir``, which is owned by this flow and rebuilt on every run
    so that re-cloning a name never mixes in the previous attempt's audio.
    """
    folder = clean_folder_path(folder)
    uploads = [path for path in map(_uploaded_path, files or []) if path]
    if recording:
        uploads.append(_uploaded_path(recording))
    uploads = [path for path in uploads if path and os.path.isfile(path)]

    folder_files = []
    if folder:
        if not os.path.isdir(folder):
            raise ValueError("素材文件夹不存在：%s" % folder)
        folder_files = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, name)) and _is_audio(name)
        ]
        if not folder_files and not uploads:
            raise ValueError("素材文件夹里没有找到音频文件：%s" % folder)

    rejected = [os.path.basename(path) for path in uploads if not _is_audio(path)]
    if rejected:
        raise ValueError("不支持的文件类型：%s" % "、".join(rejected[:5]))
    if not uploads and not folder_files:
        raise ValueError("请先上传人声音频、录一段音，或填写素材文件夹路径")

    if uploads:
        sources = folder_files + uploads
        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
        for number, source in enumerate(sources):
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(source))
            shutil.copyfile(
                source, os.path.join(dataset_dir, "%03d_%s" % (number, stem))
            )
        trainset_dir = dataset_dir
    else:
        sources = folder_files
        trainset_dir = os.path.abspath(folder)

    seconds = sum(audio_seconds(path) for path in sources)
    if 0 < seconds < MIN_SECONDS:
        raise ValueError(
            "素材总时长只有 %.0f 秒，至少需要 %d 秒，建议 1~10 分钟以上"
            % (seconds, MIN_SECONDS)
        )
    return trainset_dir, len(sources), seconds


def describe_dataset(count, seconds):
    if seconds <= 0:
        return "素材：%d 个文件" % count
    text = "素材：%d 个文件，共 %d 分 %02d 秒" % (count, seconds // 60, seconds % 60)
    if seconds < RECOMMENDED_SECONDS:
        text += "\n提示：素材不足 1 分钟，音色相似度会比较有限，建议准备 1~10 分钟以上的干净人声"
    return text


# ---------------------------------------------------------------- epochs

# Training slices the material into pieces; about this many reach each epoch
# per second of audio (measured: 36 of 60 s at batch size 4).
SEGMENTS_PER_SECOND = 0.6
# With little material an epoch is only a few steps, so a fixed epoch count
# under-trains. On a 1 minute clone the similarity to the real voice kept
# rising up to roughly 360 steps and was flat after that.
MIN_STEPS = 400
MAX_EPOCH_FACTOR = 4
GPU_CACHE_SECONDS = 600


def plan_epochs(preset_epochs, seconds, batch_size):
    """Epoch count for a preset, raised when the material is short."""
    if seconds <= 0:
        return preset_epochs
    steps_per_epoch = max(1, int(seconds * SEGMENTS_PER_SECOND / max(1, batch_size)))
    # the quick preset stays quick: it gets a proportionally smaller floor
    steps = MIN_STEPS * min(preset_epochs, 50) // 50
    needed = -(-steps // steps_per_epoch)
    return int(min(max(preset_epochs, needed), preset_epochs * MAX_EPOCH_FACTOR))


# ---------------------------------------------------------------- quality

# Seconds analysed per file; enough to judge the recording conditions.
ANALYSIS_SECONDS = 120


def analyze_file(path):
    """Measure the recording conditions of one file, or None if unreadable."""
    import librosa
    import numpy as np

    try:
        audio, _ = librosa.load(path, sr=16000, mono=True, duration=ANALYSIS_SECONDS)
    except Exception:
        return None
    if audio.size < 1600:
        return None
    frames = audio[: audio.size // 320 * 320].reshape(-1, 320)
    level = 10 * np.log10((frames**2).mean(axis=1) + 1e-10)
    loud = float(np.percentile(level, 95))
    return {
        "clipped": float((np.abs(audio) >= 0.99).mean()),
        "floor_db": float(np.percentile(level, 5)),
        "loud_db": loud,
        # frames more than 40 dB under the loud parts count as silence
        "silence": float((level < loud - 40).mean()),
    }


def quality_report(paths, max_files=30):
    """Warnings about the training audio as a list of human readable lines."""
    step = max(1, len(paths) // max_files)
    stats = [s for s in (analyze_file(p) for p in paths[::step]) if s]
    if not stats:
        return []
    import numpy as np

    clipped = float(np.mean([s["clipped"] for s in stats]))
    floor = float(np.median([s["floor_db"] for s in stats]))
    loud = float(np.median([s["loud_db"] for s in stats]))
    silence = float(np.mean([s["silence"] for s in stats]))
    warnings = []
    if clipped > 0.001:
        warnings.append(
            "⚠ 录音有削波失真（%.1f%% 的采样点顶格），建议降低录音音量后重录" % (clipped * 100)
        )
    if loud - floor < 30:
        warnings.append(
            "⚠ 背景声较大（人声只比背景高 %d dB），可能带有伴奏、混响或噪声，建议先做人声分离"
            % round(loud - floor)
        )
    if loud < -35:
        warnings.append("⚠ 录音音量很小，建议靠近麦克风或提高录音音量")
    if silence > 0.6:
        warnings.append("⚠ 素材中大部分是静音（%d%%），有效人声偏少" % round(silence * 100))
    if not warnings:
        warnings.append("✓ 素材质量检查通过（无削波，背景干净）")
    return warnings


def list_dataset_files(trainset_dir):
    return [
        os.path.join(trainset_dir, name)
        for name in sorted(os.listdir(trainset_dir))
        if os.path.isfile(os.path.join(trainset_dir, name)) and _is_audio(name)
    ]


# ---------------------------------------------------------------- cleanup

# Intermediate training data inside logs/<name>; the index and logs are kept.
INTERMEDIATE_DIRS = (
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature256",
    "3_feature768",
)


def cleanup_experiment(exp_dir):
    """Delete checkpoints and extracted features of a finished clone.

    Returns the number of bytes freed. After this the voice can still be used
    (model, index) but training cannot be resumed.
    """
    freed = 0
    if not os.path.isdir(exp_dir):
        return freed
    for name in os.listdir(exp_dir):
        path = os.path.join(exp_dir, name)
        if os.path.isfile(path) and re.match(r"^[GD]_\d+\.pth$", name):
            freed += os.path.getsize(path)
            os.remove(path)
        elif os.path.isdir(path) and name in INTERMEDIATE_DIRS:
            for root, _, files in os.walk(path):
                freed += sum(os.path.getsize(os.path.join(root, f)) for f in files)
            shutil.rmtree(path)
    return freed
