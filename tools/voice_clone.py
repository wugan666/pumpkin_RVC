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
