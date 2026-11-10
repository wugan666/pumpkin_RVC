"""Automatic transposition: match the speaker's pitch to the target voice.

A model trained with this code base stores the median F0 of its training set
(``f0_median``). Comparing it with the median F0 of the incoming voice gives
the number of semitones that puts the source into the target's natural range,
which is what users otherwise find by trial and error with the pitch slider.
"""

import glob
import math
import os
import re

import numpy as np

MAX_SEMITONES = 24
# Voiced frames (10 ms each) needed before an estimate is shown / frozen.
MIN_FRAMES = 60
LOCK_FRAMES = 300


def dataset_f0_median(exp_dir, max_files=400):
    """Median F0 in Hz of an experiment's extracted pitch, or None."""
    files = sorted(glob.glob(os.path.join(exp_dir, "2b-f0nsf", "*.npy")))
    if not files:
        return None
    step = max(1, len(files) // max_files)
    values = []
    for path in files[::step]:
        try:
            f0 = np.load(path)
        except Exception:
            continue
        values.append(f0[f0 > 0])
    values = np.concatenate(values) if values else np.zeros(0)
    if values.size < MIN_FRAMES:
        return None
    return float(np.median(values))


def model_f0_median(checkpoint, model_path=""):
    """Target median F0 for a loaded model checkpoint, or None when unknown.

    Models saved before this metadata existed fall back to the pitch files of
    their training experiment when that still exists under logs/.
    """
    value = checkpoint.get("f0_median") if isinstance(checkpoint, dict) else None
    if value:
        return float(value)
    if isinstance(checkpoint, dict) and checkpoint.get("speaker_info"):
        return None  # one number cannot describe several speakers
    stem = os.path.splitext(os.path.basename(str(model_path or "")))[0]
    experiment = re.sub(r"_e\d+_s\d+$", "", stem, flags=re.IGNORECASE)
    if not experiment:
        return None
    return dataset_f0_median(os.path.join(os.getenv("index_root", "logs"), experiment))


def suggest_transpose(source_hz, target_hz):
    """Semitones that move ``source_hz`` onto ``target_hz``."""
    if not source_hz or not target_hz or source_hz <= 0 or target_hz <= 0:
        return 0
    semitones = int(round(12 * math.log2(target_hz / source_hz)))
    return max(-MAX_SEMITONES, min(MAX_SEMITONES, semitones))
