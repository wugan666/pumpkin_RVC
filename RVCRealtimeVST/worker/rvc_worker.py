"""RVC inference worker for the RVC Realtime VST2/VST3 plugin."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import time
import traceback
from pathlib import Path

MAGIC = 0x50564352
PROTOCOL_VERSION = 1
HEADER_BYTES = 4096
MAX_FRAMES = 131072
MAP_BYTES = HEADER_BYTES + MAX_FRAMES * 4 * 2
INPUT_OFFSET = HEADER_BYTES
OUTPUT_OFFSET = HEADER_BYTES + MAX_FRAMES * 4
STATUS_TEXT_OFFSET = 128
STATUS_TEXT_BYTES = 512

STATUS_STARTING = 1
STATUS_LOADING = 2
STATUS_READY = 3
STATUS_ERROR = -1
STATUS_STOP = -2

SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258


def write_value(shared: mmap.mmap, offset: int, fmt: str, value) -> None:
    struct.pack_into("<" + fmt, shared, offset, value)


def read_value(shared: mmap.mmap, offset: int, fmt: str):
    return struct.unpack_from("<" + fmt, shared, offset)[0]


def write_status(shared: mmap.mmap, state: int, message: str) -> None:
    write_value(shared, 8, "i", state)
    encoded = message.encode("utf-8", errors="replace")[: STATUS_TEXT_BYTES - 1]
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + STATUS_TEXT_BYTES] = b"\0" * STATUS_TEXT_BYTES
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + len(encoded)] = encoded
    shared.flush()


class WinEvent:
    def __init__(self, name: str):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        kernel32.OpenEventW.restype = ctypes.c_void_p
        self._wait = kernel32.WaitForSingleObject
        self._wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._wait.restype = ctypes.c_uint32
        self._set = kernel32.SetEvent
        self._set.argtypes = [ctypes.c_void_p]
        self._set.restype = ctypes.c_int
        self._close = kernel32.CloseHandle
        self._close.argtypes = [ctypes.c_void_p]
        self._close.restype = ctypes.c_int
        self.handle = kernel32.OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, False, name)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), f"OpenEventW failed: {name}")

    def wait(self, timeout_ms: int) -> int:
        return int(self._wait(self.handle, timeout_ms))

    def set(self) -> None:
        if not self._set(self.handle):
            raise OSError(ctypes.get_last_error(), "SetEvent failed")

    def close(self) -> None:
        if self.handle:
            self._close(self.handle)
            self.handle = None


class RVCStreamEngine:
    """Thin adapter between the plugin's protocol and RVC's realtime engine.

    All signal processing lives in infer/realtime_engine.py, shared with the
    standalone realtime GUI, so both behave identically.
    """

    F0_METHODS = ("rmvpe", "fcpe", "pm")

    def __init__(self, cfg: dict):
        self.root = Path(cfg["rvc_root"]).resolve()
        os.chdir(self.root)
        sys.path.insert(0, str(self.root))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "4")

        import torch

        from configs.config import Config
        from infer import rtrvc
        from infer.realtime_engine import RealtimeEngine, RealtimeSettings

        self.torch = torch
        self.sample_rate = int(cfg["sample_rate"])
        zc = max(1, self.sample_rate // 100)
        # The plugin sends blocks of exactly this many frames.
        self.block_frame = int(round(float(cfg["block_ms"]) / 1000 * self.sample_rate / zc) * zc)

        self.settings = RealtimeSettings()
        self.settings.block_time = float(cfg["block_ms"]) / 1000
        self.settings.crossfade_time = float(cfg["crossfade_ms"]) / 1000
        self.settings.extra_time = float(cfg["extra_ms"]) / 1000

        self.config = Config()
        self.rvc = rtrvc.RVC(
            0.0,
            0.0,
            str(Path(cfg["model_path"]).resolve()),
            str(Path(cfg["index_path"]).resolve()) if cfg.get("index_path") else "",
            0.0,
            self.config,
            None,
        )
        self.engine = RealtimeEngine(
            self.rvc,
            self.settings,
            self.config.device,
            self.sample_rate,
            block_frame=self.block_frame,
        )
        self.effective_crossfade_ms = 1000.0 * self.engine.sola_buffer_frame / self.sample_rate
        self.last_pitch = None
        self.last_formant = None
        self.last_index_rate = None

    def prewarm(self) -> None:
        self.engine.prewarm()
        if self.torch.device(self.config.device).type == "cuda":
            self.torch.cuda.synchronize(self.config.device)

    def process(self, audio, pitch: float, formant: float, index_rate: float,
                rms_mix: float, threshold: float, f0_method: int):
        if len(audio) != self.block_frame:
            raise ValueError(f"block mismatch: worker={self.block_frame}, request={len(audio)}")

        if self.last_pitch != pitch:
            self.rvc.change_key(pitch)
            self.last_pitch = pitch
        if self.last_formant != formant:
            self.rvc.change_formant(formant)
            self.last_formant = formant
        if self.last_index_rate != index_rate:
            self.rvc.change_index_rate(index_rate)
            self.last_index_rate = index_rate
        self.settings.rms_mix_rate = float(rms_mix)
        self.settings.threhold = float(threshold)
        self.settings.f0method = self.F0_METHODS[max(0, min(2, int(f0_method)))]
        return self.engine.process(audio)


def run(args: argparse.Namespace) -> int:
    shared = None
    request_event = None
    response_event = None
    try:
        shared = mmap.mmap(-1, MAP_BYTES, tagname=args.map, access=mmap.ACCESS_WRITE)
        request_event = WinEvent(args.request)
        response_event = WinEvent(args.response)
        if read_value(shared, 0, "I") != MAGIC or read_value(shared, 4, "I") != PROTOCOL_VERSION:
            raise RuntimeError("RVC VST protocol mismatch")
        write_status(shared, STATUS_LOADING, "Loading Python and CUDA")
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        # RVC's Config parses process-wide CLI flags intended for its WebUI.
        sys.argv = [sys.argv[0]]
        engine = RVCStreamEngine(cfg)
        write_status(shared, STATUS_LOADING, "Prewarming CUDA and F0")
        engine.prewarm()
        write_status(shared, STATUS_READY, f"Ready (actual CF {engine.effective_crossfade_ms:.0f} ms)")

        import numpy as np

        input_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=INPUT_OFFSET)
        output_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=OUTPUT_OFFSET)
        last_sequence = 0
        while True:
            result = request_event.wait(1000)
            if result == WAIT_TIMEOUT:
                continue
            if result != WAIT_OBJECT_0:
                raise OSError("WaitForSingleObject failed")
            if read_value(shared, 8, "i") == STATUS_STOP:
                break
            sequence = read_value(shared, 12, "I")
            if sequence == last_sequence:
                continue
            frames = read_value(shared, 20, "I")
            if frames <= 0 or frames > MAX_FRAMES:
                raise ValueError(f"invalid frame count: {frames}")
            started = time.perf_counter()
            processed = engine.process(
                input_view[:frames],
                read_value(shared, 32, "f"),
                read_value(shared, 36, "f"),
                read_value(shared, 40, "f"),
                read_value(shared, 44, "f"),
                read_value(shared, 48, "f"),
                read_value(shared, 64, "I"),
            )
            output_view[:frames] = processed
            write_value(shared, 56, "f", (time.perf_counter() - started) * 1000.0)
            write_value(shared, 16, "I", sequence)
            write_value(shared, 8, "i", STATUS_READY)
            last_sequence = sequence
            response_event.set()
        return 0
    except BaseException:
        message = traceback.format_exc()
        try:
            Path(args.config + ".log").write_text(message, encoding="utf-8")
        except Exception:
            pass
        try:
            if shared is not None:
                write_status(shared, STATUS_ERROR, message[-500:])
            if response_event is not None:
                response_event.set()
        except Exception:
            pass
        return 1
    finally:
        if request_event is not None:
            request_event.close()
        if response_event is not None:
            response_event.close()
        if shared is not None:
            shared.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
