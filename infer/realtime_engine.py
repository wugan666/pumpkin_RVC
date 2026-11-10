"""Streaming voice-conversion engine behind the realtime GUI.

The engine owns every buffer between "one block of microphone samples" and
"one block of converted samples", so the GUI's audio callback is a single
``process()`` call and the exact same code path can be run over a wav file:

    python -m infer.realtime_engine --model assets/weights/a.pth \
        --input in.wav --output out.wav

which is how the realtime path is tested without an audio device.
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph
from tools.torchgate import TorchGate

# Noise gate timing, in 10 ms hops.
GATE_HOLD_HOPS = 15  # keep the gate open this long after the level drops
GATE_RELEASE_HOPS = 6  # fade-out length once the hold expires
GATE_HYSTERESIS_DB = 4.0  # an open gate closes this far below the threshold
GATE_PREROLL_HOPS = 2  # re-open this much audio before the detected onset
GATE_OUTPUT_SPREAD_HOPS = 5  # the output mute is this much wider than the gate
GATE_OFF_DB = -60  # slider minimum: gate disabled

# Automatic threshold: this far above the measured background level.
AUTO_GATE_MARGIN_DB = 8.0
AUTO_GATE_RANGE_DB = (-55.0, -30.0)
AUTO_GATE_HISTORY_HOPS = 3000  # 30 s of level history
AUTO_GATE_MIN_HOPS = 100  # history needed before the estimate is used

# Loudness matching may not boost the model output by more than this factor.
MAX_ENVELOPE_GAIN = 6.0
# Output limiter: peaks are held under the ceiling by turning the gain down
# smoothly (no distortion); the soft clipper above the knee is only a last
# safety net for what slips through.
LIMITER_CEILING = 0.7  # about -3 dBFS
LIMITER_RELEASE_HOPS = 20  # gain recovers over 200 ms
LIMITER_KNEE = 0.85
# The input is considered clipped by the microphone above this level.
INPUT_CLIP_LEVEL = 0.985


class RealtimeSettings:
    """Tunable parameters. Fields marked (live) are re-read on every block."""

    def __init__(self):
        self.pitch = 0
        self.formant = 0.0
        self.block_time = 0.25  # s
        self.crossfade_time = 0.05
        self.extra_time = 2.5
        # Seconds HuBERT re-encodes per block (rest is cached); 0 = everything.
        self.hubert_context = 1.0
        self.threhold = GATE_OFF_DB  # (live) noise gate threshold in dB
        # (live) derive the threshold from the measured background noise
        self.auto_threshold = False
        # (live) run no model at all while the gate is closed
        self.idle_skip = True
        self.I_noise_reduce = False  # (live)
        self.O_noise_reduce = False  # (live)
        self.rms_mix_rate = 0.0  # (live)
        self.index_rate = 0.0
        self.protect = 0.33
        self.f0method = "rmvpe"  # (live)
        # (live) pick the transposition from the speaker's and the model's pitch
        self.auto_pitch = False
        self.function = "vc"  # (live) "vc" converts, "im" monitors the input


def soft_limit(x, knee=LIMITER_KNEE):
    """Leave |x| <= knee untouched and bend the rest smoothly below 1.0."""
    magnitude = x.abs()
    room = 1.0 - knee
    bent = knee + room * torch.tanh((magnitude - knee) / room)
    return torch.where(magnitude > knee, torch.sign(x) * bent, x)


class RealtimeEngine:
    def __init__(self, rvc, settings, device, samplerate, block_frame=None):
        """``block_frame`` fixes the block size in samples for hosts that
        dictate it (the VST plugin); otherwise it follows ``block_time``."""
        self.rvc = rvc
        self.cfg = settings
        rvc.hubert_context = settings.hubert_context
        self.device = device
        self.samplerate = int(samplerate)
        cfg = settings

        self.zc = self.samplerate // 100
        # Blocks are a whole number of 20 ms HuBERT frames so that features
        # computed for earlier blocks stay aligned and can be reused.
        self.block_frame = max(1, round(cfg.block_time * 50)) * 2 * self.zc
        if block_frame is not None:
            self.block_frame = int(block_frame)
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = self._frames(cfg.crossfade_time)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = self._frames(cfg.extra_time)

        def zeros(length):
            return torch.zeros(length, device=device, dtype=torch.float32)

        self.input_wav = zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        self.input_wav_denoise = self.input_wav.clone()
        # Per-sample noise gate gain, aligned with input_wav.
        self.gate_env = torch.ones_like(self.input_wav)
        self.input_wav_res = zeros(160 * self.input_wav.shape[0] // self.zc)
        self.sola_buffer = zeros(self.sola_buffer_frame)
        self.sola_den_kernel = torch.ones(
            1, 1, self.sola_buffer_frame, device=device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.samplerate, new_freq=16000, dtype=torch.float32
        ).to(device)
        if rvc.tgt_sr != self.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=rvc.tgt_sr, new_freq=self.samplerate, dtype=torch.float32
            ).to(device)
        else:
            self.resampler2 = None
        # Bundled torch.istft is not CUDA Graph-capturable, so TorchGate
        # stays eager while resampling and RVC inference still use graphs.
        self.tg = TorchGate(
            sr=self.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(device)

        self.gate_tail = np.zeros(3 * self.zc, dtype=np.float32)
        self.gate_hold = 0
        self.level_history = np.zeros(0, dtype=np.float32)
        self.auto_threshold_db = None
        self.limiter_gain = 1.0
        # Gain at the end of each of the most recent hops (oldest first).
        self.gate_history = np.zeros(GATE_PREROLL_HOPS + 1, dtype=np.float32)
        # Read by the GUI thread; plain values so no locking is needed.
        self.stats = {
            "blocks": 0,
            "total_ms": 0.0,
            "overruns": 0,
            "input_db": -120.0,
            "output_db": -120.0,
            "gate_open": True,
            "sola_offset": 0,
            "auto_key": None,  # transposition chosen by auto pitch, if any
            "threshold": None,  # gate threshold in use when it is automatic
            "idle": False,  # True while the gate is closed and models are skipped
            "idle_blocks": 0,
            "input_clipped": 0,  # blocks in which the microphone signal clipped
            "limited_db": 0.0,  # gain reduction applied by the output limiter
            "glitches": 0,  # blocks muted because the model produced NaN/inf
        }

    def _frames(self, seconds):
        return int(np.round(seconds * self.samplerate / self.zc)) * self.zc

    def algorithm_delay(self):
        """Latency added by the engine itself, in seconds."""
        delay = (self.block_frame + self.crossfade_frame) / self.samplerate + 0.01
        if self.cfg.I_noise_reduce:
            delay += min(self.cfg.crossfade_time, 0.04)
        return delay

    def reset(self):
        for buffer in (
            self.input_wav,
            self.input_wav_denoise,
            self.input_wav_res,
            self.output_buffer,
            self.sola_buffer,
            self.nr_buffer,
        ):
            buffer.zero_()
        self.gate_env.fill_(1.0)
        self.gate_tail[:] = 0
        self.gate_hold = 0
        self.gate_history[:] = 0
        self.level_history = np.zeros(0, dtype=np.float32)
        self.auto_threshold_db = None
        self.limiter_gain = 1.0
        self.rvc.reset_state()

    def prewarm(self):
        """Run one block so lazy model loading and CUDA Graph capture happen
        before the audio stream starts instead of inside the first callbacks."""
        try:
            phase = torch.arange(
                self.input_wav_res.shape[0], device=self.device, dtype=torch.float32
            )
            self.input_wav_res.copy_(
                0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
            )
            if cuda_graph_enabled(self.device):
                run_cuda_graph(
                    self.resampler,
                    "realtime-input-resample",
                    lambda audio: self.resampler(audio),
                    self.input_wav[
                        -self.block_frame - (GATE_PREROLL_HOPS + 2) * self.zc :
                    ],
                )
            if self.cfg.I_noise_reduce:
                short = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
                self.tg(short.unsqueeze(0), self.input_wav.unsqueeze(0))
            self.rvc.prepare_silence()
            inferred = self._convert()
            # The second pass takes the cached-feature path, whose tensor
            # shapes differ; warm it too so the first live block is not slow.
            inferred = self._convert()
            if self.cfg.O_noise_reduce:
                self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
            if torch.cuda.is_available() and "cuda" in str(self.device):
                torch.cuda.synchronize(self.device)
        finally:
            self.reset()

    # ------------------------------------------------------------------ steps

    def _gate(self, block):
        """Noise gate with hysteresis, hold, ramps and a short pre-roll.

        Returns ``(samples, gain)`` covering the last GATE_PREROLL_HOPS hops of
        the previous block followed by this block, or ``(block, None)`` when
        the gate is off. Those earlier hops are returned again because an onset
        is only recognised once it is loud enough: re-opening the audio just
        before it keeps soft word beginnings in the model's context.

        Muting whole 10 ms slices the instant they fall under the threshold
        (the previous behaviour) chops word endings and clicks at every
        transition.
        """
        threshold = self.cfg.threhold
        zc = self.zc
        pre = GATE_PREROLL_HOPS
        if self.cfg.auto_threshold:
            threshold = self._auto_threshold(block, threshold)
        self.stats["threshold"] = threshold if self.cfg.auto_threshold else None
        if threshold <= GATE_OFF_DB:
            self.gate_history[:] = 1.0
            self.gate_hold = 0
            self.gate_tail = np.concatenate((self.gate_tail, block))[-3 * zc :]
            self.stats["gate_open"] = True
            return block, None
        hops = block.shape[0] // zc
        joined = np.concatenate((self.gate_tail, block))
        self.gate_tail = joined[-3 * zc :].copy()
        energy = np.concatenate(([0.0], np.cumsum(joined.astype(np.float64) ** 2)))
        ends = 3 * zc + (np.arange(hops) + 1) * zc

        def level_db(window):
            return 10 * np.log10((energy[ends] - energy[ends - window]) / window + 1e-12)

        # A single loud hop opens the gate quickly; the 40 ms average keeps it
        # open through the dips inside a word.
        level = np.maximum(level_db(zc), level_db(4 * zc))
        gains = np.empty(pre + 1 + hops, dtype=np.float32)
        gains[: pre + 1] = self.gate_history
        for i in range(hops):
            close_at = threshold - (GATE_HYSTERESIS_DB if self.gate_hold else 0.0)
            opening = not self.gate_hold and level[i] >= close_at
            if level[i] >= close_at:
                self.gate_hold = GATE_HOLD_HOPS
            elif self.gate_hold:
                self.gate_hold -= 1
            at = pre + 1 + i
            if self.gate_hold:
                gains[at] = 1.0
            else:
                gains[at] = max(0.0, gains[at - 1] - 1.0 / GATE_RELEASE_HOPS)
            if opening:
                for back in range(1, pre + 1):
                    gains[at - back] = max(gains[at - back], 1.0 - back / (pre + 1.0))
        self.gate_history = gains[-(pre + 1) :].copy()
        self.stats["gate_open"] = bool(self.gate_hold)
        covered = pre + hops
        envelope = np.interp(
            np.arange(covered * zc) / zc, np.arange(covered + 1), gains
        ).astype(np.float32)
        return joined[-covered * zc :] * envelope, envelope

    def _auto_threshold(self, block, fallback):
        """Gate threshold that follows the room: a fixed margin above the
        background level, taken as the 10th percentile of the last 30 s (speech
        sits far above that as long as the speaker pauses now and then)."""
        zc = self.zc
        hops = block[: block.shape[0] // zc * zc].reshape(-1, zc).astype(np.float64)
        level = 10 * np.log10((hops**2).mean(axis=1) + 1e-12)
        self.level_history = np.concatenate((self.level_history, level))[
            -AUTO_GATE_HISTORY_HOPS:
        ].astype(np.float32)
        if self.level_history.shape[0] >= AUTO_GATE_MIN_HOPS:
            floor = float(np.percentile(self.level_history, 10))
            low, high = AUTO_GATE_RANGE_DB
            self.auto_threshold_db = min(high, max(low, floor + AUTO_GATE_MARGIN_DB))
        if self.auto_threshold_db is None:
            return fallback if fallback > GATE_OFF_DB else AUTO_GATE_RANGE_DB[0]
        return self.auto_threshold_db

    def _limit(self, wav):
        """Peak limiter. ``wav`` is this block followed by the crossfade tail
        (the start of the next block), which serves as look-ahead so the gain
        is already down when a peak arrives; returns the limited block.

        Without it a shout or a hit on the desk leaves the sound card far
        above full scale and is hard-clipped there.
        """
        zc = self.zc
        hops = wav.shape[0] // zc
        peaks = wav[: hops * zc].abs().view(hops, zc).amax(dim=1).cpu().numpy()
        need = np.minimum(1.0, LIMITER_CEILING / (peaks + 1e-9))
        # Each hop also honours the next one, so its ramp ends low enough.
        need = np.minimum(need, np.append(need[1:], need[-1]))
        gains = np.empty(hops + 1, dtype=np.float32)
        gains[0] = gain = self.limiter_gain
        for i in range(hops):
            gain = min(need[i], gain + 1.0 / LIMITER_RELEASE_HOPS)
            gains[i + 1] = gain
        block_hops = self.block_frame // zc
        self.limiter_gain = float(gains[block_hops])
        self.stats["limited_db"] = float(-20 * np.log10(gains[1 : block_hops + 1].min()))
        ramp = np.interp(
            np.arange(self.block_frame) / zc, np.arange(hops + 1), gains
        ).astype(np.float32)
        return wav[: self.block_frame] * torch.from_numpy(ramp).to(wav.device)

    def _rms(self, wav):
        """Frame RMS matching librosa.feature.rms(frame=4*zc, hop=zc)."""
        frame = 4 * self.zc
        power = F.pad(wav[None, None] ** 2, (frame // 2, frame // 2))
        rms = torch.sqrt(F.avg_pool1d(power, frame, self.zc))
        return F.interpolate(
            rms, size=wav.shape[0] + 1, mode="linear", align_corners=True
        )[0, 0, :-1]

    def _convert(self):
        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.cfg.f0method,
        )
        if self.resampler2 is not None:
            infer_wav = run_cuda_graph(
                self.resampler2,
                "realtime-output-resample",
                lambda audio: self.resampler2(audio),
                infer_wav,
            )
        return infer_wav

    def process(self, indata):
        """Convert one block. ``indata`` is mono float32 of ``block_frame``
        samples; returns a mono float32 numpy array of the same length."""
        start_time = time.perf_counter()
        cfg = self.cfg
        block = self.block_frame
        converting = cfg.function == "vc"
        # A driver hiccup can deliver NaN or wildly out-of-range samples.
        indata = np.nan_to_num(
            np.asarray(indata, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        peak = float(np.max(np.abs(indata))) if indata.size else 0.0
        self.stats["input_db"] = 20 * np.log10(peak + 1e-6)
        if peak >= INPUT_CLIP_LEVEL:
            self.stats["input_clipped"] += 1
            indata = np.clip(indata, -1.0, 1.0)

        indata, envelope = self._gate(indata)
        self.input_wav[:-block] = self.input_wav[block:].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
        self.gate_env[:-block] = self.gate_env[block:].clone()
        if envelope is None:
            self.gate_env[-block:] = 1.0
        else:
            self.gate_env[-envelope.shape[0] :] = torch.from_numpy(envelope).to(
                self.device
            )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # input noise reduction and resampling
        if cfg.I_noise_reduce:
            self.input_wav_denoise[:-block] = self.input_wav_denoise[block:].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - block :]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-block:] = input_wav[:block]
            self.nr_buffer[:] = input_wav[block:]
            refreshed = block
            resample_input = self.input_wav_denoise[-block - 2 * self.zc :]
        else:
            # The gate may have rewritten its pre-roll, so resample that too.
            refreshed = block + GATE_PREROLL_HOPS * self.zc
            resample_input = self.input_wav[-refreshed - 2 * self.zc :]
        self.input_wav_res[-160 * (refreshed // self.zc + 1) :] = run_cuda_graph(
            self.resampler,
            "realtime-input-resample",
            lambda audio: self.resampler(audio),
            resample_input,
        )[160:]
        source = self.input_wav_denoise if cfg.I_noise_reduce else self.input_wav
        source = source[self.extra_frame :]
        # While the gate is closed across everything this block could output,
        # the result is muted anyway: skip all models and leave the GPU alone.
        spread = GATE_OUTPUT_SPREAD_HOPS * self.zc
        idle = (
            converting
            and cfg.idle_skip
            and envelope is not None
            and float(self.gate_env[max(0, self.extra_frame - spread) :].max()) == 0.0
        )
        self.stats["idle"] = idle
        # infer
        if idle:
            self.rvc.skip_block(self.block_frame_16k)
            self.stats["idle_blocks"] += 1
            infer_wav = torch.zeros(
                self.return_length * self.zc, device=self.device, dtype=torch.float32
            )
            converting = False
        elif converting:
            self.rvc.auto_pitch = bool(cfg.auto_pitch)
            infer_wav = self._convert()
            self.stats["auto_key"] = self.rvc.auto_key if cfg.auto_pitch else None
            # Half precision can overflow on extreme input. One bad value would
            # otherwise be played as a full-scale burst and, through the pitch
            # and feature caches, poison the following seconds as well.
            if not bool(torch.isfinite(infer_wav).all()):
                self.stats["glitches"] += 1
                self.rvc.reset_state()
                infer_wav = torch.zeros_like(infer_wav)
        else:
            infer_wav = source.clone()
        # output noise reduction
        if cfg.O_noise_reduce and converting:
            self.output_buffer[:-block] = self.output_buffer[block:].clone()
            self.output_buffer[-block:] = infer_wav[-block:]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        if converting:
            length = infer_wav.shape[0]
            # volume envelope mixing
            if cfg.rms_mix_rate < 1:
                rms_in = self._rms(source[:length])
                rms_out = torch.clamp(self._rms(infer_wav), min=1e-3)
                gain = torch.pow(rms_in / rms_out, 1.0 - cfg.rms_mix_rate)
                infer_wav = infer_wav * torch.clamp(gain, max=MAX_ENVELOPE_GAIN)
            # Whatever the model produces well inside a closed gate is noise.
            # The mute is wider than the gate so soft onsets and tails survive.
            if envelope is not None:
                mute = F.max_pool1d(
                    self.gate_env[None, None, self.extra_frame :],
                    2 * spread + 1,
                    stride=1,
                    padding=spread,
                )[0, 0, :length]
                infer_wav = infer_wav * mute
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0], dim=0)
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        sola_offset = int(sola_offset)
        infer_wav = infer_wav[sola_offset:].clone()
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[block : block + self.sola_buffer_frame]
        out = soft_limit(
            self._limit(infer_wav[: block + self.sola_buffer_frame])
        )
        out = out.cpu().numpy()

        total = time.perf_counter() - start_time
        stats = self.stats
        stats["sola_offset"] = sola_offset
        stats["output_db"] = 20 * np.log10(float(np.max(np.abs(out))) + 1e-6)
        stats["total_ms"] = total * 1000
        stats["blocks"] += 1
        # A block that takes longer than it lasts cannot be played in time.
        if total > block / self.samplerate:
            stats["overruns"] += 1
        return out


# Block lengths tried by auto_block_engine, shortest (lowest latency) first.
AUTO_BLOCK_TIMES = (0.1, 0.14, 0.2, 0.24, 0.32, 0.4, 0.5, 0.7, 1.0)


def auto_block_engine(rvc, settings, device, samplerate, headroom=0.7):
    """Build an engine with the shortest block this machine can sustain.

    Each candidate is timed on a synthetic voice-like signal; the first whose
    processing time stays under ``headroom`` of the block length wins. Returns
    ``(engine, block_time, milliseconds_per_block)`` and leaves the choice in
    ``settings.block_time``. When nothing fits, the longest block is used.
    """
    engine = cost = None
    for block_time in AUTO_BLOCK_TIMES:
        settings.block_time = block_time
        engine = RealtimeEngine(rvc, settings, device, samplerate)
        engine.prewarm()  # also absorbs one-off costs (lazy loads, graph capture)
        t = np.arange(engine.block_frame) / engine.samplerate
        probe = (0.1 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
        cost = 0.0
        for _ in range(2):
            engine.process(probe)
            cost = max(cost, engine.stats["total_ms"])
        engine.reset()
        engine.stats.update(blocks=0, overruns=0)
        if cost <= headroom * 1000 * engine.block_frame / engine.samplerate:
            break
    return engine, settings.block_time, cost


def simulate_file(engine, audio, block_times=None):
    """Feed ``audio`` (mono float32 at engine.samplerate) through the engine
    block by block, exactly as the sound card callback would. When given,
    ``block_times`` receives the processing time of every block in ms."""
    block = engine.block_frame
    blocks = int(np.ceil(audio.shape[0] / block)) + 1  # +1 flushes the tail
    padded = np.zeros(blocks * block, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    out = np.empty_like(padded)
    times = []
    for i in range(blocks):
        out[i * block : (i + 1) * block] = engine.process(
            padded[i * block : (i + 1) * block]
        )
        times.append(engine.stats["total_ms"])
    if block_times is not None:
        block_times.extend(times)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a wav file through the realtime engine (offline test)."
    )
    parser.add_argument("--model", required=True, help=".pth voice model")
    parser.add_argument("--index", default="", help="optional .index file")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument(
        "--auto-pitch",
        action="store_true",
        help="transpose automatically (needs a model with pitch statistics)",
    )
    parser.add_argument("--formant", type=float, default=0.0)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--rms-mix-rate", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=GATE_OFF_DB)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--no-idle-skip", action="store_true")
    parser.add_argument("--f0method", default="rmvpe", choices=("pm", "rmvpe", "fcpe"))
    parser.add_argument("--block-time", type=float, default=0.25)
    parser.add_argument(
        "--auto-block", action="store_true", help="pick the block length by timing"
    )
    parser.add_argument("--crossfade-time", type=float, default=0.05)
    parser.add_argument("--extra-time", type=float, default=2.5)
    parser.add_argument(
        "--hubert-context",
        type=float,
        default=None,
        help="seconds re-encoded per block; 0 re-encodes the whole window",
    )
    parser.add_argument("--seed", type=int, default=None, help="fixed sampling noise")
    parser.add_argument("--input-denoise", action="store_true")
    parser.add_argument("--output-denoise", action="store_true")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # Config() parses the WebUI's own arguments

    import librosa
    import soundfile as sf

    from configs.config import Config
    from infer.rtrvc import RVC

    config = Config()
    cfg = RealtimeSettings()
    cfg.pitch = args.pitch
    cfg.auto_pitch = args.auto_pitch
    cfg.formant = args.formant
    cfg.index_rate = args.index_rate if args.index else 0.0
    cfg.protect = args.protect
    cfg.rms_mix_rate = args.rms_mix_rate
    cfg.threhold = args.threshold
    cfg.auto_threshold = args.auto_threshold
    cfg.idle_skip = not args.no_idle_skip
    cfg.f0method = args.f0method
    cfg.block_time = args.block_time
    cfg.crossfade_time = args.crossfade_time
    cfg.extra_time = args.extra_time
    cfg.I_noise_reduce = args.input_denoise
    cfg.O_noise_reduce = args.output_denoise

    rvc = RVC(
        cfg.pitch,
        cfg.formant,
        args.model,
        args.index,
        cfg.index_rate,
        config,
        protect=cfg.protect,
    )
    if args.hubert_context is not None:
        cfg.hubert_context = args.hubert_context
    if args.auto_block:
        engine, block_time, cost = auto_block_engine(rvc, cfg, config.device, rvc.tgt_sr)
        print("auto block: %.2fs (%.0f ms per block)" % (block_time, cost))
    else:
        engine = RealtimeEngine(rvc, cfg, config.device, rvc.tgt_sr)
        engine.prewarm()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    audio, _ = librosa.load(args.input, sr=engine.samplerate, mono=True)
    times = []
    out = simulate_file(engine, audio.astype(np.float32), times)
    sf.write(args.output, out, engine.samplerate, subtype="FLOAT")
    if args.auto_pitch:
        print("auto pitch: %s (model median F0: %s)" % (rvc.auto_key, rvc.f0_median))
    block_ms = 1000 * engine.block_frame / engine.samplerate
    print(
        "blocks=%d idle=%d threshold=%s block=%.0fms infer mean=%.0fms max=%.0fms overruns=%d delay=%.0fms"
        % (
            len(times),
            engine.stats["idle_blocks"],
            engine.stats["threshold"],
            block_ms,
            float(np.mean(times)),
            float(np.max(times)),
            engine.stats["overruns"],
            1000 * engine.algorithm_delay(),
        )
    )


if __name__ == "__main__":
    main()
