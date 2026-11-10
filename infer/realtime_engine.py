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

# Loudness matching may not boost the model output by more than this factor.
MAX_ENVELOPE_GAIN = 6.0
LIMITER_KNEE = 0.85


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
        self.I_noise_reduce = False  # (live)
        self.O_noise_reduce = False  # (live)
        self.rms_mix_rate = 0.0  # (live)
        self.index_rate = 0.0
        self.protect = 0.33
        self.f0method = "rmvpe"  # (live)
        self.function = "vc"  # (live) "vc" converts, "im" monitors the input


def soft_limit(x, knee=LIMITER_KNEE):
    """Leave |x| <= knee untouched and bend the rest smoothly below 1.0."""
    magnitude = x.abs()
    room = 1.0 - knee
    bent = knee + room * torch.tanh((magnitude - knee) / room)
    return torch.where(magnitude > knee, torch.sign(x) * bent, x)


class RealtimeEngine:
    def __init__(self, rvc, settings, device, samplerate):
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
        indata = np.asarray(indata, dtype=np.float32)
        peak = float(np.max(np.abs(indata))) if indata.size else 0.0
        self.stats["input_db"] = 20 * np.log10(peak + 1e-6)

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
        # infer
        if converting:
            infer_wav = self._convert()
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
                spread = GATE_OUTPUT_SPREAD_HOPS * self.zc
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
        out = soft_limit(infer_wav[:block]).cpu().numpy()

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
    parser.add_argument("--formant", type=float, default=0.0)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--rms-mix-rate", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=GATE_OFF_DB)
    parser.add_argument("--f0method", default="rmvpe", choices=("pm", "rmvpe", "fcpe"))
    parser.add_argument("--block-time", type=float, default=0.25)
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
    cfg.formant = args.formant
    cfg.index_rate = args.index_rate if args.index else 0.0
    cfg.protect = args.protect
    cfg.rms_mix_rate = args.rms_mix_rate
    cfg.threhold = args.threshold
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
    engine = RealtimeEngine(rvc, cfg, config.device, rvc.tgt_sr)
    engine.prewarm()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    audio, _ = librosa.load(args.input, sr=engine.samplerate, mono=True)
    times = []
    out = simulate_file(engine, audio.astype(np.float32), times)
    sf.write(args.output, out, engine.samplerate)
    block_ms = 1000 * engine.block_frame / engine.samplerate
    print(
        "blocks=%d block=%.0fms infer mean=%.0fms max=%.0fms overruns=%d delay=%.0fms"
        % (
            len(times),
            block_ms,
            float(np.mean(times)),
            float(np.max(times)),
            engine.stats["overruns"],
            1000 * engine.algorithm_delay(),
        )
    )


if __name__ == "__main__":
    main()
