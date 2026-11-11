"""Built-in voices that need no trained model.

They are plain signal processing, so they work on a fresh install and cost
almost nothing to run. The realtime engine drives them through the same
interface as an RVC model (``infer.rtrvc.RVC``), except that they receive the
input at the sound card's sample rate (``fullband``) instead of at 16 kHz.
"""

import math

import torch

# Breathing helium raises the resonances of the vocal tract by the ratio of
# the speeds of sound while the vocal folds keep their pitch. Pure helium would
# be close to 3x; what people recognise as the party-balloon voice is a mix
# with air, around 1.5x to 1.8x.
HELIUM_RATIO = 1.6
# The "声线粗细" slider (-2..2) moves the ratio by this many slider units per octave.
SLIDER_UNITS_PER_OCTAVE = 6.0
ANALYSIS_SECONDS = 0.04  # long enough to hold two periods of a low voice
# Cepstrum below this quefrency is the vocal tract; above it, the pitch. Shorter
# smooths the envelope so much that narrow formants stay where they were; longer
# starts to follow the harmonics of voices above about 400 Hz.
ENVELOPE_SECONDS = 0.0025
MAX_GAIN_LOG = 3.5  # about 30 dB either way


def shift_formants(wav, samplerate, ratio):
    """Move the spectral envelope of ``wav`` (1-D tensor) up by ``ratio``
    without touching pitch or timing.

    Each short-time spectrum is split into its smooth envelope (low-quefrency
    cepstrum) and the rest; the envelope is stretched along frequency and put
    back. Phases are kept, so the result stays aligned with the input.
    """
    n_fft = 2 ** int(round(math.log2(ANALYSIS_SECONDS * samplerate)))
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
    log_mag = torch.log(spec.abs() + 1e-7)
    cepstrum = torch.fft.irfft(log_mag, n=n_fft, dim=0)
    keep = max(2, int(ENVELOPE_SECONDS * samplerate))
    cepstrum[keep : n_fft - keep + 1] = 0
    envelope = torch.fft.rfft(cepstrum, n=n_fft, dim=0).real
    bins = envelope.shape[0]
    source = torch.arange(bins, device=wav.device, dtype=wav.dtype) / ratio
    source = source.clamp(max=bins - 1)
    low = source.floor().long()
    high = (low + 1).clamp(max=bins - 1)
    mix = (source - low).unsqueeze(1)
    warped = envelope[low] * (1 - mix) + envelope[high] * mix
    gain = torch.exp((warped - envelope).clamp(-MAX_GAIN_LOG, MAX_GAIN_LOG))
    return torch.istft(spec * gain, n_fft, hop, window=window, length=wav.shape[0])


class HeliumVoice:
    """Helium voice: formants up, pitch unchanged."""

    fullband = True
    tgt_sr = 48000  # only a preference for the stream; any rate works
    f0_median = None

    def __init__(self, formant=0.0):
        self.formant_shift = formant
        self.hubert_context = 0.0
        self.auto_pitch = False
        self.auto_pitch_locked = False
        self.auto_key = None
        self.source_f0 = []

    def ratio(self):
        return HELIUM_RATIO * 2 ** (self.formant_shift / SLIDER_UNITS_PER_OCTAVE)

    def infer_fullband(self, wav, samplerate, start, length):
        """Convert ``wav[start : start + length]``; the audio around it is
        context for the analysis windows."""
        n_fft = 2 ** int(round(math.log2(ANALYSIS_SECONDS * samplerate)))
        begin = max(0, start - n_fft)
        shifted = shift_formants(wav[begin:], samplerate, self.ratio())
        return shifted[start - begin : start - begin + length]

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    # The rest of the model interface has nothing to act on here.
    def change_key(self, new_key):
        pass

    def change_index_rate(self, new_index_rate):
        pass

    def change_protect(self, new_protect):
        pass

    def reset_state(self):
        pass

    def prepare_silence(self):
        pass

    def skip_block(self, block_frame_16k):
        pass
