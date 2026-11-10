import os
import traceback
from time import time as ttime
import faiss
import numpy as np
import parselmouth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from infer.hubert import extract_hubert_features, load_hubert_model
from i18n.i18n import I18nAuto
from tools.cuda_graph import run_cuda_graph


i18n = I18nAuto()

HUBERT_HOP = 320  # samples at 16 kHz between two HuBERT frames
HUBERT_EDGE_FRAMES = 8  # left-edge frames of a partial window that are discarded


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        index_path,
        index_rate,
        config,
        last_rvc=None,
        protect=0.33,
    ) :
        """
        初始化

        模型加载失败会直接抛出异常，由调用方（GUI）提示用户；索引是可选的，
        缺失或损坏时只打印警告并在不使用检索的情况下继续。
        """
        self.config = config
        self.device = config.device
        self.f0_up_key = key
        self.formant_shift = formant
        self.protect = protect
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.is_half = config.is_half
        self.pth_path = pth_path
        self.index_path = index_path
        self.index_rate = index_rate
        self.index = None
        self.big_npy = None
        if (
            last_rvc is not None
            and getattr(last_rvc, "index", None) is not None
            and last_rvc.index_path == index_path
        ):
            self.index = last_rvc.index
            self.big_npy = last_rvc.big_npy
        elif index_rate != 0:
            self.load_index()
        self.cache_pitch = torch.zeros(1024, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(
            1024, device=self.device, dtype=torch.float32
        )
        # Raw voiced/unvoiced decision per 10 ms frame, aligned with cache_pitch.
        self.cache_voiced = torch.zeros(1024, device=self.device, dtype=torch.bool)
        # Last voiced F0 (Hz, before transposition) seen by the extractor.
        self.last_voiced_f0 = 0.0
        self.infer_count = 0
        # Seconds of audio re-encoded by HuBERT ahead of each new block; the
        # rest of the window comes from the feature cache. 0 disables caching.
        self.hubert_context = 1.0
        self.feat_cache = None
        self.feat_cache_key = None

        self.resample_kernel = {}

        if last_rvc is None:
            self.model = load_hubert_model(self.device, self.is_half)
        else:
            self.model = last_rvc.model

        self.net_g = None

        def set_synthesizer():
            self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
            self.tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            if self.is_half:
                self.net_g = self.net_g.half()
            else:
                self.net_g = self.net_g.float()

        if (
            last_rvc is None
            or last_rvc.pth_path != self.pth_path
            or getattr(last_rvc, "net_g", None) is None
        ):
            set_synthesizer()
        else:
            self.tgt_sr = last_rvc.tgt_sr
            self.if_f0 = last_rvc.if_f0
            self.version = last_rvc.version
            self.is_half = last_rvc.is_half
            self.net_g = last_rvc.net_g

        if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
            self.model_rmvpe = last_rvc.model_rmvpe
        if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
            self.model_fcpe = last_rvc.model_fcpe

    def load_index(self):
        """Load the optional faiss index. Returns True when it is usable."""
        self.index = None
        self.big_npy = None
        if not self.index_path:
            return False
        if not os.path.isfile(self.index_path):
            printt(i18n("索引文件不存在，将不使用索引继续推理：%s"), self.index_path)
            return False
        try:
            index = faiss.read_index(self.index_path)
            self.big_npy = index.reconstruct_n(0, index.ntotal)
            self.index = index
            printt(i18n("已启用索引检索"))
            return True
        except Exception:
            printt(traceback.format_exc())
            printt(i18n("索引检索失败"))
            return False

    def reset_state(self):
        """Forget the streaming history (pitch cache) before a new stream."""
        self.cache_pitch.zero_()
        self.cache_pitchf.zero_()
        self.cache_voiced.zero_()
        self.last_voiced_f0 = 0.0
        self.infer_count = 0
        self.feat_cache = None

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index is None:
            self.load_index()
        self.index_rate = new_index_rate

    def change_protect(self, new_protect):
        self.protect = new_protect

    def finalize_f0(self, f0, f0_up_key):
        """Fill unvoiced frames the way training does, then transpose.

        Training interpolates F0 across unvoiced frames, so the synthesizer has
        rarely seen F0 == 0 next to speech. A streaming window that happens to
        contain no voiced frame at all (a pause, a long consonant) would feed it
        exactly that, which is heard as a rough/hoarse burst. Holding the last
        voiced pitch keeps the input in-distribution.
        """
        f0 = np.array(f0, dtype=np.float32).reshape(-1)
        voiced = f0 > 0
        if voiced.any():
            voiced_idx = np.where(voiced)[0]
            f0[~voiced] = np.interp(np.where(~voiced)[0], voiced_idx, f0[voiced])
            self.last_voiced_f0 = float(f0[voiced_idx[-1]])
        elif self.last_voiced_f0 > 0:
            f0[:] = self.last_voiced_f0
        f0 *= pow(2, f0_up_key / 12)
        coarse, f0 = self.get_f0_post(f0)
        return coarse, f0, torch.from_numpy(voiced).to(self.device)

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, method="rmvpe"):
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        if method != "pm":
            raise ValueError(f"Unsupported F0 method: {method}")
        x = x.cpu().numpy()
        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=1100,
        )
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        f0 = f0[:p_len]
        return self.finalize_f0(f0, f0_up_key)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.rmvpe import RMVPE

            printt(i18n("正在加载RMVPE模型"))
            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return self.finalize_f0(f0, f0_up_key)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from infer.fcpe import FCPEInfer

            printt("Loading fcpe model")
            self.model_fcpe = FCPEInfer(self.device)
        f0 = self.model_fcpe.infer(
            x.unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        return self.finalize_f0(f0, f0_up_key)

    def _hubert(self, wav):
        if self.config.is_half:
            wav = wav.half().view(1, -1)
        else:
            wav = wav.float().view(1, -1)
        padding_mask = torch.BoolTensor(wav.shape).to(self.device).fill_(False)
        return extract_hubert_features(
            self.model, wav, self.version, padding_mask=padding_mask
        )

    def extract_features(self, input_wav, block_frame_16k):
        """HuBERT features for the whole window, reusing earlier blocks.

        The window slides by one block per call, so all but the newest frames
        were already computed last time. Only the tail (``hubert_context``
        seconds plus the new block) is run through HuBERT and spliced onto the
        shifted cache. The frames that are actually synthesized always sit at
        the right edge of that tail; the older ones only serve as context for
        the synthesizer's text encoder.
        """
        samples = input_wav.shape[0]
        hop = HUBERT_HOP
        wanted = int(self.hubert_context * 16000) + block_frame_16k
        # Start on the frame grid of the full window so both line up exactly.
        start = max(0, (samples - wanted) // hop * hop)
        reusable = (
            self.hubert_context > 0
            and self.feat_cache is not None
            and self.feat_cache_key == (samples, block_frame_16k, self.version)
            and block_frame_16k % hop == 0
            and start > 0
        )
        if not reusable:
            feats = self._hubert(input_wav)
        else:
            tail = self._hubert(input_wav[start:])
            shift = block_frame_16k // hop
            feats = torch.roll(self.feat_cache, -shift, dims=1)
            # The first frames of the tail lack left context; keep the cache there.
            at = start // hop + HUBERT_EDGE_FRAMES
            feats[:, at:] = tail[:, HUBERT_EDGE_FRAMES:]
        self.feat_cache = feats.clone()
        self.feat_cache_key = (samples, block_frame_16k, self.version)
        return feats

    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) :
        report_status = self.infer_count < 3 or self.infer_count % 100 == 0
        self.infer_count += 1
        t1 = ttime()
        with torch.no_grad():
            feats = self.extract_features(input_wav, block_frame_16k)
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
        t2 = ttime()
        feats0 = None
        try:
            if self.index is not None and self.index_rate != 0:
                if self.if_f0 == 1 and self.protect < 0.5:
                    feats0 = feats.clone()
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                if (ix >= 0).all():
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(
                        self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                    )
                    if self.config.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
                else:
                    printt(
                        i18n("索引无效：必须使用added_xxxx.index，不能使用trained_xxxx.index")
                    )
            else:
                if report_status:
                    printt(i18n("索引检索失败或未启用"))
        except Exception:
            traceback.print_exc()
            printt(i18n("索引检索失败"))
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf, voiced = self.get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key - self.formant_shift,
                f0method,
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_voiced[:-shift] = self.cache_voiced[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            self.cache_voiced[4 - pitch.shape[0] :] = voiced[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        if feats0 is not None:
            # Protect voiceless consonants and breaths: retrieval tends to smear
            # them into the nearest voiced timbre, so keep more of the source
            # features on frames the pitch tracker marked unvoiced.
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )[:, :p_len, :]
            keep = torch.where(
                self.cache_voiced[-p_len:],
                torch.ones((), device=self.device),
                torch.full((), float(self.protect), device=self.device),
            ).to(feats.dtype)[None, :, None]
            feats = feats * keep + feats0 * (1 - keep)
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_value = int(skip_head)
        return_length_value = int(return_length)
        return_length2_value = int(return_length2)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, coarse, continuous, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        coarse,
                        continuous,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                )
            else:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        t5 = ttime()
        if report_status:
            printt(
                i18n("耗时：特征=%.3f秒，索引=%.3f秒，音高=%.3f秒，模型=%.3f秒"),
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio.squeeze()
