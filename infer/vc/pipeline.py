import os
import traceback
import logging

logger = logging.getLogger(__name__)

from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from scipy import signal

from infer.autopitch import MIN_FRAMES, suggest_transpose
from infer.hubert import extract_hubert_features
from infer.retrieval import retrieve_features
from tools.cuda_graph import run_cuda_graph

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device
        self._index_key = None
        self._index = self._index_vectors = None

    def load_index(self, path):
        stat = os.stat(path)
        key = (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)
        if key != self._index_key:
            index = faiss.read_index(path)
            if index.ntotal < 1:
                raise ValueError("索引没有音色特征，请使用 added 索引或重新克隆")
            vectors = index.reconstruct_n(0, index.ntotal)
            self._index, self._index_vectors, self._index_key = index, vectors, key
        return self._index, self._index_vectors

    def get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
        target_f0_median=None,
    ):
        if f0_method not in ("pm", "rmvpe", "fcpe"):
            raise ValueError(f"Unsupported F0 method: {f0_method}")
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.rmvpe import RMVPE

                logger.info(
                    "Loading rmvpe model,%s" % "%s/rmvpe.pt" % os.environ["rmvpe_root"]
                )
                self.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=self.is_half,
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")
        elif f0_method == "fcpe":
            if not hasattr(self, "model_fcpe"):
                from infer.fcpe import FCPEInfer

                logger.info("Loading fcpe model")
                self.model_fcpe = FCPEInfer(self.device)
            f0 = self.model_fcpe.infer(
                torch.from_numpy(x).unsqueeze(0).float(),
                sr=self.sr,
                decoder_mode="local_argmax",
                threshold=0.006,
            ).squeeze().detach().cpu().numpy()

        # Remember which frames the tracker called voiced before the gaps are
        # interpolated away: consonant protection needs the real decision.
        voiced = np.asarray(f0) > 0
        if target_f0_median and voiced.sum() >= MIN_FRAMES:
            f0_up_key = suggest_transpose(
                float(np.median(f0[voiced])), target_f0_median
            )
        self.applied_f0_up_key = f0_up_key
        try:
            uv = f0 == 0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        except Exception:
            traceback.print_exc()
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak, voiced

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        index_vectors,
        index_rate,
        version,
        protect,
        voiced=None,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        t0 = ttime()
        with torch.no_grad():
            feats = extract_hubert_features(
                model,
                feats.to(self.device),
                version,
            )
        # Protection only matters where retrieval replaced the features.
        use_protect = (
            protect < 0.5
            and voiced is not None
            and index is not None
            and index_vectors is not None
            and index_rate != 0
        )
        if use_protect:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(index_vectors, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            npy = retrieve_features(index, index_vectors, npy)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if use_protect:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if use_protect:
            # Voiceless consonants and breaths keep more of the source features.
            # (The interpolated F0 is never 0, so it cannot be used for this.)
            frames = min(p_len, voiced.shape[1], feats.shape[1])
            keep = torch.ones(1, feats.shape[1], 1, device=feats.device)
            keep[0, :frames, 0] = torch.where(voiced[0, :frames], 1.0, float(protect))
            keep = keep.to(feats0.dtype)
            feats = feats * keep + feats0 * (1 - keep)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            if hasp:
                synthesized = run_cuda_graph(
                    net_g,
                    "rvc-synth-f0",
                    lambda phone, lengths, coarse, continuous, speaker: net_g.infer(
                        phone, lengths, coarse, continuous, speaker
                    )[0],
                    feats,
                    p_len,
                    pitch,
                    pitchf,
                    sid,
                )
            else:
                synthesized = run_cuda_graph(
                    net_g,
                    "rvc-synth-no-f0",
                    lambda phone, lengths, speaker: net_g.infer(
                        phone, lengths, speaker
                    )[0],
                    feats,
                    p_len,
                    sid,
                )
            audio1 = synthesized[0, 0].data.cpu().float().numpy()
            del hasp, synthesized
        del feats, p_len
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        target_f0_median=None,
    ):
        """``target_f0_median`` (Hz) replaces ``f0_up_key`` by the transposition
        that moves this audio into the target voice's range; the value used is
        left in ``self.applied_f0_up_key``."""
        self.applied_f0_up_key = f0_up_key
        if file_index and index_rate != 0:
            index, index_vectors = self.load_index(file_index)
        else:
            index = index_vectors = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            cumulative = np.pad(np.cumsum(np.abs(audio_pad), dtype=np.float64), (1, 0))
            audio_sum = cumulative[self.window : self.window + audio.size] - cumulative[:audio.size]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf, voiced = None, None, None
        if if_f0 == 1:
            pitch, pitchf, voiced = self.get_f0(
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                target_f0_median,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
            voiced = torch.tensor(voiced[:p_len], device=self.device).unsqueeze(0)
        t2 = ttime()
        times[1] += t2 - t1

        def convert(segment, first_frame, last_frame):
            frames = slice(first_frame, last_frame)
            return self.vc(
                model,
                net_g,
                sid,
                segment,
                pitch[:, frames] if pitch is not None else None,
                pitchf[:, frames] if pitchf is not None else None,
                times,
                index,
                index_vectors,
                index_rate,
                version,
                protect,
                voiced[:, frames] if voiced is not None else None,
            )[self.t_pad_tgt : -self.t_pad_tgt]

        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(
                convert(
                    audio_pad[s : t + self.t_pad2 + self.window],
                    s // self.window,
                    (t + self.t_pad2) // self.window,
                )
            )
            s = t
        audio_opt.append(
            convert(audio_pad[t:], t // self.window if t is not None else None, None)
        )
        audio_opt = np.concatenate(audio_opt)
        if not audio_opt.size or not np.isfinite(audio_opt).all():
            raise ValueError("转换未生成有效音频，请检查模型和索引是否匹配")
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, voiced, sid
        return audio_opt
