import traceback
import logging
from functools import wraps
from threading import RLock

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.audio import load_audio, wav2
from infer.module.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.autopitch import model_f0_median
from infer.vc.pipeline import Pipeline
from infer.vc.utils import *
from i18n.i18n import I18nAuto
from tools.progress import batch_status, should_report
from tools.cuda_graph import clear_cuda_graph_cache


i18n = I18nAuto()


def inference_status(title, state, detail=""):
    lines = ["【%s】" % i18n(title), "%s：%s" % (i18n("状态"), i18n(state))]
    if detail:
        lines.extend(["", str(detail).strip()])
    return "\n".join(lines)


def normalized_speaker_info(checkpoint, n_spk):
    speaker_info = []
    seen = set()
    for item in checkpoint.get("speaker_info", []):
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            speaker_id < 0
            or speaker_id >= n_spk
            or not speaker_name
            or speaker_id in seen
        ):
            continue
        seen.add(speaker_id)
        speaker_info.append({"id": speaker_id, "name": speaker_name})
    speaker_info.sort(key=lambda item: item["id"])
    return speaker_info


def speaker_selector_updates(checkpoint, n_spk):
    speaker_info = normalized_speaker_info(checkpoint, n_spk)
    if speaker_info:
        choices = [
            i18n("说话人：%s（ID：%s）") % (item["name"], item["id"])
            for item in speaker_info
        ]
        return (
            {
                "visible": False,
                "value": speaker_info[0]["id"],
                "__type__": "update",
            },
            {
                "visible": True,
                "choices": choices,
                "value": choices[0],
                "__type__": "update",
            },
        )
    return (
        {
            "visible": True,
            "maximum": max(n_spk - 1, 0),
            "__type__": "update",
        },
        {"visible": False, "value": None, "__type__": "update"},
    )


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.f0_median = None
        self.hubert_model = None

        self.config = config
        self.lock = RLock()
        self._model_key = None

    @synchronized
    def get_vc(self, sid, *to_return_protect):
        if not sid:
            clear_cuda_graph_cache(self.net_g)
            clear_cuda_graph_cache(self.hubert_model)
            self.net_g = self.hubert_model = self.pipeline = self.cpt = None
            self.n_spk = self.tgt_sr = self.if_f0 = self.f0_median = None
            self._model_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hidden = {"visible": False, "__type__": "update"}
            return hidden, dict(hidden, value=None), dict(hidden), dict(hidden), "", ""

        person = os.path.abspath(os.path.join(os.getenv("weight_root", "assets/weights"), sid))
        stat = os.stat(person)
        key = (person, stat.st_mtime_ns, stat.st_size)
        if key != self._model_key:
            logger.info("%s: %s", i18n("正在加载模型"), person)
            checkpoint = torch.load(person, map_location="cpu")
            checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
            if_f0 = checkpoint.get("f0", 1)
            version = checkpoint.get("version", "v1")
            synthesizer_class = {
                ("v1", 1): SynthesizerTrnMs256NSFsid,
                ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
                ("v2", 1): SynthesizerTrnMs768NSFsid,
                ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
            }
            clear_cuda_graph_cache(self.net_g)
            self.net_g = None
            self._model_key = None
            net_g = synthesizer_class[(version, if_f0)](
                *checkpoint["config"], is_half=self.config.is_half
            )
            del net_g.enc_q
            net_g.load_state_dict(checkpoint["weight"], strict=False)
            net_g = net_g.eval().to(device=self.config.device, dtype=self.config.dtype)
            target_sr = checkpoint["config"][-1]
            if self.pipeline is None or self.tgt_sr != target_sr:
                pipeline = Pipeline(target_sr, self.config)
                if self.pipeline is not None:
                    for attr in ("model_rmvpe", "model_fcpe"):
                        if hasattr(self.pipeline, attr):
                            setattr(pipeline, attr, getattr(self.pipeline, attr))
                self.pipeline = pipeline
            self.cpt, self.net_g = checkpoint, net_g
            self.tgt_sr, self.if_f0, self.version = target_sr, if_f0, version
            self.n_spk = checkpoint["config"][-3]
            self.f0_median = model_f0_median(checkpoint, sid)
            self._model_key = key

        slider, dropdown = speaker_selector_updates(self.cpt, self.n_spk)
        speaker_info = normalized_speaker_info(self.cpt, self.n_spk)
        speaker_id = speaker_info[0]["id"] if speaker_info else 0
        index = {"value": get_index_path_from_model(sid, speaker_id), "__type__": "update"}
        protect = [
            {
                "visible": self.if_f0 != 0,
                "value": to_return_protect[i] if self.if_f0 and len(to_return_protect) > i else 0.5,
                "__type__": "update",
            }
            for i in range(2)
        ]
        return (slider, dropdown, *protect, index, dict(index)) if to_return_protect else slider

    @synchronized
    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        auto_pitch=False,
    ):
        if input_audio_path is None:
            return inference_status("单次推理", "等待输入", i18n("请上传音频文件")), None
        try:
            if self.net_g is None or self.pipeline is None:
                raise ValueError("请先选择或克隆一个音色")
            if not 0 <= int(sid) < self.n_spk:
                raise ValueError("说话人编号不属于当前模型，请重新选择音色")
            f0_up_key = int(f0_up_key)
            audio = load_audio(input_audio_path, 16000)
            if audio.size < 1600 or not np.isfinite(audio).all():
                raise ValueError("录音为空、过短或损坏，请重新录制至少一秒的人声")
            if np.max(np.abs(audio)) < 1e-5:
                raise ValueError("录音是静音，请检查麦克风后重新录制")
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                self.f0_median if auto_pitch and self.if_f0 == 1 else None,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "%s：%s" % (i18n("索引"), file_index)
                if os.path.exists(file_index)
                else "%s：%s" % (i18n("索引"), i18n("未使用"))
            )
            if auto_pitch and self.if_f0 == 1:
                index_info += "\n%s：%s" % (
                    i18n("自动变调"),
                    "%+d" % self.pipeline.applied_f0_up_key
                    if self.f0_median
                    else i18n("该模型没有音高统计，已使用手动变调"),
                )
            return (
                inference_status(
                    "单次推理",
                    "成功",
                    "%s\n%s：%s %.2fs | F0 %.2fs | %s %.2fs"
                    % (
                        index_info,
                        i18n("耗时"),
                        i18n("特征"),
                        times[0],
                        times[1],
                        i18n("合成"),
                        times[2],
                    ),
                ),
                (tgt_sr, audio_opt),
            )
        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return inference_status("单次推理", "失败", info), None

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
        model_name=None,
    ):
        try:
            dir_path = (
                (dir_path or "")
                .strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = (
                (opt_root or "")
                .strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
            )
            if not opt_root:
                yield inference_status(
                    "批量推理", "等待输入", i18n("请填写输出文件夹路径")
                )
                return
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path if isinstance(path, str) else path.name for path in (paths or [])]
            except Exception:
                traceback.print_exc()
                paths = [
                    path if isinstance(path, str) else path.name for path in (paths or [])
                ]
            total = len(paths)
            if total == 0:
                yield batch_status(i18n("批量推理"), 0, 0, 0, 0)
                return
            success = 0
            failed = 0
            failures = []
            for idx, path in enumerate(paths):
                item_failed = False
                with self.lock:
                    if model_name:
                        self.get_vc(model_name)
                    info, opt = self.vc_single(
                        sid,
                        path,
                        f0_up_key,
                        f0_method,
                        file_index,
                        index_rate,
                        resample_sr,
                        rms_mix_rate,
                        protect,
                    )
                if opt and opt[0] is not None and opt[1] is not None:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (
                                    opt_root,
                                    os.path.splitext(os.path.basename(path))[0],
                                    format1,
                                ),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.splitext(os.path.basename(path))[0],
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                        success += 1
                    except Exception:
                        info = "%s\n%s" % (info, traceback.format_exc())
                        failed += 1
                        item_failed = True
                        failures.append("%s：%s" % (os.path.basename(path), info))
                else:
                    failed += 1
                    item_failed = True
                    failures.append("%s：%s" % (os.path.basename(path), info))
                if should_report(idx, total) or item_failed:
                    yield batch_status(
                        i18n("批量推理"),
                        idx + 1,
                        total,
                        success,
                        failed,
                        os.path.basename(path),
                        failures,
                    )
        except Exception:
            yield inference_status("批量推理", "失败", traceback.format_exc())
