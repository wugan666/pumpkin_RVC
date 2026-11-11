import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))

from tools.file_io import read_text

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")

# CPU inference scales with cores; beyond 8 threads the gain is gone.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(8, os.cpu_count() or 4))))

realtime_config_path = os.path.join(now_dir, "configs", "config.json")

flag_vc = False


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


if __name__ == "__main__":
    import gc
    import json
    from collections import deque
    import re
    import time
    import traceback

    import librosa
    import numpy as np
    import FreeSimpleGUI as sg
    import sounddevice as sd
    import torch

    from configs.config import Config
    from infer import rtrvc as rvc_for_realtime
    from infer.effect_voice import HeliumVoice
    from infer.realtime_engine import (
        RealtimeEngine,
        RealtimeSettings,
        auto_block_engine,
    )
    from infer.realtime_engine import AUTO_BLOCK_TIMES
    from tools.audio_devices import (
        find_virtual_cable,
        is_virtual_input,
        match_device,
        preferred_hostapi,
    )
    from tools import virtual_cable
    from tools.download_models import download_assets, inference_assets

    IS_WINDOWS = sys.platform == "win32"
    # A game will take most of the GPU once it runs, so the block length is
    # chosen with this much of each block left spare.
    AUTO_BLOCK_HEADROOM = 0.5
    # This many late blocks within ADAPT_WINDOW seconds lengthen the block.
    ADAPT_OVERRUNS = 3
    ADAPT_WINDOW = 5.0

    def raise_process_priority():
        """Keep audio flowing while a game saturates the CPU (Windows only)."""
        if not IS_WINDOWS:
            return
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            kernel32.SetPriorityClass(
                kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS
            )
        except Exception:
            printt(traceback.format_exc())
    from infer.vc.utils import get_index_path_from_model
    from i18n.i18n import I18nAuto

    i18n = I18nAuto()

    F0_METHODS = ("pm", "rmvpe", "fcpe")
    # Changing any of these needs new buffers or a new stream.
    RESTART_KEYS = {
        "pth_path",
        "index_path",
        "sg_hostapi",
        "sg_wasapi_exclusive",
        "sg_input_device",
        "sg_output_device",
        "reload_devices",
        "sr_model",
        "sr_device",
        "block_time",
        "crossfade_length",
        "extra_time",
    }
    # (block_time, crossfade_length, extra_time, f0method, hubert_context)
    # hubert_context is how many seconds HuBERT re-encodes per block; 0 means
    # the whole window every time (slowest, marginally closer to offline).
    PRESETS = {
        "preset_fast": (0.10, 0.04, 1.5, "fcpe", 0.5),
        "preset_balanced": (0.24, 0.05, 2.5, "rmvpe", 2.0),
        "preset_quality": (0.50, 0.08, 3.0, "rmvpe", 0.0),
    }
    DEFAULTS = {
        "pth_path": "",
        "index_path": "",
        "sg_hostapi": "",
        "sg_wasapi_exclusive": False,
        "sg_input_device": "",
        "sg_output_device": "",
        "sr_type": "sr_model",
        "threhold": -55,
        "auto_threshold": True,
        "auto_pitch": True,
        "auto_block": True,
        "auto_start": True,
        "started_ok": False,
        "cable_asked": False,
        "show_advanced": False,
        "pitch": 0,
        "formant": 0.0,
        "index_rate": 0.3,
        "protect": 0.33,
        "rms_mix_rate": 0.25,
        "block_time": 0.25,
        "crossfade_length": 0.05,
        "extra_time": 2.5,
        "hubert_context": 2.0,
        "f0method": "rmvpe",
        "I_noise_reduce": False,
        "O_noise_reduce": False,
    }
    COLOR_OK = "#1a7f37"
    COLOR_WARN = "#b35900"
    COLOR_BAD = "#c62828"
    COLOR_IDLE = "#57606a"

    # A voice that needs no model file, so a fresh install can be used at once.
    BUILTIN_HELIUM = i18n("氦气音（内置）")

    def list_models():
        """Trained voices first, then the built-in one."""
        root = os.environ["weight_root"]
        models = []
        if os.path.isdir(root):
            models = sorted(
                os.path.join(root, name).replace("\\", "/")
                for name in os.listdir(root)
                if name.endswith(".pth")
            )
        return models + [BUILTIN_HELIUM]

    def is_voice(path):
        return path == BUILTIN_HELIUM or os.path.isfile(str(path))

    class GUIConfig(RealtimeSettings):
        def __init__(self):
            super().__init__()
            self.pth_path = ""
            self.index_path = ""
            self.sr_type = "sr_model"
            self.sg_hostapi = ""
            self.wasapi_exclusive = False
            self.sg_input_device = ""
            self.sg_output_device = ""
            self.samplerate = 0
            self.channels = 1

    class GUI:
        def __init__(self):
            self.gui_config = GUIConfig()
            self.config = Config()
            printt("RVC_CUDA_GRAPH=%s", os.environ.get("RVC_CUDA_GRAPH", "0"))
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.engine = None
            self.shown_overruns = 0
            self.overrun_time = float("-inf")
            self.auto_block = True
            self.shown_threshold = None
            self.shown_input_clipped = 0
            self.notice = ""
            self.notice_time = float("-inf")
            self.overrun_marks = deque([(0.0, 0)])
            self.auto_pitch_unavailable = False
            self.show_advanced = False
            raise_process_priority()
            self.update_devices()
            self.launcher()

        def default_device(self, kind):
            """Best first-run choice: the virtual cable for output when there
            is one, otherwise the system default device (found by name, since
            its index belongs to the default host API)."""
            names = self.input_devices if kind == 0 else self.output_devices
            indices = (
                self.input_devices_indices if kind == 0 else self.output_devices_indices
            )
            if kind == 1 and find_virtual_cable(names):
                return find_virtual_cable(names)
            if sd.default.device[kind] in indices:
                return names[indices.index(sd.default.device[kind])]
            try:
                system_name = sd.query_devices(device=sd.default.device[kind])["name"]
            except Exception:
                system_name = ""
            matched = match_device(system_name, names)
            if matched:
                return matched
            real = [name for name in names if kind == 1 or not is_virtual_input(name)]
            return (real or names or [""])[0]

        def load(self):
            data = dict(DEFAULTS)
            try:
                data.update(json.loads(read_text(realtime_config_path)))
            except Exception:
                pass  # first run or unreadable file: start from the defaults
            if data.get("f0method") not in F0_METHODS:
                data["f0method"] = "rmvpe"
            if data.get("sr_type") not in ("sr_model", "sr_device"):
                data["sr_type"] = "sr_model"
            if data["sg_hostapi"] in self.hostapis:
                self.update_devices(hostapi_name=data["sg_hostapi"])
            if (
                data["sg_hostapi"] not in self.hostapis
                or data["sg_input_device"] not in self.input_devices
                or data["sg_output_device"] not in self.output_devices
            ):
                data["sg_hostapi"] = preferred_hostapi(self.hostapis, IS_WINDOWS)
                self.update_devices(hostapi_name=data["sg_hostapi"])
                data["sg_input_device"] = self.default_device(0)
                data["sg_output_device"] = self.default_device(1)
            # A remembered model that has since been moved or deleted would only
            # produce an error on start; fall back to what is actually there.
            models = list_models()
            if not is_voice(data["pth_path"]):
                data["pth_path"] = models[0]
                data["index_path"] = ""
            if not os.path.isfile(str(data["index_path"])):
                data["index_path"] = get_index_path_from_model(data["pth_path"])
            return data

        def launcher(self):
            data = self.load()
            self.gui_config.hubert_context = float(data["hubert_context"])
            self.started_ok = bool(data["started_ok"])
            sg.theme("SystemDefaultForReal")
            if sys.platform == "win32":
                sg.set_options(font=("Microsoft YaHei UI", 10))
            sg.set_options(element_padding=(6, 4))
            label = lambda text: sg.Text(i18n(text), size=(20, 1))

            def slider(key, value_range, resolution):
                return sg.Slider(
                    range=value_range,
                    key=key,
                    resolution=resolution,
                    orientation="h",
                    default_value=data[key],
                    enable_events=True,
                    size=(28, 14),
                    expand_x=True,
                )

            quick_frame = sg.Frame(
                title=i18n("快速开始"),
                expand_x=True,
                layout=[
                    [
                        label("音色模型"),
                        sg.Combo(
                            list_models(),
                            default_value=data["pth_path"],
                            key="pth_path",
                            enable_events=True,
                            size=(52, 1),
                            expand_x=True,
                        ),
                        sg.FileBrowse(
                            i18n("选择.pth文件"),
                            target="pth_path",
                            initial_folder=os.path.join(os.getcwd(), "assets/weights"),
                            file_types=(("pth", "*.pth"),),
                        ),
                        sg.Button(i18n("刷新"), key="refresh_models"),
                    ],
                    [
                        label("输入设备"),
                        sg.Combo(
                            self.input_devices,
                            key="sg_input_device",
                            default_value=data["sg_input_device"],
                            enable_events=True,
                            size=(52, 1),
                            expand_x=True,
                        ),
                    ],
                    [
                        label("输出设备"),
                        sg.Combo(
                            self.output_devices,
                            key="sg_output_device",
                            default_value=data["sg_output_device"],
                            enable_events=True,
                            size=(52, 1),
                            expand_x=True,
                        ),
                        sg.Button(
                            i18n("安装虚拟声卡"),
                            key="install_cable",
                            visible=IS_WINDOWS
                            and not find_virtual_cable(self.output_devices),
                        ),
                    ],
                    [
                        label("音调设置"),
                        sg.Checkbox(
                            i18n("自动变调"),
                            key="auto_pitch",
                            default=data["auto_pitch"],
                            enable_events=True,
                        ),
                        slider("pitch", (-24, 24), 1),
                    ],
                ],
            )
            model_frame = sg.Frame(
                title=i18n("索引与设备"),
                expand_x=True,
                layout=[
                    [
                        label("索引(可选)"),
                        sg.Input(
                            default_text=data["index_path"],
                            key="index_path",
                            enable_events=True,
                            size=(52, 1),
                            expand_x=True,
                        ),
                        sg.FileBrowse(
                            i18n("选择.index文件"),
                            target="index_path",
                            initial_folder=os.path.join(os.getcwd(), "logs"),
                            file_types=(("index", "*.index"),),
                        ),
                        sg.Button(i18n("自动匹配"), key="match_index"),
                    ],
                    [
                        label("设备类型"),
                        sg.Combo(
                            self.hostapis,
                            key="sg_hostapi",
                            default_value=data["sg_hostapi"],
                            enable_events=True,
                            size=(20, 1),
                        ),
                        sg.Checkbox(
                            i18n("独占 WASAPI 设备"),
                            key="sg_wasapi_exclusive",
                            default=data["sg_wasapi_exclusive"],
                            enable_events=True,
                        ),
                        sg.Button(i18n("重载设备列表"), key="reload_devices"),
                    ],
                    [
                        label("采样率"),
                        sg.Radio(
                            i18n("使用模型采样率"),
                            "sr_type",
                            key="sr_model",
                            default=data["sr_type"] == "sr_model",
                            enable_events=True,
                        ),
                        sg.Radio(
                            i18n("使用设备采样率"),
                            "sr_type",
                            key="sr_device",
                            default=data["sr_type"] == "sr_device",
                            enable_events=True,
                        ),
                        sg.Text("", key="sr_stream", size=(8, 1)),
                    ],
                ],
            )
            voice_frame = sg.Frame(
                title=i18n("常规设置"),
                expand_x=True,
                expand_y=True,
                layout=[
                    [
                        label("响应阈值"),
                        sg.Checkbox(
                            i18n("自动"),
                            key="auto_threshold",
                            default=data["auto_threshold"],
                            enable_events=True,
                        ),
                        slider("threhold", (-60, 0), 1),
                    ],
                    [label("性别因子/声线粗细"), slider("formant", (-2, 2), 0.05)],
                    [label("Index Rate"), slider("index_rate", (0.0, 1.0), 0.01)],
                    [label("清辅音保护"), slider("protect", (0.0, 0.5), 0.01)],
                    [label("响度因子"), slider("rms_mix_rate", (0.0, 1.0), 0.01)],
                    [
                        label("音高算法"),
                        *[
                            sg.Radio(
                                name,
                                "f0method",
                                key=name,
                                default=data["f0method"] == name,
                                enable_events=True,
                            )
                            for name in F0_METHODS
                        ],
                    ],
                ],
            )
            perf_frame = sg.Frame(
                title=i18n("性能设置"),
                expand_x=True,
                expand_y=True,
                layout=[
                    [
                        label("预设"),
                        sg.Button(i18n("低延迟"), key="preset_fast"),
                        sg.Button(i18n("均衡"), key="preset_balanced"),
                        sg.Button(i18n("高音质"), key="preset_quality"),
                    ],
                    [
                        sg.Checkbox(
                            i18n("自动选择采样长度（启动时实测本机速度）"),
                            key="auto_block",
                            default=data["auto_block"],
                            enable_events=True,
                        )
                    ],
                    [label("采样长度"), slider("block_time", (0.02, 1.5), 0.01)],
                    [
                        label("淡入淡出长度"),
                        slider("crossfade_length", (0.01, 0.15), 0.01),
                    ],
                    [label("额外推理时长"), slider("extra_time", (0.05, 5.00), 0.01)],
                    [
                        sg.Checkbox(
                            i18n("输入降噪"),
                            key="I_noise_reduce",
                            default=data["I_noise_reduce"],
                            enable_events=True,
                        ),
                        sg.Checkbox(
                            i18n("输出降噪"),
                            key="O_noise_reduce",
                            default=data["O_noise_reduce"],
                            enable_events=True,
                        ),
                    ],
                ],
            )
            meter = lambda key: sg.ProgressBar(
                60, orientation="h", size=(18, 12), key=key, expand_x=True
            )
            status_frame = sg.Frame(
                title=i18n("运行状态"),
                expand_x=True,
                layout=[
                    [
                        sg.Button(
                            i18n("开始音频转换"),
                            key="start_vc",
                            button_color=("white", COLOR_OK),
                        ),
                        sg.Button(
                            i18n("停止音频转换"),
                            key="stop_vc",
                            disabled=True,
                        ),
                        sg.Checkbox(
                            i18n("打开软件后自动开始"),
                            key="auto_start",
                            default=data["auto_start"],
                            enable_events=True,
                        ),
                        sg.Radio(
                            i18n("输入监听"),
                            "function",
                            key="im",
                            default=False,
                            enable_events=True,
                        ),
                        sg.Radio(
                            i18n("输出变声"),
                            "function",
                            key="vc",
                            default=True,
                            enable_events=True,
                        ),
                    ],
                    [
                        sg.Text(
                            i18n("未启动"),
                            key="status",
                            size=(70, 1),
                            text_color=COLOR_IDLE,
                            expand_x=True,
                        ),
                    ],
                    [
                        sg.Text(i18n("输入电平")),
                        meter("input_meter"),
                        sg.Text(i18n("输出电平")),
                        meter("output_meter"),
                    ],
                    [
                        sg.Text(i18n("算法延迟(ms):")),
                        sg.Text("0", key="delay_time", size=(5, 1)),
                        sg.Text(i18n("推理时间(ms):")),
                        sg.Text("0", key="infer_time", size=(5, 1)),
                        sg.Text(i18n("过载次数:")),
                        sg.Text("0", key="overruns", size=(6, 1)),
                    ],
                ],
            )
            # Everything a first-time user needs is above the fold; the rest
            # has working defaults and stays folded away.
            advanced = sg.Column(
                [[model_frame], [voice_frame, perf_frame]],
                key="advanced",
                visible=bool(data["show_advanced"]),
                expand_x=True,
                pad=(0, 0),
            )
            layout = [
                [quick_frame],
                [status_frame],
                [sg.Button(self.advanced_label(data["show_advanced"]), key="toggle_advanced")],
                [sg.pin(advanced, expand_x=True)],
            ]
            self.show_advanced = bool(data["show_advanced"])
            self.window = sg.Window(
                i18n("RVC 实时变声"), layout=layout, finalize=True, resizable=True
            )
            # Once a setup has worked, opening the program is all it takes.
            self.cable_asked = bool(data["cable_asked"])
            if IS_WINDOWS and not find_virtual_cable(self.output_devices):
                self.offer_virtual_cable()
            if data["auto_start"] and data["started_ok"] and data["pth_path"]:
                self.window.write_event_value("start_vc", None)
            self.event_handler()

        def offer_virtual_cable(self):
            """Without a virtual sound card nobody else can hear the converted
            voice, so offer to install one (asked once; the button stays)."""
            missing = i18n("未检测到虚拟声卡：点“安装虚拟声卡”后队友才能听到变声")
            if self.cable_asked:
                self.set_status(missing, COLOR_WARN)
                return
            self.cable_asked = True
            answer = sg.popup_yes_no(
                i18n(
                    "没有检测到虚拟声卡（VB-CABLE）。\n"
                    "游戏和语音软件要通过它才能收到变声后的声音。\n\n"
                    "现在自动下载并安装吗？（免费，约 1MB，安装时请在弹出的窗口里点“是”和“安装”）"
                ),
                title=i18n("安装虚拟声卡"),
            )
            if answer == "Yes":
                self.install_virtual_cable()
            else:
                self.set_status(missing, COLOR_WARN)

        def install_virtual_cable(self):
            self.set_status(i18n("正在下载并安装虚拟声卡，请稍候…"), COLOR_WARN)
            self.window.refresh()
            try:
                virtual_cable.install()
            except virtual_cable.Cancelled:
                self.set_status(i18n("已取消安装虚拟声卡"), COLOR_WARN)
                return
            except virtual_cable.InstallError as error:
                printt("VB-CABLE: %s", error)
                sg.popup(
                    i18n("虚拟声卡自动安装失败，请手动安装：%s") % virtual_cable.HOME_URL
                )
                return
            # The new device takes a moment to reach the audio system.
            _, values = self.window.read(timeout=0)
            cable = ""
            for _ in range(10):
                self.refresh_device_lists(values)
                cable = find_virtual_cable(self.output_devices)
                if cable:
                    break
                time.sleep(1)
            if not cable:
                sg.popup(i18n("虚拟声卡已安装，重启电脑后生效。重启后再打开本程序即可。"))
                return
            self.window["sg_output_device"].Update(value=cable)
            self.window["install_cable"].update(visible=False)
            self.set_status(i18n("虚拟声卡已装好，输出设备已自动选好"), COLOR_IDLE)
            sg.popup(
                i18n(
                    "虚拟声卡已装好，输出设备已自动选为 %s。\n\n"
                    "最后一步：在游戏或语音软件的设置里，把麦克风改成 CABLE Output。"
                )
                % cable
            )

        # ----------------------------------------------------------- events

        def advanced_label(self, shown):
            return i18n("收起高级设置 ▲") if shown else i18n("高级设置 ▼")

        def set_status(self, text, color=COLOR_IDLE):
            self.window["status"].update(text, text_color=color)

        def set_running(self, running):
            self.window["start_vc"].update(disabled=running)
            self.window["stop_vc"].update(disabled=not running)
            if not running:
                self.window["input_meter"].update(0)
                self.window["output_meter"].update(0)

        def collect_settings(self, values):
            settings = {key: values[key] for key in DEFAULTS if key in values}
            settings["hubert_context"] = self.gui_config.hubert_context
            settings["show_advanced"] = self.show_advanced
            settings["started_ok"] = self.started_ok
            settings["cable_asked"] = self.cable_asked
            # The devices in use can differ from the form after an MME fallback.
            cfg = self.gui_config
            if cfg.sg_hostapi:
                settings["sg_hostapi"] = cfg.sg_hostapi
                settings["sg_input_device"] = cfg.sg_input_device
                settings["sg_output_device"] = cfg.sg_output_device
            settings["sr_type"] = "sr_model" if values["sr_model"] else "sr_device"
            settings["f0method"] = next(
                (name for name in F0_METHODS if values[name]), "rmvpe"
            )
            return settings

        def save_settings(self, values):
            try:
                with open(realtime_config_path, "w", encoding="utf8") as j:
                    json.dump(self.collect_settings(values), j, ensure_ascii=False)
            except Exception:
                printt(traceback.format_exc())

        def refresh_stats(self):
            """Mirror the audio thread's numbers; Tk must only be touched here."""
            if not flag_vc or self.engine is None:
                return
            stats = self.engine.stats
            self.window["infer_time"].update(int(stats["total_ms"]))
            self.window["input_meter"].update(int(max(0, stats["input_db"] + 60)))
            self.window["output_meter"].update(int(max(0, stats["output_db"] + 60)))
            self.window["overruns"].update(stats["overruns"])
            auto_key = stats["auto_key"]
            if auto_key is not None and auto_key != self.gui_config.pitch:
                self.gui_config.pitch = auto_key
                self.window["pitch"].update(auto_key)
            threshold = stats["threshold"]
            if threshold is not None and round(threshold) != self.shown_threshold:
                self.shown_threshold = round(threshold)
                self.window["threhold"].update(self.shown_threshold)
            now = time.monotonic()
            # The limiter keeps the output clean, but a clipped microphone is
            # already distorted before it gets here: tell the user to back off.
            if stats["input_clipped"] > self.shown_input_clipped:
                self.shown_input_clipped = stats["input_clipped"]
                self.notice = i18n("麦克风输入过载（削波）：请调低麦克风音量或离麦克风远一点")
                self.notice_time = now
            if stats["overruns"] > self.shown_overruns:
                self.shown_overruns = stats["overruns"]
                self.overrun_time = now
            # A game that starts later takes GPU time away: when blocks keep
            # arriving late, lengthen them instead of letting the voice stutter.
            self.overrun_marks.append((now, stats["overruns"]))
            while now - self.overrun_marks[0][0] > ADAPT_WINDOW:
                self.overrun_marks.popleft()
            if (
                self.auto_block
                and stats["overruns"] - self.overrun_marks[0][1] >= ADAPT_OVERRUNS
                and self.grow_block()
            ):
                return
            # Keep notices up long enough to be read.
            if now - self.notice_time < 4:
                self.set_status(self.notice, COLOR_WARN)
            elif now - self.overrun_time < 3:
                self.set_status(
                    i18n("过载：推理慢于采样长度，请加大采样长度"), COLOR_BAD
                )
            elif stats["idle"]:
                self.set_status(i18n("待机 · 未说话（不占用显卡）"), COLOR_IDLE)
            elif not stats["gate_open"]:
                self.set_status(i18n("运行中 · 静音(低于响应阈值)"), COLOR_WARN)
            elif self.auto_pitch_unavailable and self.gui_config.auto_pitch:
                self.set_status(
                    i18n("运行中 · 该模型没有音高统计，请手动设置音调"), COLOR_WARN
                )
            else:
                self.set_status(i18n("运行中"), COLOR_OK)

        def grow_block(self):
            """Switch to the next longer block length. Returns False when the
            longest one is already in use."""
            cfg = self.gui_config
            longer = [t for t in AUTO_BLOCK_TIMES if t > cfg.block_time + 1e-6]
            if not longer:
                return False
            self.stop_stream()
            cfg.block_time = longer[0]
            self.engine = RealtimeEngine(
                self.rvc, cfg, self.config.device, cfg.samplerate
            )
            self.engine.prewarm()
            self.open_stream()
            self.window["block_time"].update(cfg.block_time)
            self.update_delay()
            self.reset_run_state()
            self.notice = i18n("显卡繁忙，已自动把采样长度加大到 %.2f 秒") % cfg.block_time
            self.notice_time = time.monotonic()
            printt(self.notice)
            return True

        def reset_run_state(self):
            self.shown_input_clipped = 0
            self.shown_overruns = 0
            self.overrun_time = float("-inf")
            self.overrun_marks = deque([(time.monotonic(), 0)])

        def update_delay(self):
            if self.stream is None or self.engine is None:
                return
            self.delay_time = self.stream.latency[-1] + self.engine.algorithm_delay()
            self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))

        def refresh_device_lists(self, values):
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.update_devices(hostapi_name=values["sg_hostapi"])
            if self.gui_config.sg_hostapi not in self.hostapis:
                self.gui_config.sg_hostapi = self.hostapis[0] if self.hostapis else ""
            self.window["sg_hostapi"].Update(
                values=self.hostapis, value=self.gui_config.sg_hostapi
            )
            for key, names in (
                ("sg_input_device", self.input_devices),
                ("sg_output_device", self.output_devices),
            ):
                current = values[key]
                if current not in names:
                    current = names[0] if names else ""
                self.window[key].Update(values=names, value=current)

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read(timeout=100)
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                if event == sg.TIMEOUT_KEY:
                    self.refresh_stats()
                    continue
                if event in RESTART_KEYS and flag_vc:
                    # Other parameters do not support hot update
                    self.stop_stream()
                    self.set_running(False)
                    self.set_status(i18n("设置已更改，请重新开始音频转换"), COLOR_WARN)
                if event in ("reload_devices", "sg_hostapi"):
                    self.refresh_device_lists(values)
                elif event == "install_cable":
                    self.install_virtual_cable()
                elif event == "refresh_models":
                    self.window["pth_path"].update(
                        values=list_models(), value=values["pth_path"]
                    )
                elif event in ("pth_path", "match_index"):
                    index_path = get_index_path_from_model(values["pth_path"])
                    if index_path or event == "pth_path":
                        self.window["index_path"].update(index_path)
                    else:
                        self.set_status(i18n("没有找到与该模型匹配的索引"), COLOR_WARN)
                elif event in PRESETS:
                    block_time, crossfade, extra_time, f0method, context = PRESETS[
                        event
                    ]
                    self.gui_config.hubert_context = context
                    self.window["block_time"].update(block_time)
                    self.window["crossfade_length"].update(crossfade)
                    self.window["extra_time"].update(extra_time)
                    self.window[f0method].update(True)
                    self.gui_config.f0method = f0method
                    if flag_vc:
                        self.stop_stream()
                        self.set_running(False)
                        self.set_status(
                            i18n("设置已更改，请重新开始音频转换"), COLOR_WARN
                        )
                elif event == "start_vc" and not flag_vc:
                    if self.set_values(values):
                        printt(i18n("CUDA可用：%s"), torch.cuda.is_available())
                        self.set_status(i18n("正在加载模型…"), COLOR_WARN)
                        self.window.refresh()
                        try:
                            self.start_vc()
                        except Exception as error:
                            printt(traceback.format_exc())
                            self.stop_stream()
                            self.set_status(i18n("启动失败"), COLOR_BAD)
                            sg.popup(
                                "%s\n\n%s" % (i18n("启动失败"), error),
                                title=i18n("启动失败"),
                            )
                            continue
                        self.started_ok = True
                        self.save_settings(values)
                        self.reset_run_state()
                        self.set_running(True)
                        self.set_status(i18n("运行中"), COLOR_OK)
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.update_delay()
                elif event == "stop_vc":
                    self.stop_stream()
                    self.set_running(False)
                    self.set_status(i18n("已停止"))
                # Parameter hot update
                elif event == "threhold":
                    # Moving the slider by hand takes over from the automatic one.
                    self.gui_config.threhold = values["threhold"]
                    self.gui_config.auto_threshold = False
                    self.window["auto_threshold"].update(False)
                elif event == "auto_threshold":
                    self.gui_config.auto_threshold = values["auto_threshold"]
                elif event == "toggle_advanced":
                    self.show_advanced = not self.show_advanced
                    self.window["advanced"].update(visible=self.show_advanced)
                    self.window["toggle_advanced"].update(
                        self.advanced_label(self.show_advanced)
                    )
                elif event == "pitch":
                    # Moving the slider by hand takes over from auto pitch.
                    self.gui_config.pitch = values["pitch"]
                    self.gui_config.auto_pitch = False
                    self.window["auto_pitch"].update(False)
                    if hasattr(self, "rvc"):
                        self.rvc.auto_pitch = False
                        self.rvc.change_key(values["pitch"])
                elif event == "auto_pitch":
                    self.gui_config.auto_pitch = values["auto_pitch"]
                    if values["auto_pitch"] and hasattr(self, "rvc"):
                        self.rvc.auto_pitch_locked = False
                        self.rvc.source_f0.clear()
                elif event == "auto_start":
                    self.save_settings(values)
                elif event == "auto_block":
                    pass  # read when the conversion starts
                elif event == "formant":
                    self.gui_config.formant = values["formant"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_formant(values["formant"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "protect":
                    self.gui_config.protect = values["protect"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_protect(values["protect"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in F0_METHODS:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    self.update_delay()
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event in ["vc", "im"]:
                    self.gui_config.function = event

        def set_values(self, values):
            pth_path = values["pth_path"].strip()
            index_path = values["index_path"].strip()
            if len(pth_path) == 0:
                sg.popup(i18n("请选择pth文件"))
                return False
            if not is_voice(pth_path):
                sg.popup(i18n("pth文件不存在：%s") % pth_path)
                return False
            if index_path and not os.path.isfile(index_path):
                sg.popup(i18n("索引文件不存在，将不使用索引继续推理：%s") % index_path)
                index_path = ""
                self.window["index_path"].update("")
            # faiss opens the index through a narrow-character path on Windows.
            if sys.platform == "win32" and re.search("[^\x00-\x7F]", index_path):
                sg.popup(i18n("index文件路径不可包含中文"))
                return False
            if not self.input_devices or not self.output_devices:
                sg.popup(i18n("没有找到可用的音频输入/输出设备"))
                return False
            if (
                values["sg_input_device"] not in self.input_devices
                or values["sg_output_device"] not in self.output_devices
            ):
                sg.popup(i18n("请选择输入设备和输出设备"))
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            settings = self.collect_settings(values)
            cfg = self.gui_config
            cfg.sg_hostapi = values["sg_hostapi"]
            cfg.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            cfg.sg_input_device = values["sg_input_device"]
            cfg.sg_output_device = values["sg_output_device"]
            cfg.pth_path = pth_path
            cfg.index_path = index_path
            cfg.sr_type = settings["sr_type"]
            cfg.threhold = values["threhold"]
            cfg.auto_threshold = values["auto_threshold"]
            cfg.pitch = values["pitch"]
            cfg.auto_pitch = values["auto_pitch"]
            self.auto_block = values["auto_block"]
            cfg.formant = values["formant"]
            cfg.block_time = values["block_time"]
            cfg.crossfade_time = values["crossfade_length"]
            cfg.extra_time = values["extra_time"]
            cfg.I_noise_reduce = values["I_noise_reduce"]
            cfg.O_noise_reduce = values["O_noise_reduce"]
            cfg.rms_mix_rate = values["rms_mix_rate"]
            cfg.index_rate = values["index_rate"] if index_path else 0.0
            cfg.protect = values["protect"]
            cfg.f0method = settings["f0method"]
            cfg.function = "vc" if values["vc"] else "im"
            return True

        def ensure_base_models(self):
            """Fetch HuBERT / RMVPE on first use instead of failing to start."""
            for line in download_assets(inference_assets()):
                printt(line)
                self.set_status(line[:70], COLOR_WARN)
                self.window.refresh()

        def start_vc(self):
            torch.cuda.empty_cache()
            cfg = self.gui_config
            if cfg.pth_path == BUILTIN_HELIUM:
                self.rvc = HeliumVoice(cfg.formant)
            else:
                self.ensure_base_models()
                last = getattr(self, "rvc", None)
                self.rvc = rvc_for_realtime.RVC(
                    cfg.pitch,
                    cfg.formant,
                    cfg.pth_path,
                    cfg.index_path,
                    cfg.index_rate,
                    self.config,
                    last if isinstance(last, rvc_for_realtime.RVC) else None,
                    protect=cfg.protect,
                )
            # The previous voice model is unreferenced now; give its memory back.
            self.engine = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.build_engine()
            try:
                self.start_stream()
            except Exception as error:
                # WASAPI refuses e.g. a microphone and an output that run at
                # different rates. MME is slower but converts anything.
                failed_hostapi = cfg.sg_hostapi
                if not self.fallback_hostapi():
                    raise
                printt("%s -> %s (%s)", failed_hostapi, cfg.sg_hostapi, error)
                self.build_engine()
                self.start_stream()

        def build_engine(self):
            cfg = self.gui_config
            # Shared-mode WASAPI only opens a device at its own sample rate.
            cfg.samplerate = (
                self.rvc.tgt_sr
                if cfg.sr_type == "sr_model" and "WASAPI" not in cfg.sg_hostapi
                else self.get_device_samplerate()
            )
            cfg.channels = self.get_device_channels()
            self.auto_pitch_unavailable = bool(
                cfg.auto_pitch
                and not self.rvc.f0_median
                and not getattr(self.rvc, "fullband", False)
            )
            printt(i18n("正在预热模型"))
            if self.auto_block:
                self.set_status(i18n("正在测试本机速度…"), COLOR_WARN)
                self.window.refresh()
                self.engine, block_time, cost = auto_block_engine(
                    self.rvc,
                    cfg,
                    self.config.device,
                    cfg.samplerate,
                    headroom=AUTO_BLOCK_HEADROOM,
                )
                printt("auto block: %.2fs (%.0f ms per block)", block_time, cost)
                self.window["block_time"].update(block_time)
            else:
                self.engine = RealtimeEngine(
                    self.rvc, cfg, self.config.device, cfg.samplerate
                )
                self.engine.prewarm()

        def fallback_hostapi(self):
            """Re-select the same devices under MME. False when not possible."""
            cfg = self.gui_config
            mme = next((name for name in self.hostapis if "MME" in name), "")
            if not mme or mme == cfg.sg_hostapi:
                return False
            wanted_in, wanted_out = cfg.sg_input_device, cfg.sg_output_device
            self.update_devices(hostapi_name=mme)
            cfg.sg_hostapi = mme
            cfg.sg_input_device = (
                match_device(wanted_in, self.input_devices) or self.default_device(0)
            )
            cfg.sg_output_device = (
                match_device(wanted_out, self.output_devices) or self.default_device(1)
            )
            if not cfg.sg_input_device or not cfg.sg_output_device:
                return False
            self.window["sg_hostapi"].Update(values=self.hostapis, value=mme)
            self.window["sg_input_device"].Update(
                values=self.input_devices, value=cfg.sg_input_device
            )
            self.window["sg_output_device"].Update(
                values=self.output_devices, value=cfg.sg_output_device
            )
            self.set_devices(cfg.sg_input_device, cfg.sg_output_device)
            return True

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                self.open_stream()

        def open_stream(self):
            global flag_vc
            if (
                "WASAPI" in self.gui_config.sg_hostapi
                and self.gui_config.sg_wasapi_exclusive
            ):
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.engine.block_frame,
                samplerate=self.gui_config.samplerate,
                channels=self.gui_config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()
            flag_vc = True

        def stop_stream(self):
            global flag_vc
            flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None

        def audio_callback(self, indata, outdata, frames, times, status):
            """
            音频处理

            Runs on the sound card thread: no printing and no GUI calls here,
            both stall the callback and are heard as crackles.
            """
            try:
                out = self.engine.process(librosa.to_mono(indata.T))
                outdata[:] = out[:, None]
            except Exception:
                outdata.fill(0)
                printt(traceback.format_exc())

        def update_devices(self, hostapi_name=None):
            """获取设备列表"""
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0] if self.hostapis else None
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]

        def set_devices(self, input_device, output_device):
            """设置输出设备"""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt(i18n("输入设备：%s:%s"), str(sd.default.device[0]), input_device)
            printt(i18n("输出设备：%s:%s"), str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)

    gui = GUI()
