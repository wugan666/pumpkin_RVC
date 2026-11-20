<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
简单易用的 语音音色转换/变声器 框架<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)


[**更新日志**](./docs/cn/Changelog_CN.md) | [**常见问题解答**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E8%A7%A3%E7%AD%94) | [**AutoDL·5毛钱训练AI歌手**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**对照实验记录**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**在线演示**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](./docs/en/README.en.md) | [**中文简体**](./README.md) | [**日本語**](./docs/jp/README.ja.md) | [**한국어**](./docs/kr/README.ko.md) ([**韓國語**](./docs/kr/README.ko.han.md)) | [**Français**](./docs/fr/README.fr.md) | [**Türkçe**](./docs/tr/README.tr.md) | [**Português**](./docs/pt/README.pt.md)

</div>

> 底模使用接近50小时的开源高质量VCTK训练集训练，无版权方面的顾虑，请大家放心使用

> 请期待RVCv3的底模，参数更大，数据更大，效果更好，基本持平的推理速度，需要训练数据量更少。

<table>
   <tr>
		<td align="center">训练推理界面</td>
		<td align="center">实时变声界面</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-webui.bat</td>
		<td align="center">go-realtime_gui.bat</td>
	</tr>
  <tr>
    <td align="center">可以自由选择想要执行的操作。</td>
		<td align="center">我们已经实现端到端170ms延迟。如使用ASIO输入输出设备，已能实现端到端90ms延迟，但非常依赖硬件驱动支持。</td>
	</tr>
</table>

## 简介
本仓库具有以下特点
+ 使用top1检索替换输入源特征为训练集特征来杜绝音色泄漏
+ 即便在相对较差的显卡上也能快速训练
+ 使用少量数据进行训练也能得到较好结果(推荐至少收集10分钟低底噪语音数据)
+ 可以通过模型融合来改变音色(借助ckpt处理选项卡中的ckpt-merge)
+ 简单易用的网页界面
+ 可调用pymss/MSST模型来快速分离人声和伴奏
+ 使用最先进的[人声音高提取算法InterSpeech2023-RMVPE](#参考项目)根绝哑音问题，速度快、资源占用小
+ A卡/I卡使用 CPU 依赖方案；Windows 可使用 DirectML，Linux 使用 CPU

点此查看我们的[演示视频](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

## Windows 一键使用

装好 NVIDIA 显卡驱动后，下载本项目并双击 **`一键启动.bat`**：

- 首次运行会自动完成全部准备：创建运行环境 `runtime`、按显卡型号安装对应的依赖（RTX 50 系用 CUDA 12.8，其余 NVIDIA 显卡用 CUDA 11.8，没有 NVIDIA 显卡用 CPU 版）、下载基础模型和 ffmpeg。需要下载约 3~5GB，只做一次。
- 电脑上没有 Python 3.12 时会询问是否用 winget 自动安装。
- 之后双击直接打开统一网页主界面：一键克隆、音频转换、素材处理、更多工具。实时变声从“更多工具 → 实时变声”打开，也可直接运行 `go-realtime_gui.bat`。
- 实时变声成功开始过一次后，下次打开软件会自动开始变声（可在界面上取消勾选“打开软件后自动开始”）。

推荐顺序：先在“一键克隆”做出音色并播放对比，再从“更多工具”打开实时变声。下面的“环境配置”是手动安装方式，使用一键启动时可以跳过。

## 环境配置

本分支面向 **Python 3.12 x64**，请先进入仓库根目录。Ubuntu 推荐使用 Ubuntu 24.04 x86_64。

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

安装 Python 3.12 x64 后创建虚拟环境：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 按硬件选择依赖

| 硬件 | 安装方式 |
| --- | --- |
| CPU、AMD、Intel | 使用 `requirments_cpu_py312.txt`；Windows 可使用 DirectML，Linux 使用 CPU |
| NVIDIA RTX 50 系 | 先安装 CUDA 12.8 版 Torch，再安装 `requirments_cu128_py312.txt` |
| NVIDIA RTX 50 系以前 | 先安装 CUDA 11.8 版 Torch，再安装 `requirments_cu118_py312.txt` |

#### CPU、AMD、Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX 50 系：两阶段安装

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### NVIDIA RTX 50 系以前：两阶段安装

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

检查 Torch 与 CUDA 状态：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```


### 修改下载源

三个 `requirments_*.txt` 顶部已经包含下载源。中国大陆用户可保留默认镜像；需要使用官方源时，只替换 `--index-url` 和 `--extra-index-url`，保留包版本、CUDA 后缀和两阶段顺序。

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## 模型与运行目录

WebUI 会自动创建运行目录。模型请从 [Hugging Face 模型仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 下载，并保持以下路径：

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── rmvpe/rmvpe.pt
├── pretrained/
├── pretrained_v2/
├── pymss_weights/
├── weights/        # user RVC .pth models
└── indices/        # user .index files
logs/
└── mute/           # training silence samples

# Exact paths used by the code
assets/hubert_base/config.json
assets/hubert_base/preprocessor_config.json
assets/hubert_base/pytorch_model.bin
assets/rmvpe/rmvpe.pt
assets/pretrained/*.pth
assets/pretrained_v2/*.pth
assets/pymss_weights/*
assets/weights/*.pth
assets/indices/*.index
logs/mute/*
```

### 下载模型

```bash
python -m pip install --upgrade huggingface_hub

# Required for inference and feature extraction
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Required for v1/v2 training
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Required only for pymss/MSST vocal separation
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pymss_weights/*" --local-dir assets
```

仅 Windows AMD/Intel DirectML 环境还需要：

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

### FFmpeg

Ubuntu 已在前面的系统依赖命令中安装 FFmpeg。Windows 用户可把下面两个文件放到项目根目录：

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## 开始使用

启动 WebUI：

```bash
python webui.py        # Linux / macOS 也可以直接运行 ./go-webui.sh
```

首次启动时如果缺少 HuBERT、RMVPE 等推理必需的基础模型，程序会在后台自动下载，无需手动准备。

无桌面的 Ubuntu 服务器：

```bash
python webui.py --noautoopen
```

默认服务监听端口为 `7865`。用户自己的 `.pth` 模型放入 `assets/weights/`，`.index` 文件放入 `assets/indices/`。

WebUI 默认监听所有网卡（`0.0.0.0`）。在公网服务器上只想本机访问时，设置 `RVC_WEBUI_HOST=127.0.0.1` 再启动。

### 一键克隆音色

WebUI 首页“一键克隆”把录音、播放、训练和试听放在同一页：

1. 填一个音色名称，上传或录制**同一个人**的干净人声，至少 30 秒，推荐 3~10 分钟、覆盖不同句子和语气。可选择上传文件逐个播放，麦克风录音结束后也能直接回放。素材很多时可以填写文件夹路径。
2. 默认选择“精细克隆”。程序自动下载缺少的基础模型，完成切分、RMVPE 音高与特征提取、训练和索引。快速、标准、精细分别至少训练 20/80/120 轮，同时至少完成 160/800/1600 个实际更新步骤；最终轮数按切分后有效批次数计算。这是训练预算，不是相似度保证。
3. 右侧明确选择试听来源：麦克风录音、上传音频或克隆素材。转换后可分别播放本次原声和变声结果。建议用新录的一句话判断效果；只转换训练素材不能说明对新句子的表现。

重复使用同一个名称时，程序自动生成 `my_voice_2` 等新名称，保留旧音色。每次一键克隆使用新的训练目录并生成匹配索引，避免旧特征、检查点和索引污染新录音。旧版本训练出的音色需要重新克隆才能获得训练修复的效果。

开始前会提示削波、音量过小等录音问题；少于 10 秒有效有声音高帧时停止训练。录音条件提示不能识别所有混响、伴奏或多人声音，仍需先播放确认。训练完成后默认清理中间文件，保留模型、索引、原素材和日志；需要继续训练时取消“清理”并在“更多工具 → 训练”里操作。

试听默认自动变调，按训练人声的音高中位数匹配目标音域。唱歌时建议关闭并手动调整。高相似度还取决于素材时长、干净程度、发音覆盖和源声音域，几秒录音或带伴奏的录音不能靠增加轮数补足。

“音频转换”保留单次和批量处理，“素材处理”提供人声分离与去混响；训练、多说话人、模型处理和实时变声入口集中在“更多工具”。本机实时窗口负责低延迟音频设备处理，不需要另开一个网页服务。`tools/pymss/server` 是内嵌依赖提供的独立服务，日常使用无需启动。

连续试听会复用 HuBERT、音高模型、已选音色和索引，模型或索引更新后自动重新加载；转换切片之间保留 PyTorch 显存缓存。重任务串行排队，停止按钮与素材播放保持可用，避免训练和多个转换请求争抢同一显卡。切换到克隆训练时释放推理模型占用。

基础模型也可以用命令行提前下载（无法访问 Hugging Face 时会自动改用镜像，也可以用 `HF_ENDPOINT` 指定）：

```bash
python -m tools.download_models            # 推理 + v2 40k 训练所需
python -m tools.download_models --check    # 只检查缺少哪些文件
```

### 实时变声

```bash
python realtime_gui.py        # Linux / macOS 也可以直接运行 ./go-realtime_gui.sh
```

默认界面只有四样东西：音色模型、输入/输出设备、音调、开始按钮。选好模型和设备后点“开始音频转换”即可，其余都是自动的：

还没有自己的音色时，音色模型里默认选中 **氦气音（内置）**：不需要任何模型文件，打开就能用。它把共振峰整体上移约 1.6 倍而音高不变（吸氦气的真实效果），用“声线粗细”滑块可以调强弱，音调、索引等设置对它不起作用。

- 缺少基础模型时自动下载；索引按模型名自动匹配（索引是可选的）。
- **自动变调**：说几秒话后按你的音高和目标音色自动设定音调并锁定；手动拖动音调滑块即切回手动。旧模型没有音高统计时会提示手动设置。
- **自动选择采样长度**：启动时实测本机速度，选出不会断续的最短采样长度（延迟最低）。
- 状态栏显示输入/输出电平、推理时间和过载次数。**推理时间必须小于采样长度**，否则声音会断续。

#### 打游戏时和队友语音（Windows）

变声后的声音要先进“虚拟声卡”，游戏或语音软件再把虚拟声卡当麦克风用：

1. 打开实时变声界面。电脑上没有虚拟声卡时会弹窗询问，点“是”即自动下载并安装 [VB-CABLE](https://vb-audio.com/Cable/)（免费；安装需要管理员权限，按提示点“是”和“安装”）。之后也可以点输出设备旁的“安装虚拟声卡”按钮。自动安装失败时可从官网手动安装。
2. 输入设备选你的真实麦克风，输出设备会自动选好 `CABLE Input`，点“开始音频转换”。
3. 在 Discord / KOOK / YY / 游戏内语音的设置里，把**麦克风（输入设备）**改成 `CABLE Output`。建议关掉这些软件自带的降噪和回声消除，它们容易把变声后的声音当噪声削掉。

为这个场景做的自动处理（都默认开启，不用设置）：

- **不说话时不占显卡**：噪声门关闭期间完全跳过模型推理，把显卡留给游戏；状态栏显示“待机”。
- **自动噪声门**：按环境底噪自动设定响应阈值，键盘、风扇等低于阈值的声音不会被变声外放。
- **抗抢占**：采样长度按“只用一半时间”来选，给游戏留余量；游戏开始后如果仍连续超时，会自动加大采样长度并提示，而不是一直断续。
- **低延迟设备**：Windows 上优先使用 WASAPI；如果打不开（常见原因是麦克风和虚拟声卡采样率不同），自动退回 MME。想要最低延迟，可在 Windows 声音设置里把麦克风和 `CABLE Input` 都设成 48000 Hz。
- 程序在 Windows 上以“高于正常”的优先级运行，避免游戏占满 CPU 时声音卡顿。
- **防炸麦**：输出带限幅器，喊叫、大笑、拍桌时峰值被平滑压在 -3 dB 以内，不会再被声卡硬削波；模型偶发的异常输出会被静音而不是爆音。

如果仍然觉得炸麦，按下面顺序排查：

1. 状态栏提示“麦克风输入过载（削波）”：声音在进入变声前就已经失真，请在 Windows 声音设置里调低麦克风音量，并关闭“麦克风增强”。
2. 语音软件里关闭“自动增益控制”“自动调节输入灵敏度”，否则它会把已经限幅的声音再放大。
3. 确认没有开着扬声器外放让麦克风录回去，也没有在 Windows 里勾选“侦听此设备”。

点“高级设置”可以展开其余选项：

- “预设”提供低延迟 / 均衡 / 高音质三组参数。均衡和低延迟会复用前面音频块已算过的 HuBERT 特征（更快），高音质每块完整重算。
- “响应阈值”是噪声门：低于阈值的环境噪声不会被送去变声，说话停顿时输出保持安静。
- “清辅音保护”在开启索引检索时保留清辅音和气声，减少咬字发糊。

VST 插件（`RVCRealtimeVST`）的工作进程使用同一套实时引擎，效果与独立界面一致。

没有声卡也可以用文件验证实时效果，下面的命令走的是和 GUI 完全相同的逐块处理链路：

```bash
python -m infer.realtime_engine --model assets/weights/my_voice.pth \
    --input in.wav --output out.wav --threshold -45 --auto-pitch --auto-block
```

## 参考项目
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [pymss-project/pymss](https://github.com/pymss-project/pymss)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).

## 感谢所有贡献者作出的努力
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
