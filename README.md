# 双音区音频分离（座位绑定串音对消）

给同事接手用的原型仓库：两路麦克风进、两路按座位绑定的干净语音出。不用声纹，全因果。本仓库包含可运行代码、15 段实采，以及评测与交接说明。

实采是内部评测录音，本仓为私有，请勿外传到公开网络。

## 一句话结论

方案可行，已有可试听原型。现行默认是 **16 ms 窗（N=256 / hop=64）+ 因果 GSC（从零自适应、带空间门控）+ 动态后置抑制 β=1.00**。单说 10 段平均可把对侧串音再压约 **14.93 dB**（固定 λ=0.80 时为 13.33 dB），本路目标基本无损。

预期是「减轻串音」，不是把对侧抹掉。回声消除（AEC）、长时间叠说、以及接真实云端 ASR 的串词，都还没有验证。

更完整的方案说明见 [docs/HANDOFF.md](docs/HANDOFF.md)。数字出处：

- [RESULT-gsc.md](RESULT-gsc.md) — 因果 GSC + 空间门控，以及已放弃的做法
- [RESULT-postfilter.md](RESULT-postfilter.md) — 固定后置强度 λ，默认停在 0.80
- [RESULT-window.md](RESULT-window.md) — 窗长扫描，用户选定 16 ms
- [RESULT-dynamic-postfilter.md](RESULT-dynamic-postfilter.md) — 动态因子 β，建议默认 1.00

## 怎么跑

需要 Python 3，以及 `numpy`、`scipy`（见 `requirements.txt`）。第一次启动会按仓库里的 `pcm/` 预计算 15 段，可能要一两分钟。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python ./start.sh
```

默认监听 [http://127.0.0.1:8766/](http://127.0.0.1:8766/)。端口被占用时换一个：

```bash
HOST=127.0.0.1 PORT=8767 PYTHON=.venv/bin/python ./start.sh
```

只预计算、不启动服务：

```bash
PYTHON=.venv/bin/python server.py --precompute-only
```

打开浏览器后，选一段实采，对比「主驾原麦 / 主驾输出 / 副驾原麦 / 副驾输出」。页面上的滑杆是动态因子 β：0 等于固定 λ=0.80，默认停在建议值 1.00。另有「只做 GSC」开关作对照。

健康检查：`http://127.0.0.1:8766/api/health`。

## 测试音频

15 段无头 PCM，在 [`pcm/`](pcm/) 下，按采集条件分三个目录，每段一个时间戳子目录：

- `pcm/Only driver/` — 5 段，只有主驾说
- `pcm/Only Passenger/` — 5 段，只有副驾说
- `pcm/Both driver and passenger/` — 5 段，两人会话（有交替，也有短暂重叠）

每段通常包含 `mic_raw.pcm`（主驾麦）、`mic2_raw.pcm`（副驾麦），以及采集时留下的 `echo_ref.pcm`、`aec_output.pcm`、`aec2_output.pcm`（这批里回采几乎是静音，AEC 未验证）。

**格式假设（文件本身没有头、也没有采样率标注）：** 16 kHz、16-bit little-endian、单声道。若实际不是 16 kHz，时间和听感、全部 dB 结论都要重算。

服务启动时只要仓库里有 `pcm/`，就会用它，不必再改路径。也可以用环境变量 `AUDIO_PCM_ROOT` 指向另一份同样结构的数据。

## 现行算法参数

| 项 | 值 |
|---|---|
| 通路 | 两进两出，成对处理，输出绑座位 |
| 声纹 | 不用，只靠空间特征 |
| 窗长 / hop | 256 / 64（16 kHz，周期 Hann，hop=N/4） |
| 算法时延 | 16 ms（按 N/fs；OLA 块内约 12.0–15.9 ms） |
| GSC | 从零自适应，`mu=0.25`，先出本帧再更新；空间门控（ILD + 帧级 GCC-PHAT）保护本路 |
| 后置 | 因果残差 Wiener；默认 β=1.00。β=0 即固定 λ=0.80。控制量用 ILD、TDOA 和阻塞支路，不用对侧原麦 RMS |

## HTTP API

`POST /api/cancel`，`multipart/form-data`：

- `mic_driver`（别名 `mic_raw`）：主驾单声道 wav 或 pcm
- `mic_passenger`（别名 `mic2_raw`）：副驾单声道 wav 或 pcm

成功时返回 zip，内含 `driver_out.wav`、`passenger_out.wav`、`info.json`。上传的 WAV 必须是 16 kHz，否则接口会拒绝，不会擅自重采样。

```bash
curl -sS -X POST http://127.0.0.1:8766/api/cancel \
  -F "mic_driver=@pcm/Only driver/20261126_152912/mic_raw.pcm" \
  -F "mic_passenger=@pcm/Only driver/20261126_152912/mic2_raw.pcm" \
  -o cancel_out.zip
```

## 目录

```
cancel.py / server.py / start.sh / static/   算法与试听服务
pcm/                                         15 段实采
docs/HANDOFF.md                              给接手人的完整说明
RESULT-*.md、summary_*.csv、eval_*_figs/     评测结论、表格与图
run_*_eval.py                                复现评测的脚本
```

`.venv/` 和首次运行生成的 `cache/` 不进仓库，本地跑过就会出现。
