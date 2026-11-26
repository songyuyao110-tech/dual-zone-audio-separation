#!/usr/bin/env python3
"""Seat-bound crosstalk-cancellation demo service (Web UI + simple HTTP API).

v1.1 is a file/session demo: the 15 captures are precomputed with the
evaluation's recommended method (causal STFT-domain GSC from zero +
spatial gating) plus a causal residual Wiener post-filter. The UI slider
mixes GSC-only and full-strength post-filter outputs; it does not rerun
GSC. Live mic streaming is out of scope.
"""
from __future__ import annotations

import argparse
import io
import json
import posixpath
import sys
import threading
import traceback
import zipfile
from email import message_from_bytes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
CACHE = HERE / "cache"
CACHE_WAV = CACHE / "wav"
SESSIONS_JSON = CACHE / "sessions.json"
DEFAULT_PCM_ROOT = HERE / "pcm"
EVAL_WAV = HERE / "eval_ref_wav"  # optional; omitted from this repo

HOST_DEFAULT = "0.0.0.0"
PORT_DEFAULT = 8766

# Evaluation numbers quoted in the UI — from RESULT-window.md / RESULT-postfilter.md,
# not measured live here. Suggested slider default remains λ=0.80.
DEFAULT_POSTFILTER_STRENGTH = 0.80  # β=0 floor; still the fixed-λ baseline
DEFAULT_DYNAMIC_BETA = 1.00         # RESULT-dynamic-postfilter.md suggestion
# Operating STFT: user-selected 16 ms point from the window sweep
# (RESULT-window.md). N=256 is the shortest window whose single-talk
# mean CTA is not below the N=512 / 32 ms baseline.
OPERATING_N_FFT = 256
OPERATING_HOP = 64

EVAL_SUMMARY = {
    "source": "动态后置 RESULT-dynamic-postfilter.md（N=256、16 ms，默认 β=1.00）；固定 λ=0.80 对照见同文与 RESULT-window.md。不是本页现场测量。",
    "default_postfilter_strength": DEFAULT_POSTFILTER_STRENGTH,
    "default_dynamic_beta": DEFAULT_DYNAMIC_BETA,
    "n_fft": OPERATING_N_FFT,
    "hop": OPERATING_HOP,
    "algo_delay_ms": OPERATING_N_FFT / 16000 * 1000.0,
    "gsc_only": {
        "cta_ten_session_mean_db": 6.83,
        "cta_driver_only_mean_db": 5.57,
        "cta_passenger_only_mean_db": 8.09,
        "target_eratio_mean_db": -0.04,
    },
    "fixed_lambda_080": {
        "strength": 0.80,
        "cta_ten_session_mean_db": 13.33,
        "cta_driver_only_mean_db": 13.45,
        "cta_passenger_only_mean_db": 13.22,
        "both_alternate_cta_driver_talk_db": 9.81,
        "both_alternate_cta_passenger_talk_db": 12.16,
        "target_eratio_mean_db": -0.12,
        "target_corr_mean": 0.997,
        "target_sisdr_min_db": 13.85,
    },
    "default_strength": {
        # kept for older UI fields; now means "β=0 / fixed 0.80"
        "strength": DEFAULT_POSTFILTER_STRENGTH,
        "cta_ten_session_mean_db": 13.33,
        "cta_driver_only_mean_db": 13.45,
        "cta_passenger_only_mean_db": 13.22,
        "both_alternate_cta_driver_talk_db": 9.81,
        "both_alternate_cta_passenger_talk_db": 12.16,
        "target_eratio_mean_db": -0.12,
        "target_corr_mean": 0.997,
        "target_sisdr_min_db": 13.85,
        "cta_delta_vs_gsc_db": 6.50,
        "cta_delta_vs_n512_db": 0.17,
    },
    "default_dynamic": {
        "beta": DEFAULT_DYNAMIC_BETA,
        "cta_ten_session_mean_db": 14.93,
        "cta_driver_only_mean_db": 15.69,
        "cta_passenger_only_mean_db": 14.17,
        "both_alternate_cta_driver_talk_db": 10.71,
        "both_alternate_cta_passenger_talk_db": 12.97,
        "target_eratio_mean_db": -0.12,
        "target_corr_mean": 0.997,
        "target_sisdr_min_db": 13.85,
        "cta_delta_vs_fixed080_db": 1.60,
        "mapping": "λ(t)=clip(0.80+β·δ(t),0,1)；对侧占优→1.0，叠说→0.60，本路/静音→0.80",
    },
    "baseline_n512_lambda080": {
        "cta_ten_session_mean_db": 13.16,
        "algo_delay_ms": 32.0,
    },
    "note": "窗长 N=256（16 ms）。默认动态后置 β=1.00：单说平均 CTA 14.93 dB（相对固定 λ=0.80 的 13.33 dB 多 1.60 dB），本路能量比/相关/LSD 几乎不变。β=0 即固定 0.80。重叠段有 λ 上限 0.60；对侧仍清不干净。",
}


COND_SLUGS = {
    "Only driver": "only_driver",
    "Only Passenger": "only_passenger",
    "Both driver and passenger": "both",
}

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wav": "audio/wav",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
    ".md": "text/markdown; charset=utf-8",
}

_state_lock = threading.Lock()
_engine = None  # lazy: pack + sessions, only if numpy/scipy available
_algo_error = None


def try_import_cancel():
    global _algo_error
    try:
        import cancel as cancel_mod  # noqa: WPS433
        return cancel_mod
    except Exception as exc:
        _algo_error = f"{type(exc).__name__}: {exc}"
        return None


cancel = try_import_cancel()
if cancel is not None:
    cancel.set_stft_params(OPERATING_N_FFT, OPERATING_HOP)


def pcm_root() -> Path:
    env = Path(__file__).resolve().parent / "pcm"
    if env.exists():
        return env
    return Path(__import__("os").environ.get("AUDIO_PCM_ROOT", str(DEFAULT_PCM_ROOT)))


def ensure_cache(force: bool = False) -> dict:
    """Precompute 15-session wavs if the cache is missing. Returns sessions.json."""
    if SESSIONS_JSON.exists() and not force:
        data = json.loads(SESSIONS_JSON.read_text())
        want_n = cancel.N_FFT if cancel is not None else OPERATING_N_FFT
        want_h = cancel.HOP if cancel is not None else OPERATING_HOP
        same_stft = int(data.get("n_fft", 0)) == int(want_n) and int(data.get("hop", 0)) == int(want_h)
        # Basic integrity: all listed wavs present
        ok = same_stft
        if ok:
            required = (
                "mic_raw", "mic2_raw", "driver_out", "passenger_out",
                "driver_out_pf", "passenger_out_pf",
                "driver_out_base", "passenger_out_base",
                "driver_out_exc", "passenger_out_exc",
            )
            for s in data.get("sessions", []):
                files = s.get("files", {})
                for key in required:
                    rel = files.get(key)
                    if not rel or not (CACHE / rel).exists():
                        ok = False
                        break
                if not ok:
                    break
        if ok and len(data.get("sessions", [])) >= 15:
            return data
        if not same_stft:
            print(
                f"[precompute] cache STFT {data.get('n_fft')}/{data.get('hop')} "
                f"!= operating {want_n}/{want_h}, rebuilding",
                flush=True,
            )
    if cancel is None:
        raise RuntimeError(
            "缓存不完整，且无法导入算法依赖（需要 numpy / scipy）。"
            + (f" 导入错误：{_algo_error}" if _algo_error else "")
        )
    return precompute_all(pcm_root(), CACHE, verify_dir=EVAL_WAV if EVAL_WAV.exists() else None)


def precompute_all(root: Path, cache_dir: Path, verify_dir: Path | None = None) -> dict:
    print(f"[precompute] loading sessions from {root}", flush=True)
    sessions = cancel.list_sessions(root)
    if len(sessions) != 15:
        print(f"[precompute] warning: expected 15 sessions, got {len(sessions)}", flush=True)
    print(f"[precompute] {len(sessions)} sessions, calibrating + RTF (LOSO / pool)...", flush=True)
    pack = cancel.prepare_rtfs(sessions)
    calib = pack["calib"]
    wav_root = cache_dir / "wav"
    rows = []
    verify = []
    for s in sessions:
        sid, cond, label = s["sid"], s["cond"], s["label"]
        rtf_d, rtf_p = cancel.rtf_for_session(pack, sid)
        yd, yp, st = cancel.cancel_pair(s["mic1"], s["mic2"], rtf_d, rtf_p, pack["freqs"])
        out_dir = wav_root / cond / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        yd_pf, yp_pf, pf = cancel.postfilter_full(
            s["mic1"], s["mic2"], yd, yp, labels=st["labels"]
        )
        yd_base, yp_base, yd_exc, yp_exc, _dyn = cancel.dynamic_postfilter_components(
            s["mic1"], s["mic2"], yd, yp, rtf_d, rtf_p,
            labels=st["labels"], pf_state=pf,
        )
        cancel.save_wav(out_dir / "mic_raw.wav", s["mic1"])
        cancel.save_wav(out_dir / "mic2_raw.wav", s["mic2"])
        cancel.save_wav(out_dir / "driver_out.wav", yd)
        cancel.save_wav(out_dir / "passenger_out.wav", yp)
        cancel.save_wav(out_dir / "driver_out_pf.wav", yd_pf)
        cancel.save_wav(out_dir / "passenger_out_pf.wav", yp_pf)
        cancel.save_wav(out_dir / "driver_out_base.wav", yd_base)
        cancel.save_wav(out_dir / "passenger_out_base.wav", yp_base)
        cancel.save_wav(out_dir / "driver_out_exc.wav", yd_exc)
        cancel.save_wav(out_dir / "passenger_out_exc.wav", yp_exc)
        slug_dir = wav_root / COND_SLUGS.get(cond, cond.replace(" ", "_").lower())
        if not slug_dir.exists():
            slug_dir.symlink_to(cond)
        # compact label counts for the UI
        labs = list(st["labels"])
        counts = {}
        for lab in labs:
            counts[str(lab)] = counts.get(str(lab), 0) + 1
        slug = COND_SLUGS.get(cond, cond.replace(" ", "_").lower())
        rec = {
            "id": f"{slug}/{sid}",
            "sid": sid,
            "cond": cond,
            "label": label,
            "duration_s": round(len(s["mic1"]) / cancel.FS, 3),
            "n_samples": int(len(s["mic1"])),
            "frame_labels": counts,
            "files": {
                "mic_raw": f"wav/{slug}/{sid}/mic_raw.wav",
                "mic2_raw": f"wav/{slug}/{sid}/mic2_raw.wav",
                "driver_out": f"wav/{slug}/{sid}/driver_out.wav",
                "passenger_out": f"wav/{slug}/{sid}/passenger_out.wav",
                "driver_out_pf": f"wav/{slug}/{sid}/driver_out_pf.wav",
                "passenger_out_pf": f"wav/{slug}/{sid}/passenger_out_pf.wav",
                "driver_out_base": f"wav/{slug}/{sid}/driver_out_base.wav",
                "passenger_out_base": f"wav/{slug}/{sid}/passenger_out_base.wav",
                "driver_out_exc": f"wav/{slug}/{sid}/driver_out_exc.wav",
                "passenger_out_exc": f"wav/{slug}/{sid}/passenger_out_exc.wav",
            },
        }
        rows.append(rec)
        if verify_dir is not None and cancel.N_FFT == 512 and cancel.HOP == 128:
            ev = verify_dir / cond / sid / "gsc_online_gated"
            if ev.exists():
                import numpy as np
                from scipy.io import wavfile
                for name, arr in (("driver_out.wav", yd), ("passenger_out.wav", yp)):
                    ev_path = ev / name
                    if not ev_path.exists():
                        continue
                    sr, ref = wavfile.read(str(ev_path))
                    ref_f = ref.astype(np.float64) / 32768.0
                    ours = np.clip(arr, -1.0, 1.0)
                    n = min(len(ours), len(ref_f))
                    diff = float(np.max(np.abs(ours[:n] - ref_f[:n])))
                    verify.append({"sid": sid, "file": name, "max_abs_diff": diff})
                    print(f"  verify {cond}/{sid}/{name} max_abs_diff={diff:.6g}", flush=True)
        print(f"  wrote {cond}/{sid} ({rec['duration_s']}s)", flush=True)

    meta = {
        "fs": cancel.FS,
        "n_fft": cancel.N_FFT,
        "hop": cancel.HOP,
        "algo_delay_ms": cancel.ALGO_DELAY_MS,
        "method": "gsc_online_gated_dynamic_postfilter",
        "method_label": "GSC 从零 + 空间门控 + 动态残差后置抑制",
        "mu": 0.25,
        "gate": True,
        "adapt": True,
        "postfilter": {
            "type": "causal residual Wiener with spatial dynamic λ(t)",
            "strength_range": [0.0, 1.0],
            "default_strength": DEFAULT_POSTFILTER_STRENGTH,
            "default_dynamic_beta": DEFAULT_DYNAMIC_BETA,
            "mix": "y(β)=y_base+β·y_exc；β=0 即固定 λ=0.80；滑杆只调 β，不重跑 GSC",
            "mapping": "λ(t)=clip(0.80+β·δ(t),0,1)；对侧占优→1.0，叠说→0.60，本路/静音→0.80",
            "control": "ILD + TDOA + GSC blocking-branch energy；不用对侧原麦 RMS",
            "max_atten_db": getattr(cancel, "PF_MAX_ATTEN_DB", 18.0),
            "target_floor_db": getattr(cancel, "PF_TARGET_FLOOR_DB", 2.5),
            "overlap_floor_db": getattr(cancel, "PF_OVERLAP_FLOOR_DB", 6.0),
            "lam_base": getattr(cancel, "DYN_LAM_BASE", 0.80),
            "lam_opp": getattr(cancel, "DYN_LAM_OPP", 1.00),
            "lam_ovl": getattr(cancel, "DYN_LAM_OVL", 0.60),
        },
        "default_postfilter_strength": DEFAULT_POSTFILTER_STRENGTH,
        "default_dynamic_beta": DEFAULT_DYNAMIC_BETA,
        "assumption": "实采 PCM 无头，按 16 kHz、16-bit little-endian、单声道解读（与评测一致）。文件本身没有采样率标注。",
        "channels": {"mic_raw": "主驾麦", "mic2_raw": "副驾麦"},
        "tau_d_ms": calib["tau_d"] * 1000.0,
        "tau_p_ms": calib["tau_p"] * 1000.0,
        "eval": EVAL_SUMMARY,
        "verify_vs_eval": verify,
        "sessions": rows,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    SESSIONS_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[precompute] wrote {SESSIONS_JSON}", flush=True)
    return meta


def get_engine():
    """Lazy-load sessions + RTFs for the POST /api/cancel path."""
    global _engine
    with _state_lock:
        if _engine is not None:
            return _engine
        if cancel is None:
            raise RuntimeError(_algo_error or "算法依赖不可用")
        root = pcm_root()
        if not root.exists():
            raise RuntimeError(
                f"找不到实采目录 {root}，POST /api/cancel 需要单说段来估计阻塞矩阵 RTF。"
            )
        sessions = cancel.list_sessions(root)
        pack = cancel.prepare_rtfs(sessions)
        _engine = pack
        return _engine


def load_sessions_meta() -> dict:
    return ensure_cache(force=False)


def parse_multipart(content_type: str, body: bytes) -> dict:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = message_from_bytes(header + body)
    files = {}
    parts = msg.get_payload()
    if not isinstance(parts, list):
        parts = msg.iter_parts() if hasattr(msg, "iter_parts") else []
    for part in parts:
        cd = part.get("Content-Disposition", "")
        name = part.get_param("name", header="Content-Disposition")
        filename = part.get_filename() or ""
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if name:
            files[str(name)] = {"filename": filename, "data": payload, "cd": cd}
    return files


def safe_join(root: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/")
    if ".." in Path(rel).parts:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = "AudioPcmCancelService/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj, extra_headers=None):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8", extra_headers)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in ("/", "/index.html"):
            self._serve_file(STATIC / "index.html")
            return
        if path == "/api/health":
            meta_ok = SESSIONS_JSON.exists()
            n = 0
            if meta_ok:
                try:
                    n = len(json.loads(SESSIONS_JSON.read_text()).get("sessions", []))
                except Exception:
                    meta_ok = False
            self._send_json(200, {
                "ok": True,
                "algo_available": cancel is not None,
                "algo_error": _algo_error,
                "cache_ready": meta_ok,
                "n_sessions": n,
                "method": "gsc_online_gated_dynamic_postfilter",
                "fs": 16000,
                "n_fft": cancel.N_FFT if cancel is not None else OPERATING_N_FFT,
                "hop": cancel.HOP if cancel is not None else OPERATING_HOP,
                "algo_delay_ms": (cancel.ALGO_DELAY_MS if cancel is not None
                                  else OPERATING_N_FFT / 16000 * 1000.0),
                "default_postfilter_strength": DEFAULT_POSTFILTER_STRENGTH,
                "default_dynamic_beta": DEFAULT_DYNAMIC_BETA,
                "postfilter": "causal residual Wiener with spatial dynamic λ(t); β=0 is fixed λ=0.80",
            })
            return
        if path == "/api/sessions":
            try:
                meta = load_sessions_meta()
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, meta)
            return
        if path.startswith("/audio/"):
            rel = path[len("/audio/"):]
            target = safe_join(CACHE, rel)
            if target is None or not target.is_file():
                self._send_json(404, {"ok": False, "error": "file not found"})
                return
            self._serve_file(target)
            return
        if path.startswith("/static/"):
            target = safe_join(STATIC, path[len("/static/"):])
            if target is None or not target.is_file():
                self._send_json(404, {"ok": False, "error": "file not found"})
                return
            self._serve_file(target)
            return
        if path == "/README.md":
            self._serve_file(HERE / "README.md")
            return
        if path in ("/RESULT-postfilter.md", "/RESULT.md"):
            self._serve_file(HERE / "RESULT-postfilter.md")
            return
        if path in ("/RESULT-window.md",):
            self._serve_file(HERE / "RESULT-window.md")
            return
        if path in ("/RESULT-dynamic-postfilter.md",):
            self._serve_file(HERE / "RESULT-dynamic-postfilter.md")
            return
        self._send_json(404, {"ok": False, "error": "not found", "path": path})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/cancel":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        if cancel is None:
            self._send_json(503, {
                "ok": False,
                "error": "算法依赖不可用（需要 numpy 和 scipy）。试听 UI 仍可播放预计算 wav。",
                "detail": _algo_error,
            })
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"ok": False, "error": "empty body"})
            return
        if length > 40 * 1024 * 1024:
            self._send_json(413, {"ok": False, "error": "body too large (max 40 MiB)"})
            return
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype.lower():
            self._send_json(400, {
                "ok": False,
                "error": "请用 multipart/form-data，字段 mic_driver（或 mic_raw）与 mic_passenger（或 mic2_raw）。",
            })
            return
        try:
            files = parse_multipart(ctype, body)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"无法解析 multipart: {exc}"})
            return

        def pick(*names):
            for n in names:
                if n in files and files[n]["data"]:
                    return files[n]
            return None

        d_part = pick("mic_driver", "mic_raw", "driver", "mic1")
        p_part = pick("mic_passenger", "mic2_raw", "passenger", "mic2")
        if d_part is None or p_part is None:
            self._send_json(400, {
                "ok": False,
                "error": "缺少两路音频。字段名：mic_driver 与 mic_passenger（或 mic_raw / mic2_raw）。",
                "got_fields": list(files.keys()),
            })
            return
        try:
            x, sr1, k1 = cancel.load_wav_or_pcm(d_part["data"], d_part["filename"])
            y, sr2, k2 = cancel.load_wav_or_pcm(p_part["data"], p_part["filename"])
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"读音频失败: {exc}"})
            return
        warnings = []
        if k1 == "pcm":
            warnings.append("主驾按无头 16 kHz 16-bit LE 单声道 PCM 解读。")
        if k2 == "pcm":
            warnings.append("副驾按无头 16 kHz 16-bit LE 单声道 PCM 解读。")
        if sr1 != cancel.FS or sr2 != cancel.FS:
            self._send_json(400, {
                "ok": False,
                "error": f"采样率必须是 {cancel.FS} Hz（收到主驾 {sr1}、副驾 {sr2}）。本服务不重采样。",
                "assumption": "与评测一致：16 kHz、16-bit LE、单声道。",
            })
            return
        n = min(len(x), len(y))
        if len(x) != len(y):
            warnings.append(f"两路长度不同（{len(x)} vs {len(y)}），已截到较短的 {n} 点。")
            x, y = x[:n], y[:n]
        if n < cancel.N_FFT:
            self._send_json(400, {"ok": False, "error": f"音频太短（{n} 点），至少需要 {cancel.N_FFT} 点。"})
            return
        try:
            pack = get_engine()
            beta = DEFAULT_DYNAMIC_BETA
            use_gsc_only = False
            qs = parse_qs(urlparse(self.path).query)
            raw_b = None
            if "beta" in files and files["beta"]["data"]:
                raw_b = files["beta"]["data"].decode("utf-8", errors="replace").strip()
            elif qs.get("beta"):
                raw_b = qs.get("beta", [""])[0]
            raw_s = None
            if "strength" in files and files["strength"]["data"]:
                raw_s = files["strength"]["data"].decode("utf-8", errors="replace").strip()
            elif qs.get("strength"):
                raw_s = qs.get("strength", [""])[0]
            # Prefer beta. Legacy strength: 0 → GSC only; otherwise treat as
            # fixed-λ request by setting beta=0 and mixing afterward.
            fixed_strength = None
            if raw_b not in (None, ""):
                try:
                    beta = float(raw_b)
                except ValueError:
                    self._send_json(400, {"ok": False, "error": f"beta 无法解析: {raw_b}"})
                    return
                if not (0.0 <= beta <= 1.0):
                    self._send_json(400, {"ok": False, "error": "beta 须在 0 到 1 之间（0=固定 λ=0.80）。"})
                    return
            elif raw_s not in (None, ""):
                try:
                    fixed_strength = float(raw_s)
                except ValueError:
                    self._send_json(400, {"ok": False, "error": f"strength 无法解析: {raw_s}"})
                    return
                if not (0.0 <= fixed_strength <= 1.0):
                    self._send_json(400, {"ok": False, "error": "strength 须在 0 到 1 之间（0=只做 GSC）。"})
                    return
                if fixed_strength <= 0.0:
                    use_gsc_only = True
                else:
                    beta = 0.0  # base is λ=0.80; if strength≠0.80, remixed below
            yd, yp, st = cancel.cancel_pair_with_dynamic(
                x, y, pack["rtf_d_pool"], pack["rtf_p_pool"], pack["freqs"],
                beta=beta, gsc_only=use_gsc_only,
            )
            if (not use_gsc_only) and fixed_strength is not None and abs(fixed_strength - 0.80) > 1e-9:
                yd = cancel.apply_postfilter_strength(st["yd_gsc"], st["yd_full"], fixed_strength)
                yp = cancel.apply_postfilter_strength(st["yp_gsc"], st["yp_full"], fixed_strength)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": f"对消失败: {exc}"})
            return
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("driver_out.wav", cancel.write_wav_bytes(yd))
            zf.writestr("passenger_out.wav", cancel.write_wav_bytes(yp))
            if "yd_gsc" in st:
                zf.writestr("driver_out_gsc.wav", cancel.write_wav_bytes(st["yd_gsc"]))
                zf.writestr("passenger_out_gsc.wav", cancel.write_wav_bytes(st["yp_gsc"]))
            if "yd_base" in st:
                zf.writestr("driver_out_base.wav", cancel.write_wav_bytes(st["yd_base"]))
                zf.writestr("passenger_out_base.wav", cancel.write_wav_bytes(st["yp_base"]))
                zf.writestr("driver_out_exc.wav", cancel.write_wav_bytes(st["yd_exc"]))
                zf.writestr("passenger_out_exc.wav", cancel.write_wav_bytes(st["yp_exc"]))
            info = {
                "method": "gsc_online_gated_dynamic_postfilter",
                "method_label": "GSC 从零 + 空间门控 + 动态残差后置抑制",
                "fs": cancel.FS,
                "n_fft": cancel.N_FFT,
                "hop": cancel.HOP,
                "algo_delay_ms": cancel.ALGO_DELAY_MS,
                "mu": 0.25,
                "gate": True,
                "dynamic_beta": None if use_gsc_only else (None if fixed_strength not in (None,) and abs((fixed_strength or 0)-0.80)>1e-9 else beta),
                "postfilter_strength": 0.0 if use_gsc_only else (fixed_strength if fixed_strength is not None else None),
                "default_dynamic_beta": DEFAULT_DYNAMIC_BETA,
                "default_postfilter_strength": DEFAULT_POSTFILTER_STRENGTH,
                "mapping": "λ(t)=clip(0.80+β·δ(t),0,1)；β=0 即固定 0.80",
                "rtf": "pooled from single-talk captures (same as evaluation 'Both' path)",
                "n_samples": int(n),
                "duration_s": n / cancel.FS,
                "warnings": warnings,
                "frame_labels": {str(k): int(v) for k, v in __import__("collections").Counter(map(str, st["labels"])).items()},
            }
            zf.writestr("info.json", json.dumps(info, ensure_ascii=False, indent=2))
        data = buf.getvalue()
        self._send(
            200,
            data,
            "application/zip",
            {"Content-Disposition": "attachment; filename=cancel_out.zip"},
        )

    def _serve_file(self, path: Path):
        if not path.is_file():
            self._send_json(404, {"ok": False, "error": "file not found"})
            return
        data = path.read_bytes()
        ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, data, ctype)


def main():
    parser = argparse.ArgumentParser(description="2-in-2-out seat-bound cancel demo service")
    parser.add_argument("--host", default=HOST_DEFAULT)
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--force-precompute", action="store_true")
    args = parser.parse_args()
    if args.precompute_only or args.force_precompute or not SESSIONS_JSON.exists():
        ensure_cache(force=args.force_precompute or args.precompute_only)
        if args.precompute_only:
            return
    else:
        # still validate cache
        ensure_cache(force=False)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"listening on http://{args.host}:{args.port}/  (UI)  "
        f"POST /api/cancel  GET /api/sessions  GET /api/health",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()
