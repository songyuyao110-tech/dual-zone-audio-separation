#!/usr/bin/env python3
"""Causal 2-in-2-out seat-bound crosstalk cancellation.

Extracted from run_cancel.py
(recommended method: GSC from zero + spatial gating). Algorithm is
copied, not reimplemented.

Assumed format (no header in raw PCM files): 16 kHz, 16-bit LE, mono.
Channels: mic_raw = driver mic, mic2_raw = passenger mic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import stft as sp_stft, istft as sp_istft
from scipy.signal.windows import hann as hann_win

DEFAULT_ROOT = Path(__file__).resolve().parent / "pcm"

FS = 16000  # assumed
EPS = 1e-12
N_FFT = 512  # 32 ms
HOP = 128    # 8 ms
WIN = hann_win(N_FFT, sym=False).astype(np.float64)
ALGO_DELAY_MS = N_FFT / FS * 1000.0  # 32 ms, full analysis window


def set_stft_params(n_fft: int, hop: int | None = None) -> dict:
    """Switch analysis window / hop (Hann, hop defaults to N/4).

    Algorithm delay is reported as N/fs, same convention as the 32 ms
    baseline (full analysis window). After hop-OLA, a hop-block of
    samples becomes complete when the current frame arrives; delays
    inside that block range from (N-H)/fs to (N-1)/fs.
    """
    global N_FFT, HOP, WIN, ALGO_DELAY_MS
    n_fft = int(n_fft)
    if hop is None:
        hop = n_fft // 4
    hop = int(hop)
    if n_fft < 32 or hop < 1 or hop >= n_fft:
        raise ValueError(f"invalid STFT params: n_fft={n_fft}, hop={hop}")
    N_FFT = n_fft
    HOP = hop
    WIN = hann_win(N_FFT, sym=False).astype(np.float64)
    ALGO_DELAY_MS = N_FFT / FS * 1000.0
    return {
        "n_fft": N_FFT,
        "hop": HOP,
        "algo_delay_ms": ALGO_DELAY_MS,
        "hop_ms": HOP / FS * 1000.0,
        "ola_delay_ms_min": (N_FFT - HOP) / FS * 1000.0,
        "ola_delay_ms_max": (N_FFT - 1) / FS * 1000.0,
    }


def stft_params() -> dict:
    return {
        "n_fft": N_FFT,
        "hop": HOP,
        "algo_delay_ms": ALGO_DELAY_MS,
        "hop_ms": HOP / FS * 1000.0,
        "ola_delay_ms_min": (N_FFT - HOP) / FS * 1000.0,
        "ola_delay_ms_max": (N_FFT - 1) / FS * 1000.0,
    }

CONDS = {
    "Only driver": "只主驾说",
    "Only Passenger": "只副驾说",
    "Both driver and passenger": "两人",
}

def load_pcm(path: Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(x, -1.0, 1.0)
    wavfile.write(str(path), FS, (y * 32767.0).astype(np.int16))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + EPS))


def db(v: float) -> float:
    return 20.0 * math.log10(abs(v) + EPS)


def db10(v: float) -> float:
    return 10.0 * math.log10(abs(v) + EPS)


def frame_rms(x: np.ndarray, win_s=0.02, hop_s=0.01):
    w, h = int(FS * win_s), int(FS * hop_s)
    n = 1 + max(0, (len(x) - w) // h)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * h : i * h + w]
        out[i] = np.sqrt(np.mean(seg * seg) + EPS)
    t = (np.arange(n) * h + w / 2) / FS
    return t, out, w, h


def vad_from_near(near: np.ndarray, far: np.ndarray | None = None):
    t, frm, w, h = frame_rms(near)
    if far is not None:
        _, frm2, _, _ = frame_rms(far)
        combo = np.maximum(frm, frm2)
    else:
        combo = frm
    ref = np.percentile(combo, 95)
    thr = max(ref * 0.15, 10 ** (-45 / 20))
    m = combo > thr
    out = m.copy()
    hold = 0
    for i in range(len(m)):
        if m[i]:
            hold = 2
            out[i] = True
        elif hold > 0:
            out[i] = True
            hold -= 1
    hold = 0
    for i in range(len(m) - 1, -1, -1):
        if m[i]:
            hold = 2
            out[i] = True
        elif hold > 0:
            out[i] = True
            hold -= 1
    mask = np.zeros(len(near), dtype=bool)
    for i, flag in enumerate(out):
        if flag:
            a = i * h
            b = min(len(near), a + w)
            mask[a:b] = True
    return mask, t, frm, out, thr


def gcc_phat(x, y, max_ms=6.0):
    n = int(2 ** math.ceil(math.log2(len(x) + len(y))))
    X = rfft(x, n)
    Y = rfft(y, n)
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-12
    cc = irfft(R, n)
    cc = np.concatenate([cc[-(n // 2) :], cc[: n // 2 + 1]])
    lags = np.arange(-(n // 2), n // 2 + 1)
    max_lag = int(FS * max_ms / 1000.0)
    mid = n // 2
    sl = slice(mid - max_lag, mid + max_lag + 1)
    cc_w, lags_w = cc[sl], lags[sl]
    k = int(np.argmax(cc_w))
    if 0 < k < len(cc_w) - 1:
        a, b, c = cc_w[k - 1], cc_w[k], cc_w[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-18 else 0.0
        delta = float(np.clip(delta, -1, 1))
    else:
        delta = 0.0
    lag = lags_w[k] + delta
    peak = float(cc_w[k])
    sharp = peak / (np.median(np.abs(cc_w)) + 1e-12)
    return lag / FS, lag, peak, sharp, lags_w / FS, cc_w


def peak_in_window(lags_s, cc, t0, t1, earliest=False, rel=0.45):
    """If earliest, take the first peak that reaches `rel` of the window max
    (direct path), not a later stronger reflection inside the same window."""
    m = (lags_s >= t0) & (lags_s <= t1)
    if not np.any(m):
        return None
    sl_s = lags_s[m]
    sl_c = cc[m]
    idxs = np.where(m)[0]
    if earliest:
        mx = float(np.max(sl_c))
        thr = max(mx * rel, 0.0)
        # simple peak: greater than neighbors and above thr
        chosen = None
        for j in range(len(sl_c)):
            left = sl_c[j - 1] if j > 0 else -np.inf
            right = sl_c[j + 1] if j < len(sl_c) - 1 else -np.inf
            if sl_c[j] >= left and sl_c[j] >= right and sl_c[j] >= thr:
                chosen = j
                break
        k = int(chosen) if chosen is not None else int(np.argmax(sl_c))
    else:
        k = int(np.argmax(sl_c))
    i = idxs[k]
    if 0 < i < len(cc) - 1:
        a, b, c = cc[i - 1], cc[i], cc[i + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-18 else 0.0
        delta = float(np.clip(delta, -1, 1))
        ds = lags_s[1] - lags_s[0] if len(lags_s) > 1 else 1.0 / FS
        tau = sl_s[k] + delta * ds
    else:
        tau = sl_s[k]
    return float(tau), float(sl_c[k])


def stft_pair(x, y):
    f, t, X = sp_stft(
        x, fs=FS, window=WIN, nperseg=N_FFT, noverlap=N_FFT - HOP,
        nfft=N_FFT, boundary=None, padded=False, return_onesided=True,
    )
    _, _, Y = sp_stft(
        y, fs=FS, window=WIN, nperseg=N_FFT, noverlap=N_FFT - HOP,
        nfft=N_FFT, boundary=None, padded=False, return_onesided=True,
    )
    return f, t, X, Y


def istft_one(X, n_out):
    _, x = sp_istft(
        X, fs=FS, window=WIN, nperseg=N_FFT, noverlap=N_FFT - HOP,
        nfft=N_FFT, input_onesided=True, boundary=False,
    )
    if len(x) < n_out:
        x = np.pad(x, (0, n_out - len(x)))
    return x[:n_out]


def speech_stft_stats(x, y, vad_mask):
    f, t, X, Y = stft_pair(x, y)
    n = X.shape[1]
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        a = i * HOP
        b = min(len(vad_mask), a + N_FFT)
        if b > a and vad_mask[a:b].mean() > 0.4:
            keep[i] = True
    if not np.any(keep):
        keep[:] = True
    Xk, Yk = X[:, keep], Y[:, keep]
    pxx = np.mean(np.abs(Xk) ** 2, axis=1)
    pyy = np.mean(np.abs(Yk) ** 2, axis=1)
    pxy = np.mean(Xk * np.conj(Yk), axis=1)
    return f, pxx, pyy, pxy, int(np.sum(keep))


def list_sessions(root: Path | None = None):
    root = Path(root) if root is not None else DEFAULT_ROOT
    rows = []
    for cond, label in CONDS.items():
        d = root / cond
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
            mic1 = sub / "mic_raw.pcm"
            mic2 = sub / "mic2_raw.pcm"
            if mic1.exists() and mic2.exists():
                rows.append({
                    "cond": cond, "label": label, "sid": sub.name, "dir": sub,
                    "mic1": load_pcm(mic1), "mic2": load_pcm(mic2),
                })
    return rows


def calibrate(sessions):
    """Estimate direct-path tau (mic1 relative to mic2) from single-talk.

    Driver: search [-2.0, -0.5] ms (ignore 3-5 ms reflections).
    Passenger: search [+0.5, +2.0] ms.
    """
    recs = []
    for s in sessions:
        if s["cond"] not in ("Only driver", "Only Passenger"):
            continue
        x, y = s["mic1"], s["mic2"]
        near = x if s["cond"] == "Only driver" else y
        far = y if s["cond"] == "Only driver" else x
        mask, t, frm, fvad, thr = vad_from_near(near)
        if mask.sum() < FS * 0.2:
            mask, _, _, _, _ = vad_from_near(near, far)
        xs, ys = x[mask], y[mask]
        if len(xs) < 256:
            xs, ys = x, y
        tau, lag, peak, sharp, lags_s, cc = gcc_phat(xs, ys, max_ms=6.0)
        k = int(np.argmax(cc))
        tau_global = float(lags_s[k])
        if s["cond"] == "Only driver":
            dp = peak_in_window(lags_s, cc, -0.0020, -0.0005, earliest=True)
        else:
            dp = peak_in_window(lags_s, cc, 0.0005, 0.0020, earliest=True)
        tau_dp, pk_dp = dp if dp else (tau_global, peak)
        e1 = rms(xs)
        e2 = rms(ys)
        recs.append({
            "cond": s["cond"], "sid": s["sid"],
            "tau_global_ms": tau_global * 1000,
            "tau_direct_ms": tau_dp * 1000,
            "gcc_sharp": sharp,
            "ratio_db": db(e1) - db(e2),
            "speech_s": mask.mean() * len(x) / FS,
            "lags_s": lags_s, "cc": cc,
        })
        s["vad_mask"] = mask
        s["tau_direct_ms"] = tau_dp * 1000
        s["ratio_db"] = db(e1) - db(e2)

    drv = [r for r in recs if r["cond"] == "Only driver"]
    pax = [r for r in recs if r["cond"] == "Only Passenger"]
    tau_d = float(np.median([r["tau_direct_ms"] for r in drv])) / 1000.0
    tau_p = float(np.median([r["tau_direct_ms"] for r in pax])) / 1000.0
    # Prefer sessions whose global peak already sits in the direct-path window
    # (these are not pulled by a 3-5 ms reflection).
    clean_d = [r["tau_direct_ms"] for r in drv if abs(r["tau_global_ms"] - r["tau_direct_ms"]) < 0.4]
    clean_p = [r["tau_direct_ms"] for r in pax if abs(r["tau_global_ms"] - r["tau_direct_ms"]) < 0.4]
    if len(clean_d) >= 2:
        tau_d = float(np.median(clean_d)) / 1000.0
    if len(clean_p) >= 2:
        tau_p = float(np.median(clean_p)) / 1000.0
    g_p = float(np.median([10 ** (r["ratio_db"] / 20) for r in pax]))
    g_d = float(np.median([10 ** (-r["ratio_db"] / 20) for r in drv]))
    return {
        "tau_d": tau_d, "tau_p": tau_p, "g_d": g_d, "g_p": g_p, "recs": recs,
        "drv_direct_ms": [r["tau_direct_ms"] for r in drv],
        "pax_direct_ms": [r["tau_direct_ms"] for r in pax],
        "drv_global_ms": [r["tau_global_ms"] for r in drv],
        "pax_global_ms": [r["tau_global_ms"] for r in pax],
        "drv_ratio_db": [r["ratio_db"] for r in drv],
        "pax_ratio_db": [r["ratio_db"] for r in pax],
    }


def estimate_rtf_from(sessions, cond):
    """Pooled RTF from sessions of one single-talk condition.

    Driver: RTF_d = E[X2 conj(X1)] / E[|X1|^2]
    Passenger: RTF_p = E[X1 conj(X2)] / E[|X2|^2]
    """
    acc_num = None
    acc_den = None
    n_frames = 0
    freqs = None
    for s in sessions:
        x, y = s["mic1"], s["mic2"]
        if "vad_mask" in s:
            mask = s["vad_mask"]
        else:
            near = x if cond == "Only driver" else y
            mask, *_ = vad_from_near(near)
        f, pxx, pyy, pxy, nf = speech_stft_stats(x, y, mask)
        freqs = f
        n_frames += nf
        if cond == "Only driver":
            num, den = np.conj(pxy), pxx
        else:
            num, den = pxy, pyy
        if acc_num is None:
            acc_num = num * nf
            acc_den = den * nf
        else:
            acc_num += num * nf
            acc_den += den * nf
    rtf = acc_num / (acc_den + 1e-18)
    return freqs, rtf, n_frames


def blend_rtf_with_delay(freqs, rtf, tau):
    """Regularize measured RTF toward anechoic delay at low freq / wild bins."""
    model = np.exp(1j * 2 * np.pi * freqs * tau)
    mag = np.abs(rtf)
    mag_clip = np.clip(mag, 0.05, 4.0)
    rtf_u = rtf / (mag + 1e-18) * mag_clip
    f = freqs
    w_model = 1.0 / (1.0 + (f / 250.0) ** 2)
    clip_pen = (np.abs(np.log((mag + 1e-8) / (mag_clip + 1e-8))) > 0.2).astype(np.float64)
    w = np.clip(w_model + 0.3 * clip_pen, 0, 1)
    return (1 - w) * rtf_u + w * model


def _gcc_phase_matrix(freqs, max_ms=5.0):
    max_lag = int(FS * max_ms / 1000.0)
    lags = np.arange(-max_lag, max_lag + 1)
    phase = np.exp(1j * 2 * np.pi * freqs[:, None] * lags[None, :] / FS)
    return lags, phase


def frame_tdoa_gcc_batch(X1, X2, freqs, max_ms=5.0):
    """Vectorized per-frame GCC-PHAT. X is n_freq x n_frm."""
    R = X1 * np.conj(X2)
    R /= np.abs(R) + 1e-12
    band = (freqs >= 300) & (freqs <= 4000)
    R = R.copy()
    R[~band, :] = 0
    lags, phase = _gcc_phase_matrix(freqs, max_ms)
    gcc = np.real(R.T @ phase)  # n_frm x n_lags
    k = np.argmax(gcc, axis=1)
    # parabolic interpolation
    nL = gcc.shape[1]
    k_cl = np.clip(k, 1, nL - 2)
    a = gcc[np.arange(len(k)), k_cl - 1]
    b = gcc[np.arange(len(k)), k_cl]
    c = gcc[np.arange(len(k)), k_cl + 1]
    denom = a - 2 * b + c
    delta = np.where(np.abs(denom) > 1e-18, 0.5 * (a - c) / denom, 0.0)
    delta = np.clip(delta, -1, 1)
    delta = np.where((k > 0) & (k < nL - 1), delta, 0.0)
    tdoa = (lags[k] + delta) / FS
    peak = gcc[np.arange(len(k)), k]
    return tdoa, peak


def frame_tdoa_gcc(X1, X2, freqs, max_ms=5.0):
    tdoa, peak = frame_tdoa_gcc_batch(X1[:, None], X2[:, None], freqs, max_ms)
    return float(tdoa[0]), float(peak[0])


def classify_frames(X1, X2, freqs, e_sil=None):
    n = X1.shape[1]
    band = (freqs >= 300) & (freqs <= 3500)
    e1 = np.sum(np.abs(X1[band, :]) ** 2, axis=0)
    e2 = np.sum(np.abs(X2[band, :]) ** 2, axis=0)
    ild = 10 * np.log10((e1 + 1e-18) / (e2 + 1e-18))
    emax = np.maximum(e1, e2)
    emin = np.minimum(e1, e2)
    if e_sil is None:
        e_sil = np.percentile(emax, 20) * 8.0
        e_sil = max(e_sil, 1e-6)
    tdoa, _ = frame_tdoa_gcc_batch(X1, X2, freqs)
    tdoa = tdoa.astype(float)
    tdoa[emax < e_sil * 0.3] = np.nan
    labels = np.array(["silence"] * n, dtype=object)
    for i in range(n):
        if emax[i] < e_sil:
            labels[i] = "silence"
            continue
        td = tdoa[i]
        td_ok = td == td
        if ild[i] > 2.5 and (not td_ok or td < -0.0003):
            labels[i] = "driver"
        elif ild[i] < -2.5 and (not td_ok or td > 0.0003):
            labels[i] = "passenger"
        elif ild[i] > 4.0:
            labels[i] = "driver"
        elif ild[i] < -4.0:
            labels[i] = "passenger"
        elif td_ok and td < -0.0007 and ild[i] > 0.5:
            labels[i] = "driver"
        elif td_ok and td > 0.0007 and ild[i] < -0.5:
            labels[i] = "passenger"
        elif emin[i] > e_sil * 0.5 and abs(ild[i]) <= 3.0:
            labels[i] = "overlap"
        else:
            labels[i] = "uncertain"
    return labels, ild, tdoa, e1, e2, e_sil


def run_gsc(mic1, mic2, rtf_d, rtf_p, freqs, mu=0.2, leak=0.0005,
            adapt=True, H_d=None, H_p=None, freeze_overlap=True, mu_slow=0.02,
            gate=False):
    """Causal STFT-domain GSC, near-mic FBF, 1 complex tap/bin.

    Driver: y_d = X1 - H_d * u_d, u_d = X2 - RTF_d * X1 (driver blocked)
    Passenger: y_p = X2 - H_p * u_p, u_p = X1 - RTF_p * X2 (passenger blocked)
    Adapt H_d only when passenger is dominant; H_p only when driver is dominant.
    If gate=True, spatially protect the target seat: when a frame is classified
    as driver-only, driver output is the near mic (no subtraction); symmetric
    for passenger. Overlap/uncertain still use the canceller.
    """
    f, t, X1, X2 = stft_pair(mic1, mic2)
    n_freq, n_frm = X1.shape
    labels, ild, tdoa, e1, e2, e_sil = classify_frames(X1, X2, f)

    H_d = np.zeros(n_freq, dtype=np.complex128) if H_d is None else H_d.copy()
    H_p = np.zeros(n_freq, dtype=np.complex128) if H_p is None else H_p.copy()
    Pd = np.ones(n_freq) * 1e-6
    Pp = np.ones(n_freq) * 1e-6
    Yd = np.zeros_like(X1)
    Yp = np.zeros_like(X2)
    alpha_p = 0.7
    Ud = X2 - rtf_d[:, None] * X1
    Up = X1 - rtf_p[:, None] * X2

    for i in range(n_frm):
        lab = labels[i]
        ud = Ud[:, i]
        up = Up[:, i]
        yd_c = X1[:, i] - H_d * ud
        yp_c = X2[:, i] - H_p * up
        if gate:
            if lab == "driver":
                Yd[:, i], Yp[:, i] = X1[:, i], yp_c
            elif lab == "passenger":
                Yd[:, i], Yp[:, i] = yd_c, X2[:, i]
            else:
                # overlap, uncertain, silence: keep cancellation on
                Yd[:, i], Yp[:, i] = yd_c, yp_c
        else:
            Yd[:, i], Yp[:, i] = yd_c, yp_c
        if not adapt:
            continue
        Pd = alpha_p * Pd + (1 - alpha_p) * (np.abs(ud) ** 2)
        Pp = alpha_p * Pp + (1 - alpha_p) * (np.abs(up) ** 2)
        if lab == "passenger":
            mu_d, mu_p = mu, 0.0
        elif lab == "driver":
            mu_d, mu_p = 0.0, mu
        elif lab == "overlap":
            mu_d = mu_p = 0.0 if freeze_overlap else mu_slow
        elif lab == "uncertain":
            mu_d = mu_p = mu_slow * 0.25
        else:
            mu_d = mu_p = 0.0
        if mu_d > 0:
            e = Yd[:, i]
            H_d = (1 - leak) * H_d + mu_d * np.conj(ud) * e / (Pd + 1e-8)
            mag = np.abs(H_d)
            too = mag > 4.0
            H_d[too] *= 4.0 / (mag[too] + 1e-18)
        if mu_p > 0:
            e = Yp[:, i]
            H_p = (1 - leak) * H_p + mu_p * np.conj(up) * e / (Pp + 1e-8)
            mag = np.abs(H_p)
            too = mag > 4.0
            H_p[too] *= 4.0 / (mag[too] + 1e-18)

    yd = istft_one(Yd, len(mic1))
    yp = istft_one(Yp, len(mic2))
    state = {"H_d": H_d, "H_p": H_p, "labels": labels, "ild": ild,
             "tdoa": tdoa, "t_frames": t, "e_sil": e_sil}
    return yd, yp, state


# ---------------------------------------------------------------------------
# Service-facing helpers (same GSC call as evaluation gsc_online_gated)
# ---------------------------------------------------------------------------

def prepare_rtfs(sessions, calib=None):
    """Build pooled and leave-one-session-out RTFs exactly as in run_cancel.main."""
    if calib is None:
        calib = calibrate(sessions)
    freqs = rfftfreq(N_FFT, 1 / FS)
    drv_all = [s for s in sessions if s["cond"] == "Only driver"]
    pax_all = [s for s in sessions if s["cond"] == "Only Passenger"]
    _, rtf_d_pool, nd = estimate_rtf_from(drv_all, "Only driver")
    _, rtf_p_pool, np_ = estimate_rtf_from(pax_all, "Only Passenger")
    rtf_d_pool = blend_rtf_with_delay(freqs, rtf_d_pool, calib["tau_d"])
    rtf_p_pool = blend_rtf_with_delay(freqs, rtf_p_pool, -calib["tau_p"])
    loso = {}
    for s in drv_all + pax_all:
        others_d = [u for u in drv_all if u["sid"] != s["sid"]]
        others_p = [u for u in pax_all if u["sid"] != s["sid"]]
        _, rd, _ = estimate_rtf_from(others_d if others_d else drv_all, "Only driver")
        _, rp, _ = estimate_rtf_from(others_p if others_p else pax_all, "Only Passenger")
        rd = blend_rtf_with_delay(freqs, rd, calib["tau_d"])
        rp = blend_rtf_with_delay(freqs, rp, -calib["tau_p"])
        loso[s["sid"]] = {"rtf_d": rd, "rtf_p": rp}
    return {
        "freqs": freqs,
        "rtf_d_pool": rtf_d_pool,
        "rtf_p_pool": rtf_p_pool,
        "loso": loso,
        "n_drv_frames": nd,
        "n_pax_frames": np_,
        "calib": calib,
    }


def rtf_for_session(pack, sid):
    loso = pack["loso"]
    if sid in loso:
        return loso[sid]["rtf_d"], loso[sid]["rtf_p"]
    return pack["rtf_d_pool"], pack["rtf_p_pool"]


def cancel_pair(mic1, mic2, rtf_d, rtf_p, freqs, mu=0.25, gate=True):
    """Recommended online path: GSC from zero + spatial gating, mu=0.25."""
    return run_gsc(
        mic1, mic2, rtf_d, rtf_p, freqs, mu=mu, adapt=True, gate=gate,
    )


def load_wav_or_pcm(data: bytes, name: str = "") -> tuple[np.ndarray, int, str]:
    """Load mono audio. WAV uses its header; raw PCM is 16-bit LE.

    Returns (float64 in [-1, 1], sample_rate, kind) where kind is 'wav' or 'pcm'.
    Stereo WAV: first channel only.
    """
    lower = (name or "").lower()
    is_wav = lower.endswith(".wav") or data[:4] == b"RIFF"
    if is_wav:
        import io
        sr, x = wavfile.read(io.BytesIO(data))
        if x.ndim > 1:
            x = x[:, 0]
        if x.dtype == np.int16:
            x = x.astype(np.float64) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float64) / 2147483648.0
        elif np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)
        else:
            raise ValueError(f"unsupported WAV dtype: {x.dtype}")
        return x, int(sr), "wav"
    x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    return x, FS, "pcm"


def write_wav_bytes(x: np.ndarray, sr: int = FS) -> bytes:
    import io
    buf = io.BytesIO()
    y = np.clip(x, -1.0, 1.0)
    wavfile.write(buf, sr, (y * 32767.0).astype(np.int16))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Causal residual Wiener post-filter (after GSC). No voiceprint.
#
# Full-strength masks are computed once. A later intensity λ in [0, 1]
# is applied as y(λ) = (1-λ) y_gsc + λ y_full, which equals ISTFT of
# Y_gsc * ((1-λ) + λ G_full) because ISTFT is linear. λ = 0 is GSC only.
# ---------------------------------------------------------------------------

# Max suppression of the residual Wiener at λ = 1, in dB (gain floor).
PF_MAX_ATTEN_DB = 18.0
PF_G_MIN = 10.0 ** (-PF_MAX_ATTEN_DB / 20.0)
# On frames classified as this seat's single-talk, do not punch below this.
PF_TARGET_FLOOR_DB = 2.5
# Overlap has no clean reference — keep a higher floor.
PF_OVERLAP_FLOOR_DB = 6.0
PF_UNCERTAIN_FLOOR_DB = 10.0
PF_ALPHA_PN = 0.7
PF_ALPHA_DD = 0.98
PF_GAIN_SMOOTH = 0.55  # G = smooth * G_prev + (1-smooth) * G_inst
PF_FREQ_W = 0.15
# Limit how fast the interference PSD may rise (mis-tagged target protection).
PF_PN_RISE = 3.0


def _smooth_freq(g: np.ndarray, w: float = PF_FREQ_W) -> np.ndarray:
    out = g.copy()
    if len(g) >= 3:
        out[1:-1] = (1.0 - 2.0 * w) * g[1:-1] + w * g[:-2] + w * g[2:]
    return out


def _wiener_gain(py: np.ndarray, pn: np.ndarray, g_prev: np.ndarray) -> np.ndarray:
    """Decision-directed a priori SNR → Wiener gain, causal in time."""
    gamma = py / (pn + 1e-18)
    eta = PF_ALPHA_DD * (g_prev * g_prev) * gamma + (1.0 - PF_ALPHA_DD) * np.maximum(
        gamma - 1.0, 0.0
    )
    return np.clip(eta / (eta + 1.0), 0.0, 1.0)


def _update_pn(pn: np.ndarray, inst: np.ndarray, alpha: float = PF_ALPHA_PN) -> np.ndarray:
    capped = np.minimum(inst, PF_PN_RISE * pn + 1e-8)
    return alpha * pn + (1.0 - alpha) * capped


def postfilter_full(mic1, mic2, yd, yp, labels=None):
    """Causal full-strength residual Wiener on GSC outputs.

    Interference PSD for a seat's output is learned when the *other* seat is
    spatially dominant (that output is then mostly residual crosstalk), and
    during silence. Target-labeled frames get a high gain floor. Overlap
    frames are conservative. Returns (yd_full, yp_full, state) with masks
    Gd, Gp in [G_min, 1].
    """
    f, t, X1, X2 = stft_pair(mic1, mic2)
    _, _, Yd, Yp = stft_pair(yd, yp)
    n_freq, n_frm = Yd.shape
    cls_labels, ild, tdoa, e1, e2, e_sil = classify_frames(X1, X2, f)
    if labels is None or len(np.asarray(labels)) != n_frm:
        labels = cls_labels
    else:
        labels = np.asarray(labels, dtype=object)

    g_min = PF_G_MIN
    g_tgt = 10.0 ** (-PF_TARGET_FLOOR_DB / 20.0)
    g_ovl = 10.0 ** (-PF_OVERLAP_FLOOR_DB / 20.0)
    g_unc = 10.0 ** (-PF_UNCERTAIN_FLOOR_DB / 20.0)

    # Tiny causal init — first frames stay close to passthrough until Pn is learned.
    Pn_d = np.full(n_freq, 1e-8, dtype=np.float64)
    Pn_p = np.full(n_freq, 1e-8, dtype=np.float64)
    G_d_prev = np.ones(n_freq, dtype=np.float64)
    G_p_prev = np.ones(n_freq, dtype=np.float64)
    Gd = np.ones((n_freq, n_frm), dtype=np.float64)
    Gp = np.ones((n_freq, n_frm), dtype=np.float64)
    mag_d = np.abs(Yd) ** 2
    mag_p = np.abs(Yp) ** 2

    for i in range(n_frm):
        lab = labels[i]
        pd = mag_d[:, i]
        pp = mag_p[:, i]
        if lab == "passenger":
            Pn_d = _update_pn(Pn_d, pd)
        elif lab == "driver":
            Pn_p = _update_pn(Pn_p, pp)
        elif lab == "silence":
            Pn_d = _update_pn(Pn_d, pd)
            Pn_p = _update_pn(Pn_p, pp)
        else:
            # overlap / uncertain: slow, conservative residual candidate
            cand_d = np.minimum(pd, 0.5 * pp)
            cand_p = np.minimum(pp, 0.5 * pd)
            Pn_d = 0.92 * Pn_d + 0.08 * np.minimum(cand_d, PF_PN_RISE * Pn_d + 1e-8)
            Pn_p = 0.92 * Pn_p + 0.08 * np.minimum(cand_p, PF_PN_RISE * Pn_p + 1e-8)

        gd = _wiener_gain(pd, Pn_d, G_d_prev)
        gp = _wiener_gain(pp, Pn_p, G_p_prev)

        if lab == "driver":
            floor_d, floor_p = g_tgt, g_min
        elif lab == "passenger":
            floor_d, floor_p = g_min, g_tgt
        elif lab == "overlap":
            floor_d, floor_p = g_ovl, g_ovl
        elif lab == "uncertain":
            floor_d, floor_p = g_unc, g_unc
        else:
            floor_d, floor_p = g_min, g_min

        gd = np.maximum(_smooth_freq(gd), floor_d)
        gp = np.maximum(_smooth_freq(gp), floor_p)
        gd = PF_GAIN_SMOOTH * G_d_prev + (1.0 - PF_GAIN_SMOOTH) * gd
        gp = PF_GAIN_SMOOTH * G_p_prev + (1.0 - PF_GAIN_SMOOTH) * gp
        gd = np.clip(np.maximum(gd, floor_d), g_min, 1.0)
        gp = np.clip(np.maximum(gp, floor_p), g_min, 1.0)
        Gd[:, i] = gd
        Gp[:, i] = gp
        G_d_prev = gd
        G_p_prev = gp

    yd_f = istft_one(Yd * Gd, len(yd))
    yp_f = istft_one(Yp * Gp, len(yp))
    state = {
        "Gd": Gd,
        "Gp": Gp,
        "labels": labels,
        "ild": ild,
        "tdoa": tdoa,
        "t_frames": t,
        "e_sil": e_sil,
        "g_min": g_min,
        "max_atten_db": PF_MAX_ATTEN_DB,
    }
    return yd_f, yp_f, state


def apply_postfilter_strength(y_gsc: np.ndarray, y_full: np.ndarray, strength: float) -> np.ndarray:
    """Mix GSC-only and full post-filter. strength 0 = GSC only, 1 = full."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return np.asarray(y_gsc, dtype=np.float64)
    if s >= 1.0:
        return np.asarray(y_full, dtype=np.float64)
    return (1.0 - s) * np.asarray(y_gsc, dtype=np.float64) + s * np.asarray(y_full, dtype=np.float64)


def cancel_pair_with_postfilter(mic1, mic2, rtf_d, rtf_p, freqs, mu=0.25, gate=True, strength=1.0):
    """GSC (same as cancel_pair) plus residual post-filter at the given strength."""
    yd, yp, st = cancel_pair(mic1, mic2, rtf_d, rtf_p, freqs, mu=mu, gate=gate)
    yd_f, yp_f, pf = postfilter_full(mic1, mic2, yd, yp, labels=st["labels"])
    st["postfilter"] = pf
    st["yd_gsc"] = yd
    st["yp_gsc"] = yp
    st["yd_full"] = yd_f
    st["yp_full"] = yp_f
    s = float(np.clip(strength, 0.0, 1.0))
    return apply_postfilter_strength(yd, yd_f, s), apply_postfilter_strength(yp, yp_f, s), st



# ---------------------------------------------------------------------------
# Dynamic post-filter strength, driven by *spatial* opposite-seat activity.
#
# A single user factor β in [0, 1]:
#   β = 0 → λ(t) stays at LAM_BASE (0.80), i.e. the current fixed default
#   β > 0 → λ(t) = clip(LAM_BASE + β * δ(t), 0, 1)
#       δ > 0 when the opposite seat is spatially dominant (raise suppression
#             toward 1.0)
#       δ < 0 when both seats look active (cap / pull back toward LAM_OVL,
#             so overlap does not go to full strength and crush the target)
#       δ ≈ 0 when this seat is talking alone, or in silence
#
# Opposite-seat activity is NOT the opposite mic's RMS (both mics pick up
# both talkers). It is a causal combination of ILD, TDOA, and the GSC
# blocking-branch energy (driver-blocked ≈ passenger remnant, and vice versa).
# All of this lives on the same STFT frame as GSC / the residual Wiener;
# no extra look-ahead.
#
# Because λ(t, β) = LAM_BASE + β δ(t) stays inside [LAM_OVL, LAM_OPP] for
# β in [0, 1], the STFT-domain mix is exactly affine in β:
#   Y(β) = Y_base + β Y_exc
# with Y_base = Y_gsc * ((1-LAM_BASE) + LAM_BASE G)
# and  Y_exc  = Y_gsc * δ * (G - 1).
# The UI can therefore mix two precomputed wavs and never rerun GSC.
# ---------------------------------------------------------------------------

DYN_LAM_BASE = 0.80
DYN_LAM_OPP = 1.00
DYN_LAM_OVL = 0.60
DYN_LAM_OWN = 0.80
# Causal smoothers (higher = more memory of the past).
DYN_A_ATTACK = 0.45   # when activity rises
DYN_A_RELEASE = 0.75  # when activity falls
DYN_DELTA_SMOOTH = 0.55
DYN_FLOOR_ALPHA = 0.04
# Speech-band used for ILD / blocking-branch energy (same as classify_frames).
DYN_F_LO = 300.0
DYN_F_HI = 3500.0


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _causal_attack_release(x, att=DYN_A_ATTACK, rel=DYN_A_RELEASE):
    """Causal one-pole with faster attack than release. x is 1-D."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    prev = float(x[0]) if len(x) else 0.0
    for i, v in enumerate(x):
        a = att if v >= prev else rel
        prev = a * prev + (1.0 - a) * float(v)
        out[i] = prev
    return out


def spatial_opposite_activity(X1, X2, freqs, rtf_d, rtf_p, e_sil=None):
    """Causal per-frame opposite-seat activity and the β=1 delta tracks.

    Control features: ILD, TDOA, and blocking-branch (null) energy.
    Does not use the opposite microphone RMS as a loudness estimate.
    """
    n_freq, n_frm = X1.shape
    band = (freqs >= DYN_F_LO) & (freqs <= DYN_F_HI)
    e1 = np.sum(np.abs(X1[band, :]) ** 2, axis=0)
    e2 = np.sum(np.abs(X2[band, :]) ** 2, axis=0)
    ild = 10.0 * np.log10((e1 + 1e-18) / (e2 + 1e-18))
    emax = np.maximum(e1, e2)
    tdoa, _ = frame_tdoa_gcc_batch(X1, X2, freqs)
    tdoa = tdoa.astype(float)

    Ud = X2 - rtf_d[:, None] * X1
    Up = X1 - rtf_p[:, None] * X2
    e_ud = np.sum(np.abs(Ud[band, :]) ** 2, axis=0)
    e_up = np.sum(np.abs(Up[band, :]) ** 2, axis=0)

    # Causal noise floors (updated only on likely-quiet frames).
    floor_e = max(float(emax[0]) if n_frm else 1e-8, 1e-10)
    floor_ud = max(float(e_ud[0]) if n_frm else 1e-8, 1e-10)
    floor_up = max(float(e_up[0]) if n_frm else 1e-8, 1e-10)
    g_speech = np.zeros(n_frm, dtype=np.float64)
    level_ud = np.zeros(n_frm, dtype=np.float64)
    level_up = np.zeros(n_frm, dtype=np.float64)
    for i in range(n_frm):
        # Quiet if this frame is close to the running floor.
        quiet = emax[i] < 4.0 * floor_e
        if quiet:
            floor_e = (1.0 - DYN_FLOOR_ALPHA) * floor_e + DYN_FLOOR_ALPHA * emax[i]
            floor_ud = (1.0 - DYN_FLOOR_ALPHA) * floor_ud + DYN_FLOOR_ALPHA * e_ud[i]
            floor_up = (1.0 - DYN_FLOOR_ALPHA) * floor_up + DYN_FLOOR_ALPHA * e_up[i]
        else:
            # Very slow upward creep so a long session cannot freeze a too-low floor.
            floor_e = min(floor_e * 1.001, floor_e * 0.999 + 0.001 * emax[i])
        # ~12–15 dB above the floor → "speech present" / "branch active".
        g_speech[i] = _sigmoid(
            (10.0 * np.log10(emax[i] + 1e-18) - 10.0 * np.log10(floor_e + 1e-18) - 12.0) / 3.0
        )
        level_ud[i] = _sigmoid(
            (10.0 * np.log10(e_ud[i] + 1e-18) - 10.0 * np.log10(floor_ud + 1e-18) - 12.0) / 4.0
        )
        level_up[i] = _sigmoid(
            (10.0 * np.log10(e_up[i] + 1e-18) - 10.0 * np.log10(floor_up + 1e-18) - 12.0) / 4.0
        )

    # Soft seat membership. ILD is the primary cue; TDOA confirms when reliable.
    # tdoa < 0 → source closer to mic1 (driver); tdoa > 0 → passenger.
    td_ok = tdoa == tdoa
    td_p = np.where(td_ok, _sigmoid(tdoa / 0.0005), 0.5)
    td_d = np.where(td_ok, _sigmoid(-tdoa / 0.0005), 0.5)
    s_pax = _sigmoid(-ild / 2.0) * (0.45 + 0.55 * td_p)
    s_drv = _sigmoid(ild / 2.0) * (0.45 + 0.55 * td_d)

    # Level comes from the *blocking branch*, not the opposite raw mic.
    a_pax = g_speech * s_pax * (0.35 + 0.65 * level_ud)
    a_drv = g_speech * s_drv * (0.35 + 0.65 * level_up)
    a_pax = np.clip(_causal_attack_release(a_pax), 0.0, 1.0)
    a_drv = np.clip(_causal_attack_release(a_drv), 0.0, 1.0)

    # Exclusive opposite / own / both.
    # both uses a product so a single dominant seat does not trigger the cap.
    both = a_pax * a_drv
    opp_for_d = a_pax * (1.0 - a_drv)   # passenger alone → raise λ on driver out
    own_for_d = a_drv * (1.0 - a_pax)
    opp_for_p = a_drv * (1.0 - a_pax)
    own_for_p = a_pax * (1.0 - a_drv)

    d_opp = DYN_LAM_OPP - DYN_LAM_BASE
    d_ovl = DYN_LAM_OVL - DYN_LAM_BASE
    d_own = DYN_LAM_OWN - DYN_LAM_BASE
    delta_d = opp_for_d * d_opp + both * d_ovl + own_for_d * d_own
    delta_p = opp_for_p * d_opp + both * d_ovl + own_for_p * d_own
    delta_d = _causal_attack_release(delta_d, att=DYN_DELTA_SMOOTH, rel=DYN_DELTA_SMOOTH)
    delta_p = _causal_attack_release(delta_p, att=DYN_DELTA_SMOOTH, rel=DYN_DELTA_SMOOTH)
    # Keep λ(β) inside [LAM_OVL, LAM_OPP] so the affine mix stays exact.
    lo = DYN_LAM_OVL - DYN_LAM_BASE
    hi = DYN_LAM_OPP - DYN_LAM_BASE
    delta_d = np.clip(delta_d, lo, hi)
    delta_p = np.clip(delta_p, lo, hi)

    return {
        "a_drv": a_drv, "a_pax": a_pax,
        "s_drv": s_drv, "s_pax": s_pax,
        "g_speech": g_speech,
        "level_ud": level_ud, "level_up": level_up,
        "e_ud": e_ud, "e_up": e_up,
        "ild": ild, "tdoa": tdoa,
        "delta_d": delta_d, "delta_p": delta_p,
        "both": both,
        "lam_base": DYN_LAM_BASE,
        "lam_opp": DYN_LAM_OPP,
        "lam_ovl": DYN_LAM_OVL,
    }


def _stft_gain_mix(Y, G, lam_frames, n_out):
    """Y_out = Y * ((1-λ) + λ G) with per-frame λ, then ISTFT."""
    lam = np.clip(np.asarray(lam_frames, dtype=np.float64), 0.0, 1.0)
    gain = (1.0 - lam)[None, :] + lam[None, :] * G
    return istft_one(Y * gain, n_out)


def dynamic_postfilter_components(mic1, mic2, yd, yp, rtf_d, rtf_p, labels=None, pf_state=None):
    """Build the affine pair (y_base, y_exc) for each seat.

    y(β) = y_base + β * y_exc is the exact STFT-domain mix at
    λ(t) = LAM_BASE + β δ(t). y_base is identical to a fixed λ = 0.80 mix.
    """
    if pf_state is None:
        yd_f, yp_f, pf_state = postfilter_full(mic1, mic2, yd, yp, labels=labels)
    else:
        yd_f = yp_f = None
        if labels is None:
            labels = pf_state.get("labels")
    f, t, X1, X2 = stft_pair(mic1, mic2)
    _, _, Yd, Yp = stft_pair(yd, yp)
    Gd = np.asarray(pf_state["Gd"])
    Gp = np.asarray(pf_state["Gp"])
    act = spatial_opposite_activity(X1, X2, f, rtf_d, rtf_p)
    n = len(yd)
    yd_base = _stft_gain_mix(Yd, Gd, np.full(Yd.shape[1], DYN_LAM_BASE), n)
    yp_base = _stft_gain_mix(Yp, Gp, np.full(Yp.shape[1], DYN_LAM_BASE), n)
    # Y_exc = Y * δ * (G - 1)
    Yd_exc = Yd * act["delta_d"][None, :] * (Gd - 1.0)
    Yp_exc = Yp * act["delta_p"][None, :] * (Gp - 1.0)
    yd_exc = istft_one(Yd_exc, n)
    yp_exc = istft_one(Yp_exc, n)
    if yd_f is None:
        yd_f = _stft_gain_mix(Yd, Gd, np.ones(Yd.shape[1]), n)
        yp_f = _stft_gain_mix(Yp, Gp, np.ones(Yp.shape[1]), n)
    state = {
        "activity": act,
        "postfilter": pf_state,
        "labels": labels if labels is not None else pf_state.get("labels"),
        "t_frames": t,
        "lam_base": DYN_LAM_BASE,
        "lam_opp": DYN_LAM_OPP,
        "lam_ovl": DYN_LAM_OVL,
        "yd_full": yd_f,
        "yp_full": yp_f,
    }
    return yd_base, yp_base, yd_exc, yp_exc, state


def apply_dynamic_beta(y_base, y_exc, beta):
    """y(β) = y_base + β y_exc. β=0 is fixed λ=0.80."""
    b = float(np.clip(beta, 0.0, 1.0))
    if b <= 0.0:
        return np.asarray(y_base, dtype=np.float64)
    return np.asarray(y_base, dtype=np.float64) + b * np.asarray(y_exc, dtype=np.float64)


def cancel_pair_with_dynamic(mic1, mic2, rtf_d, rtf_p, freqs, mu=0.25, gate=True, beta=0.0, gsc_only=False):
    """GSC + residual Wiener, with dynamic λ(t) at the given β.

    beta=0 is exactly the fixed λ=0.80 default. gsc_only returns GSC outputs.
    """
    yd, yp, st = cancel_pair(mic1, mic2, rtf_d, rtf_p, freqs, mu=mu, gate=gate)
    yd_f, yp_f, pf = postfilter_full(mic1, mic2, yd, yp, labels=st["labels"])
    yd_base, yp_base, yd_exc, yp_exc, dyn = dynamic_postfilter_components(
        mic1, mic2, yd, yp, rtf_d, rtf_p, labels=st["labels"], pf_state=pf,
    )
    st["postfilter"] = pf
    st["dynamic"] = dyn
    st["yd_gsc"] = yd
    st["yp_gsc"] = yp
    st["yd_full"] = yd_f
    st["yp_full"] = yp_f
    st["yd_base"] = yd_base
    st["yp_base"] = yp_base
    st["yd_exc"] = yd_exc
    st["yp_exc"] = yp_exc
    if gsc_only:
        return yd, yp, st
    return apply_dynamic_beta(yd_base, yd_exc, beta), apply_dynamic_beta(yp_base, yp_exc, beta), st
