#!/usr/bin/env python3
"""Sweep residual post-filter strength on the existing 15-session set.

Metrics match RESULT.md:
speech-frame CTA, target SI-SDR / correlation / energy ratio / LSD;
Both sessions split into alternate / overlap (no invented overlap SIR).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import stft as sp_stft

import cancel as C

HERE = Path(__file__).resolve().parent
PCM_ROOT = HERE / "pcm"
if not PCM_ROOT.exists():
    PCM_ROOT = Path(__file__).resolve().parent / "pcm"
OUT = HERE
WAV_OUT = HERE / "eval_pf_wav"
CSV_PATH = HERE / "summary_postfilter.csv"
JSON_PATH = HERE / "postfilter_sweep.json"

STRENGTHS = [round(x * 0.10, 2) for x in range(0, 11)]  # 0.00 .. 1.00
EPS = C.EPS
FS = C.FS


def align_to_ref(est, ref, max_ms=8.0):
    max_lag = int(FS * max_ms / 1000.0)
    n = min(len(est), len(ref))
    a, b = est[:n], ref[:n]
    if n < 200000:
        c = np.correlate(a, b, mode="full")
    else:
        N = int(2 ** math.ceil(math.log2(2 * n - 1)))
        Cspec = rfft(a, N) * np.conj(rfft(b, N))
        c = irfft(Cspec, N)
        c = np.concatenate([c[-(n - 1) :], c[:n]])
    lags = np.arange(-(n - 1), n)
    mid = n - 1
    sl = slice(mid - max_lag, mid + max_lag + 1)
    k = int(np.argmax(np.abs(c[sl])))
    lag = lags[sl][k]
    if lag >= 0:
        est_a = a[lag:]
        ref_a = b[: len(est_a)]
    else:
        ref_a = b[-lag:]
        est_a = a[: len(ref_a)]
    return est_a, ref_a, lag


def _mask_aligned(mask, ref_len, est_len, lag, n_aligned):
    mm = mask[: min(len(mask), ref_len)]
    if lag >= 0:
        mm = mm[:n_aligned]
    else:
        start = -lag
        mm = mm[start : start + n_aligned] if start < len(mm) else mm[:0]
    if len(mm) < n_aligned:
        mm = np.pad(mm, (0, n_aligned - len(mm)), constant_values=False)
    return mm[:n_aligned].astype(bool)


def si_sdr(est, ref, mask=None):
    e, r, lag = align_to_ref(est, ref)
    if mask is not None:
        mm = _mask_aligned(mask, min(len(mask), len(ref)), len(est), lag, len(r))
        if mm.sum() > 100:
            e, r = e[mm], r[mm]
    if len(r) < 64:
        return float("nan"), 0.0, 0
    alpha = float(np.dot(e, r) / (np.dot(r, r) + EPS))
    e_true = alpha * r
    e_res = e - e_true
    sdr = 10 * np.log10((np.dot(e_true, e_true) + EPS) / (np.dot(e_res, e_res) + EPS))
    if np.std(e) > 1e-12 and np.std(r) > 1e-12:
        corr = float(np.corrcoef(e, r)[0, 1])
    else:
        corr = 0.0
    return float(sdr), corr, int(len(r))


def energy_db_on_mask(x, mask):
    if mask is not None and mask.sum() > 16:
        v = x[: len(mask)][mask[: len(x)]]
    else:
        v = x
    return C.db10(float(np.mean(v * v) + EPS))


def lsd_db(est, ref, mask=None):
    e, r, lag = align_to_ref(est, ref)
    _, _, E = sp_stft(
        e, fs=FS, window=C.WIN, nperseg=C.N_FFT, noverlap=C.N_FFT - C.HOP,
        boundary=None, padded=False,
    )
    f, _, R = sp_stft(
        r, fs=FS, window=C.WIN, nperseg=C.N_FFT, noverlap=C.N_FFT - C.HOP,
        boundary=None, padded=False,
    )
    band = (f >= 300) & (f <= 7000)
    re = np.abs(R[band]) ** 2 + 1e-10
    ee = np.abs(E[band]) ** 2 + 1e-10
    lre = 10 * np.log10(re)
    lee = 10 * np.log10(ee)
    lre = lre - np.mean(lre, axis=0, keepdims=True)
    lee = lee - np.mean(lee, axis=0, keepdims=True)
    d = np.sqrt(np.mean((lre - lee) ** 2, axis=0))
    er = np.mean(re, axis=0)
    keep = er > np.percentile(er, 40)
    if keep.sum() < 3:
        return float(np.mean(d))
    return float(np.mean(d[keep]))


def sample_mask_from_frame_labels(n, labels):
    mask_map = {k: np.zeros(n, dtype=bool) for k in ("driver", "passenger", "overlap")}
    for i, lab in enumerate(labels):
        if lab in mask_map:
            a = i * C.HOP
            b = min(n, a + C.N_FFT)
            mask_map[lab][a:b] = True
    return mask_map


def add_row(results, cond, sid, method, region, yd, yp, mask_cta, mask_tgt,
            cta_in, tgt_ref, cta_is_pax_out, extra=None):
    out_cta = yp if cta_is_pax_out else yd
    out_tgt = yd if cta_is_pax_out else yp
    e_in = energy_db_on_mask(cta_in, mask_cta)
    e_out = energy_db_on_mask(out_cta, mask_cta)
    cta = e_in - e_out
    sdr, corr, nsam = si_sdr(out_tgt, tgt_ref, mask=mask_tgt)
    e_tgt_in = energy_db_on_mask(tgt_ref, mask_tgt)
    e_tgt_out = energy_db_on_mask(out_tgt, mask_tgt)
    eratio = e_tgt_out - e_tgt_in
    try:
        lsd = lsd_db(out_tgt, tgt_ref, mask_tgt)
    except Exception:
        lsd = float("nan")
    row = {
        "cond": cond, "sid": sid, "method": method, "region": region,
        "cta_db": cta, "cta_in_db": e_in, "cta_out_db": e_out,
        "target_sisdr_db": sdr, "target_corr": corr,
        "target_eratio_db": eratio, "target_lsd_db": lsd,
        "speech_s": float((mask_tgt.sum() if mask_tgt is not None else len(tgt_ref)) / FS),
        "cta_which": "pax_out vs mic2" if cta_is_pax_out else "drv_out vs mic1",
        "tgt_which": "yd vs mic1" if cta_is_pax_out else "yp vs mic2",
    }
    if extra:
        row.update(extra)
    results.append(row)


def nanmean(xs):
    a = np.asarray(list(xs), float)
    return float(np.nanmean(a)) if a.size else float("nan")


def nanmedian(xs):
    a = np.asarray(list(xs), float)
    return float(np.nanmedian(a)) if a.size else float("nan")


def nanmin(xs):
    a = np.asarray(list(xs), float)
    return float(np.nanmin(a)) if a.size else float("nan")


def summarize(results, strength, region, cond=None):
    rs = [
        r for r in results
        if r["method"] == f"gsc_pf_{strength:.2f}"
        and r["region"] == region
        and (cond is None or r["cond"] == cond)
    ]
    if not rs:
        return None
    return {
        "n": len(rs),
        "cta_mean": nanmean(r["cta_db"] for r in rs),
        "cta_med": nanmedian(r["cta_db"] for r in rs),
        "cta_min": nanmin(r["cta_db"] for r in rs),
        "cta_max": float(np.nanmax([r["cta_db"] for r in rs])),
        "sisdr_mean": nanmean(r["target_sisdr_db"] for r in rs),
        "sisdr_min": nanmin(r["target_sisdr_db"] for r in rs),
        "corr_mean": nanmean(r["target_corr"] for r in rs),
        "corr_min": nanmin(r["target_corr"] for r in rs),
        "eratio_mean": nanmean(r["target_eratio_db"] for r in rs),
        "eratio_min": nanmin(r["target_eratio_db"] for r in rs),
        "lsd_mean": nanmean(r["target_lsd_db"] for r in rs),
        "lsd_max": float(np.nanmax([r["target_lsd_db"] for r in rs])),
        "rows": rs,
    }


def overlap_summary(results, strength):
    rs = [
        r for r in results
        if r["method"] == f"gsc_pf_{strength:.2f}" and r["region"] == "both_overlap"
    ]
    if not rs:
        return None
    yd_drop = [r["overlap_mic1_db"] - r["overlap_yd_db"] for r in rs]
    yp_drop = [r["overlap_mic2_db"] - r["overlap_yp_db"] for r in rs]
    return {
        "n": len(rs),
        "speech_s_mean": nanmean(r["speech_s"] for r in rs),
        "yd_drop_mean": nanmean(yd_drop),
        "yp_drop_mean": nanmean(yp_drop),
        "yd_drop_min": nanmin(yd_drop),
        "yp_drop_min": nanmin(yp_drop),
        "rows": rs,
    }


def pick_default(table_by_s):
    """Pick a default strength from the sweep.

    Prefer more CTA as long as the target is not clearly hurt.
    Hard-ish gates vs λ=0 (GSC only):
      - single-talk mean energy ratio >= -1.0 dB
      - single-talk mean correlation stays high (>= 0.980) and no session < 0.95
      - single-talk worst SI-SDR does not collapse into single digits if it
        was healthy at λ=0; allow a drop from the numerically huge GSC
        pass-through values, but not below ~12 dB mean or ~8 dB worst
        unless λ=0 was already that low
      - alternate-region mean energy ratio >= -1.0 dB, mean corr >= 0.970
      - mean LSD should not jump past ~2.0 dB on single-talk
    If the next stronger step starts failing a gate, stop at the previous.
    """
    strengths = sorted(table_by_s)
    base = table_by_s[0.0]
    chosen = 0.0
    reasons = []

    def single_ok(s, pack, verbose=False):
        fails = []
        st = pack["single_all"]
        drv = pack["single_driver"]
        pax = pack["single_pax"]
        alt_d = pack["alt_driver"]
        alt_p = pack["alt_pax"]
        if st["eratio_mean"] < -1.0:
            fails.append(f"单说平均能量比 {st['eratio_mean']:.2f} dB < -1.0 dB")
        if st["eratio_min"] < -2.0:
            fails.append(f"单说最差能量比 {st['eratio_min']:.2f} dB < -2.0 dB")
        if st["corr_mean"] < 0.980:
            fails.append(f"单说平均相关 {st['corr_mean']:.3f} < 0.980")
        if st["corr_min"] < 0.95:
            fails.append(f"单说最差相关 {st['corr_min']:.3f} < 0.95")
        # SI-SDR: GSC λ=0 is often 30–90 dB because of gate passthrough.
        # Collapse = falling to a clearly damaged regime.
        base_min = base["single_all"]["sisdr_min"]
        floor_min = min(8.0, base_min - 1.0) if base_min == base_min else 8.0
        floor_mean = 12.0
        if st["sisdr_mean"] < floor_mean and (base["single_all"]["sisdr_mean"] - st["sisdr_mean"]) > 8.0:
            # only fail if mean actually entered a low regime, not just dropped from 80 to 25
            if st["sisdr_mean"] < 15.0:
                fails.append(f"单说平均 SI-SDR {st['sisdr_mean']:.2f} dB 进入低区")
        if st["sisdr_min"] < 8.0 and st["sisdr_min"] < (base_min - 3.0):
            fails.append(f"单说最差 SI-SDR {st['sisdr_min']:.2f} dB 明显坍塌（λ=0 为 {base_min:.2f}）")
        if st["lsd_mean"] > 2.0 and st["lsd_mean"] > base["single_all"]["lsd_mean"] + 0.8:
            fails.append(f"单说平均 LSD {st['lsd_mean']:.2f} dB，可能有音乐噪声/谱伤")
        if alt_d and alt_d["eratio_mean"] < -1.0:
            fails.append(f"主驾交替平均能量比 {alt_d['eratio_mean']:.2f} dB < -1.0 dB")
        if alt_p and alt_p["eratio_mean"] < -1.0:
            fails.append(f"副驾交替平均能量比 {alt_p['eratio_mean']:.2f} dB < -1.0 dB")
        if alt_d and alt_d["corr_mean"] < 0.970:
            fails.append(f"主驾交替平均相关 {alt_d['corr_mean']:.3f} < 0.970")
        if alt_p and alt_p["corr_mean"] < 0.970:
            fails.append(f"副驾交替平均相关 {alt_p['corr_mean']:.3f} < 0.970")
        if alt_d and alt_d["sisdr_min"] < 8.0 and alt_d["sisdr_min"] < (base["alt_driver"]["sisdr_min"] - 3.0):
            fails.append(f"主驾交替最差 SI-SDR {alt_d['sisdr_min']:.2f} dB 坍塌")
        if alt_p and alt_p["sisdr_min"] < 8.0 and alt_p["sisdr_min"] < (base["alt_pax"]["sisdr_min"] - 3.0):
            fails.append(f"副驾交替最差 SI-SDR {alt_p['sisdr_min']:.2f} dB 坍塌")
        return fails

    last_ok = 0.0
    first_fail = None
    for s in strengths:
        fails = single_ok(s, table_by_s[s])
        if not fails:
            last_ok = s
        else:
            first_fail = (s, fails)
            reasons.append(f"强度 {s:.2f} 未过门槛：" + "；".join(fails))
            # keep scanning only to record, but chosen is last_ok
            # stop choosing higher
            for t in strengths:
                if t > s:
                    # remaining will also likely fail; still record
                    pass
            break

    # Among passing strengths, prefer the one that still adds CTA vs previous
    # but if a stronger passing step adds almost no CTA and starts denting
    # fidelity, stay earlier. We already stopped at first fail. Now maybe
    # back off one step if the last_ok is only marginally better in CTA
    # but noticeably worse in fidelity than the previous passing step.
    chosen = last_ok
    if first_fail is not None and first_fail[0] > 0:
        reasons.append(
            f"更强一档 {first_fail[0]:.2f} 开始越线，停在前一档 {last_ok:.2f}。"
        )

    # Back off if last_ok already shows a clear fidelity dent vs mid-level
    # with little CTA gain (<0.4 dB extra mean CTA over previous 0.10 step
    # AND eratio dropped more than 0.3 dB). Optional refinement:
    if chosen > 0:
        prevs = [s for s in strengths if s < chosen]
        if prevs:
            prev = prevs[-1]
            cur = table_by_s[chosen]["single_all"]
            prv = table_by_s[prev]["single_all"]
            d_cta = cur["cta_mean"] - prv["cta_mean"]
            d_er = cur["eratio_mean"] - prv["eratio_mean"]
            d_corr = cur["corr_mean"] - prv["corr_mean"]
            d_lsd = cur["lsd_mean"] - prv["lsd_mean"]
            if d_cta < 0.35 and (d_er < -0.25 or d_lsd > 0.35 or d_corr < -0.005):
                reasons.append(
                    f"{chosen:.2f} 相对 {prev:.2f} 只多压 CTA {d_cta:+.2f} dB，"
                    f"但能量比 {d_er:+.2f} dB、LSD {d_lsd:+.2f}、相关 {d_corr:+.3f}，回退一档。"
                )
                chosen = prev

    return chosen, reasons, first_fail


def fmt(v, nd=2):
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def main():
    print(f"loading sessions from {PCM_ROOT}", flush=True)
    sessions = C.list_sessions(PCM_ROOT)
    print(f"loaded {len(sessions)} sessions", flush=True)
    pack = C.prepare_rtfs(sessions)
    results = []
    gsc_cache = {}

    example_sids = {"20261126_152912", "20261126_152929", "20261126_153105",
                    "20261126_153134", "20261126_153348", "20261126_153322"}

    for s in sessions:
        x, y = s["mic1"], s["mic2"]
        sid, cond = s["sid"], s["cond"]
        n = len(x)
        rtf_d, rtf_p = C.rtf_for_session(pack, sid)
        yd, yp, st = C.cancel_pair(x, y, rtf_d, rtf_p, pack["freqs"])
        yd_f, yp_f, pf = C.postfilter_full(x, y, yd, yp, labels=st["labels"])
        gsc_cache[sid] = (x, y, yd, yp, yd_f, yp_f, st)

        if cond == "Only driver":
            vad, *_ = C.vad_from_near(x)
        elif cond == "Only Passenger":
            vad, *_ = C.vad_from_near(y)
        else:
            vad, *_ = C.vad_from_near(x, y)
        s["vad_mask"] = vad
        both_tags = None
        if cond.startswith("Both"):
            f, t, X1, X2 = C.stft_pair(x, y)
            labs, ild, tdoa, e1, e2, e_sil = C.classify_frames(X1, X2, f)
            both_tags = {"labels": labs}

        for strength in STRENGTHS:
            yds = C.apply_postfilter_strength(yd, yd_f, strength)
            yps = C.apply_postfilter_strength(yp, yp_f, strength)
            method = f"gsc_pf_{strength:.2f}"
            if cond == "Only driver":
                add_row(results, cond, sid, method, "single", yds, yps, vad, vad, y, x, True)
            elif cond == "Only Passenger":
                add_row(results, cond, sid, method, "single", yds, yps, vad, vad, x, y, False)
            else:
                masks = sample_mask_from_frame_labels(n, both_tags["labels"])
                m_drv, m_pax, m_ovl = masks["driver"], masks["passenger"], masks["overlap"]
                if m_drv.sum() > FS * 0.15:
                    add_row(results, cond, sid, method, "both_alt_driver",
                            yds, yps, m_drv, m_drv, y, x, True)
                if m_pax.sum() > FS * 0.15:
                    add_row(results, cond, sid, method, "both_alt_passenger",
                            yds, yps, m_pax, m_pax, x, y, False)
                if m_ovl.sum() > FS * 0.1:
                    results.append({
                        "cond": cond, "sid": sid, "method": method, "region": "both_overlap",
                        "cta_db": float("nan"),
                        "cta_in_db": energy_db_on_mask(x, m_ovl),
                        "cta_out_db": energy_db_on_mask(y, m_ovl),
                        "target_sisdr_db": float("nan"), "target_corr": float("nan"),
                        "target_eratio_db": energy_db_on_mask(yds, m_ovl) - energy_db_on_mask(x, m_ovl),
                        "target_lsd_db": float("nan"),
                        "speech_s": float(m_ovl.sum() / FS),
                        "overlap_yd_db": energy_db_on_mask(yds, m_ovl),
                        "overlap_yp_db": energy_db_on_mask(yps, m_ovl),
                        "overlap_mic1_db": energy_db_on_mask(x, m_ovl),
                        "overlap_mic2_db": energy_db_on_mask(y, m_ovl),
                        "cta_which": "n/a (no clean ref)", "tgt_which": "n/a",
                    })
            if sid in example_sids and strength in (0.0, 0.3, 0.5, 0.7, 1.0):
                outd = WAV_OUT / cond / sid / f"pf_{strength:.2f}"
                C.save_wav(outd / "driver_out.wav", yds)
                C.save_wav(outd / "passenger_out.wav", yps)

        # always keep GSC and full for examples
        if sid in example_sids:
            base = WAV_OUT / cond / sid
            C.save_wav(base / "mic_raw.wav", x)
            C.save_wav(base / "mic2_raw.wav", y)

        print(f"  done {cond}/{sid}", flush=True)

    keys = [
        "cond", "sid", "method", "region", "cta_db", "cta_in_db", "cta_out_db",
        "target_sisdr_db", "target_corr", "target_eratio_db", "target_lsd_db",
        "speech_s", "cta_which", "tgt_which",
        "overlap_yd_db", "overlap_yp_db", "overlap_mic1_db", "overlap_mic2_db",
    ]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in results:
            out = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    out[k] = "" if (v != v) else f"{v:.6f}"
                else:
                    out[k] = v
            w.writerow(out)
    print(f"wrote {CSV_PATH} ({len(results)} rows)", flush=True)

    table = {}
    for s in STRENGTHS:
        table[s] = {
            "single_driver": summarize(results, s, "single", "Only driver"),
            "single_pax": summarize(results, s, "single", "Only Passenger"),
            "single_all": None,
            "alt_driver": summarize(results, s, "both_alt_driver"),
            "alt_pax": summarize(results, s, "both_alt_passenger"),
            "overlap": overlap_summary(results, s),
        }
        # combine 10 single-talk sessions
        rs = [r for r in results if r["method"] == f"gsc_pf_{s:.2f}" and r["region"] == "single"]
        table[s]["single_all"] = {
            "n": len(rs),
            "cta_mean": nanmean(r["cta_db"] for r in rs),
            "cta_med": nanmedian(r["cta_db"] for r in rs),
            "cta_min": nanmin(r["cta_db"] for r in rs),
            "cta_max": float(np.nanmax([r["cta_db"] for r in rs])),
            "sisdr_mean": nanmean(r["target_sisdr_db"] for r in rs),
            "sisdr_min": nanmin(r["target_sisdr_db"] for r in rs),
            "corr_mean": nanmean(r["target_corr"] for r in rs),
            "corr_min": nanmin(r["target_corr"] for r in rs),
            "eratio_mean": nanmean(r["target_eratio_db"] for r in rs),
            "eratio_min": nanmin(r["target_eratio_db"] for r in rs),
            "lsd_mean": nanmean(r["target_lsd_db"] for r in rs),
            "lsd_max": float(np.nanmax([r["target_lsd_db"] for r in rs])),
            "rows": rs,
        }

    def dump_pack(p):
        if p is None:
            return None
        return {k: v for k, v in p.items() if k != "rows"}

    print("\n======= 单说 10 段（只主驾 5 + 只副驾 5）=======")
    print(f"{'λ':>6} {'CTA平均':>8} {'CTA中位':>8} {'CTA最小':>8} {'SI-SDR均':>9} {'SI-SDR差':>9} {'相关均':>8} {'相关差':>8} {'能量比均':>8} {'能量比差':>8} {'LSD均':>7}")
    for s in STRENGTHS:
        p = table[s]["single_all"]
        print(
            f"{s:6.2f} {p['cta_mean']:8.2f} {p['cta_med']:8.2f} {p['cta_min']:8.2f} "
            f"{p['sisdr_mean']:9.2f} {p['sisdr_min']:9.2f} {p['corr_mean']:8.3f} {p['corr_min']:8.3f} "
            f"{p['eratio_mean']:8.2f} {p['eratio_min']:8.2f} {p['lsd_mean']:7.2f}"
        )

    for title, key, cond in (
        ("只主驾说", "single_driver", "Only driver"),
        ("只副驾说", "single_pax", "Only Passenger"),
    ):
        print(f"\n======= {title} =======")
        print(f"{'λ':>6} {'CTA平均':>8} {'CTA中位':>8} {'SI-SDR均':>9} {'SI-SDR差':>9} {'相关均':>8} {'能量比均':>8} {'能量比差':>8} {'LSD均':>7}")
        for s in STRENGTHS:
            p = table[s][key]
            print(
                f"{s:6.2f} {p['cta_mean']:8.2f} {p['cta_med']:8.2f} "
                f"{p['sisdr_mean']:9.2f} {p['sisdr_min']:9.2f} {p['corr_mean']:8.3f} "
                f"{p['eratio_mean']:8.2f} {p['eratio_min']:8.2f} {p['lsd_mean']:7.2f}"
            )
        # per-session at 0, mid, 1 for inspection
        for s in (0.0, 0.4, 0.6, 0.8, 1.0):
            print(f"  -- λ={s:.2f} 逐段 --")
            for r in table[s][key]["rows"]:
                print(
                    f"    {r['sid']}  CTA={r['cta_db']:.2f}  SI-SDR={r['target_sisdr_db']:.2f}  "
                    f"corr={r['target_corr']:.3f}  Eratio={r['target_eratio_db']:.2f}  "
                    f"LSD={r['target_lsd_db']:.2f}"
                )

    print("\n======= 两人交替 =======")
    for title, key in (("主驾交替 → 压副驾输出", "alt_driver"), ("副驾交替 → 压主驾输出", "alt_pax")):
        print(f"\n-- {title} --")
        print(f"{'λ':>6} {'CTA平均':>8} {'CTA中位':>8} {'SI-SDR均':>9} {'SI-SDR差':>9} {'相关均':>8} {'能量比均':>8} {'能量比差':>8} {'LSD均':>7}")
        for s in STRENGTHS:
            p = table[s][key]
            if not p:
                continue
            print(
                f"{s:6.2f} {p['cta_mean']:8.2f} {p['cta_med']:8.2f} "
                f"{p['sisdr_mean']:9.2f} {p['sisdr_min']:9.2f} {p['corr_mean']:8.3f} "
                f"{p['eratio_mean']:8.2f} {p['eratio_min']:8.2f} {p['lsd_mean']:7.2f}"
            )

    print("\n======= 两人重叠（无干净参考，不报 CTA/SIR）=======")
    print(f"{'λ':>6} {'n':>3} {'重叠s均':>8} {'主驾出相对原麦下降':>20} {'副驾出相对原麦下降':>20}")
    for s in STRENGTHS:
        p = table[s]["overlap"]
        if not p:
            continue
        print(
            f"{s:6.2f} {p['n']:3d} {p['speech_s_mean']:8.2f} "
            f"{p['yd_drop_mean']:20.2f} {p['yp_drop_mean']:20.2f}"
        )

    # Verify λ=0 matches prior GSC numbers roughly
    p0 = table[0.0]
    print("\n======= λ=0 对照原 GSC（RESULT.md：单说总均 CTA≈7.9，主驾 7.49，副驾 8.29）=======", flush=True)
    print(
        f"  本轮 λ=0  十段 CTA={p0['single_all']['cta_mean']:.2f}  "
        f"只主驾 {p0['single_driver']['cta_mean']:.2f}  只副驾 {p0['single_pax']['cta_mean']:.2f}  "
        f"交替主驾 {p0['alt_driver']['cta_mean']:.2f}  交替副驾 {p0['alt_pax']['cta_mean']:.2f}"
    )

    chosen, reasons, first_fail = pick_default(table)
    print("\n======= 建议默认值 =======")
    print(f"chosen default strength = {chosen:.2f}")
    for line in reasons:
        print("  ", line)
    if first_fail:
        print(f"  first failing strength = {first_fail[0]:.2f}: {first_fail[1]}")
    cpack = table[chosen]
    bpack = table[0.0]
    print(
        f"  vs GSC-only: 单说 CTA {bpack['single_all']['cta_mean']:.2f} → {cpack['single_all']['cta_mean']:.2f} "
        f"(Δ {cpack['single_all']['cta_mean']-bpack['single_all']['cta_mean']:+.2f} dB); "
        f"能量比 {bpack['single_all']['eratio_mean']:.2f} → {cpack['single_all']['eratio_mean']:.2f}; "
        f"相关 {bpack['single_all']['corr_mean']:.3f} → {cpack['single_all']['corr_mean']:.3f}; "
        f"SI-SDR {bpack['single_all']['sisdr_mean']:.2f} → {cpack['single_all']['sisdr_mean']:.2f} "
        f"(min {bpack['single_all']['sisdr_min']:.2f} → {cpack['single_all']['sisdr_min']:.2f}); "
        f"LSD {bpack['single_all']['lsd_mean']:.2f} → {cpack['single_all']['lsd_mean']:.2f}"
    )

    slim = {}
    for s, pack_s in table.items():
        slim[f"{s:.2f}"] = {k: dump_pack(v) for k, v in pack_s.items()}
    payload = {
        "strengths": STRENGTHS,
        "default_strength": chosen,
        "reasons": reasons,
        "first_fail": None if not first_fail else {"strength": first_fail[0], "fails": first_fail[1]},
        "table": slim,
        "pf_params": {
            "max_atten_db": C.PF_MAX_ATTEN_DB,
            "target_floor_db": C.PF_TARGET_FLOOR_DB,
            "overlap_floor_db": C.PF_OVERLAP_FLOOR_DB,
            "uncertain_floor_db": C.PF_UNCERTAIN_FLOOR_DB,
            "alpha_pn": C.PF_ALPHA_PN,
            "alpha_dd": C.PF_ALPHA_DD,
            "gain_smooth": C.PF_GAIN_SMOOTH,
        },
        "criteria": {
            "single_eratio_mean_min_db": -1.0,
            "single_eratio_session_min_db": -2.0,
            "single_corr_mean_min": 0.980,
            "single_corr_session_min": 0.95,
            "single_sisdr_collapse_min_db": 8.0,
            "single_lsd_mean_max_db": 2.0,
            "alt_eratio_mean_min_db": -1.0,
            "alt_corr_mean_min": 0.970,
        },
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {JSON_PATH}", flush=True)
    return chosen, table, results


if __name__ == "__main__":
    main()
