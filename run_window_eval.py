#!/usr/bin/env python3
"""Sweep STFT window length on the existing 15-session set.

Main pipeline matches the current service: GSC from zero + spatial gate
+ residual Wiener post-filter at λ=0.80 (and λ=0 as a GSC-only reference).
Hop defaults to N/4, same ratio as the 512/128 baseline.

Algorithm delay is reported as N/fs (same convention as the published
32 ms figure). OLA reconstruction delay within each completed hop block
is also recorded: (N-H)/fs to (N-1)/fs.

Metrics match RESULT.md / RESULT-postfilter.md (speech-frame CTA,
target SI-SDR / correlation / energy ratio / LSD; Both sessions split
into alternate / overlap, no invented overlap SIR). LSD and residual
peakiness use a *fixed* 512/128 analysis STFT so the meter does not
change with the processing window; at N=512 this coincides with the
previous evaluation.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import stft as sp_stft
from scipy.signal.windows import hann as hann_win

import cancel as C
import run_postfilter_eval as E

HERE = Path(__file__).resolve().parent
PCM_ROOT = HERE / "pcm"
if not PCM_ROOT.exists():
    PCM_ROOT = Path(__file__).resolve().parent / "pcm"
OUT = HERE
FIG_DIR = HERE / "eval_window_figs"
CSV_PATH = HERE / "summary_window.csv"
JSON_PATH = HERE / "window_sweep.json"

# Processing windows. 512 is the baseline; 640/768 only to see if 512
# is already on a quality plateau. hop = N/4.
WINDOWS = [128, 192, 256, 320, 384, 512, 640, 768]
STRENGTHS_MAIN = (0.0, 0.80)
# Extra λ values scored cheaply (mix only) if a short window hurts at 0.80.
STRENGTHS_EXTRA = (0.50, 0.60, 0.70, 0.90)

# Fixed metric STFT (same as the published baseline eval).
METRIC_N = 512
METRIC_HOP = 128
METRIC_WIN = hann_win(METRIC_N, sym=False).astype(np.float64)

FS = C.FS
EPS = C.EPS

# "No compromise" gates vs baseline (N=512, λ=0.80). Written before looking
# at the table. Primary CTA budget is 0.5 dB; may relax to 1.0 dB if no
# shorter window survives, and that relaxation will be stated explicitly.
CTA_DROP_TIGHT_DB = 0.5
CTA_DROP_RELAX_DB = 1.0
ERATIO_MEAN_MIN_DB = -1.0
ERATIO_SESS_MIN_DB = -2.0
CORR_MEAN_MIN = 0.980
CORR_SESS_MIN = 0.95
SISDR_COLLAPSE_DB = 8.0
ALT_CORR_MEAN_MIN = 0.970
# LSD / musical-noise: not clearly worse than baseline.
LSD_MEAN_SLACK_DB = 0.25
P2M_REL_SLACK = 1.15  # far residual peak/median may rise up to +15%
P2M_ABS_SLACK = 4.0   # or +4.0 absolute, whichever is looser


def lsd_db_fixed(est, ref, mask=None):
    e, r, lag = E.align_to_ref(est, ref)
    _, _, Espec = sp_stft(
        e, fs=FS, window=METRIC_WIN, nperseg=METRIC_N,
        noverlap=METRIC_N - METRIC_HOP, boundary=None, padded=False,
    )
    f, _, Rspec = sp_stft(
        r, fs=FS, window=METRIC_WIN, nperseg=METRIC_N,
        noverlap=METRIC_N - METRIC_HOP, boundary=None, padded=False,
    )
    band = (f >= 300) & (f <= 7000)
    re = np.abs(Rspec[band]) ** 2 + 1e-10
    ee = np.abs(Espec[band]) ** 2 + 1e-10
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


def residual_tonalness(x):
    f, t, X = sp_stft(
        x, fs=FS, window=METRIC_WIN, nperseg=METRIC_N,
        noverlap=METRIC_N - METRIC_HOP, nfft=METRIC_N,
        boundary=None, padded=False, return_onesided=True,
    )
    mag = np.abs(X)
    band = (f >= 800) & (f <= 7000)
    M = mag[band, :]
    e = np.mean(M ** 2, axis=0)
    keep = e > np.percentile(e, 30)
    if keep.sum() < 3:
        keep[:] = True
    Mk = M[:, keep] + 1e-12
    p2m = np.max(Mk, axis=0) / (np.median(Mk, axis=0) + 1e-12)
    m2 = np.mean(Mk ** 2, axis=0)
    m4 = np.mean(Mk ** 4, axis=0)
    kurt = m4 / (m2 ** 2 + 1e-18)
    sparse = np.mean(Mk > (0.3 * np.max(Mk, axis=0, keepdims=True)), axis=0)
    return float(np.mean(p2m)), float(np.mean(kurt)), float(np.mean(sparse))


def add_row(results, cond, sid, method, region, yd, yp, mask_cta, mask_tgt,
            cta_in, tgt_ref, cta_is_pax_out, extra=None):
    # Same as E.add_row but LSD uses the fixed metric STFT.
    out_cta = yp if cta_is_pax_out else yd
    out_tgt = yd if cta_is_pax_out else yp
    e_in = E.energy_db_on_mask(cta_in, mask_cta)
    e_out = E.energy_db_on_mask(out_cta, mask_cta)
    cta = e_in - e_out
    sdr, corr, nsam = E.si_sdr(out_tgt, tgt_ref, mask=mask_tgt)
    e_tgt_in = E.energy_db_on_mask(tgt_ref, mask_tgt)
    e_tgt_out = E.energy_db_on_mask(out_tgt, mask_tgt)
    eratio = e_tgt_out - e_tgt_in
    try:
        lsd = lsd_db_fixed(out_tgt, tgt_ref, mask_tgt)
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
    return row


def score_outputs(results, s, cond, sid, method, extra, yds, yps, vad, both_masks):
    x, y = s["mic1"], s["mic2"]
    if cond == "Only driver":
        add_row(results, cond, sid, method, "single", yds, yps, vad, vad, y, x, True, extra)
    elif cond == "Only Passenger":
        add_row(results, cond, sid, method, "single", yds, yps, vad, vad, x, y, False, extra)
    else:
        m_drv, m_pax, m_ovl = both_masks["driver"], both_masks["passenger"], both_masks["overlap"]
        if m_drv.sum() > FS * 0.15:
            add_row(results, cond, sid, method, "both_alt_driver",
                    yds, yps, m_drv, m_drv, y, x, True, extra)
        if m_pax.sum() > FS * 0.15:
            add_row(results, cond, sid, method, "both_alt_passenger",
                    yds, yps, m_pax, m_pax, x, y, False, extra)
        if m_ovl.sum() > FS * 0.1:
            results.append({
                "cond": cond, "sid": sid, "method": method, "region": "both_overlap",
                "cta_db": float("nan"),
                "cta_in_db": E.energy_db_on_mask(x, m_ovl),
                "cta_out_db": E.energy_db_on_mask(y, m_ovl),
                "target_sisdr_db": float("nan"), "target_corr": float("nan"),
                "target_eratio_db": E.energy_db_on_mask(yds, m_ovl) - E.energy_db_on_mask(x, m_ovl),
                "target_lsd_db": float("nan"),
                "speech_s": float(m_ovl.sum() / FS),
                "overlap_yd_db": E.energy_db_on_mask(yds, m_ovl),
                "overlap_yp_db": E.energy_db_on_mask(yps, m_ovl),
                "overlap_mic1_db": E.energy_db_on_mask(x, m_ovl),
                "overlap_mic2_db": E.energy_db_on_mask(y, m_ovl),
                "cta_which": "n/a (no clean ref)", "tgt_which": "n/a",
                **(extra or {}),
            })


def pack_summary(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "cta_mean": E.nanmean(r["cta_db"] for r in rows),
        "cta_med": E.nanmedian(r["cta_db"] for r in rows),
        "cta_min": E.nanmin(r["cta_db"] for r in rows),
        "cta_max": float(np.nanmax([r["cta_db"] for r in rows])),
        "sisdr_mean": E.nanmean(r["target_sisdr_db"] for r in rows),
        "sisdr_min": E.nanmin(r["target_sisdr_db"] for r in rows),
        "corr_mean": E.nanmean(r["target_corr"] for r in rows),
        "corr_min": E.nanmin(r["target_corr"] for r in rows),
        "eratio_mean": E.nanmean(r["target_eratio_db"] for r in rows),
        "eratio_min": E.nanmin(r["target_eratio_db"] for r in rows),
        "lsd_mean": E.nanmean(r["target_lsd_db"] for r in rows),
        "lsd_max": float(np.nanmax([r["target_lsd_db"] for r in rows])),
        "speech_s_mean": E.nanmean(r["speech_s"] for r in rows),
    }


def overlap_pack(rows):
    if not rows:
        return None
    yd_drop = [r["overlap_mic1_db"] - r["overlap_yd_db"] for r in rows]
    yp_drop = [r["overlap_mic2_db"] - r["overlap_yp_db"] for r in rows]
    return {
        "n": len(rows),
        "speech_s_mean": E.nanmean(r["speech_s"] for r in rows),
        "yd_drop_mean": E.nanmean(yd_drop),
        "yp_drop_mean": E.nanmean(yp_drop),
        "yd_drop_min": E.nanmin(yd_drop),
        "yp_drop_min": E.nanmin(yp_drop),
    }


def tonal_pack(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "far_p2m_mean": E.nanmean(r["far_p2m"] for r in rows),
        "far_kurt_mean": E.nanmean(r["far_kurt"] for r in rows),
        "far_sparse_mean": E.nanmean(r["far_sparse"] for r in rows),
        "near_p2m_mean": E.nanmean(r["near_p2m"] for r in rows),
        "near_kurt_mean": E.nanmean(r["near_kurt"] for r in rows),
    }


def method_name(n_fft, hop, strength):
    return f"n{n_fft}_h{hop}_l{strength:.2f}"


def summarize_one(results, tonal_rows, n_fft, hop, strength):
    meth = method_name(n_fft, hop, strength)
    rs_all = [r for r in results if r["method"] == meth]
    def sel(region, cond=None):
        return [
            r for r in rs_all
            if r["region"] == region and (cond is None or r["cond"] == cond)
        ]
    pack = {
        "n_fft": n_fft,
        "hop": hop,
        "strength": strength,
        "algo_delay_ms": n_fft / FS * 1000.0,
        "hop_ms": hop / FS * 1000.0,
        "ola_delay_ms_min": (n_fft - hop) / FS * 1000.0,
        "ola_delay_ms_max": (n_fft - 1) / FS * 1000.0,
        "single_all": pack_summary(sel("single")),
        "single_driver": pack_summary(sel("single", "Only driver")),
        "single_pax": pack_summary(sel("single", "Only Passenger")),
        "alt_driver": pack_summary(sel("both_alt_driver")),
        "alt_pax": pack_summary(sel("both_alt_passenger")),
        "overlap": overlap_pack(sel("both_overlap")),
        "tonal_single": tonal_pack([r for r in tonal_rows if r["method"] == meth]),
        "rows_single": sel("single"),
        "rows_alt_d": sel("both_alt_driver"),
        "rows_alt_p": sel("both_alt_passenger"),
    }
    return pack


def judge(pack, baseline, cta_budget):
    """Return (ok, fails) against the pre-declared no-compromise gates."""
    fails = []
    st = pack["single_all"]
    bst = baseline["single_all"]
    if st is None or bst is None:
        return False, ["missing single-talk summary"]

    d_cta = st["cta_mean"] - bst["cta_mean"]
    if d_cta < -cta_budget:
        fails.append(
            f"单说平均 CTA {st['cta_mean']:.2f} 相对基线 {bst['cta_mean']:.2f} "
            f"下降 {(-d_cta):.2f} dB，超过预算 {cta_budget:.1f} dB"
        )

    if st["eratio_mean"] < ERATIO_MEAN_MIN_DB:
        fails.append(f"单说平均能量比 {st['eratio_mean']:.2f} dB < {ERATIO_MEAN_MIN_DB:.1f} dB")
    if st["eratio_min"] < ERATIO_SESS_MIN_DB:
        fails.append(f"单说最差能量比 {st['eratio_min']:.2f} dB < {ERATIO_SESS_MIN_DB:.1f} dB")
    if st["corr_mean"] < CORR_MEAN_MIN:
        fails.append(f"单说平均相关 {st['corr_mean']:.3f} < {CORR_MEAN_MIN:.3f}")
    if st["corr_min"] < CORR_SESS_MIN:
        fails.append(f"单说最差相关 {st['corr_min']:.3f} < {CORR_SESS_MIN:.2f}")

    # SI-SDR: do not fall into single digits, and the worst session should
    # not clearly deteriorate vs the same session's baseline.
    if st["sisdr_min"] < SISDR_COLLAPSE_DB and st["sisdr_min"] < (bst["sisdr_min"] - 3.0):
        fails.append(
            f"单说最差 SI-SDR {st['sisdr_min']:.2f} dB 掉进个位数/明显差于基线 "
            f"{bst['sisdr_min']:.2f}"
        )
    # Per-session: the previously weakest session should not crash further.
    base_by = {r["sid"]: r for r in baseline.get("rows_single", [])}
    for r in pack.get("rows_single", []):
        b = base_by.get(r["sid"])
        if not b:
            continue
        if (r["target_sisdr_db"] < 8.0) and (r["target_sisdr_db"] < b["target_sisdr_db"] - 3.0):
            fails.append(
                f"{r['sid']} SI-SDR {r['target_sisdr_db']:.2f} 相对基线 "
                f"{b['target_sisdr_db']:.2f} 明显恶化并进入低区"
            )

    alt_d, alt_p = pack["alt_driver"], pack["alt_pax"]
    b_ad, b_ap = baseline["alt_driver"], baseline["alt_pax"]
    if alt_d and alt_d["eratio_mean"] < ERATIO_MEAN_MIN_DB:
        fails.append(f"主驾交替平均能量比 {alt_d['eratio_mean']:.2f} dB < {ERATIO_MEAN_MIN_DB:.1f}")
    if alt_p and alt_p["eratio_mean"] < ERATIO_MEAN_MIN_DB:
        fails.append(f"副驾交替平均能量比 {alt_p['eratio_mean']:.2f} dB < {ERATIO_MEAN_MIN_DB:.1f}")
    if alt_d and alt_d["corr_mean"] < ALT_CORR_MEAN_MIN:
        fails.append(f"主驾交替平均相关 {alt_d['corr_mean']:.3f} < {ALT_CORR_MEAN_MIN:.3f}")
    if alt_p and alt_p["corr_mean"] < ALT_CORR_MEAN_MIN:
        fails.append(f"副驾交替平均相关 {alt_p['corr_mean']:.3f} < {ALT_CORR_MEAN_MIN:.3f}")
    if alt_d and b_ad and alt_d["sisdr_min"] < SISDR_COLLAPSE_DB and alt_d["sisdr_min"] < (b_ad["sisdr_min"] - 3.0):
        fails.append(f"主驾交替最差 SI-SDR {alt_d['sisdr_min']:.2f} 坍塌")
    if alt_p and b_ap and alt_p["sisdr_min"] < SISDR_COLLAPSE_DB and alt_p["sisdr_min"] < (b_ap["sisdr_min"] - 3.0):
        fails.append(f"副驾交替最差 SI-SDR {alt_p['sisdr_min']:.2f} 坍塌")
    # Alternate CTA should not collapse either (same  budget).
    if alt_d and b_ad and (alt_d["cta_mean"] - b_ad["cta_mean"]) < -cta_budget:
        fails.append(
            f"主驾交替 CTA {alt_d['cta_mean']:.2f} 相对基线 {b_ad['cta_mean']:.2f} "
            f"下降超过 {cta_budget:.1f} dB"
        )
    if alt_p and b_ap and (alt_p["cta_mean"] - b_ap["cta_mean"]) < -cta_budget:
        fails.append(
            f"副驾交替 CTA {alt_p['cta_mean']:.2f} 相对基线 {b_ap['cta_mean']:.2f} "
            f"下降超过 {cta_budget:.1f} dB"
        )

    if st["lsd_mean"] > bst["lsd_mean"] + LSD_MEAN_SLACK_DB:
        fails.append(
            f"单说平均 LSD {st['lsd_mean']:.2f} 明显高于基线 {bst['lsd_mean']:.2f} "
            f"(+{st['lsd_mean']-bst['lsd_mean']:.2f} > {LSD_MEAN_SLACK_DB:.2f})"
        )

    ton, bton = pack.get("tonal_single"), baseline.get("tonal_single")
    if ton and bton:
        lim = max(bton["far_p2m_mean"] * P2M_REL_SLACK, bton["far_p2m_mean"] + P2M_ABS_SLACK)
        if ton["far_p2m_mean"] > lim:
            fails.append(
                f"对侧残差峰值/中位数 {ton['far_p2m_mean']:.1f} 明显高于基线 "
                f"{bton['far_p2m_mean']:.1f}（上限 {lim:.1f}）"
            )

    return (len(fails) == 0), fails


def choose(table_pf, windows):
    """Pick the shortest-delay config that does not compromise vs baseline."""
    baseline = table_pf[512]
    notes = []
    # Sanity: baseline itself must be present.
    if baseline is None:
        raise RuntimeError("missing N=512 λ=0.80 baseline")

    def collect(budget):
        ok = []
        detail = []
        for n in windows:
            pack = table_pf[n]
            passed, fails = judge(pack, baseline, budget)
            detail.append((n, passed, fails, pack))
            if passed:
                ok.append(n)
        return ok, detail

    ok_tight, detail_tight = collect(CTA_DROP_TIGHT_DB)
    notes.append({
        "budget_db": CTA_DROP_TIGHT_DB,
        "passing_n": ok_tight,
        "detail": [
            {"n_fft": n, "ok": p, "fails": f} for n, p, f, _ in detail_tight
        ],
    })
    budget_used = CTA_DROP_TIGHT_DB
    passing = ok_tight
    detail = detail_tight
    if not passing:
        ok_relax, detail_relax = collect(CTA_DROP_RELAX_DB)
        notes.append({
            "budget_db": CTA_DROP_RELAX_DB,
            "passing_n": ok_relax,
            "detail": [
                {"n_fft": n, "ok": p, "fails": f} for n, p, f, _ in detail_relax
            ],
            "reason": "0.5 dB CTA 预算下无更短（或任何）通过项，放宽到 1.0 dB",
        })
        budget_used = CTA_DROP_RELAX_DB
        passing = ok_relax
        detail = detail_relax

    if not passing:
        # Fall back to baseline if nothing else passes (including if 512
        # somehow failed a self-check — it should pass).
        chosen = 512
        why = "没有配置通过不妥协门槛，维持基线 N=512"
    else:
        chosen = min(passing)  # shortest delay among passing
        why = (
            f"在 CTA 下降不超过 {budget_used:.1f} dB 且保真/LSD/音乐噪声达标的"
            f"配置里，选算法时延最短的 N={chosen}"
        )
    return chosen, budget_used, why, notes, detail


def plot_curves(table0, table80, chosen, baseline_n=512):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fp in ["/tmp/fonts/NotoSansCJKsc-Regular.ttf"]:
        if Path(fp).exists():
            from matplotlib import font_manager
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = "Noto Sans CJK SC"
    plt.rcParams["axes.unicode_minus"] = False

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ns = sorted(table80)
    delay = [table80[n]["algo_delay_ms"] for n in ns]

    def ys(table, keypath):
        out = []
        for n in ns:
            p = table[n]
            cur = p
            for k in keypath.split("."):
                cur = None if cur is None else cur.get(k)
            out.append(cur if cur is not None else math.nan)
        return out

    # --- CTA vs delay ---
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(delay, ys(table80, "single_all.cta_mean"), "o-", color="#1f4e79",
            label="λ=0.80 单说平均 CTA")
    ax.plot(delay, ys(table0, "single_all.cta_mean"), "s--", color="#7a9bb8",
            label="λ=0 GSC-only 单说平均 CTA")
    ax.plot(delay, ys(table80, "alt_driver.cta_mean"), "^-", color="#c45911",
            label="λ=0.80 主驾交替 CTA")
    ax.plot(delay, ys(table80, "alt_pax.cta_mean"), "v-", color="#548235",
            label="λ=0.80 副驾交替 CTA")
    bcta = table80[baseline_n]["single_all"]["cta_mean"]
    ax.axhline(bcta - CTA_DROP_TIGHT_DB, color="#c00000", ls=":", lw=1,
               label=f"基线 CTA − {CTA_DROP_TIGHT_DB:.1f} dB")
    ax.axhline(bcta - CTA_DROP_RELAX_DB, color="#c00000", ls="--", lw=0.8, alpha=0.5,
               label=f"基线 CTA − {CTA_DROP_RELAX_DB:.1f} dB")
    ax.axvline(table80[baseline_n]["algo_delay_ms"], color="#888", ls="--", lw=0.8,
               label="基线 32 ms")
    if chosen != baseline_n:
        ax.axvline(table80[chosen]["algo_delay_ms"], color="#2e7d32", ls="-", lw=1.2,
                   label=f"推荐 N={chosen}")
    ax.set_xlabel("算法时延（窗长 N/fs，ms）")
    ax.set_ylabel("CTA (dB)")
    ax.set_title("窗长 / 算法时延 vs 串音衰减")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p1 = FIG_DIR / "delay_vs_cta.png"
    fig.savefig(p1, dpi=140)
    plt.close()

    # --- fidelity vs delay ---
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    ax = axes[0, 0]
    ax.plot(delay, ys(table80, "single_all.eratio_mean"), "o-", color="#1f4e79",
            label="平均")
    ax.plot(delay, ys(table80, "single_all.eratio_min"), "s--", color="#c45911",
            label="最差段")
    ax.axhline(ERATIO_MEAN_MIN_DB, color="#c00000", ls=":", lw=1, label="平均门槛 −1 dB")
    ax.axhline(ERATIO_SESS_MIN_DB, color="#c00000", ls="--", lw=0.8, alpha=0.6,
               label="单段门槛 −2 dB")
    ax.set_ylabel("能量比 (dB)")
    ax.set_title("目标能量比（λ=0.80）")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(delay, ys(table80, "single_all.corr_mean"), "o-", color="#1f4e79",
            label="平均")
    ax.plot(delay, ys(table80, "single_all.corr_min"), "s--", color="#c45911",
            label="最差段")
    ax.axhline(CORR_MEAN_MIN, color="#c00000", ls=":", lw=1)
    ax.set_ylabel("相关")
    ax.set_title("目标相关（λ=0.80）")
    ax.set_ylim(0.94, 1.002)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(delay, ys(table80, "single_all.sisdr_mean"), "o-", color="#1f4e79",
            label="平均")
    ax.plot(delay, ys(table80, "single_all.sisdr_min"), "s--", color="#c45911",
            label="最差段")
    ax.axhline(SISDR_COLLAPSE_DB, color="#c00000", ls=":", lw=1, label="个位数门槛")
    ax.set_xlabel("算法时延（N/fs，ms）")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("目标 SI-SDR（λ=0.80）")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(delay, ys(table80, "single_all.lsd_mean"), "o-", color="#1f4e79",
            label="LSD 平均")
    p2m = ys(table80, "tonal_single.far_p2m_mean")
    ax2 = ax.twinx()
    ax2.plot(delay, p2m, "^-", color="#548235", label="对侧峰值/中位")
    ax.set_xlabel("算法时延（N/fs，ms）")
    ax.set_ylabel("LSD (dB)")
    ax2.set_ylabel("对侧残差 peak/median")
    ax.set_title("谱伤与音乐噪声线索（λ=0.80）")
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)

    for ax in axes.ravel():
        ax.axvline(table80[baseline_n]["algo_delay_ms"], color="#888", ls="--", lw=0.8)
        if chosen != baseline_n:
            ax.axvline(table80[chosen]["algo_delay_ms"], color="#2e7d32", ls="-", lw=1.0)

    fig.suptitle("窗长 / 算法时延 vs 本路保真", fontsize=12)
    fig.tight_layout()
    p2 = FIG_DIR / "delay_vs_fidelity.png"
    fig.savefig(p2, dpi=140)
    plt.close()
    return [str(p1), str(p2)]


def main():
    print(f"loading sessions from {PCM_ROOT}", flush=True)
    # Start from a known baseline STFT so VAD/calib and region masks
    # are identical to the published evaluation.
    C.set_stft_params(512, 128)
    sessions = C.list_sessions(PCM_ROOT)
    print(f"loaded {len(sessions)} sessions", flush=True)
    calib = C.calibrate(sessions)

    # Time-domain VAD is independent of STFT; compute once.
    for s in sessions:
        x, y = s["mic1"], s["mic2"]
        if s["cond"] == "Only driver":
            vad, *_ = C.vad_from_near(x)
        elif s["cond"] == "Only Passenger":
            vad, *_ = C.vad_from_near(y)
        else:
            vad, *_ = C.vad_from_near(x, y)
        s["vad_mask"] = vad

    # Both-talk region masks from the baseline classifier (N=512), so
    # alternate/overlap metrics compare the same time intervals.
    C.set_stft_params(512, 128)
    both_masks = {}
    for s in sessions:
        if not s["cond"].startswith("Both"):
            continue
        f, t, X1, X2 = C.stft_pair(s["mic1"], s["mic2"])
        labs, *_ = C.classify_frames(X1, X2, f)
        both_masks[s["sid"]] = E.sample_mask_from_frame_labels(len(s["mic1"]), labs)
        print(
            f"  baseline regions {s['sid']}: "
            f"alt_d={both_masks[s['sid']]['driver'].sum()/FS:.2f}s "
            f"alt_p={both_masks[s['sid']]['passenger'].sum()/FS:.2f}s "
            f"ovl={both_masks[s['sid']]['overlap'].sum()/FS:.2f}s",
            flush=True,
        )

    results = []
    tonal_rows = []

    # Process 512 first as a calibration check, then the rest.
    order = [512] + [n for n in WINDOWS if n != 512]
    for n_fft in order:
        hop = n_fft // 4
        params = C.set_stft_params(n_fft, hop)
        print(
            f"\n=== N={n_fft} hop={hop}  algo_delay={params['algo_delay_ms']:.1f} ms "
            f"OLA {params['ola_delay_ms_min']:.1f}–{params['ola_delay_ms_max']:.1f} ms ===",
            flush=True,
        )
        pack = C.prepare_rtfs(sessions, calib=calib)
        extra_base = {
            "n_fft": n_fft,
            "hop": hop,
            "algo_delay_ms": params["algo_delay_ms"],
            "hop_ms": params["hop_ms"],
            "ola_delay_ms_min": params["ola_delay_ms_min"],
            "ola_delay_ms_max": params["ola_delay_ms_max"],
        }
        for s in sessions:
            x, y = s["mic1"], s["mic2"]
            sid, cond = s["sid"], s["cond"]
            rtf_d, rtf_p = C.rtf_for_session(pack, sid)
            yd, yp, st = C.cancel_pair(x, y, rtf_d, rtf_p, pack["freqs"])
            yd_f, yp_f, pf = C.postfilter_full(x, y, yd, yp, labels=st["labels"])

            strengths = list(STRENGTHS_MAIN)
            # Always also score the extra λ values; they are cheap mixes
            # and let us retune only if 0.80 clearly hurts a short window.
            for lam in STRENGTHS_EXTRA:
                if lam not in strengths:
                    strengths.append(lam)

            for strength in strengths:
                yds = C.apply_postfilter_strength(yd, yd_f, strength)
                yps = C.apply_postfilter_strength(yp, yp_f, strength)
                meth = method_name(n_fft, hop, strength)
                extra = dict(extra_base)
                extra["strength"] = strength
                score_outputs(
                    results, s, cond, sid, meth, extra, yds, yps,
                    s["vad_mask"], both_masks.get(sid),
                )
                if cond in ("Only driver", "Only Passenger"):
                    if cond == "Only driver":
                        far, near = yps, yds
                    else:
                        far, near = yds, yps
                    fp2m, fk, fs_ = residual_tonalness(far)
                    np2m, nk, ns_ = residual_tonalness(near)
                    tonal_rows.append({
                        "cond": cond, "sid": sid, "method": meth,
                        "n_fft": n_fft, "hop": hop, "strength": strength,
                        "far_p2m": fp2m, "far_kurt": fk, "far_sparse": fs_,
                        "near_p2m": np2m, "near_kurt": nk, "near_sparse": ns_,
                    })
            print(f"  done {cond}/{sid}", flush=True)

        # Immediate check for N=512 λ=0.80 vs published 13.16
        if n_fft == 512:
            tmp = summarize_one(results, tonal_rows, 512, 128, 0.80)
            cta = tmp["single_all"]["cta_mean"]
            print(
                f"  CALIBRATION N=512 λ=0.80 single CTA mean={cta:.2f} dB "
                f"(published 13.16); eratio={tmp['single_all']['eratio_mean']:.2f}, "
                f"corr={tmp['single_all']['corr_mean']:.3f}, "
                f"lsd={tmp['single_all']['lsd_mean']:.2f}",
                flush=True,
            )
            if abs(cta - 13.16) > 0.15:
                print(
                    f"  WARNING: CTA {cta:.2f} is off the published 13.16 by "
                    f"{cta-13.16:+.2f} dB",
                    flush=True,
                )

    # Build tables
    table = {}  # n_fft -> strength -> pack
    for n_fft in WINDOWS:
        hop = n_fft // 4
        table[n_fft] = {}
        for strength in list(STRENGTHS_MAIN) + list(STRENGTHS_EXTRA):
            table[n_fft][strength] = summarize_one(
                results, tonal_rows, n_fft, hop, strength
            )

    table0 = {n: table[n][0.0] for n in WINDOWS}
    table80 = {n: table[n][0.80] for n in WINDOWS}

    print("\n======= 主对比：GSC + 后置 λ=0.80 =======")
    hdr = (
        f"{'N':>5} {'时延':>6} {'hop':>5} {'CTA均':>7} {'CTA中':>7} {'CTA最小':>7} "
        f"{'SI均':>7} {'SI差':>7} {'相关均':>7} {'相关差':>7} "
        f"{'能比均':>7} {'能比差':>7} {'LSD均':>7} {'p2m':>7}"
    )
    print(hdr)
    for n in WINDOWS:
        p = table80[n]
        st = p["single_all"]
        ton = p["tonal_single"] or {}
        print(
            f"{n:5d} {p['algo_delay_ms']:6.1f} {p['hop']:5d} "
            f"{st['cta_mean']:7.2f} {st['cta_med']:7.2f} {st['cta_min']:7.2f} "
            f"{st['sisdr_mean']:7.2f} {st['sisdr_min']:7.2f} "
            f"{st['corr_mean']:7.3f} {st['corr_min']:7.3f} "
            f"{st['eratio_mean']:7.2f} {st['eratio_min']:7.2f} "
            f"{st['lsd_mean']:7.2f} {ton.get('far_p2m_mean', float('nan')):7.2f}"
        )

    print("\n======= 参照：GSC-only λ=0 =======")
    print(hdr)
    for n in WINDOWS:
        p = table0[n]
        st = p["single_all"]
        ton = p["tonal_single"] or {}
        print(
            f"{n:5d} {p['algo_delay_ms']:6.1f} {p['hop']:5d} "
            f"{st['cta_mean']:7.2f} {st['cta_med']:7.2f} {st['cta_min']:7.2f} "
            f"{st['sisdr_mean']:7.2f} {st['sisdr_min']:7.2f} "
            f"{st['corr_mean']:7.3f} {st['corr_min']:7.3f} "
            f"{st['eratio_mean']:7.2f} {st['eratio_min']:7.2f} "
            f"{st['lsd_mean']:7.2f} {ton.get('far_p2m_mean', float('nan')):7.2f}"
        )

    print("\n======= 只主驾 / 只副驾 / 交替（λ=0.80）=======")
    for n in WINDOWS:
        p = table80[n]
        d, x, ad, ap = p["single_driver"], p["single_pax"], p["alt_driver"], p["alt_pax"]
        ov = p["overlap"]
        print(
            f"N={n}  只主驾 CTA={d['cta_mean']:.2f} SI={d['sisdr_mean']:.2f} "
            f"Er={d['eratio_mean']:.2f} LSD={d['lsd_mean']:.2f} | "
            f"只副驾 CTA={x['cta_mean']:.2f} SI={x['sisdr_mean']:.2f} "
            f"Er={x['eratio_mean']:.2f} LSD={x['lsd_mean']:.2f} | "
            f"交替主 CTA={ad['cta_mean']:.2f} SI={ad['sisdr_mean']:.2f} Er={ad['eratio_mean']:.2f} / "
            f"交替副 CTA={ap['cta_mean']:.2f} SI={ap['sisdr_mean']:.2f} Er={ap['eratio_mean']:.2f}"
            + (f" | 重叠主掉={ov['yd_drop_mean']:.2f} 副掉={ov['yp_drop_mean']:.2f}" if ov else "")
        )

    chosen, budget_used, why, notes, detail = choose(table80, WINDOWS)
    print(f"\nCHOSEN N={chosen} hop={chosen//4}  delay={chosen/FS*1000:.1f} ms")
    print(f"  budget={budget_used} dB; {why}")
    for n, passed, fails, pack in detail:
        tag = "PASS" if passed else "FAIL"
        print(f"  {tag} N={n}: " + ("; ".join(fails) if fails else "ok"))

    # If a shorter window failed only because λ=0.80 hurt the target /
    # musical noise, peek at extra λ for that N (appendix, not the main pick).
    lambda_notes = []
    for n in WINDOWS:
        if n >= 512:
            continue
        pack80 = table80[n]
        ok80, fails80 = judge(pack80, table80[512], CTA_DROP_TIGHT_DB)
        if ok80:
            continue
        hurt = any(("能量比" in f or "相关" in f or "SI-SDR" in f or "LSD" in f
                    or "峰值/中位数" in f) for f in fails80)
        cta_only = all("CTA" in f for f in fails80)
        if not hurt or cta_only:
            continue
        for lam in STRENGTHS_EXTRA:
            pack = table[n][lam]
            ok, fails = judge(pack, table80[512], CTA_DROP_TIGHT_DB)
            lambda_notes.append({
                "n_fft": n, "strength": lam, "ok_tight": ok, "fails": fails,
                "cta_mean": pack["single_all"]["cta_mean"],
                "eratio_mean": pack["single_all"]["eratio_mean"],
                "lsd_mean": pack["single_all"]["lsd_mean"],
            })
            print(
                f"  extra λ={lam:.2f} at N={n}: "
                + ("PASS" if ok else "FAIL " + "; ".join(fails))
            )

    fig_paths = plot_curves(table0, table80, chosen)

    # Write CSV
    keys = [
        "n_fft", "hop", "algo_delay_ms", "hop_ms", "ola_delay_ms_min", "ola_delay_ms_max",
        "strength", "cond", "sid", "method", "region",
        "cta_db", "cta_in_db", "cta_out_db",
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

    tonal_csv = HERE / "summary_window_tonal.csv"
    tkeys = ["n_fft", "hop", "strength", "cond", "sid", "method",
             "far_p2m", "far_kurt", "far_sparse", "near_p2m", "near_kurt", "near_sparse"]
    with tonal_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tkeys, extrasaction="ignore")
        w.writeheader()
        for row in tonal_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]) for k in tkeys})
    print(f"wrote {tonal_csv}", flush=True)

    def slim(p):
        if p is None:
            return None
        return {k: v for k, v in p.items() if not str(k).startswith("rows")}

    payload = {
        "windows": WINDOWS,
        "hop_rule": "N/4",
        "strengths_main": list(STRENGTHS_MAIN),
        "strengths_extra": list(STRENGTHS_EXTRA),
        "metric_stft": {"n_fft": METRIC_N, "hop": METRIC_HOP, "fixed": True},
        "region_masks": "baseline N=512 classifier, reused for all windows",
        "delay_convention": {
            "algo_delay_ms": "N/fs, same as the published 32 ms (full analysis window)",
            "ola_delay": "after hop-OLA, a hop block completes when the current frame arrives; per-sample delay in that block is (N-H)/fs to (N-1)/fs",
        },
        "criteria": {
            "cta_drop_tight_db": CTA_DROP_TIGHT_DB,
            "cta_drop_relax_db": CTA_DROP_RELAX_DB,
            "eratio_mean_min_db": ERATIO_MEAN_MIN_DB,
            "eratio_session_min_db": ERATIO_SESS_MIN_DB,
            "corr_mean_min": CORR_MEAN_MIN,
            "corr_session_min": CORR_SESS_MIN,
            "sisdr_collapse_db": SISDR_COLLAPSE_DB,
            "alt_corr_mean_min": ALT_CORR_MEAN_MIN,
            "lsd_mean_slack_db": LSD_MEAN_SLACK_DB,
            "p2m_rel_slack": P2M_REL_SLACK,
            "p2m_abs_slack": P2M_ABS_SLACK,
        },
        "baseline": {"n_fft": 512, "hop": 128, "strength": 0.80},
        "chosen": {
            "n_fft": chosen,
            "hop": chosen // 4,
            "algo_delay_ms": chosen / FS * 1000.0,
            "budget_used_db": budget_used,
            "why": why,
        },
        "judge_notes": notes,
        "lambda_notes": lambda_notes,
        "figures": fig_paths,
        "table_l0": {str(n): slim(table0[n]) for n in WINDOWS},
        "table_l80": {str(n): slim(table80[n]) for n in WINDOWS},
        "table_extra": {
            str(n): {f"{lam:.2f}": slim(table[n][lam]) for lam in STRENGTHS_EXTRA}
            for n in WINDOWS
        },
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {JSON_PATH}", flush=True)
    # Restore baseline params in the module.
    C.set_stft_params(512, 128)
    return chosen, table80, table0


if __name__ == "__main__":
    main()
