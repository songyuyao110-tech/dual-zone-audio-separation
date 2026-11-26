#!/usr/bin/env python3
"""Sweep the dynamic post-filter factor β on the 15-session set.

Operating STFT is the adopted 16 ms point: N=256, hop=64.
β=0 is exactly the fixed λ=0.80 default (must reproduce ~13.33 dB CTA).
β>0 lets λ(t) follow a spatial opposite-seat activity track.

Also scores fixed λ=0.90 and λ=1.00 as references, so a dynamic win is
not just "we turned the knob up globally".
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import cancel as C
import run_postfilter_eval as E
import run_window_eval as W

HERE = Path(__file__).resolve().parent
PCM_ROOT = HERE / "pcm"
if not PCM_ROOT.exists():
    PCM_ROOT = Path(__file__).resolve().parent / "pcm"

OUT = HERE
FIG_DIR = HERE / "eval_dyn_figs"
CSV_PATH = HERE / "summary_dynamic_pf.csv"
TONAL_CSV = HERE / "summary_dynamic_pf_tonal.csv"
JSON_PATH = HERE / "dynamic_pf_sweep.json"

OPERATING_N = 256
OPERATING_HOP = 64
BETAS = (0.00, 0.25, 0.50, 0.75, 1.00)
FIXED_REFS = (0.80, 0.90, 1.00)  # constant-λ references
FS = C.FS

# Pre-declared gates vs fixed λ=0.80 at N=256. Written before looking at the table.
CTA_GAIN_NOTICEABLE_DB = 0.25
ERATIO_MEAN_MIN_DB = -1.0
ERATIO_SESS_MIN_DB = -2.0
CORR_MEAN_MIN = 0.980
CORR_SESS_MIN = 0.95
SISDR_COLLAPSE_DB = 8.0
ALT_CORR_MEAN_MIN = 0.970
LSD_MEAN_SLACK_DB = 0.25
P2M_REL_SLACK = 1.15
P2M_ABS_SLACK = 4.0
# Do not accept a "win" that gains almost no CTA but dents fidelity.
MARGINAL_CTA_DB = 0.25
MARGINAL_ER_DB = -0.20
MARGINAL_LSD_DB = 0.20


def method_dyn(beta):
    return f"dyn_b{beta:.2f}"


def method_fix(lam):
    return f"fix_l{lam:.2f}"


def pack_of(results, tonal, method):
    rs_all = [r for r in results if r["method"] == method]
    def sel(region, cond=None):
        return [r for r in rs_all if r["region"] == region and (cond is None or r["cond"] == cond)]
    return {
        "method": method,
        "single_all": W.pack_summary(sel("single")),
        "single_driver": W.pack_summary(sel("single", "Only driver")),
        "single_pax": W.pack_summary(sel("single", "Only Passenger")),
        "alt_driver": W.pack_summary(sel("both_alt_driver")),
        "alt_pax": W.pack_summary(sel("both_alt_passenger")),
        "overlap": W.overlap_pack(sel("both_overlap")),
        "tonal_single": W.tonal_pack([r for r in tonal if r["method"] == method]),
        "rows_single": sel("single"),
        "rows_alt_d": sel("both_alt_driver"),
        "rows_alt_p": sel("both_alt_passenger"),
    }


def judge(pack, baseline, require_cta_gain=False):
    """Hard gates vs fixed λ=0.80. Written before seeing the table."""
    fails = []
    st, bst = pack["single_all"], baseline["single_all"]
    if st is None or bst is None:
        return False, ["missing single-talk summary"]

    if st["eratio_mean"] < ERATIO_MEAN_MIN_DB:
        fails.append(f"单说平均能量比 {st['eratio_mean']:.2f} dB < {ERATIO_MEAN_MIN_DB:.1f}")
    if st["eratio_min"] < ERATIO_SESS_MIN_DB:
        fails.append(f"单说最差能量比 {st['eratio_min']:.2f} dB < {ERATIO_SESS_MIN_DB:.1f}")
    if st["corr_mean"] < CORR_MEAN_MIN:
        fails.append(f"单说平均相关 {st['corr_mean']:.3f} < {CORR_MEAN_MIN:.3f}")
    if st["corr_min"] < CORR_SESS_MIN:
        fails.append(f"单说最差相关 {st['corr_min']:.3f} < {CORR_SESS_MIN:.2f}")
    if st["sisdr_min"] < SISDR_COLLAPSE_DB and st["sisdr_min"] < (bst["sisdr_min"] - 3.0):
        fails.append(
            f"单说最差 SI-SDR {st['sisdr_min']:.2f} 掉进个位数/明显差于基线 {bst['sisdr_min']:.2f}"
        )
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
        fails.append(f"主驾交替平均能量比 {alt_d['eratio_mean']:.2f} dB 过低")
    if alt_p and alt_p["eratio_mean"] < ERATIO_MEAN_MIN_DB:
        fails.append(f"副驾交替平均能量比 {alt_p['eratio_mean']:.2f} dB 过低")
    if alt_d and alt_d["corr_mean"] < ALT_CORR_MEAN_MIN:
        fails.append(f"主驾交替平均相关 {alt_d['corr_mean']:.3f} 过低")
    if alt_p and alt_p["corr_mean"] < ALT_CORR_MEAN_MIN:
        fails.append(f"副驾交替平均相关 {alt_p['corr_mean']:.3f} 过低")
    if alt_d and b_ad and alt_d["sisdr_min"] < SISDR_COLLAPSE_DB and alt_d["sisdr_min"] < (b_ad["sisdr_min"] - 3.0):
        fails.append(f"主驾交替最差 SI-SDR {alt_d['sisdr_min']:.2f} 坍塌")
    if alt_p and b_ap and alt_p["sisdr_min"] < SISDR_COLLAPSE_DB and alt_p["sisdr_min"] < (b_ap["sisdr_min"] - 3.0):
        fails.append(f"副驾交替最差 SI-SDR {alt_p['sisdr_min']:.2f} 坍塌")
    if st["lsd_mean"] > bst["lsd_mean"] + LSD_MEAN_SLACK_DB:
        fails.append(
            f"单说平均 LSD {st['lsd_mean']:.2f} 明显高于基线 {bst['lsd_mean']:.2f} "
            f"(+{st['lsd_mean']-bst['lsd_mean']:.2f})"
        )
    ton, bton = pack.get("tonal_single"), baseline.get("tonal_single")
    if ton and bton:
        lim = max(bton["far_p2m_mean"] * P2M_REL_SLACK, bton["far_p2m_mean"] + P2M_ABS_SLACK)
        if ton["far_p2m_mean"] > lim:
            fails.append(
                f"对侧残差峰值/中位 {ton['far_p2m_mean']:.1f} 明显高于基线 "
                f"{bton['far_p2m_mean']:.1f}（上限 {lim:.1f}）"
            )
    # Fidelity should not clearly regress even if still above absolute floors.
    if st["eratio_mean"] < bst["eratio_mean"] - 0.40:
        fails.append(
            f"单说平均能量比相对固定 0.80 多掉 {bst['eratio_mean']-st['eratio_mean']:.2f} dB"
        )
    if require_cta_gain and (st["cta_mean"] - bst["cta_mean"]) < CTA_GAIN_NOTICEABLE_DB:
        # not a hard fail for "allowed", used only when picking a winner
        pass
    return len(fails) == 0, fails


def is_better(pack, best, baseline):
    """Among passing packs, prefer more CTA; if CTA gain is tiny, keep the milder one."""
    st, bst, ost = pack["single_all"], best["single_all"], baseline["single_all"]
    d_cta = st["cta_mean"] - bst["cta_mean"]
    # If this pack barely beats `best` in CTA but dents fidelity more, skip.
    d_er = st["eratio_mean"] - bst["eratio_mean"]
    d_lsd = st["lsd_mean"] - bst["lsd_mean"]
    if d_cta < MARGINAL_CTA_DB and (d_er < MARGINAL_ER_DB or d_lsd > MARGINAL_LSD_DB):
        return False
    # Must not lose CTA vs current best by more than a hair.
    if d_cta < -0.05:
        return False
    return d_cta > 0.05 or (abs(d_cta) <= 0.05 and d_lsd < -0.05)


def choose(dyn_table, baseline):
    notes = []
    passing = []
    for beta, pack in dyn_table.items():
        ok, fails = judge(pack, baseline)
        notes.append({"beta": beta, "ok": ok, "fails": fails,
                      "cta": pack["single_all"]["cta_mean"] if pack["single_all"] else None})
        if ok:
            passing.append(beta)
    # Default is β=0 if nothing else is a clear improvement.
    if not passing:
        return 0.0, "没有档位通过保真门槛，维持固定 λ=0.80（β=0）", notes, passing
    # Among passing, start from β=0 (always should pass) and move up only
    # while CTA improves enough and fidelity stays healthy.
    ordered = sorted(passing)
    chosen = 0.0 if 0.0 in passing else ordered[0]
    for beta in ordered:
        if beta <= chosen:
            continue
        if is_better(dyn_table[beta], dyn_table[chosen], baseline):
            chosen = beta
        else:
            # later stronger β may still win; keep scanning
            if dyn_table[beta]["single_all"]["cta_mean"] > dyn_table[chosen]["single_all"]["cta_mean"] + 0.25:
                # bigger CTA and it passed gates — already handled by is_better;
                # if is_better said no, the CTA gain was offset by fidelity.
                pass
    # If the best passing β does not beat β=0 by a noticeable CTA (or a
    # clear fidelity win at equal CTA), stay at 0 and say so.
    if chosen > 0.0:
        d_cta = dyn_table[chosen]["single_all"]["cta_mean"] - baseline["single_all"]["cta_mean"]
        if d_cta < CTA_GAIN_NOTICEABLE_DB:
            # only keep it if fidelity clearly improved at similar CTA
            d_er = dyn_table[chosen]["single_all"]["eratio_mean"] - baseline["single_all"]["eratio_mean"]
            d_lsd = dyn_table[chosen]["single_all"]["lsd_mean"] - baseline["single_all"]["lsd_mean"]
            if not (d_er >= 0.10 or d_lsd <= -0.15):
                why = (
                    f"通过门槛的动态档最多只多压 CTA {d_cta:+.2f} dB（不足 {CTA_GAIN_NOTICEABLE_DB:.2f}），"
                    f"也没有明显更好的本路保真，维持固定 λ=0.80（β=0）"
                )
                return 0.0, why, notes, passing
        why = (
            f"在本路不被明显伤害的通过档里，β={chosen:.2f} 相对固定 0.80 "
            f"单说平均 CTA {d_cta:+.2f} dB，取该档为建议默认"
        )
        return chosen, why, notes, passing
    return 0.0, "通过门槛的档里，没有比固定 λ=0.80 更值得换成默认的选择", notes, passing


def setup_font():
    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/tmp/fonts/NotoSansCJKsc-Regular.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            name = font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_all(dyn_table, fix_table, chosen):
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    betas = sorted(dyn_table)
    cta = [dyn_table[b]["single_all"]["cta_mean"] for b in betas]
    cta_d = [dyn_table[b]["single_driver"]["cta_mean"] for b in betas]
    cta_p = [dyn_table[b]["single_pax"]["cta_mean"] for b in betas]
    alt_d = [dyn_table[b]["alt_driver"]["cta_mean"] for b in betas]
    alt_p = [dyn_table[b]["alt_pax"]["cta_mean"] for b in betas]
    base_cta = fix_table[0.80]["single_all"]["cta_mean"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(betas, cta, "o-", color="#1f4e79", label="动态 β · 单说平均 CTA")
    ax.plot(betas, cta_d, "s--", color="#5b9bd5", label="只主驾")
    ax.plot(betas, cta_p, "s--", color="#ed7d31", label="只副驾")
    ax.plot(betas, alt_d, "^-", color="#c45911", label="主驾交替")
    ax.plot(betas, alt_p, "v-", color="#548235", label="副驾交替")
    ax.axhline(base_cta, color="#888", ls="--", lw=1, label=f"固定 λ=0.80（{base_cta:.2f} dB）")
    if 0.90 in fix_table:
        ax.axhline(fix_table[0.90]["single_all"]["cta_mean"], color="#888", ls=":", lw=0.8,
                   label=f"固定 λ=0.90（{fix_table[0.90]['single_all']['cta_mean']:.2f} dB）")
    ax.axvline(chosen, color="#2e7d32", ls="-", lw=1.2, label=f"建议 β={chosen:.2f}")
    ax.set_xlabel("动态强度 β（0 = 固定 0.80）")
    ax.set_ylabel("CTA (dB)")
    ax.set_title("动态后置强度 vs 串音衰减")
    ax.set_xticks(betas)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = FIG_DIR / "beta_vs_cta.png"
    fig.savefig(p1, dpi=140)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    ax = axes[0, 0]
    ax.plot(betas, [dyn_table[b]["single_all"]["eratio_mean"] for b in betas], "o-", color="#1f4e79", label="平均")
    ax.plot(betas, [dyn_table[b]["single_all"]["eratio_min"] for b in betas], "s--", color="#c45911", label="最差段")
    ax.axhline(ERATIO_MEAN_MIN_DB, color="#c00000", ls=":", lw=1)
    ax.set_ylabel("能量比 (dB)")
    ax.set_title("目标能量比")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(betas, [dyn_table[b]["single_all"]["corr_mean"] for b in betas], "o-", color="#1f4e79", label="平均")
    ax.plot(betas, [dyn_table[b]["single_all"]["corr_min"] for b in betas], "s--", color="#c45911", label="最差段")
    ax.axhline(CORR_MEAN_MIN, color="#c00000", ls=":", lw=1)
    ax.set_ylabel("相关")
    ax.set_ylim(0.94, 1.002)
    ax.set_title("目标相关")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(betas, [dyn_table[b]["single_all"]["sisdr_mean"] for b in betas], "o-", color="#1f4e79", label="平均")
    ax.plot(betas, [dyn_table[b]["single_all"]["sisdr_min"] for b in betas], "s--", color="#c45911", label="最差段")
    ax.axhline(SISDR_COLLAPSE_DB, color="#c00000", ls=":", lw=1)
    ax.set_xlabel("动态强度 β")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("目标 SI-SDR")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(betas, [dyn_table[b]["single_all"]["lsd_mean"] for b in betas], "o-", color="#1f4e79", label="LSD 平均")
    ax2 = ax.twinx()
    p2m = [((dyn_table[b]["tonal_single"] or {}).get("far_p2m_mean", math.nan)) for b in betas]
    ax2.plot(betas, p2m, "^-", color="#548235", label="对侧峰值/中位")
    ax.set_xlabel("动态强度 β")
    ax.set_ylabel("LSD (dB)")
    ax2.set_ylabel("对侧残差 peak/median")
    ax.set_title("谱伤与音乐噪声线索")
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)

    for ax in axes.ravel():
        ax.axvline(chosen, color="#2e7d32", ls="-", lw=1.0)
        ax.set_xticks(betas)
    fig.suptitle("动态后置强度 vs 本路保真", fontsize=12)
    fig.tight_layout()
    p2 = FIG_DIR / "beta_vs_fidelity.png"
    fig.savefig(p2, dpi=140)
    plt.close()
    return [str(p1), str(p2)]


def main():
    C.set_stft_params(OPERATING_N, OPERATING_HOP)
    print(f"STFT {C.stft_params()}", flush=True)
    print(f"loading sessions from {PCM_ROOT}", flush=True)
    sessions = C.list_sessions(PCM_ROOT)
    print(f"loaded {len(sessions)} sessions", flush=True)
    calib = C.calibrate(sessions)
    pack = C.prepare_rtfs(sessions, calib=calib)

    for s in sessions:
        x, y = s["mic1"], s["mic2"]
        if s["cond"] == "Only driver":
            vad, *_ = C.vad_from_near(x)
        elif s["cond"] == "Only Passenger":
            vad, *_ = C.vad_from_near(y)
        else:
            vad, *_ = C.vad_from_near(x, y)
        s["vad_mask"] = vad

    # Region masks at the operating window (N=256), matching how the
    # service classifies. Single-talk CTA uses time-domain VAD (same as
    # the window sweep that reported 13.33 dB).
    both_masks = {}
    # Also build N=512 masks for an optional sensitivity check? Not needed
    # for the 13.33 calibration (single-talk). Use operating-window labels
    # for alternate/overlap so they match the 16 ms pipeline.
    for s in sessions:
        if not s["cond"].startswith("Both"):
            continue
        f, t, X1, X2 = C.stft_pair(s["mic1"], s["mic2"])
        labs, *_ = C.classify_frames(X1, X2, f)
        both_masks[s["sid"]] = E.sample_mask_from_frame_labels(len(s["mic1"]), labs)
        print(
            f"  regions {s['sid']}: "
            f"alt_d={both_masks[s['sid']]['driver'].sum()/FS:.2f}s "
            f"alt_p={both_masks[s['sid']]['passenger'].sum()/FS:.2f}s "
            f"ovl={both_masks[s['sid']]['overlap'].sum()/FS:.2f}s",
            flush=True,
        )

    results = []
    tonal_rows = []
    calib_diffs = []
    act_stats = []

    for s in sessions:
        x, y = s["mic1"], s["mic2"]
        sid, cond = s["sid"], s["cond"]
        rtf_d, rtf_p = C.rtf_for_session(pack, sid)
        yd, yp, st = C.cancel_pair(x, y, rtf_d, rtf_p, pack["freqs"])
        yd_f, yp_f, pf = C.postfilter_full(x, y, yd, yp, labels=st["labels"])
        yd_base, yp_base, yd_exc, yp_exc, dyn = C.dynamic_postfilter_components(
            x, y, yd, yp, rtf_d, rtf_p, labels=st["labels"], pf_state=pf,
        )
        # β=0 must match a constant-λ 0.80 time-domain (and STFT) mix.
        ydb80 = C.apply_postfilter_strength(yd, yd_f, 0.80)
        ypb80 = C.apply_postfilter_strength(yp, yp_f, 0.80)
        dmax = max(float(np.max(np.abs(yd_base - ydb80))), float(np.max(np.abs(yp_base - ypb80))))
        calib_diffs.append({"sid": sid, "max_abs_diff_vs_fixed080": dmax})
        act = dyn["activity"]
        act_stats.append({
            "sid": sid, "cond": cond,
            "a_drv_mean": float(np.mean(act["a_drv"])),
            "a_pax_mean": float(np.mean(act["a_pax"])),
            "delta_d_mean": float(np.mean(act["delta_d"])),
            "delta_p_mean": float(np.mean(act["delta_p"])),
            "delta_d_min": float(np.min(act["delta_d"])),
            "delta_d_max": float(np.max(act["delta_d"])),
            "delta_p_min": float(np.min(act["delta_p"])),
            "delta_p_max": float(np.max(act["delta_p"])),
            "both_mean": float(np.mean(act["both"])),
        })

        def score(method, yds, yps, extra):
            W.score_outputs(
                results, s, cond, sid, method, extra, yds, yps,
                s["vad_mask"], both_masks.get(sid),
            )
            if cond in ("Only driver", "Only Passenger"):
                far, near = (yps, yds) if cond == "Only driver" else (yds, yps)
                fp2m, fk, fs_ = W.residual_tonalness(far)
                np2m, nk, ns_ = W.residual_tonalness(near)
                tonal_rows.append({
                    "cond": cond, "sid": sid, "method": method,
                    "far_p2m": fp2m, "far_kurt": fk, "far_sparse": fs_,
                    "near_p2m": np2m, "near_kurt": nk, "near_sparse": ns_,
                    **extra,
                })

        for beta in BETAS:
            yds = C.apply_dynamic_beta(yd_base, yd_exc, beta)
            yps = C.apply_dynamic_beta(yp_base, yp_exc, beta)
            score(method_dyn(beta), yds, yps, {"kind": "dynamic", "beta": beta, "lambda": ""})
        for lam in FIXED_REFS:
            yds = C.apply_postfilter_strength(yd, yd_f, lam)
            yps = C.apply_postfilter_strength(yp, yp_f, lam)
            score(method_fix(lam), yds, yps, {"kind": "fixed", "beta": "", "lambda": lam})

        print(f"  done {cond}/{sid}  β=0 vs fixed 0.80 max|Δ|={dmax:.4g}", flush=True)

    max_cal = max(c["max_abs_diff_vs_fixed080"] for c in calib_diffs)
    print(f"\nCalibration: max |y_base - y(λ=0.80)| over 15 sessions = {max_cal:.6g}", flush=True)

    dyn_table = {b: pack_of(results, tonal_rows, method_dyn(b)) for b in BETAS}
    fix_table = {l: pack_of(results, tonal_rows, method_fix(l)) for l in FIXED_REFS}

    def dump_line(tag, p):
        st = p["single_all"]
        d, x = p["single_driver"], p["single_pax"]
        ad, ap = p["alt_driver"], p["alt_pax"]
        ton = p["tonal_single"] or {}
        ov = p["overlap"]
        print(
            f"{tag:>14} CTA={st['cta_mean']:6.2f} (drv {d['cta_mean']:5.2f}/pax {x['cta_mean']:5.2f}) "
            f"SImin={st['sisdr_min']:6.2f} Er={st['eratio_mean']:6.2f}/{st['eratio_min']:6.2f} "
            f"C={st['corr_mean']:.3f}/{st['corr_min']:.3f} LSD={st['lsd_mean']:.2f} "
            f"p2m={ton.get('far_p2m_mean', float('nan')):5.2f} "
            f"alt={ad['cta_mean']:.2f}/{ap['cta_mean']:.2f}"
            + (f" ovDrop={ov['yd_drop_mean']:.2f}/{ov['yp_drop_mean']:.2f}" if ov else "")
        )

    print("\n======= 动态 β =======")
    for b in BETAS:
        dump_line(f"β={b:.2f}", dyn_table[b])
    print("\n======= 固定 λ 参照 =======")
    for l in FIXED_REFS:
        dump_line(f"λ={l:.2f}", fix_table[l])

    # Calibration CTA
    cta0 = dyn_table[0.0]["single_all"]["cta_mean"]
    cta080 = fix_table[0.80]["single_all"]["cta_mean"]
    print(f"\nβ=0 CTA={cta0:.2f}  fixed 0.80 CTA={cta080:.2f}  published N=256 λ=0.80 ≈ 13.33")
    if abs(cta080 - 13.33) > 0.20:
        print(f"  WARNING: fixed 0.80 CTA {cta080:.2f} is off 13.33 by {cta080-13.33:+.2f}")
    if abs(cta0 - cta080) > 0.05:
        print(f"  WARNING: β=0 CTA differs from fixed 0.80 by {cta0-cta080:+.2f} dB")

    baseline = fix_table[0.80]
    # Use the same rows for judging β=0 vs fixed (should pass).
    chosen, why, notes, passing = choose(dyn_table, baseline)
    print(f"\nCHOSEN β={chosen:.2f}")
    print(" ", why)
    for n in notes:
        tag = "PASS" if n["ok"] else "FAIL"
        fails = ("; " + "; ".join(n["fails"])) if n["fails"] else ""
        print(f"  {tag} β={n['beta']:.2f} CTA={n['cta']:.2f}{fails}")

    figs = plot_all(dyn_table, fix_table, chosen)

    keys = [
        "kind", "beta", "lambda", "cond", "sid", "method", "region",
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

    tkeys = ["kind", "beta", "lambda", "cond", "sid", "method",
             "far_p2m", "far_kurt", "far_sparse", "near_p2m", "near_kurt", "near_sparse"]
    with TONAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tkeys, extrasaction="ignore")
        w.writeheader()
        for row in tonal_rows:
            out = {}
            for k in tkeys:
                v = row.get(k, "")
                out[k] = f"{v:.6f}" if isinstance(v, float) else v
            w.writerow(out)

    def slim(p):
        if p is None:
            return None
        return {k: v for k, v in p.items() if not str(k).startswith("rows")}

    payload = {
        "operating": {"n_fft": OPERATING_N, "hop": OPERATING_HOP, "algo_delay_ms": OPERATING_N / FS * 1000.0},
        "mapping": {
            "lambda": "λ(t) = clip(0.80 + β·δ(t), 0, 1)",
            "beta_0": "fixed λ=0.80",
            "delta": "δ>0 opposite-seat spatially dominant (toward λ=1.0); "
                     "δ<0 both seats active (toward λ=0.60); δ≈0 own-seat or silence",
            "control": "ILD + TDOA + GSC blocking-branch energy; not opposite-mic RMS",
            "lam_base": C.DYN_LAM_BASE,
            "lam_opp": C.DYN_LAM_OPP,
            "lam_ovl": C.DYN_LAM_OVL,
            "causal": True,
            "same_stft_frame": True,
        },
        "criteria": {
            "prefer": "more CTA if the target is not clearly hurt",
            "eratio_mean_min_db": ERATIO_MEAN_MIN_DB,
            "eratio_session_min_db": ERATIO_SESS_MIN_DB,
            "corr_mean_min": CORR_MEAN_MIN,
            "cta_gain_noticeable_db": CTA_GAIN_NOTICEABLE_DB,
            "lsd_mean_slack_db": LSD_MEAN_SLACK_DB,
        },
        "calibration": {
            "published_n256_l080_cta": 13.33,
            "fixed_l080_cta": cta080,
            "beta0_cta": cta0,
            "max_abs_diff_base_vs_fixed080": max_cal,
            "per_session": calib_diffs,
        },
        "chosen": {"beta": chosen, "why": why, "passing_betas": passing},
        "judge_notes": notes,
        "activity_session_means": act_stats,
        "figures": figs,
        "table_dynamic": {f"{b:.2f}": slim(dyn_table[b]) for b in BETAS},
        "table_fixed": {f"{l:.2f}": slim(fix_table[l]) for l in FIXED_REFS},
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {JSON_PATH}", flush=True)
    return chosen, dyn_table, fix_table


if __name__ == "__main__":
    main()
