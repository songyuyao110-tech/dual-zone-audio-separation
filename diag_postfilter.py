"""Musical-noise / residual peakiness diagnostics + a few spectrograms."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cancel as C

HERE = Path(__file__).resolve().parent
FIG = HERE / "eval_pf_figs"
FIG.mkdir(exist_ok=True)

for fp in ["/tmp/fonts/NotoSansCJKsc-Regular.ttf"]:
    if Path(fp).exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

sessions = C.list_sessions(HERE / "pcm")
pack = C.prepare_rtfs(sessions)

def residual_tonalness(x):
    f, t, X = C.sp_stft(
        x, fs=C.FS, window=C.WIN, nperseg=C.N_FFT,
        noverlap=C.N_FFT - C.HOP, nfft=C.N_FFT,
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

examples = [
    ("Only driver", "20261126_152912", "pax"),
    ("Only driver", "20261126_152929", "pax"),
    ("Only Passenger", "20261126_153105", "drv"),
    ("Only Passenger", "20261126_153134", "drv"),
    ("Both driver and passenger", "20261126_153348", "both"),
    ("Both driver and passenger", "20261126_153322", "both"),
]

rows = []
cache = {}
for s in sessions:
    sid, cond = s["sid"], s["cond"]
    rtf_d, rtf_p = C.rtf_for_session(pack, sid)
    yd, yp, st = C.cancel_pair(s["mic1"], s["mic2"], rtf_d, rtf_p, pack["freqs"])
    ydf, ypf, pf = C.postfilter_full(s["mic1"], s["mic2"], yd, yp, labels=st["labels"])
    cache[sid] = (s, yd, yp, ydf, ypf)
    if cond == "Only driver":
        far_gsc, far_full, near_gsc, near_full = yp, ypf, yd, ydf
    elif cond == "Only Passenger":
        far_gsc, far_full, near_gsc, near_full = yd, ydf, yp, ypf
    else:
        continue
    for lam in (0.0, 0.5, 0.7, 0.8, 0.9, 1.0):
        far = C.apply_postfilter_strength(far_gsc, far_full, lam)
        near = C.apply_postfilter_strength(near_gsc, near_full, lam)
        p2m, kurt, sparse = residual_tonalness(far)
        np2m, nkurt, nsparse = residual_tonalness(near)
        rows.append((cond, sid, lam, p2m, kurt, sparse, np2m, nkurt, nsparse))

from collections import defaultdict
by = defaultdict(list)
print("FAR residual tonalness (peak/median, kurt, sparse-frac) / NEAR")
for cond, sid, lam, p2m, kurt, sparse, np2m, nkurt, nsparse in rows:
    by[lam].append((p2m, kurt, sparse, np2m, nkurt, nsparse))
    if sid in {e[1] for e in examples}:
        print(
            f"  {cond:28s} {sid} λ={lam:.1f}  "
            f"far p2m={p2m:.2f} kurt={kurt:.2f} sparse={sparse:.3f} | "
            f"near p2m={np2m:.2f} kurt={nkurt:.2f}"
        )

print("\nMean over 10 single-talk sessions:")
for lam in (0.0, 0.5, 0.7, 0.8, 0.9, 1.0):
    a = np.array(by[lam])
    print(
        f"  λ={lam:.1f}  far p2m={a[:,0].mean():.2f} kurt={a[:,1].mean():.2f} "
        f"sparse={a[:,2].mean():.3f} | near p2m={a[:,3].mean():.2f} kurt={a[:,4].mean():.2f}"
    )


def spec(ax, x, title, vmin=-90, vmax=-20):
    f, t, X = C.sp_stft(
        x, fs=C.FS, window=C.WIN, nperseg=C.N_FFT,
        noverlap=C.N_FFT - C.HOP, nfft=C.N_FFT,
        boundary=None, padded=False, return_onesided=True,
    )
    S = 20 * np.log10(np.abs(X) + 1e-12)
    ax.pcolormesh(t, f / 1000, S, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("kHz")
    ax.set_ylim(0, 8)


for cond, sid, which in examples:
    s, yd, yp, ydf, ypf = cache[sid]
    if which == "both":
        pairs = [
            (s["mic1"], "主驾原麦"),
            (C.apply_postfilter_strength(yd, ydf, 0.0), "主驾出 λ=0"),
            (C.apply_postfilter_strength(yd, ydf, 0.7), "主驾出 λ=0.7"),
            (C.apply_postfilter_strength(yd, ydf, 1.0), "主驾出 λ=1.0"),
            (s["mic2"], "副驾原麦"),
            (C.apply_postfilter_strength(yp, ypf, 0.0), "副驾出 λ=0"),
            (C.apply_postfilter_strength(yp, ypf, 0.7), "副驾出 λ=0.7"),
            (C.apply_postfilter_strength(yp, ypf, 1.0), "副驾出 λ=1.0"),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.2), sharex=True, sharey=True)
        for ax, (x, title) in zip(axes.ravel(), pairs):
            spec(ax, x, title)
        for ax in axes[1]:
            ax.set_xlabel("s")
        fig.suptitle(f"两人 {sid}", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / f"spec_both_{sid}.png")
        plt.close()
        print("wrote", FIG / f"spec_both_{sid}.png")
        continue

    if which == "pax":
        far0, far1 = yp, ypf
        near0, near1 = yd, ydf
        far_name, near_name = "副驾输出（对侧残差）", "主驾输出（本路）"
        ref_far, ref_near = s["mic2"], s["mic1"]
    else:
        far0, far1 = yd, ydf
        near0, near1 = yp, ypf
        far_name, near_name = "主驾输出（对侧残差）", "副驾输出（本路）"
        ref_far, ref_near = s["mic1"], s["mic2"]

    items = [
        (ref_far, f"{far_name.split('（')[0]}原麦"),
        (C.apply_postfilter_strength(far0, far1, 0.0), f"{far_name} λ=0 GSC"),
        (C.apply_postfilter_strength(far0, far1, 0.7), f"{far_name} λ=0.7"),
        (C.apply_postfilter_strength(far0, far1, 1.0), f"{far_name} λ=1.0"),
        (ref_near, f"{near_name.split('（')[0]}原麦"),
        (C.apply_postfilter_strength(near0, near1, 0.0), f"{near_name} λ=0 GSC"),
        (C.apply_postfilter_strength(near0, near1, 0.7), f"{near_name} λ=0.7"),
        (C.apply_postfilter_strength(near0, near1, 1.0), f"{near_name} λ=1.0"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.2), sharex=True, sharey=True)
    for ax, (x, title) in zip(axes.ravel(), items):
        spec(ax, x, title)
    for ax in axes[1]:
        ax.set_xlabel("s")
    fig.suptitle(f"{cond} {sid}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"spec_{sid}.png")
    plt.close()
    print("wrote", FIG / f"spec_{sid}.png")
