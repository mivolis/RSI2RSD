import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

R = pickle.load(open("results.pkl", "rb"))
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df", "#ffffff"
COL = {"II": "#2a78d6", "ID": "#eb6834", "DI": "#1baf7a", "DD": "#4a3aa7"}
TXT = {"II": "white", "ID": "white", "DI": INK, "DD": "white"}
B_COL, X_COL = "#e4e3df", "#b4b2a9"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

def boundaries(ax, bmin, bmax, kmin, kmax):
    ax.axvline(3.0, color=INK, ls="--", lw=1.3)
    bb = np.linspace(bmin, bmax, 100); ax.plot(bb, 1.5 - 0.5 * bb, color=INK, lw=1.3)
    ax.set_xlim(bmin, bmax); ax.set_ylim(kmin, kmax)

def region_map(ax, g, show_region_labels=True, title=None):
    betas, kappas = g["betas"], g["kappas"]; db, dk = betas[1] - betas[0], kappas[1] - kappas[0]
    lo, lc = g["OL"]["lab"], g["CL"]["lab"]
    for i, (b, k) in enumerate(zip(g["b"], g["k"])):
        reg = g["region"][i]
        if reg in COL: fc, hatch = COL[reg], None
        elif "X" in (lo[i], lc[i]): fc, hatch = X_COL, "////"
        else: fc, hatch = B_COL, None
        ax.add_patch(Rectangle((b - db / 2, k - dk / 2), db, dk, facecolor=fc, edgecolor=SURF, lw=0.8, hatch=hatch))
    boundaries(ax, betas[0] - db / 2, betas[-1] + db / 2, kappas[0] - dk / 2, kappas[-1] + dk / 2)
    if show_region_labels:
        for reg, (x, y) in {"II": (1.9, -1.8), "ID": (1.9, 2.2), "DI": (4.8, -2.3), "DD": (4.8, 1.6)}.items():
            ax.text(x, y, r"$\mathcal{R}_{\mathrm{%s}}$" % reg, ha="center", va="center", fontsize=13, color=TXT[reg], weight="bold")
    ax.set_xlabel(r"self-distillation sharpening $s$  ($a=0.4+0.2s$)")
    ax.set_ylabel(r"environment feedback gain $\kappa$  ($g=0.4\kappa$)")
    if title: ax.set_title(title, loc="left")

def legend_regions(fig, y=0.0, ncol=8, extra=True):
    h = [Patch(color=COL[k], label=r"$\mathcal{R}_{\mathrm{%s}}$" % k) for k in COL]
    h += [Patch(facecolor=B_COL, label="undetermined"), Patch(facecolor=X_COL, hatch="////", label="inconsistent")]
    if extra:
        h += [Line2D([], [], color=INK, ls="--", label=r"predicted OL boundary $s=3$"),
              Line2D([], [], color=INK, label=r"predicted CL boundary $\kappa=1.5-0.5s$")]
    fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))

# ------------------------------------------------------------------ Fig 2: main region map
m = R["main"]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
region_map(ax, m)
legend_regions(fig, y=-0.12, ncol=3)
fig.savefig("fig_region_map.pdf"); fig.savefig("fig_region_map.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: rho scatter + predicted vs measured
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax = axs[0]
for reg in COL:
    sel = (m["region"] == reg)
    ax.scatter(m["OL"]["rho"][sel], m["CL"]["rho"][sel], s=22, color=COL[reg], edgecolor=SURF, lw=0.8, label=r"$\mathcal{R}_{\mathrm{%s}}$" % reg, zorder=3)
sel = ~np.isin(m["region"], list(COL))
ax.scatter(m["OL"]["rho"][sel], m["CL"]["rho"][sel], s=22, color=X_COL, edgecolor=SURF, lw=0.8, label="B / X", zorder=3)
ax.axvline(1, color=INK, lw=1); ax.axhline(1, color=INK, lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.55, 1.75); ax.set_ylim(0.5, 2.9)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f")); axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.5]); ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5])
ax.set_xlabel(r"measured $\hat\varrho^{\,\mathrm{OL}}$"); ax.set_ylabel(r"measured $\hat\varrho^{\,\mathrm{CL}}$")
for (x, y, t, ha) in [(0.58, 2.6, "ID", "left"), (1.04, 2.6, "DD", "left"), (0.58, 0.52, "II", "left"), (1.7, 0.52, "DI", "right")]:
    ax.text(x, y, r"$\mathcal{R}_{\mathrm{%s}}$" % t, fontsize=12, color=INK2, ha=ha, va="bottom")
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=8)
ax.set_title(r"(a) the four quadrants are the four regions", loc="left")
ax = axs[1]
for loop, mk, z in (("CL", "s", 3), ("OL", "o", 4)):
    ax.scatter(m[f"r{loop}_pred"], m[loop]["rho"], s=14 if loop == "CL" else 26, marker=mk, facecolor=INK2 if loop == "CL" else "none",
               edgecolor=INK if loop == "OL" else INK2, lw=0.9, label=f"{loop} (441 points)", zorder=z)
lim = [0.45, 2.3]; ax.plot(lim, lim, color=MUTED, lw=1.2, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f")); axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0]); ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0])
ax.set_xlabel(r"predicted $\varrho$ from reduced coordinates"); ax.set_ylabel(r"measured $\hat\varrho$ (median over seeds)")
ax.grid(True, which="both", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.set_title(r"(b) measured vs. predicted amplification", loc="left")
fig.tight_layout(); fig.savefig("fig_rho.pdf"); fig.savefig("fig_rho.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: trajectories
tr, reps = R["traj"], R["reps"]
fig, axs = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
for j, reg in enumerate(["II", "ID", "DI", "DD"]):
    for i, loop in enumerate(["OL", "CL"]):
        ax = axs[i, j]; d = tr[reg][loop]
        for key, c, ls, lab in (("O", MUTED, "--", "operational loss $O^\\ast-O$"), ("H", INK, "-", "protected loss $H^\\ast-H$")):
            y = -d[key]; t = np.arange(y.shape[1])
            med = np.nanmedian(y, 0); lo_, hi_ = np.nanquantile(y, [0.1, 0.9], axis=0)
            ax.fill_between(t, lo_, hi_, color=c, alpha=0.12, lw=0)
            ax.plot(t, med, color=c, ls=ls, lw=1.6, label=lab)
        ax.set_yscale("log"); ax.set_ylim(1e-7, 1e13)
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        lab = d["lab"]; rho = np.nanmedian(d["rho"])
        ax.set_title(f"{loop}: label {lab},  " + r"$\hat\varrho$" + f" = {rho:.2f}", loc="left", fontsize=9)
        if j == 0: ax.set_ylabel("excess loss (log)")
        if i == 1: ax.set_xlabel("round $t$")
    b, k = reps[reg]
    axs[0, j].annotate(r"$\mathcal{R}_{\mathrm{%s}}$:  $s$=%.1f, $\kappa$=%.1f" % (reg, b, k), xy=(0, 1.22), xycoords="axes fraction",
                       fontsize=10, color=INK, weight="bold")
h = [Line2D([], [], color=MUTED, ls="--", lw=1.6, label=r"operational excess loss $O_{\rm op}-O$ (V-inputs)"),
     Line2D([], [], color=INK, lw=1.6, label=r"protected excess loss $H_{\rm op}-H$ (V- and S-inputs)")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig("fig_traj.pdf"); fig.savefig("fig_traj.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: boundary sweeps
sw = R["sweeps"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
meta = {"OL_beta": (r"OL, sweep $s$ ($\kappa$ inert)", r"$s$"),
        "CL_kappa_beta2": (r"CL, $s=2$, sweep $\kappa$", r"$\kappa$"),
        "CL_kappa_beta5": (r"CL, $s=5$, sweep $\kappa$", r"$\kappa$")}
for ax, (nm, s) in zip(axs, sw.items()):
    x = s["x"]
    ax.plot(x, s["pred"], color=MUTED, lw=1.8, label=r"predicted $\varrho$", zorder=2)
    for labv, mk, fc in (("I", "o", "none"), ("D", "o", INK), ("B", "x", INK)):
        sel = s["lab"] == labv
        if sel.any():
            ax.errorbar(x[sel], s["rho"][sel], yerr=[s["rho"][sel] - s["q"][0][sel], s["q"][1][sel] - s["rho"][sel]],
                        fmt=mk, ms=4.5, mfc=fc, mec=INK, ecolor=INK2, elinewidth=0.8, lw=0, label={"I":"measured, improving","D":"measured, deteriorating (RSD)","B":"measured, undetermined"}[labv], zorder=3)
    ax.axhline(1, color=INK, lw=0.8); ax.axvline(s["pred_cross"], color=INK, ls="--", lw=1)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    t, xl = meta[nm]
    ax.set_title(t + f"\ncrossing: measured {s['cross']:.3f}, predicted {s['pred_cross']:.3f}", loc="left", fontsize=9)
    ax.set_xlabel(xl); ax.set_ylabel(r"$\hat\varrho$")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("fig_sweeps.pdf"); fig.savefig("fig_sweeps.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig 6: ablations
ab = R["ablations"]
names = {"reference": "(a) reference (all requirements met)", "no_separability": "(b) every sample touches V and S",
         "O_sees_B": "(c) $O$ evaluated on V and S", "composition_feedback": "(d) count-only feedback",
         "frozen_OL_driver": "(e) OL driver frozen at $w_0$"}
fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))
for ax, (nm, g) in zip(axs.ravel(), ab.items()):
    region_map(ax, g, show_region_labels=False)
    s = g["summary"]; c = s["counts"]
    ax.set_title(names[nm] + f"\nII/ID/DI/DD = {c['II']}/{c['ID']}/{c['DI']}/{c['DD']},  undet. = {s['nB']},  incons. = {s['nX']}", loc="left", fontsize=9)
axs.ravel()[-1].axis("off")
h = [Patch(color=COL[k], label=r"$\mathcal{R}_{\mathrm{%s}}$" % k) for k in COL]
h += [Patch(facecolor=B_COL, label="undetermined"), Patch(facecolor=X_COL, hatch="////", label="inconsistent"),
      Line2D([], [], color=INK, ls="--", label=r"predicted OL boundary"), Line2D([], [], color=INK, label=r"predicted CL boundary")]
axs.ravel()[-1].legend(handles=h, loc="center", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig("fig_ablations.pdf"); fig.savefig("fig_ablations.png", dpi=150); plt.close(fig)
print("figures done")
