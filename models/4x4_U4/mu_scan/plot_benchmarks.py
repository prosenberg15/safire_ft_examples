import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import matplotlib

matplotlib.use('TkAgg')

MU_SHIFT = 2.0   # safire mu is shifted by -2 from the DQMC mu

# ----------------------------------------------------------------------
# Column layout (shared by every file):
#   0:mu 1:Etot 2:Etot_err 3:E_U 4:E_U_err 5:E_K 6:E_K_err
#   7:ntot 8:ntot_err 9:nn_corr 10:nn_corr_err 11:szsz_corr 12:szsz_corr_err
# ----------------------------------------------------------------------
def load(fname,shift=False):
    d = np.loadtxt(fname,skiprows=1)
    if(shift):
        return {"mu": d[:, 0] + MU_SHIFT, "d": d}
    else:
        return {"mu": d[:, 0], "d": d}
# ----------------------------------------------------------------------
# Register datasets here. method: "FT-AFQMC" or "DQMC"; beta: label value.
# ----------------------------------------------------------------------
datasets = [
    dict(file="Beta2/props_vs_mu",      method="FT-AFQMC", beta=2.0),
    dict(file="Beta5/props_vs_mu",      method="FT-AFQMC", beta=5.0),
    dict(file="props_vs_mu_Beta2_DQMC", method="DQMC",     beta=2.0),
    dict(file="props_vs_mu_Beta5_DQMC", method="DQMC",     beta=5.0),
]
for ds in datasets:
    if ds["method"] == "FT-AFQMC":
        shift = True
    else:
        shift = False
    ds.update(load(ds["file"],shift))

# Style: color encodes beta, marker/fill encodes method (matches the paper).
beta_color = {2.0: "C3", 5.0: "C0"}
def style(ds):
    c = beta_color.get(ds["beta"], "k")
    if ds["method"] == "FT-AFQMC":
        return dict(marker="o", ms=5, mfc="none", mec=c, color=c, lw=1.2, capsize=2)
    else:  # DQMC: filled markers, no connecting line
        return dict(marker="s", ms=4, mfc=c, mec=c, color=c, lw=0, capsize=2)

def label(ds):
    return rf"{ds['method']} $\beta t={ds['beta']:g}$"

# ----------------------------------------------------------------------
# Inset corner -> [x0, y0, w, h] in axes fraction coordinates.
# ----------------------------------------------------------------------
INSET_W, INSET_H = 0.4, 0.35
INSET_LOC = {
    "lower left":  [0.13,           0.13,           INSET_W, INSET_H],
    "lower right": [0.97 - INSET_W, 0.13,           INSET_W, INSET_H],
    "upper left":  [0.13,           0.97 - INSET_H, INSET_W, INSET_H],
    "upper right": [0.97 - INSET_W, 0.97 - INSET_H, INSET_W, INSET_H],
}

ylims = { 0: [-2.85,-0.95],
          1: [0.48,1.02],
          2: [0.0,0.31],
          3: [-0.062,-0.0148]
}

panels = [
    dict(col=1,  err=2,  ylabel=r"$\langle H\rangle / N_s$",                        no_xlabel=True,  yside="left", inset_loc="lower right", tag=0),
    dict(col=7,  err=8,  ylabel=r"$\langle n\rangle$",                              no_xlabel=True,  yside="right", inset_loc="lower left", tag=1),
    dict(col=9,  err=10, ylabel=r"$\langle n_{i\uparrow} n_{i+1\downarrow}\rangle$", no_xlabel=False, yside="left", inset_loc="lower left", tag=2),
    dict(col=11, err=12, ylabel=r"$\langle s^z_i s^z_{i+1}\rangle$",                no_xlabel=False, yside="right", inset_loc="lower right", tag=3),
]

def find(method, beta):
    for ds in datasets:
        if ds["method"] == method and ds["beta"] == beta:
            return ds
    return None

fig, axes = plt.subplots(2, 2, figsize=(9, 7))

for ax, p in zip(axes.ravel(), panels):
    for ds in datasets:
        ax.errorbar(ds["mu"], ds["d"][:, p["col"]], yerr=ds["d"][:, p["err"]],
                    label=label(ds), **style(ds))

    if not p["no_xlabel"]:
        ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel(p["ylabel"])
    ax.set_ylim(ylims[p["tag"]])
    ax.set_xlim(0.0, 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    # Move y-axis label + ticklabels to the right where requested.
    if p["yside"] == "right":
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    # Tickmarks pointing inward, on all four sides.
    if p["no_xlabel"]:
        ax.tick_params(direction="in", which="both", top=True, right=True, left=True, bottom=True, labelbottom=False, length=6)
    else:
        ax.tick_params(direction="in", which="both", top=True, right=True, left=True, bottom=True, length=6)

    # Relative-error inset: FT-AFQMC vs DQMC at each beta, where both exist.
    ins = None
    for beta in sorted(beta_color):
        ft, dq = find("FT-AFQMC", beta), find("DQMC", beta)
        if ft is None or dq is None:
            continue
        if ins is None:
            ins = ax.inset_axes(INSET_LOC[p["inset_loc"]])
            ins.axhline(0, color="gray", lw=0.6)
            ins.set_title(r"Rel. Err.($\times10^{-3}$)", fontsize=10)
            ins.tick_params(direction="in", which="both", labelsize=8,
                            top=True, right=True)
            ins.set_xlim(0.0, 2.0)
        ref = dq["d"][:, p["col"]]
        rel = (ft["d"][:, p["col"]] - ref) / np.abs(ref)
        ins.plot(ft["mu"], rel * 1e3, marker=".", ms=4, lw=1,
                 color=beta_color[beta], label=rf"$\beta t={beta:g}$")

axes.ravel()[0].legend(fontsize=12, loc="upper left")
fig.tight_layout()
fig.savefig("4x4_U4_FTAFQMC_vs_DQMC.png", dpi=300, bbox_inches="tight")
plt.show()
