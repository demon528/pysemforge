"""Reproduce Fig. 2: the partial invariance solution drawn in both schools.

The figure shows the model of Fig. 1 fitted with loadings constrained equal and
the intercepts of x3 and x7 released, once per group.  Every number in the
figure -- loadings, factor correlations, intercepts -- is read from the
multigroup fit, so the picture cannot disagree with Table 3.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysemforge import Model
from pysemforge.datasets import load_holzinger_swineford, HOLZINGER_SWINFORD_MODEL
from pysemforge.multigroup import fit_multigroup
from pysemforge.plotting.diagram import PathDiagram
from pysemforge.plotting.style import default_style

LATENTS = ["visual", "textual", "speed"]
INDICATORS = [f"x{i}" for i in range(1, 10)]
FREE = {"x3", "x7"}

data = load_holzinger_swineford()
fit = fit_multigroup(Model(HOLZINGER_SWINFORD_MODEL), data, "school",
                     level="scalar", release=frozenset({("intercept", n) for n in FREE}))

fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.4))
fig.subplots_adjust(hspace=0.02, top=0.98, bottom=0.075, left=0.01, right=0.99)
for ax, group, mat in zip(axes, fit["groups"], fit["matrices"]):
    lam, phi, tau = mat["Lambda"], mat["Phi"], mat["tau"]
    sd = np.sqrt(np.diag(mat["Sigma"]))
    load = {o: lam[i, j] * np.sqrt(phi[j, j]) / sd[i]
            for i, o in enumerate(INDICATORS)
            for j, f in enumerate(LATENTS) if lam[i, j] != 0}

    edges = [(f, o, load[o]) for o in INDICATORS for f in LATENTS if (f, o) in
             [(LATENTS[j], o) for j in range(len(LATENTS)) if lam[INDICATORS.index(o), j] != 0]]
    covs = [(LATENTS[j], LATENTS[k], phi[j, k] / np.sqrt(phi[j, j] * phi[k, k]))
            for j in range(len(LATENTS)) for k in range(j + 1, len(LATENTS))]

    # The loadings are constrained equal, so printing them twice would only
    # invite the reader to compare standardised values, which differ because
    # they are scaled by each group's SD.  The figure carries the intercepts.
    style = dict(default_style)
    style["show_coefficients"] = False
    diagram = PathDiagram(
        latents=LATENTS, observed=INDICATORS, edges=edges, covariances=covs,
        style=style, intercepts=dict(zip(INDICATORS, tau)),
        free_intercepts=FREE,
    )
    diagram.draw(ax)
    # Headroom for the panel title: the covariance arcs already occupy the top
    # of the drawing, so a title placed at the axes edge would sit on them.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 2.05)
    n = int((data["school"] == group).sum())
    ax.text((x0 + x1) / 2, y1 + 1.85, f"{group} (n = {n})",
            ha="center", va="bottom", fontsize=10)

fig.text(0.5, 0.012, "Numbers under the boxes are the estimated intercepts; "
                     "the two released in the partial model are shown in red.",
         ha="center", fontsize=8, color="#555555")
fig.savefig("fig2_partial_invariance.png", dpi=600, bbox_inches="tight",
            metadata={"Date": None})
print("written fig2_partial_invariance.png")

for i, group in enumerate(fit["groups"]):
    tau = fit["matrices"][i]["tau"]
    print(f"{group:12s} " + "  ".join(f"{o}={v:.2f}" for o, v in zip(INDICATORS, tau)))
