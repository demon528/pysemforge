"""End-to-end tutorial: CFA -> multigroup -> measurement invariance -> plot.

Mirrors the worked example in the user guide (proposal Section 4.5).
Run from the repo root:

    python examples/cfa_invariance_tutorial.py
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove for interactive display

import numpy as np
import pandas as pd

from pysemforge import Model


def main():
    rng = np.random.default_rng(42)
    n = 800
    f = rng.standard_normal((n, 2))
    x = f @ np.array([[0.8, 0.6, 0.4, 0, 0], [0, 0, 0.3, 0.7, 0.9]])
    x += rng.standard_normal(x.shape) * 0.3
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(1, 6)])
    df["group"] = rng.choice(["A", "B"], n)

    # ---- CFA ---------------------------------------------------------
    syntax = """
        visual =~ x1 + x2 + x3
        textual =~ x4 + x5
    """
    model = Model(syntax)
    result = model.fit(df[["x1", "x2", "x3", "x4", "x5"]])
    print(result.summary())

    # ---- Path diagram (no graphviz) ---------------------------------
    model.plot("sem_path.png", standardized=True, show=False)

    # ---- Multigroup ------------------------------------------------
    mg = model.fit(df, group="group")
    print("\n[Multigroup]")
    print(mg.summary())

    # ---- Measurement invariance (configural -> metric -> scalar) ----
    inv = model.measurement_invariance(df, "group")
    print("\n[Measurement Invariance]")
    print(inv.summary())
    inv.plot("invariance.png")

    print("\nSaved: sem_path.png, invariance.png")


if __name__ == "__main__":
    main()
