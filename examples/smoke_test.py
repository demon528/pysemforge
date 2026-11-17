"""End-to-end smoke test: CFA + multigroup + measurement invariance."""
import numpy as np
import pandas as pd
from pysemforge import Model

np.random.seed(42)
n = 600

# Two-group ordinal-ish data (binary + continuous)
g = np.random.choice(["A", "B"], n)
x1 = np.random.randn(n) + 0.3 * (g == "B")
x2 = np.random.randn(n) + 0.3 * (g == "B")
x3 = np.random.randn(n)
y1 = np.random.randn(n)
y2 = np.random.randn(n)
data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2, "group": g})

syntax = """
  f1 =~ x1 + x2 + x3
  f2 =~ y1 + y2
  f1 ~~ f2
"""

print("=== CFA fit ===")
m = Model(syntax)
res = m.fit(data[["x1", "x2", "x3", "y1", "y2"]], estimator="ML")
print(res.summary())

print("\n=== Multigroup fit ===")
res_mg = m.fit(data, group="group")
print("groups:", list(res_mg.result["groups"].keys()))
print(res_mg.summary())

print("\n=== Measurement invariance ===")
inv = m.measurement_invariance(data, "group")
for level, r in inv.levels.items():
    print(f"{level:10s} CFI={r['cfi']:.3f} RMSEA={r['rmsea']:.3f}")
print("comparisons:", {k: f"Δχ²={v['delta_chisq']:.2f} p={v['pvalue']:.3f}"
                       for k, v in inv.comparisons.items()})

print("\n=== Plot (matplotlib, saved to pathdiagram.png) ===")
import matplotlib
matplotlib.use("Agg")
fig = m.plot(style="publication", show=False, filename="pathdiagram.png")
print("figure:", fig)

print("\nALL GREEN - package works end to end")
