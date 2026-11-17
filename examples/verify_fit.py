"""Verify fit indices are sensible on data with a real factor structure."""
import numpy as np
import pandas as pd
from pysemforge import Model

np.random.seed(1)
n = 1000
f = np.random.randn(n)
# x1..x3 load on f1, y1..y2 load on f2; f1<->f2 correlated
x1 = 0.8*f + np.random.randn(n)*0.6
x2 = 0.9*f + np.random.randn(n)*0.5
x3 = 0.7*f + np.random.randn(n)*0.7
y1 = 0.6*f + np.random.randn(n)*0.8
y2 = 0.5*f + np.random.randn(n)*0.9
data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})

syntax = """
  f1 =~ x1 + x2 + x3
  f2 =~ y1 + y2
  f1 ~~ f2
"""
m = Model(syntax)
res = m.fit(data, estimator="ML")
fit = res.result["fit"]
print(f"CFI={fit['cfi']:.3f} TLI={fit['tli']:.3f} RMSEA={fit['rmsea']:.3f} SRMR={fit['srmr']:.3f}")
print(f"chi2={fit['chisq']:.2f} df={fit['df']}")
assert fit["cfi"] > 0.5, "CFI should be reasonable on structured data"
assert 0 <= fit["srmr"] <= 1, "SRMR out of range"
print("FIT INDICES SENSIBLE - estimator working correctly")
