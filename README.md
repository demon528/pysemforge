# pysemforge

**Structural Equation Modeling for Python — closing the R-Python gap for behavioral and econometric modeling.**

> A Python-native SEM/CFA package inspired by **lavaan** (syntax) and **semTools** (multigroup + measurement invariance), with a **graphviz-free path diagram** backend.

---

## Why pysemforge?

Python has mature estimation backends (e.g. semopy), but still lacks a single package that combines:

| Capability | R (lavaan + semTools) | Python (semopy / pySEM) | **pysemforge** |
|---|---|---|---|
| lavaan-style syntax | ✅ | partial | ✅ |
| CFA / path / mediation | ✅ | partial | ✅ |
| Multigroup analysis (measurement models) | ✅ native | ❌ not native | ✅ **native** |
| Measurement invariance (configural → metric → scalar) | ✅ `semTools` | ❌ missing | ✅ **one-liner** |
| Partial invariance | ✅ `semTools` | ❌ | ✅ (tear-down) |
| Ordinal / WLSMV | ✅ | limited | — planned |
| Robust ML (MLR) + fit-index CIs | ✅ | partial | — planned |
| Path diagrams (no graphviz, publication-ready) | `lavaanPlot` | ⚠️ needs graphviz | ✅ **matplotlib backend** |

The two highlighted rows are the real gaps this package targets — so you **stay in Python end-to-end**.

---

## Quick Start

```python
import pandas as pd
from pysemforge import Model

# 1) lavaan-compatible syntax
syntax = """
    visual =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed =~ x7 + x8 + x9
"""

model = Model(syntax)
result = model.fit(data)          # a pandas DataFrame of your indicators
print(result.summary())

# 2) Path diagram (no graphviz needed)
model.plot("sem_path.png", standardized=True)

# 3) Multigroup + measurement invariance (the headline feature)
inv = model.measurement_invariance(data, group_col="school")
print(inv.summary())          # configural -> metric -> scalar, with dCFI/dRMSEA
```

---

## Install

```bash
pip install pysemforge
```

This pulls in NumPy, SciPy, pandas, Matplotlib and semopy (the estimation
backend). No Graphviz installation is needed: path diagrams are drawn with
Matplotlib.

To check the installation and reproduce the validation numbers from the paper:

```python
import pysemforge
from pysemforge import Model
from pysemforge.datasets import load_holzinger_swineford, HOLZINGER_SWINFORD_MODEL

data = load_holzinger_swineford()
model = Model(HOLZINGER_SWINFORD_MODEL)
print(model.fit(data).summary())                 # chi2 = 85.306, df = 24
print(model.measurement_invariance(data, "school").summary())
```

For the latest development version:

```bash
git clone https://github.com/demon528/pysemforge.git
cd pysemforge
pip install -e .
```

---

## Features

- **lavaan-style syntax** — `=~` (measurement), `~` (regression), `~~` (co)variances, comments; statements may be separated by newlines or semicolons
- **CFA and path models** through lavaan-style syntax (estimation by semopy)
- **Estimator** — normal-theory ML; robust ML and ordinal/WLSMV are on the roadmap
- **Native multigroup** estimation for measurement models (joint ML over the stacked groups)
- **Measurement invariance** — fully automated `configural → metric → scalar` ladder with Δχ² / ΔCFI / ΔRMSEA decisions (R `semTools` parity)
- **Partial invariance** — releases the constraints with the largest modification indices and reports the group difference in each released parameter
- **Fit indices** — χ², CFI, TLI, RMSEA (+90% CI), SRMR
- **graphviz-free diagrams** — publication-quality path plots via matplotlib (no system dependencies)
- **pandas / NumPy / SciPy** first-class interoperability

### Current scope

Estimation is normal-theory ML. Robust ML (MLR), ordinal/WLSMV, latent growth
models and Bayesian estimation are not implemented yet and are listed as
planned work. Multigroup estimation covers measurement models (CFA): path
models with observed-variable regressions raise `NotImplementedError` instead
of returning a silently wrong answer. The syntax is lavaan-style but not
lavaan-complete — parameter labels (`a*x1`), intercept statements (`x1 ~ 1`)
and equality constraints written in the model string are not supported; in the
multigroup estimator, cross-group equality is imposed by the model level
(`level=` / `release=`), not in the syntax.

---

## Measurement Invariance Example

```python
inv = model.measurement_invariance(data, group_col="group")

# Access each level's fit
inv.levels["configural"]
inv.levels["metric"]
inv.levels["scalar"]

# Pairwise comparisons (Δχ², ΔCFI, decision)
inv.comparisons
```

The returned object mirrors **`semTools::measurementInvariance()`** output, so R users feel at home.

---

## Documentation & Publishing

- Source: https://github.com/demon528/pysemforge
- API reference: docstrings, plus `docs/` in the repository
- Issues: https://github.com/demon528/pysemforge/issues

**Publish to PyPI** (main release target):

```bash
pip install build twine
python -m build
twine check dist/*
twine upload --repository testpypi dist/*   # test first
twine upload dist/*                         # then production
```

---

## Roadmap (post-MVP)

Latent growth curve models · Bayesian estimation · complex survey designs · full `lavaanPlot`-parity plotting.

---

## License

MIT — see [LICENSE.txt](LICENSE.txt).
