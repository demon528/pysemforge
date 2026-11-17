"""pysemforge - Structural Equation Modeling for Python.

A lavaan-style Python interface for confirmatory factor analysis and
measurement invariance: formula-based model specification, multigroup
estimation, the configural → metric → scalar invariance ladder, partial
invariance diagnostics, and publication-ready path diagrams drawn natively
with Matplotlib (no Graphviz required). Maximum likelihood estimation is
delegated to semopy; the invariance ladder and the diagrams are implemented
here.

Quick start::

    from pysemforge import Model

    model = Model('''
        visual =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
    ''')
    result = model.fit(data)
    print(result.summary())

Proposal reference: see the accompanying design document (Section 1.3).
"""

from __future__ import annotations

from .model import Model
from .result import SEMResult, InvarianceResult
from .multigroup import (
    fit_multigroup,
    invariance_test,
    modification_indices,
    partial_invariance,
)
from .parser import parse_lavaan, collect_variables
from .estim import (
    fit_model,
    compute_fit_indices,
    polychoric_correlation,
    robust_se,
)
from . import datasets, diagnostics

__all__ = [
    "Model",
    "SEMResult",
    "InvarianceResult",
    "fit_multigroup",
    "invariance_test",
    "modification_indices",
    "partial_invariance",
    "parse_lavaan",
    "collect_variables",
    "fit_model",
    "compute_fit_indices",
    "polychoric_correlation",
    "robust_se",
    "datasets",
    "diagnostics",
]

try:
    from . import __about__
    __version__ = __about__.__version__
except Exception:  # pragma: no cover
    __version__ = "0.1.1"
