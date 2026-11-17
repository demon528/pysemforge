"""Diagnostics for fitted models.

Only the standardised solution lives here.  Modification indices are defined
with respect to cross-group constraints and therefore need the multigroup
likelihood; they are implemented in :mod:`pysemforge.multigroup`.
"""

from __future__ import annotations

import numpy as np


def standardized_solution(estimates, latent_sd, observed_sd, pairs) -> np.ndarray:
    """Completely standardised parameter estimates (lavaan ``std.all``).

    Parameters
    ----------
    estimates : array_like
        Unstandardised estimates.
    latent_sd, observed_sd : dict
        Standard deviations of the latent and observed variables.
    pairs : sequence of tuple
        For each estimate, the ``(from, to)`` variables it connects.

    Returns
    -------
    numpy.ndarray
        Estimates divided by the product of the two variables' standard
        deviations, so that they are expressed on a correlation metric.
    """
    estimates = np.asarray(estimates, dtype=float)
    sds = {**dict(latent_sd), **dict(observed_sd)}
    out = np.empty_like(estimates)
    for i, (a, b) in enumerate(pairs):
        denom = sds.get(a, 1.0) * sds.get(b, 1.0)
        out[i] = estimates[i] / denom if denom else estimates[i]
    return out
