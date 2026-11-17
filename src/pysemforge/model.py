"""Main Model class."""

from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .parser import parse_lavaan, collect_variables
from .estim import fit_model, polychoric_correlation
from .result import SEMResult


def _semopy_available() -> bool:
    try:
        import semopy  # noqa: F401
        return True
    except ImportError:
        return False


def _normalise_syntax(syntax: str) -> str:
    """Put one statement per line.

    Semicolons are accepted as statement separators so that a specification can
    be written on a single line, but the backend is handed newline-separated
    syntax, which is what lavaan and semopy both expect.
    """
    parts = [part.strip() for part in re.split(r"[\n;]", syntax)]
    return "\n".join(part for part in parts if part)


class Model:
    """A structural equation model specified in lavaan-style syntax.

    Parameters
    ----------
    syntax : str
        lavaan-compatible model specification.  Statements may be separated by
        newlines or by semicolons.
    """

    def __init__(self, syntax: str):
        self.syntax = _normalise_syntax(syntax)
        self.params_table = parse_lavaan(self.syntax)
        self.variables = collect_variables(self.params_table)
        self.result: Optional[SEMResult] = None
        self._last_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        data: pd.DataFrame,
        estimator: str = "ML",
        group: Optional[str] = None,
        **kwargs,
    ) -> SEMResult:
        """Fit the model to data.

        Parameters
        ----------
        data : pandas.DataFrame
            Observed indicators (rows = cases, columns = variables).
        estimator : {'ML', 'MLR', 'WLSMV'}
            Estimation method.
        group : str, optional
            Column name defining groups → multigroup fitting.

        Returns
        -------
        SEMResult
        """
        self._last_data = data

        if group is not None:
            return self._fit_multigroup(data, group, estimator, **kwargs)

        cov, mean, n_obs = self._prepare(data, estimator)
        # Prefer the semopy backend (proposal Section 3.1): it delivers
        # lavaan-comparable ML estimates and fit indices.  Falls back to the
        # pure-Python reference estimator when semopy is absent.
        if _semopy_available() and estimator.upper() != "WLSMV":
            raw = fit_model(cov, mean, n_obs, self.syntax,
                            estimator=estimator, backend="semopy", data=data)
        else:
            raw = fit_model(cov, mean, n_obs, self.params_table,
                            estimator=estimator, backend="native", data=data)
        self.result = SEMResult(raw, model=self)
        return self.result

    def _fit_multigroup(
        self,
        data: pd.DataFrame,
        group_col: str,
        estimator: str,
        **kwargs,
    ) -> SEMResult:
        """Joint multigroup fit (configural model: no cross-group constraints)."""
        from .multigroup import fit_multigroup

        if estimator.upper() != "ML":
            raise ValueError(
                "Multigroup estimation currently supports ML only; "
                "fit each group separately for other estimators."
            )

        fit = fit_multigroup(self, data, group_col, level="configural")
        groups = {
            name: {"n": fit["stats"][i]["n"], "chisq": fit["group_chisq"][i]}
            for i, name in enumerate(fit["groups"])
        }
        combined = {
            "groups": groups,
            "fit": {
                k: fit[k]
                for k in ("chisq", "df", "pvalue", "cfi", "tli", "rmsea",
                          "n_obs", "n_groups")
            },
            "multigroup": True,
            "converged": fit["converged"],
            "joint": fit,
        }
        self.result = SEMResult(combined, model=self, multigroup=True)
        return self.result

    def _prepare(
        self, data: pd.DataFrame, estimator: str
    ) -> tuple[np.ndarray, np.ndarray, int]:
        # ``self.variables["observed"]`` is a set, so iterating it gives an
        # arbitrary column order. Fit indices are order-invariant, but SRMR is
        # computed element-wise against the model-implied matrix, where a
        # permuted order turns 0.065 into 0.280. Use syntax order instead.
        cols = [v for v in self._ordered_observed() if v in data.columns]
        X = data[cols].to_numpy()
        n_obs = X.shape[0]
        if estimator.upper() == "WLSMV":
            corr = polychoric_correlation(X)
            cov = corr * np.diag(np.diag(X.T @ X / max(n_obs - 1, 1)))
        else:
            cov = np.cov(X, rowvar=False, ddof=1)
        # Keep column names so the semopy backend can match syntax variables.
        cov = pd.DataFrame(cov, index=cols, columns=cols)
        mean = np.zeros(len(cols))  # mean structure optional for MVP
        return cov, mean, n_obs

    # -------------------------------------------------------------- invariance
    def _ordered_observed(self) -> list:
        """Observed variable names in model-syntax order (never set order)."""
        latent = set(self.variables["latent"])
        out = []
        for e in self.params_table:
            for key in ("indicator", "independent", "dependent", "var1", "var2", "variable"):
                n = e.get(key)
                if n and n not in latent and n not in out:
                    out.append(n)
        return out

    def measurement_invariance(
        self,
        data: pd.DataFrame,
        group_col: str,
        levels: tuple = ("configural", "metric", "scalar"),
    ) -> "InvarianceResult":
        """Run a measurement invariance ladder (semTools parity).

        Parameters
        ----------
        data : pandas.DataFrame
        group_col : str
            Grouping column.
        levels : tuple
            Sequence of invariance levels to test. Default is the standard
            ``configural → metric → scalar`` ladder.

        Returns
        -------
        InvarianceResult
            With ``.levels`` (per-level fit) and ``.comparisons``
            (Δχ² / ΔCFI / decision) — mirrors ``semTools::measurementInvariance``.
        """
        from .multigroup import invariance_test as _invariance_test
        return _invariance_test(self, data, group_col, levels=levels)

    # ----------------------------------------------------------------- plotting
    def plot(
        self,
        filename: Optional[str] = None,
        standardized: bool = True,
        show: bool = True,
        **kwargs,
    ):
        """Draw a path diagram (no graphviz required).

        Uses the matplotlib backend from :mod:`pysemforge.plotting`.
        """
        from .plotting import plot_model

        return plot_model(
            self, filename=filename, standardized=standardized, show=show, **kwargs
        )

    # --------------------------------------------------------------- reporting
    def summary(self) -> str:
        """Return a text summary of the fitted model."""
        if self.result is None:
            return "Model has not been fitted yet. Call .fit(data) first."
        return self.result.summary()

