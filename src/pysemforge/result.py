"""Result containers for SEM estimation output.

Two public classes:

- :class:`SEMResult` -- returned by :meth:`pysemforge.model.Model.fit` and
  :meth:`Model.measurement_invariance`. Provides ``summary()``,
  ``to_dataframe()`` and attribute access (``.fit``, ``.result``,
  ``.multigroup``).
- :class:`InvarianceResult` -- measurement-invariance ladder output with
  ``.levels``, ``.comparisons`` and a decision-rule ``.summary()``
  (mirrors ``semTools::measurementInvariance``).
"""

from __future__ import annotations

import pandas as pd


class SEMResult:
    """Container for SEM estimation / multigroup results.

    Parameters
    ----------
    raw : dict
        Result dictionary produced by :func:`pysemforge.estim.fit_model`,
        or an aggregated multigroup dict (with key ``"groups"``).
    model : Model, optional
        The fitted model (for provenance / reporting).
    group : str, optional
        Group label when this is a per-group result.
    multigroup : bool
        Whether this is an aggregated multigroup result.
    """

    def __init__(self, raw: dict, model=None, group: str | None = None, multigroup: bool = False):
        self.result = raw  # raw dict access (used by tests / internals)
        self.model = model
        self.group = group
        self.multigroup = multigroup

    # ------------------------------ attribute accessors
    @property
    def fit(self) -> dict:
        """Model fit dictionary (single-group), or aggregated fit (multigroup)."""
        if self.multigroup:
            return self.result.get("fit", {})
        return self.result.get("fit", {})

    def summary(self) -> str:
        """Return a human-readable model / multigroup summary."""
        if self.multigroup:
            return _multigroup_summary(self.result)
        return _single_summary(self.result)

    @property
    def parameters(self) -> pd.DataFrame | None:
        """Parameter table: estimate, standardised estimate, SE, z and p.

        Populated by the semopy backend.  ``None`` when the native fallback
        estimator was used, which does not compute standard errors.
        """
        table = self.result.get("param_table")
        if isinstance(table, pd.DataFrame):
            return table
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Export key results as a :class:`pandas.DataFrame` (one row)."""
        if self.multigroup:
            rows = []
            for name, g in self.result.get("groups", {}).items():
                d = dict(g.fit) if hasattr(g, "fit") else {}
                d["group"] = name
                rows.append(d)
            return pd.DataFrame(rows)
        return pd.DataFrame([self.fit])


def _single_summary(raw: dict) -> str:
    fit = raw.get("fit", {})
    if not fit:
        return "No fit information available."
    lines = [
        "=== pysemforge Model Summary ===",
        f"  Converged : {raw.get('converged', 'n/a')}",
    ]
    # The backend wrapper does not always expose the iteration counter; showing
    # a hard zero would claim a convergence check that was not reported.
    n_iter = raw.get("n_iter")
    if n_iter:
        lines.append(f"  Iterations: {n_iter}")
    lines += [
        f"  Chi-square: {fit.get('chisq', float('nan')):.3f}  (df={fit.get('df', '?')}, p={fit.get('pvalue', float('nan')):.3g})",
        f"  CFI       : {fit.get('cfi', float('nan')):.3f}",
        f"  TLI       : {fit.get('tli', float('nan')):.3f}",
        f"  RMSEA     : {fit.get('rmsea', float('nan')):.3f}  "
        f"[{fit.get('rmsea_lower', 0):.3f}, {fit.get('rmsea_upper', 0):.3f}]",
        f"  SRMR      : {fit.get('srmr', float('nan')):.3f}",
    ]
    return "\n".join(lines)


def _multigroup_summary(raw: dict) -> str:
    fit = raw.get("fit", {})
    lines = ["=== pysemforge Multigroup Summary ==="]
    lines.append(f"  Groups    : {fit.get('n_groups', '?')}")
    for k in ("chisq", "df", "cfi", "tli", "rmsea"):
        if k in fit:
            value = fit[k]
            if isinstance(value, (int, float)):
                lines.append(f"  {k.upper():<9}: {value:.3f}")
    lines.append("")
    lines.append("  Per-group contribution:")
    for name, g in raw.get("groups", {}).items():
        gf = g.fit if hasattr(g, "fit") else g
        n = gf.get("n", "?") if isinstance(gf, dict) else "?"
        chi = gf.get("chisq", float("nan")) if isinstance(gf, dict) else float("nan")
        lines.append(f"    {name}: n={n}  chi2={chi:.2f}")
    return "\n".join(lines)


# ----------------------------------------------------------------- invariance
class InvarianceResult:
    """Result of a measurement invariance ladder (semTools parity).

    Attributes
    ----------
    levels : dict
        Mapping ``level -> fit dict`` (configural / metric / scalar).
    comparisons : dict
        Mapping ``"a_vs_b" -> {"delta_chisq", "delta_df", "pvalue", ...}``.
    decisions : dict
        Per-step decision strings (``"INVARIANT"`` / ``"NON-INVARIANT"``).
    """

    def __init__(self, levels: dict, comparisons: dict, decisions: dict | None = None):
        self.levels = levels
        self.comparisons = comparisons
        self.decisions = decisions or {}
        # Flattened mapping for dict-style access (``res["levels"]`` etc.).
        self._mapping = {
            "levels": levels,
            "comparisons": comparisons,
            "decisions": self.decisions,
        }

    def summary(self) -> str:
        """Pretty summary with Δχ² / ΔCFI and invariance decision."""
        lines = ["=== Measurement Invariance (semTools parity) ===", ""]
        header = f"{'Level':<12}{'χ²':>9}{'df':>5}{'CFI':>8}{'RMSEA':>8}"
        lines.append(header)
        prev_cfi = None
        for lvl in ("configural", "metric", "scalar"):
            if lvl not in self.levels:
                continue
            f = self.levels[lvl]
            lines.append(
                f"{lvl:<12}{f.get('chisq', float('nan')):>9.3f}"
                f"{f.get('df', '?'):>5}{f.get('cfi', float('nan')):>8.3f}"
                f"{f.get('rmsea', float('nan')):>8.3f}"
            )
            if prev_cfi is not None and "cfi" in f:
                lines[-1] += f"   ΔCFI={f['cfi'] - prev_cfi:+.4f}"
            prev_cfi = f.get("cfi")
        lines.append("")
        lines.append("--- Comparisons (Δχ², α=.05) ---")
        for step, c in self.comparisons.items():
            decision = self.decisions.get(step, "")
            lines.append(
                f"{step:<24} Δχ²={c.get('delta_chisq', float('nan')):.3f}  "
                f"Δdf={c.get('delta_df', '?')}  p={c.get('pvalue', float('nan')):.3g}  "
                f"[{decision}]"
            )
        return "\n".join(lines)

    def __getitem__(self, key):
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        # Enrich comparison dicts with their 'decision' for API parity.
        if key == "comparisons" and isinstance(value, dict):
            return {
                step: {**comp, "decision": self.decisions.get(step, "")}
                for step, comp in value.items()
            }
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self._mapping

    def to_dataframe(self) -> pd.DataFrame:
        """One row per level, with the stepwise changes on the same row.

        The first level carries no comparison, so its change columns are empty.
        The table has the shape that a manuscript's invariance table has, and
        can be written out with ``df.to_csv()`` without reassembly.
        """
        rows = []
        for lvl, f in self.levels.items():
            rows.append({"level": lvl, **f})
        df = pd.DataFrame(rows)

        change_cols = ["delta_chisq", "delta_df", "delta_cfi", "delta_rmsea", "pvalue"]
        for col in change_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64")
        df["pvalue_step"] = pd.Series(dtype="float64")

        for step, comp in self.comparisons.items():
            earlier, later = step.split("_vs_")
            mask = df["level"] == later
            if not mask.any():
                continue
            for col in ("delta_chisq", "delta_df", "delta_cfi", "delta_rmsea"):
                if col in comp:
                    df.loc[mask, col] = comp[col]
            if "pvalue" in comp:
                df.loc[mask, "pvalue_step"] = comp["pvalue"]

        ordered = ["level", "chisq", "df", "cfi", "tli", "rmsea",
                   "delta_chisq", "delta_df", "delta_cfi", "delta_rmsea",
                   "pvalue_step", "rmsea_lower", "rmsea_upper",
                   "n_free", "n_obs", "n_groups", "converged"]
        cols = [c for c in ordered if c in df.columns]
        cols += [c for c in df.columns if c not in cols and c != "pvalue"]
        return df[cols]
