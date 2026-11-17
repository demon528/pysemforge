"""Parse lavaan-compatible model syntax into a parameter table.

Supported operators
-------------------
- Measurement:   ``visual =~ x1 + x2 + x3``
- Regression:    ``y ~ x1 + x2``
- Covariance:    ``x1 ~~ x2``
- Intercept:     ``x1 ~ 1``
- Constraint:    ``b1 == b2``
- Comments:      ``# ...``
- Separators:    statements may be separated by newlines or semicolons
- Group blocks:  lines may be grouped (multigroup) by repeating the same
  syntax for each group -- handled at the Model level.
"""

from __future__ import annotations

import re
from typing import Dict, List


def _split_terms(expr: str) -> List[str]:
    """Split a rhs like 'x1 + x2 + x3' into ['x1','x2','x3']."""
    return [t.strip() for t in re.split(r"\s*\+\s*", expr) if t.strip()]


def parse_lavaan(syntax: str) -> List[Dict]:
    """Parse a lavaan-style model specification into a parameter table.

    Parameters
    ----------
    syntax : str
        The model specification (may contain multiple lines / comments).

    Returns
    -------
    list of dict
        One dict per parameter. Each dict has at least a ``type`` key plus
        operator-specific keys (``latent``, ``indicator``, ``dependent``,
        ``independent``, ``var1``, ``var2``, ``lhs``, ``rhs``).
    """
    params: List[Dict] = []
    # Statements may be separated by newlines or by semicolons, so that a
    # specification can be written on one line inside a script or a paper.
    lines = [ln.strip() for ln in re.split(r"[\n;]", syntax)]

    for raw in lines:
        # strip inline comments
        line = re.sub(r"#.*$", "", raw).strip()
        if not line:
            continue

        # Constraint:  a == b  (must come before '=' checks)
        if "==" in line:
            lhs, rhs = line.split("==", 1)
            params.append({"type": "constraint", "lhs": lhs.strip(), "rhs": rhs.strip()})
            continue

        # Measurement:  latent =~ indicators
        if "=~" in line:
            lhs, rhs = line.split("=~", 1)
            latent = lhs.strip()
            for indicator in _split_terms(rhs):
                params.append(
                    {"type": "loading", "latent": latent, "indicator": indicator}
                )
            continue

        # Covariance:  a ~~ b
        if "~~" in line:
            lhs, rhs = line.split("~~", 1)
            params.append(
                {"type": "covariance", "var1": lhs.strip(), "var2": rhs.strip()}
            )
            continue

        # Regression / intercept:  a ~ b + c   or   a ~ 1
        if "~" in line and "=~" not in line:
            lhs, rhs = line.split("~", 1)
            dependent = lhs.strip()
            terms = _split_terms(rhs)
            # Pure intercept statement "var ~ 1" → intercept parameter (used by
            # scalar measurement invariance to constrain intercepts across groups).
            if terms == ["1"]:
                params.append({"type": "intercept", "variable": dependent})
                continue
            for term in terms:
                params.append(
                    {
                        "type": "regression",
                        "dependent": dependent,
                        "independent": term,
                    }
                )
            continue

    return params


def collect_variables(params: List[Dict]) -> Dict[str, set]:
    """Collect the set of latent and observed variables from a parameter table.

    Returns a dict with keys ``"latent"`` and ``"observed"`` (plural forms
    ``"latents"`` are also provided for backwards compatibility).
    """
    latent: set = set()
    observed: set = set()
    for p in params:
        if p["type"] == "loading":
            latent.add(p["latent"])
            observed.add(p["indicator"])
        elif p["type"] == "regression":
            observed.add(p["dependent"])
            observed.add(p["independent"])
        elif p["type"] in ("covariance", "constraint"):
            observed.add(p["var1"])
            observed.add(p["var2"])
        elif p["type"] == "intercept":
            observed.add(p["variable"])
    # A latent variable that also appears in a ``~~`` statement (e.g. ``f1 ~~ f2``)
    # must not be classified as observed: it is drawn as an oval, not a box.
    observed -= latent
    return {"latent": latent, "observed": observed, "latents": latent}
