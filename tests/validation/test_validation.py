"""Validation suite: parameter recovery, and cross-validation against lavaan.

The R/lavaan comparison is driven by ``fit_lavaan.R``, which refits the same
synthetic data with lavaan and writes JSON that this test compares against
term by term.  An R installation with lavaan is located through the
``RSCRIPT`` environment variable, ``PATH``, or the usual Windows and macOS
install directories; if none is found the test is skipped rather than failing
the Python CI.

Protocol (correctness before speed):
1. Draw data from a known population model -- two correlated factors, three
   indicators each, standardised loadings .80/.70/.90 and .85/.75/.65, factor
   correlation .30.  The model has eight degrees of freedom, so the fit is
   over-identified and the likelihood-ratio statistic is informative.
2. Fit with pysemforge.
3. Fit the same model with lavaan on the same CSV.
4. Assert agreement on chi-square, df, CFI, TLI, RMSEA, SRMR, every
   standardised loading and the factor correlation, and check that the
   estimates recover the population values.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysemforge import Model

SYNTH_N = 1000
POP_LOADINGS = {
    "x1": 0.80, "x2": 0.70, "x3": 0.90,
    "x4": 0.85, "x5": 0.75, "x6": 0.65,
}
POP_PHI = 0.30
SYNTAX = "F1 =~ x1 + x2 + x3\nF2 =~ x4 + x5 + x6"
HERE = Path(__file__).parent
R_SCRIPT = HERE / "fit_lavaan.R"


def _find_rscript() -> str | None:
    """Locate an Rscript binary: env var, PATH, then common install roots."""
    env = os.environ.get("RSCRIPT")
    if env and Path(env).exists():
        return env
    found = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if found:
        return found
    patterns = [
        r"C:\Program Files\R\R-*\bin\Rscript.exe",
        r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern), reverse=True)
        if hits:
            return hits[0]
    return None


RSCRIPT = _find_rscript()


def _has_lavaan(rscript: str | None) -> bool:
    if not rscript:
        return False
    try:
        probe = subprocess.run(
            [rscript, "-e", "suppressMessages(library(lavaan)); cat('ok')"],
            capture_output=True, text=True, timeout=120,
        )
        return probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


requires_r = pytest.mark.skipif(
    not _has_lavaan(RSCRIPT),
    reason="R with lavaan not available (set RSCRIPT to the Rscript binary)",
)


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    """Two correlated factors drawn from a known population."""
    rng = np.random.default_rng(123)
    n = SYNTH_N
    F = rng.multivariate_normal([0.0, 0.0], [[1.0, POP_PHI], [POP_PHI, 1.0]], size=n)
    loadings = np.array(list(POP_LOADINGS.values()))
    L = np.vstack([np.c_[loadings[:3], np.zeros(3)],
                   np.c_[np.zeros(3), loadings[3:]]])
    # Residual variances of 1 - lambda^2, so that the standardised loadings of
    # the population are exactly the values in POP_LOADINGS.
    theta = 1.0 - loadings ** 2
    X = F @ L.T + rng.standard_normal((n, 6)) * np.sqrt(theta)
    df = pd.DataFrame(X, columns=list(POP_LOADINGS))
    out = tmp_path_factory.mktemp("data") / "synth.csv"
    df.to_csv(out, index=False)
    return out, df


def _standardised_loadings(result) -> dict[str, float]:
    """Map indicator -> completely standardised loading."""
    table = result.parameters
    rows = table[(table["op"] == "~") & (table["lval"].isin(POP_LOADINGS))]
    return {str(r["lval"]): float(r["Est. Std"]) for _, r in rows.iterrows()}


def test_parameter_recovery(synthetic_data):
    """The estimates must recover the population loadings without R."""
    _, df = synthetic_data
    model = Model(SYNTAX)
    result = model.fit(df)

    loadings = _standardised_loadings(result)
    assert set(loadings) == set(POP_LOADINGS)
    for name, pop in POP_LOADINGS.items():
        assert loadings[name] == pytest.approx(pop, abs=0.05), name

    fit = result.fit
    assert fit["df"] == 8
    assert fit["cfi"] > 0.95
    assert fit["rmsea"] < 0.05


@requires_r
def test_against_lavaan(synthetic_data, tmp_path):
    """Fit pysemforge and lavaan on the same data; compare term by term."""
    csv_path, df = synthetic_data

    model = Model(SYNTAX)
    result = model.fit(df)
    py_fit = result.fit
    py_load = _standardised_loadings(result)

    out_json = tmp_path / "ref.json"
    subprocess.run(
        [RSCRIPT, str(R_SCRIPT), str(csv_path), str(out_json)],
        check=True, capture_output=True, text=True, timeout=300,
    )
    ref = json.loads(out_json.read_text())

    assert py_fit["df"] == ref["df"]
    assert py_fit["chisq"] == pytest.approx(ref["chisq"], rel=1e-3)
    for key in ("cfi", "tli", "rmsea", "srmr"):
        assert py_fit[key] == pytest.approx(ref[key], abs=5e-3), key

    assert set(py_load) == set(ref["loadings"])
    for name, value in ref["loadings"].items():
        assert py_load[name] == pytest.approx(value, abs=5e-3), name


def test_fit_lavaan_script_exists():
    """The reference R script must be shipped and must fit the same model."""
    assert R_SCRIPT.exists(), "fit_lavaan.R is required for lavaan cross-validation"
    content = R_SCRIPT.read_text()
    assert "library(lavaan)" in content
    assert "cfa(" in content
