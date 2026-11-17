"""Bundled example data.

``load_holzinger_swineford()`` returns the classic Holzinger & Swineford (1939)
data set as distributed with lavaan (``HolzingerSwineford1939``): 301
observations on nine continuous indicators and the two schools they were
collected in.  It is used in the paper's illustration and in the test suite.
"""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

__all__ = ["load_holzinger_swineford", "HOLZINGER_SWINFORD_MODEL"]

HOLZINGER_SWINFORD_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""


def load_holzinger_swineford() -> pd.DataFrame:
    """Return the Holzinger-Swineford data (301 rows, x1-x9 plus ``school``)."""
    path = files(__package__).joinpath("holzinger_swineford.csv")
    return pd.read_csv(path)
