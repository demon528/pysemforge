"""Reproduce Tables 2 and 3 and Fig. 1 of the pysemforge paper.

Run from a directory where the outputs may be written:

    python examples/paper_replication.py
"""

import pandas as pd

from pysemforge import Model
from pysemforge.datasets import load_holzinger_swineford, HOLZINGER_SWINFORD_MODEL
from pysemforge.multigroup import modification_indices, partial_invariance

data = load_holzinger_swineford()
indicators = [f"x{i}" for i in range(1, 10)]
model = Model(HOLZINGER_SWINFORD_MODEL)

single = model.fit(data[indicators])
print(single.summary())
keys = ["chisq", "df", "cfi", "tli", "rmsea", "rmsea_lower", "rmsea_upper", "srmr"]
pd.Series({k: single.fit[k] for k in keys}).to_frame("pysemforge").to_csv("table2_fit.csv")
model.plot("fig1_hs_model.png", standardized=True, show=False, dpi=600)

ladder = model.measurement_invariance(data, group_col="school")
print(ladder.summary())
ladder.to_dataframe().to_csv("table3_invariance.csv", index=False)

print(modification_indices(model, data, "school", level="scalar")[["constraint", "mi", "epc"]])
print(partial_invariance(model, data, "school")["final"])
