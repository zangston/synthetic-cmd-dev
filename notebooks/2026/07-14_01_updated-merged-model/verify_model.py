from pathlib import Path
import numpy as np
import pandas as pd

UPDATED_ROOT = Path(
    "/home/wyz5rge/synthetic-cmd-dev/notebooks/2026/07-14_01_updated-merged-model/merged_baraffe_updated"
)

COLUMNS = [
    "M_init",
    "logT",
    "logL",
    "logg",
    "logT_WR",
    "M_curr",
    "phase",
    "source",
]


def read_updated_iso(path):
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=COLUMNS,
        engine="python",
    )


for grid_name in ["z015_norot", "z015_rot"]:
    path = UPDATED_ROOT / grid_name / "iso_6.00.dat"
    df = read_updated_iso(path)

    print(f"\n{grid_name}")
    print(f"Rows: {len(df)}")
    print(f"Mass range: {df['M_init'].min()}–{df['M_init'].max()} Msun")
    print(df.head(12))

    assert np.all(np.diff(df["M_init"].values) >= 0)
    assert np.isclose(df["M_init"].min(), 0.01)