"""Fuel-corrected OLS baselines.

The fuel correction uses an EXTERNAL assumption (gamma_ext s/kg, default 0.03)
— never a gamma fitted by the hierarchical model on the same data — so that the
baseline is an independent reference, not a derived quantity of the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data import Dataset
from .ppc import grouped_ols_slopes


def ols_baseline(ds: Dataset, gamma_ext: float, out_dir: Path, tag: str = "ols") -> Dict[str, Any]:
    S = ds.n_stints
    raw = grouped_ols_slopes(ds.stint_codes, ds.rel_age, ds.y, S)
    adj = grouped_ols_slopes(ds.stint_codes, ds.rel_age, ds.y - gamma_ext * ds.fuel_c, S)
    st = ds.stints.copy()
    st["raw_slope"] = raw; st["fuel_adj_slope"] = adj
    st["net_fuel_trend_s_per_lap"] = -gamma_ext * ds.fuel_b_of_race[ds.race_of_stint]   # expected slope contribution of fuel burn alone
    out_dir.mkdir(parents=True, exist_ok=True)
    st.to_csv(out_dir / f"{tag}_stint_slopes.csv", index=False)

    def summ(col, by):
        g = st.dropna(subset=[col]).groupby(by)[col]
        return pd.DataFrame({"n": g.size(), "mean": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.size()), "median": g.median(),
                             "q25": g.quantile(.25), "q75": g.quantile(.75), "share_negative": g.apply(lambda v: float((v < 0).mean()))}).reset_index()
    by_comp = pd.concat([summ("raw_slope", "Compound").assign(kind="raw"), summ("fuel_adj_slope", "Compound").assign(kind=f"fuel-adjusted (gamma={gamma_ext})")])
    by_track = pd.concat([summ("raw_slope", "Location").assign(kind="raw"), summ("fuel_adj_slope", "Location").assign(kind=f"fuel-adjusted (gamma={gamma_ext})")])
    by_tc = st.dropna(subset=["fuel_adj_slope"]).groupby(["Location", "Compound"])["fuel_adj_slope"].agg(["size", "mean", "median"]).reset_index()
    by_comp.to_csv(out_dir / f"{tag}_by_compound.csv", index=False); by_track.to_csv(out_dir / f"{tag}_by_track.csv", index=False)
    by_tc.to_csv(out_dir / f"{tag}_by_track_compound.csv", index=False)
    return {"gamma_ext": gamma_ext, "stint_csv": str(out_dir / f"{tag}_stint_slopes.csv"), "by_compound_csv": str(out_dir / f"{tag}_by_compound.csv"),
            "by_track_csv": str(out_dir / f"{tag}_by_track.csv"), "by_track_compound_csv": str(out_dir / f"{tag}_by_track_compound.csv"),
            "share_negative_raw": float(np.nanmean(raw < 0)), "share_negative_adj": float(np.nanmean(adj < 0)),
            "median_raw": float(np.nanmedian(raw)), "median_adj": float(np.nanmedian(adj))}
