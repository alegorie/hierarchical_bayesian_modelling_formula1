"""Data loading, audit tables, cleaning and index construction.

Everything that turns the FastF1 lap table into model arrays lives here, so
that the *same* code path serves the retrospective fit, the sensitivity
variants (thresholds, minimum stint length) and the forecast pipeline.

Definitions (see README §Data):
  TyreLife           absolute tyre age in laps as reported by FastF1 (1 = first lap on the set)
  set_start_age      TyreLife of the first lap of the *raw* stint (>1 => used/scrubbed set)
  first_retained_age TyreLife of the first lap retained after cleaning
  rel_age            TyreLife - first_retained_age   (within-stint clock used by the latent state)
  dage               TyreLife increment between consecutive *retained* laps (0 at first lap);
                     equals the LapNumber increment unless the record is damaged (audited).
  fuel_kg            110 * (L_r - lap) / (L_r - 1) with L_r = max LapNumber of the race (raw)
  fuel_c             fuel_kg - 55  (deterministic mid-race centring; beta_s absorbs the constant)
  y                  LapTime - median(LapTime over cleaned laps of the race)  [retrospective]
  y_prefix           LapTime - median(LapTime over cleaned laps with LapNumber <= ceil(L_r/4)) [forecast]
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig, canonical_json

REQUIRED_COLUMNS = [
    "RaceKey", "Year", "RoundNumber", "EventName", "Location", "Driver", "Team", "LapNumber", "Stint",
    "Compound", "TyreLife", "FreshTyre", "TrackStatus", "IsAccurate", "LapTime_seconds",
    "PitOutTime_seconds", "PitInTime_seconds", "Deleted", "FastF1Generated",
]
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_COLORS = {"SOFT": "#d62728", "MEDIUM": "#e6b800", "HARD": "#7f7f7f"}
COMPOUND_MARKERS = {"SOFT": "o", "MEDIUM": "s", "HARD": "^"}


# --------------------------------------------------------------------------- #
# loading and audit
# --------------------------------------------------------------------------- #
def load_raw(path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"data file lacks required columns: {missing}")
    for c in ("Deleted", "FreshTyre", "IsAccurate", "FastF1Generated"):
        if raw[c].dtype != bool:
            raw[c] = raw[c].map({True: True, False: False, "True": True, "False": False, 1: True, 0: False}).astype("boolean").fillna(False).astype(bool)
    return raw


def audit_raw(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Audit tables for the appendix: schema, missingness, duplicates, flag
    semantics, stint integrity, driver/team mapping, race list."""
    out: Dict[str, pd.DataFrame] = {}
    schema = pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values,
                           "missing_share": raw.isna().mean().values.round(5)})
    out["schema"] = schema[schema["column"].isin(REQUIRED_COLUMNS)].reset_index(drop=True)

    key = ["RaceKey", "Driver", "LapNumber"]
    dup = raw.duplicated(subset=key, keep=False)
    out["duplicates"] = pd.DataFrame({"metric": ["duplicate (race, driver, lap) rows"], "value": [int(dup.sum())]})

    ts = raw["TrackStatus"].astype("Int64").astype(str)
    out["track_status"] = ts.value_counts().rename_axis("TrackStatus").reset_index(name="laps").head(25)
    flags = {f: float(raw[f].mean()) for f in ("Deleted", "IsAccurate", "FastF1Generated", "FreshTyre")}
    flags["LapTime_missing"] = float(raw["LapTime_seconds"].isna().mean())
    flags["Compound_missing"] = float(raw["Compound"].isna().mean())
    out["flags"] = pd.DataFrame({"flag": list(flags), "share": [round(v, 5) for v in flags.values()]})
    out["compounds"] = raw["Compound"].value_counts(dropna=False).rename_axis("Compound").reset_index(name="laps")

    # stint integrity on the raw file
    sid = raw["RaceKey"] + "::" + raw["Driver"] + "::" + raw["Stint"].astype("Int64").astype(str)
    g = raw.assign(sid=sid).dropna(subset=["Stint", "TyreLife"]).sort_values(["sid", "LapNumber"]).groupby("sid")
    const_viol = {c: int((g[c].nunique() > 1).sum()) for c in ("Compound", "Location", "Team", "Driver")}
    dlife = g["TyreLife"].diff()
    dlap = g["LapNumber"].diff()
    integ = pd.DataFrame({
        "check": ["stints", "stints with >1 Compound", "stints with >1 Location", "stints with >1 Team", "stints with >1 Driver",
                  "within-stint TyreLife increment != LapNumber increment", "within-stint TyreLife non-increasing",
                  "stints starting on used tyres (min TyreLife > 1)", "share of laps FreshTyre=False"],
        "value": [g.ngroups, const_viol["Compound"], const_viol["Location"], const_viol["Team"], const_viol["Driver"],
                  int(((dlife != dlap) & dlife.notna()).sum()), int((dlife <= 0).sum()),
                  int((g["TyreLife"].min() > 1).sum()), round(float((~raw["FreshTyre"]).mean()), 4)],
    })
    out["stint_integrity"] = integ
    out["set_start_age"] = g["TyreLife"].min().value_counts().sort_index().rename_axis("set_start_age").reset_index(name="stints").head(12)

    dt = raw.groupby(["Driver", "Year"])["Team"].agg(lambda s: "/".join(sorted(s.unique()))).unstack().fillna("-")
    out["driver_team_by_year"] = dt.reset_index()
    races = raw.groupby("RaceKey").agg(Year=("Year", "first"), Round=("RoundNumber", "first"), Location=("Location", "first"),
                                       EventName=("EventName", "first"), max_lap=("LapNumber", "max"), laps=("LapNumber", "size"),
                                       dry_laps=("Compound", lambda s: int(s.isin(COMPOUND_ORDER).sum())),
                                       wet_laps=("Compound", lambda s: int(s.isin(["INTERMEDIATE", "WET"]).sum())),
                                       green_share=("TrackStatus", lambda s: float((s.astype("Int64").astype(str) == "1").mean()))).reset_index()
    out["races"] = races.sort_values(["Year", "Round"]).reset_index(drop=True)
    # suspected shortened races: max_lap clearly below the same track's typical length
    typ = races.groupby("Location")["max_lap"].max()
    races["typical_max_lap"] = races["Location"].map(typ)
    out["suspected_shortened"] = races[races["max_lap"] < 0.9 * races["typical_max_lap"]][["RaceKey", "max_lap", "typical_max_lap"]].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def clean(raw: pd.DataFrame, dcfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Apply the cleaning rules in the fixed order; return (clean_df, steps
    table, exclusion tables).  `min_stint_laps` counts *retained* laps (not
    necessarily consecutive) — this is the rule actually used by the old code,
    now stated explicitly."""
    d = raw.copy()
    d["Constructor"] = d["Team"].replace(dcfg.team_map)
    d["TrackStatusStr"] = d["TrackStatus"].astype("Int64").astype(str)
    n0 = len(d)
    steps: List[Tuple[str, int]] = []

    def step(name, mask):
        nonlocal d
        d = d[mask]
        steps.append((name, len(d)))

    if dcfg.years is not None:
        step(f"years in {list(dcfg.years)}", d["Year"].isin(list(dcfg.years)))
    if dcfg.tracks is not None:
        step(f"tracks in {list(dcfg.tracks)}", d["Location"].isin(list(dcfg.tracks)))
    step("valid lap time", d["LapTime_seconds"].notna())
    if dcfg.drop_deleted:
        step("not Deleted", ~d["Deleted"])
    step(f"dry compound {dcfg.dry_compounds}", d["Compound"].isin(dcfg.dry_compounds))
    if dcfg.drop_pit_laps:
        step("no in-/out-laps", d["PitInTime_seconds"].isna() & d["PitOutTime_seconds"].isna())
    if dcfg.drop_lap1:
        step("drop race lap 1", d["LapNumber"] > 1)
    if dcfg.green_only:
        step("green-flag only (TrackStatus==1)", d["TrackStatusStr"] == "1")
    step("valid TyreLife & Stint", d["TyreLife"].notna() & d["Stint"].notna())
    ref = d.groupby("RaceKey")["LapTime_seconds"].transform("min")
    d = d.assign(race_fastest=ref)
    step(f"<= {dcfg.max_rel_laptime} x race-fastest clean lap", d["LapTime_seconds"] / d["race_fastest"] <= dcfg.max_rel_laptime)
    d = d.assign(stint_id=d["RaceKey"] + "::" + d["Driver"] + "::" + d["Stint"].astype(int).astype(str))
    slen = d.groupby("stint_id")["LapNumber"].transform("size")
    before_min = d.copy()
    step(f"stint >= {dcfg.min_stint_laps} retained laps", slen >= dcfg.min_stint_laps)

    tbl = pd.DataFrame(steps, columns=["step", "rows"])
    tbl.insert(0, "order", range(1, len(tbl) + 1))
    tbl["retained_share"] = (tbl["rows"] / n0).round(4)
    tbl["removed"] = (-tbl["rows"].diff()).fillna(n0 - tbl["rows"].iloc[0]).astype(int)

    # exclusions: races and stints
    all_races = raw.groupby("RaceKey").agg(Year=("Year", "first"), Location=("Location", "first"), laps=("LapNumber", "size"),
                                           dry=("Compound", lambda s: int(s.isin(COMPOUND_ORDER).sum())),
                                           green=("TrackStatus", lambda s: int((s.astype("Int64").astype(str) == "1").sum()))).reset_index()
    kept = set(d["RaceKey"].unique())
    excl_r = all_races[~all_races["RaceKey"].isin(kept)].copy()
    if dcfg.years is not None:
        excl_r = excl_r[excl_r["Year"].isin(list(dcfg.years))]
    if dcfg.tracks is not None:
        excl_r = excl_r[excl_r["Location"].isin(list(dcfg.tracks))]

    def _reason(row):
        if row["dry"] == 0:
            return "no dry-compound laps (wet race)"
        if row["green"] == 0:
            return "no green-flag laps"
        return "no stint survived cleaning"

    excl_r["reason"] = excl_r.apply(_reason, axis=1) if len(excl_r) else []
    short = before_min.groupby("stint_id").size()
    excl_s = short[short < dcfg.min_stint_laps].rename("retained_laps").reset_index()
    excl_s["reason"] = f"< {dcfg.min_stint_laps} retained laps"
    return d.copy(), tbl, {"excluded_races": excl_r.reset_index(drop=True), "excluded_short_stints": excl_s}


# --------------------------------------------------------------------------- #
# dataset / index construction
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """Model-ready arrays.  All per-lap arrays are sorted by (stint_id, LapNumber)."""
    laps: pd.DataFrame                   # per-lap table (keys + covariates + y)
    stints: pd.DataFrame                 # per-stint table (codes, labels, counts)
    races: pd.DataFrame                  # per-race table (L_r, centring constants)
    y: np.ndarray
    fuel_c: np.ndarray
    rel_age: np.ndarray
    dage: np.ndarray
    stint_codes: np.ndarray
    first_pos: np.ndarray
    inno_idx: np.ndarray                 # lap indices with dage > 0 (latent innovations)
    track_of_stint: np.ndarray
    comp_of_stint: np.ndarray
    driver_of_stint: np.ndarray
    team_of_stint: np.ndarray
    race_of_stint: np.ndarray
    levels: Dict[str, List[str]]
    fuel_b_of_race: np.ndarray           # kg per lap burned, b_r = 110/(L_r-1)
    truth: Optional[Dict[str, Any]] = None   # synthetic datasets carry their generating parameters
    source: str = "real"
    fingerprint: str = ""

    # ---- convenience ----------------------------------------------------- #
    @property
    def n_laps(self) -> int:
        return int(len(self.y))

    @property
    def n_stints(self) -> int:
        return int(len(self.first_pos))

    def sizes(self) -> Dict[str, int]:
        return {"laps": self.n_laps, "stints": self.n_stints, "innovations": int(len(self.inno_idx)),
                **{k: len(v) for k, v in self.levels.items()}}

    def track_of_lap(self) -> np.ndarray:
        return self.track_of_stint[self.stint_codes]

    def summary_counts(self) -> pd.DataFrame:
        s = self.stints
        return pd.DataFrame({"quantity": ["laps", "stints", "races", "tracks", "drivers", "constructors", "compounds", "latent innovations"],
                             "value": [self.n_laps, self.n_stints, s["race"].nunique(), len(self.levels["track"]), len(self.levels["driver"]),
                                       len(self.levels["team"]), len(self.levels["compound"]), int(len(self.inno_idx))]})


def build_dataset(clean_df: pd.DataFrame, raw: pd.DataFrame, dcfg: DataConfig, source: str = "real") -> Dataset:
    d = clean_df.copy()
    # race length from the raw file (not from retained laps) ------------------
    total_laps = raw.groupby("RaceKey")["LapNumber"].max()
    d["L_r"] = d["RaceKey"].map(total_laps).astype(float)
    d["fuel_kg"] = dcfg.fuel_start_kg * (d["L_r"] - d["LapNumber"]) / (d["L_r"] - 1.0)
    d["fuel_c"] = d["fuel_kg"] - dcfg.fuel_start_kg / 2.0          # deterministic mid-race centring
    # outcome centring: full-race median (retrospective) and first-quarter median (prefix-safe)
    d["race_median_full"] = d.groupby("RaceKey")["LapTime_seconds"].transform("median")
    q1 = d[d["LapNumber"] <= np.ceil(d["L_r"] / 4.0)].groupby("RaceKey")["LapTime_seconds"].median()
    d["race_median_q1"] = d["RaceKey"].map(q1)
    d["race_median_q1"] = d["race_median_q1"].fillna(d["race_median_full"])
    d["y"] = d["LapTime_seconds"] - d["race_median_full"]
    d["y_prefix"] = d["LapTime_seconds"] - d["race_median_q1"]
    d["race_cut_y"] = dcfg.max_rel_laptime * d["race_fastest"] - d["race_median_full"]   # truncation point on the y scale

    d = d.sort_values(["stint_id", "LapNumber"], kind="mergesort").reset_index(drop=True)
    stint_codes, stint_levels = pd.factorize(d["stint_id"], sort=False)
    sf = d.groupby("stint_id", sort=False).agg(race=("RaceKey", "first"), Year=("Year", "first"), Location=("Location", "first"),
                                               Driver=("Driver", "first"), Constructor=("Constructor", "first"), Team=("Team", "first"),
                                               Compound=("Compound", "first"), Stint=("Stint", "first"), n_laps=("LapNumber", "size"),
                                               first_retained_age=("TyreLife", "min"), last_age=("TyreLife", "max"),
                                               first_lap=("LapNumber", "min"), last_lap=("LapNumber", "max"), FreshTyre=("FreshTyre", "first"))
    assert (sf.index.values == stint_levels).all()
    # set start age from the raw stint (before cleaning)
    raw_sid = raw["RaceKey"] + "::" + raw["Driver"] + "::" + raw["Stint"].astype("Int64").astype(str)
    set_start = raw.assign(sid=raw_sid).groupby("sid")["TyreLife"].min()
    sf["set_start_age"] = sf.index.map(set_start).astype(float)
    sf["used_set"] = sf["set_start_age"] > 1

    comp_levels = [c for c in COMPOUND_ORDER if c in set(sf["Compound"])]
    comp_codes = pd.Categorical(sf["Compound"], categories=comp_levels).codes.astype(int)
    track_codes, track_levels = pd.factorize(sf["Location"], sort=True)
    driver_codes, driver_levels = pd.factorize(sf["Driver"], sort=True)
    team_codes, team_levels = pd.factorize(sf["Constructor"], sort=True)
    race_codes, race_levels = pd.factorize(sf["race"], sort=True)

    first_pos = np.r_[0, 1 + np.flatnonzero(np.diff(stint_codes))]
    rel_age = (d["TyreLife"] - d.groupby("stint_id")["TyreLife"].transform("min")).to_numpy(float)
    dage = d.groupby("stint_id")["TyreLife"].diff().fillna(0.0).to_numpy(float)
    dlap = d.groupby("stint_id")["LapNumber"].diff().fillna(0.0).to_numpy(float)
    # integrity: age increments must be positive after the first lap and equal lap increments
    bad = np.flatnonzero((dage <= 0) & (dlap > 0))
    if len(bad):
        raise ValueError(f"{len(bad)} laps have non-increasing TyreLife inside a stint — damaged records; inspect audit before fitting")
    mism = int(np.sum((dage != dlap) & (dlap > 0)))
    inno_idx = np.flatnonzero(dage > 0)
    assert len(inno_idx) == len(d) - len(first_pos)
    assert np.all(dage[first_pos] == 0.0)

    races = d.groupby("RaceKey").agg(Year=("Year", "first"), Location=("Location", "first"), L_r=("L_r", "first"),
                                     race_median_full=("race_median_full", "first"), race_median_q1=("race_median_q1", "first"),
                                     race_fastest=("race_fastest", "first"), laps=("LapNumber", "size"), stints=("stint_id", "nunique")).reset_index()
    races["fuel_b"] = dcfg.fuel_start_kg / (races["L_r"] - 1.0)
    fuel_b_of_race = races.set_index("RaceKey").loc[list(race_levels), "fuel_b"].to_numpy(float)

    sf = sf.reset_index().rename(columns={"index": "stint_id"})
    sf["stint_code"] = np.arange(len(sf))
    sf["track_code"] = track_codes
    sf["comp_code"] = comp_codes
    sf["driver_code"] = driver_codes
    sf["team_code"] = team_codes
    sf["race_code"] = race_codes
    sf["tyrelife_lap_mismatch"] = 0
    d["lap_index"] = np.arange(len(d))
    d["stint_code"] = stint_codes
    d["rel_age"] = rel_age
    d["dage"] = dage
    levels = {"stint": list(stint_levels), "track": list(track_levels), "compound": comp_levels, "driver": list(driver_levels),
              "team": list(team_levels), "race": list(race_levels)}
    ds = Dataset(laps=d, stints=sf, races=races, y=d["y"].to_numpy(float), fuel_c=d["fuel_c"].to_numpy(float), rel_age=rel_age, dage=dage,
                 stint_codes=stint_codes.astype(int), first_pos=first_pos.astype(int), inno_idx=inno_idx.astype(int),
                 track_of_stint=track_codes.astype(int), comp_of_stint=comp_codes, driver_of_stint=driver_codes.astype(int),
                 team_of_stint=team_codes.astype(int), race_of_stint=race_codes.astype(int), levels=levels,
                 fuel_b_of_race=fuel_b_of_race, source=source)
    ds.fingerprint = dataset_fingerprint(ds, dcfg)
    ds.laps.attrs["tyrelife_lap_mismatch"] = mism
    return ds


def dataset_fingerprint(ds: Dataset, dcfg: DataConfig) -> str:
    """Hash of the observation table *in model order* plus the preprocessing
    configuration and category mappings."""
    h = hashlib.sha256()
    for arr in (ds.y, ds.fuel_c, ds.rel_age, ds.dage, ds.stint_codes, ds.track_of_stint, ds.comp_of_stint,
                ds.driver_of_stint, ds.team_of_stint, ds.race_of_stint, ds.fuel_b_of_race):
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(canonical_json({"levels": ds.levels, "stint_ids": ds.levels["stint"], "dcfg": dataclasses.asdict(dcfg), "source": ds.source}).encode())
    return h.hexdigest()


def subset_by_years(clean_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    return clean_df[clean_df["Year"].isin(list(years))].copy()


def stint_ols_slopes(ds: Dataset, ycol: str = "y", gamma_fixed: Optional[float] = None, min_n: int = 3) -> pd.DataFrame:
    """Per-stint OLS slopes of y (optionally fuel-corrected with an *external*
    gamma) on rel_age.  s per lap of tyre age."""
    d = ds.laps
    yv = d[ycol].to_numpy(float)
    if gamma_fixed is not None:
        yv = yv - gamma_fixed * d["fuel_c"].to_numpy(float)
    rows = []
    for j in range(ds.n_stints):
        m = ds.stint_codes == j
        x = ds.rel_age[m]
        if m.sum() >= min_n and np.ptp(x) > 0:
            b = np.polyfit(x, yv[m], 1)[0]
            rows.append((j, b, int(m.sum())))
    out = pd.DataFrame(rows, columns=["stint_code", "slope", "n"])
    st = ds.stints.set_index("stint_code")
    out["Compound"] = st.loc[out["stint_code"], "Compound"].values
    out["Location"] = st.loc[out["stint_code"], "Location"].values
    out["race"] = st.loc[out["stint_code"], "race"].values
    return out
