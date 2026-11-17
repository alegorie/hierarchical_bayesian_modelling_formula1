#!/usr/bin/env python
"""Reproducible re-download of the race lap tables with FastF1 (separate stage).

    pip install fastf1
    python scripts/fetch_fastf1.py --years 2022 2023 2024 --cache .fastf1_cache --out data/

Writes  data/f1_laps_snapshot_<YYYYMMDD>.csv  and  data/f1_laps_snapshot_<YYYYMMDD>.sessions.json
(the exact list of sessions, FastF1 version and fetch time).  The new snapshot is NEVER written
over the original `f1_cache_combined_2022_2024.csv`; if the original exists, the script prints a
row/race/column comparison so that any difference (FastF1 version, corrected timing data) is
visible before the snapshot is used.  To use the snapshot, set `data_csv` in config.yaml explicitly.

This script was not executed in the development environment (no network access); it mirrors the
column layout of the original export (RaceKey, Year, RoundNumber, EventName, OfficialEventName,
Location, Country, SessionName, SessionType, Driver, DriverNumber, Team, LapNumber, Stint, Compound,
TyreLife, FreshTyre, LapStartDate, TrackStatus, Position, IsAccurate, *_seconds columns, Deleted,
DeletedReason, FastF1Generated, IsPersonalBest).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

TIMEDELTA_COLS = ["Time", "LapTime", "PitOutTime", "PitInTime", "Sector1Time", "Sector2Time", "Sector3Time",
                  "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime", "LapStartTime"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--cache", default=".fastf1_cache")
    ap.add_argument("--out", default="data")
    ap.add_argument("--original", default="data/f1_cache_combined_2022_2024.csv")
    args = ap.parse_args()
    try:
        import fastf1
    except ImportError:
        sys.exit("pip install fastf1 first")
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    frames, sessions = [], []
    for year in args.years:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load(laps=True, telemetry=False, weather=False, messages=False)
            except Exception as e:  # noqa
                print(f"skip {year} round {rnd}: {e!r}")
                sessions.append({"year": year, "round": rnd, "event": ev["EventName"], "status": f"failed: {e!r}"})
                continue
            laps = s.laps.copy()
            laps.insert(0, "RaceKey", f"{year}::{ev['EventName']}::R")
            laps.insert(1, "Year", year); laps.insert(2, "RoundNumber", rnd); laps.insert(3, "EventName", ev["EventName"])
            laps.insert(4, "OfficialEventName", ev.get("OfficialEventName", "")); laps.insert(5, "Location", ev.get("Location", ""))
            laps.insert(6, "Country", ev.get("Country", "")); laps.insert(7, "SessionName", "Race"); laps.insert(8, "SessionType", "R")
            for c in TIMEDELTA_COLS:
                if c in laps:
                    laps[f"{c}_seconds"] = laps[c].dt.total_seconds()
            frames.append(laps)
            sessions.append({"year": year, "round": rnd, "event": ev["EventName"], "status": "ok", "laps": int(len(laps))})
            print(f"{year} R{rnd:02d} {ev['EventName']}: {len(laps)} laps")
    df = pd.concat(frames, ignore_index=True)
    stamp = time.strftime("%Y%m%d")
    out = Path(args.out) / f"f1_laps_snapshot_{stamp}.csv"
    df.to_csv(out, index=False)
    json.dump({"fastf1_version": fastf1.__version__, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "sessions": sessions}, open(out.with_suffix(".sessions.json"), "w"), indent=1)
    print(f"wrote {out} ({len(df)} rows, {df['RaceKey'].nunique()} races)")
    if Path(args.original).exists():
        orig = pd.read_csv(args.original, low_memory=False)
        print("comparison with the original CSV (NOT replaced):")
        print(f"  rows: original {len(orig)} vs snapshot {len(df)}; races: {orig['RaceKey'].nunique()} vs {df['RaceKey'].nunique()}")
        print(f"  columns only in original: {sorted(set(orig.columns) - set(df.columns))}")
        print(f"  columns only in snapshot: {sorted(set(df.columns) - set(orig.columns))}")
        for rk in sorted(set(orig["RaceKey"]) ^ set(df["RaceKey"])):
            print("  race only on one side:", rk)


if __name__ == "__main__":
    main()
