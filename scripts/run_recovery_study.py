#!/usr/bin/env python
"""Multi-replicate parameter-recovery study on synthetic data (reviewer remark A2.3).

The default pipeline fits a few replicates per scenario as an example.  This script runs a larger,
resumable study and writes its own tables, so that coverage and bias can be reported with their
uncertainty.  Several workers may run in parallel on disjoint replicate ranges, each with its own
output root (one stage registry per root); `--merge` then combines their tables.

    # two workers
    python scripts/run_recovery_study.py --scenarios baseline --start 0 --stop 6  --out outputs_local/recovery_w1
    python scripts/run_recovery_study.py --scenarios baseline --start 6 --stop 12 --out outputs_local/recovery_w2
    # merge (add --tag _large for the tables of the larger design)
    python scripts/run_recovery_study.py --merge outputs_local/recovery_w1 outputs_local/recovery_w2 --out outputs_local/recovery_study

    # the same scenario on a larger synthetic data set (10 tracks x 3 seasons = 480 stints instead of 160):
    # how does the bias of the overall level depend on the sample size?  Use a separate output root.
    python scripts/run_recovery_study.py --scenarios baseline --start 0 --stop 4 --tracks 10 --seasons 3 --tag _large --out outputs_local/recovery_large
    # a synthetic data set of the size of the real one (3 000 stints, about 54 000 laps); every fit takes hours, not minutes
    python scripts/run_recovery_study.py --scenarios baseline --start 0 --stop 2 --tracks 25 --seasons 3 --drivers 20 --teams 10 --laps 8 30 --tag _realsize --out outputs_local/recovery_realsize

Every fit is checkpointed: after an interruption run the same command again.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["baseline"])
    ap.add_argument("--start", type=int, default=0, help="first replicate index")
    ap.add_argument("--stop", type=int, default=6, help="one past the last replicate index")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--tracks", type=int, default=None, help="number of synthetic tracks (default: smoke.n_tracks of the configuration, 5)")
    ap.add_argument("--seasons", type=int, default=None, help="number of synthetic seasons (default: as in the configuration, 2)")
    ap.add_argument("--drivers", type=int, default=None, help="number of synthetic drivers (default: 8)")
    ap.add_argument("--teams", type=int, default=None, help="number of synthetic teams (default: 4)")
    ap.add_argument("--laps", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="range of the stint length in laps (default: 8 16)")
    ap.add_argument("--tag", default="_study", help="suffix of the tables written by a worker (default _study)")
    ap.add_argument("--out", required=True, help="output root of this worker (or of the merged tables with --merge)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--merge", nargs="+", default=None, help="worker output roots to merge (no sampling)")
    a = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    if a.merge:
        from f1deg import recovery as RC
        frames, fits = [], []
        for w in a.merge:
            # per-fit tables are written as soon as a fit is scored, so an interrupted worker still contributes its finished fits
            for p in sorted(Path(w).glob("*/recovery/recovery_rec_*.csv")):
                frames.append(pd.read_csv(p))
            for sj in sorted(Path(w).glob("*/stages.json")):
                for name, rec in (json.load(open(sj, encoding="utf-8")).get("stages") or {}).items():
                    res = rec.get("result") or {}
                    if name.startswith("recovery_rec_") and rec.get("status") == "completed" and "scenario" in res:
                        fits.append({"fit": name[len("recovery_"):], **{k: res.get(k) for k in ("scenario", "replicate", "variant", "well_specified", "divergences", "chains", "draws",
                                                                                              "coverage_share_effects", "scalar_z_abs_max", "share_true_nu_negative")}})
        if not frames:
            raise SystemExit("nothing to merge: no */recovery/recovery_rec_*.csv under the given roots")
        # design of the workers (written by each worker; older roots have no such file -> the command-line values are recorded)
        designs = []
        for w in a.merge:
            f = Path(w) / "study_design.json"
            d = json.load(open(f, encoding="utf-8")) if f.exists() else {"warmup": a.warmup, "draws": a.draws, "chains": a.chains, "max_treedepth": a.max_treedepth, "source": "command line of --merge"}
            if d not in designs:
                designs.append(d)
        if len(designs) > 1:
            print("WARNING: the merged workers used different designs:", designs, file=sys.stderr)
        allrec = pd.concat(frames, ignore_index=True)
        allrec["element"] = allrec["element"].fillna("").astype(str)
        allrec = allrec.drop_duplicates(subset=["scenario", "replicate", "variant", "parameter", "element"])
        out = Path(a.out) / "tables"; out.mkdir(parents=True, exist_ok=True)
        allrec.to_csv(out / f"synthetic_recovery{a.tag}.csv", index=False)
        summ = RC.recovery_summary(allrec); summ.to_csv(out / f"synthetic_recovery_summary{a.tag}.csv", index=False)
        if fits:
            pd.DataFrame(fits).drop_duplicates(subset=["fit"]).sort_values(["scenario", "replicate", "variant"]).to_csv(out / f"synthetic_recovery_fits{a.tag}.csv", index=False)
        eff = allrec[(allrec["element"] != "") & allrec["well_specified"]]
        n, k = len(eff), int(eff["covered"].sum())
        nfit = allrec.loc[allrec["well_specified"], ["scenario", "replicate", "variant"]].drop_duplicates()
        cov = {"n_elements": n, "covered": k, "share": k / n if n else None, "binomial_se": float(np.sqrt(k / n * (1 - k / n) / n)) if n else None,
               "n_fits": int(len(nfit)), "fits_per_scenario": {str(key): int(v) for key, v in allrec[["scenario", "replicate", "variant"]].drop_duplicates().groupby(["scenario", "variant"]).size().items()},
               "sampler": designs[0] if len(designs) == 1 else designs, "workers": list(a.merge),
               "note": "лише коректно специфіковані оцінювання; елементи однієї репліки корельовані, тому SE наближена"}
        json.dump(cov, open(out / f"synthetic_recovery_coverage{a.tag}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(json.dumps(cov, indent=1, ensure_ascii=False)); print(summ.to_string(index=False))
        return 0

    import jax
    jax.config.update("jax_enable_x64", True)
    from f1deg.config import load_config
    from f1deg.pipeline import Pipeline

    cfg = load_config(a.config if Path(a.config).exists() else None, mode="SMOKE", outputs_root=a.out)
    cfg.fits_override = []                                        # the study uses no fits of the main queue
    cfg.sampler = dataclasses.replace(cfg.sampler, max_treedepth=a.max_treedepth, chain_method="sequential")
    cfg.recovery.sampler = {"warmup": a.warmup, "draws": a.draws, "chains": a.chains}
    size = {}                                                     # size of the synthetic data set
    if a.tracks is not None:
        size["n_tracks"] = int(a.tracks)
    if a.seasons is not None:
        size["years"] = list(range(1, int(a.seasons) + 1))
    if a.drivers is not None:
        size["n_drivers"] = int(a.drivers)
    if a.teams is not None:
        size["n_teams"] = int(a.teams)
    if a.laps is not None:
        size["laps_per_stint"] = (int(a.laps[0]), int(a.laps[1]))
    if size:
        cfg.smoke = {**cfg.smoke, **size}
    # one design per output root: the per-fit tables are named by scenario and replicate only
    design = {"n_tracks": int(cfg.smoke["n_tracks"]), "years": [int(y) for y in cfg.smoke["years"]], "n_drivers": int(cfg.smoke["n_drivers"]), "n_teams": int(cfg.smoke["n_teams"]),
              "laps_per_stint": [int(v) for v in cfg.smoke["laps_per_stint"]], "warmup": a.warmup, "draws": a.draws, "chains": a.chains, "max_treedepth": a.max_treedepth}
    dfile = Path(a.out) / "study_design.json"
    if dfile.exists():
        prev = json.load(open(dfile, encoding="utf-8"))
        if prev != design:
            raise SystemExit(f"{dfile} describes another design ({prev}); this call asks for {design}. Use a different --out for a different design.")
    else:
        dfile.parent.mkdir(parents=True, exist_ok=True)
        json.dump(design, open(dfile, "w", encoding="utf-8"), indent=1)
    pl = Pipeline(cfg)
    from f1deg import synthetic as SY
    print("synthetic data set of the first replicate:", SY.generate_from_config(cfg, a.scenarios[0], a.start).sizes(), flush=True)
    t0 = time.time()
    res = pl.stage_recovery(scenarios=a.scenarios, replicates=a.stop - a.start, table_tag=a.tag, replicate_start=a.start)
    print(f"done in {(time.time()-t0)/60:.1f} min; pooled coverage of well-specified fits:", json.dumps(res.get("_coverage"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
