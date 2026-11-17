#!/usr/bin/env python
"""Command-line entry point of the pipeline (same code path as the notebook).

    python scripts/run_pipeline.py --mode SMOKE                      # synthetic data, minutes: tests the whole pipeline
    python scripts/run_pipeline.py --mode FULL --list                # what exists on disk, nothing is run
    python scripts/run_pipeline.py --mode FULL --fits clean_110      # add ONE fit to what is already there
    python scripts/run_pipeline.py --mode FULL                       # the queue `run_fits` of config.yaml
    python scripts/run_pipeline.py --mode REPORT_ONLY                # no MCMC: diagnostics, PPC, LOO, forecast, tables, figures, reports
    python scripts/run_pipeline.py --mode FULL --fits noint --extend 1   # continue the chains of a finished fit to twice the draws

Every fit is checkpointed every `checkpoint_every` transitions.  After a power cut run THE SAME command
again: completed stages are skipped, the interrupted fit continues from its last committed state.

`--fits` only chooses what is sampled in this call.  Fits that are already complete on disk are always
found again (by their fingerprint) and included in the tables, whether they are in the queue or not.

Exit code: 0 — the requested stages completed; 3 — a fit was paused by the time budget
(`budget.max_hours_per_fit`): run the same command again; 1 — a post-processing stage failed
(the others were still carried out; see reports/pipeline_problems.csv).

v1.1 keeps the options of the v1.0 script (--mode, --config, --force-refit, --no-resume, --outputs-root,
--data-csv, exit code 3) and adds --fits, --list, --stage, --extend, --report-source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None, choices=["SMOKE", "PILOT", "FULL", "REPORT_ONLY"], help="default: `mode` of the config file")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--fits", nargs="*", default=None, help="experiments to sample in this call (default: run_fits of the config; for FULL without run_fits — the whole matrix by priority)")
    ap.add_argument("--report-source", default=None, help="for REPORT_ONLY: which mode's outputs to read (default: FULL)")
    ap.add_argument("--outputs-root", default=None)
    ap.add_argument("--data-csv", default=None)
    ap.add_argument("--force-refit", action="store_true", help="DELETE the fits of the queue and sample them again")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the state of every experiment and exit (nothing is sampled)")
    ap.add_argument("--extend", type=int, default=None, metavar="ROUNDS",
                    help="allow up to ROUNDS doublings of the number of saved draws for fits of the queue whose R-hat/ESS/MCSE criteria fail "
                         "(same chains, no new warm-up; fits that fail only E-BFMI are not extended).  Default: budget.max_extension_rounds of the config")
    ap.add_argument("--stage", default="all", choices=["all", "fits", "recovery", "postprocess", "forecast", "export", "report"],
                    help="run one part only: fits (queue + diagnostics), recovery, postprocess (= REPORT_ONLY), forecast, export, report")
    a = ap.parse_args(argv)

    import yaml
    raw = (yaml.safe_load(open(a.config, encoding="utf-8")) or {}) if Path(a.config).exists() else {}
    if (raw.get("sampler") or {}).get("chain_method") == "parallel":            # pmap needs host devices BEFORE jax is imported
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={(raw.get('sampler') or {}).get('chains', 4)}"
    import jax
    jax.config.update("jax_enable_x64", True)
    import pandas as pd

    from f1deg.config import load_config
    from f1deg.pipeline import Pipeline
    from f1deg import report as RP

    over = {"data_csv": str(Path(a.data_csv).resolve()) if a.data_csv else str((ROOT / raw.get("data_csv", "data/f1_cache_combined_2022_2024.csv")).resolve()),
            "outputs_root": str(Path(a.outputs_root).resolve()) if a.outputs_root else str((ROOT / raw.get("outputs_root", "outputs")).resolve())}
    if a.mode:
        over["mode"] = a.mode
    if a.force_refit:
        over["force_refit"] = True
    if a.no_resume:
        over["resume"] = False
    if a.report_source:
        over["report_only_source"] = a.report_source
    cfg = load_config(a.config if Path(a.config).exists() else None, **over)
    if a.fits is not None:
        cfg.fits_override = list(a.fits)
    if a.extend is not None:
        cfg.budget["max_extension_rounds"] = int(a.extend)
    pl = Pipeline(cfg)
    print(f"run ID {pl.run_id} | mode {cfg.mode}{' (REPORT_ONLY)' if cfg.report_only else ''} | outputs {cfg.out_dir}")
    print("versions:", json.dumps(cfg.versions()))
    pl.stage_data()
    if a.list:
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
        st = RP.experiments_status(cfg.experiments, pl.states())
        print(st[["experiment", "tier", "пріоритет", "in_queue", "статус", "fit_id"]].to_string(index=False))
        return 0
    summary = {}
    if a.stage == "all":
        summary = pl.run_all()
    elif a.stage == "fits":
        for e in pl.experiments():
            summary[e.name] = pl.fit_with_bounded_extension(e)
    elif a.stage == "recovery":
        pl.stage_recovery()
    elif a.stage == "postprocess":
        pl.postprocess()
    elif a.stage == "forecast":
        pl.stage_forecast()
    elif a.stage == "export":
        pl.stage_export()
    elif a.stage == "report":
        pl.stage_report()
    if pl.problems:
        print("PROBLEMS (see reports/pipeline_problems.csv):")
        for p in pl.problems:
            print("  ", p["stage"], "->", p["error"])
    miss = cfg.out_dir / "paper" / "MISSING.md"
    if miss.exists():
        print(miss.read_text(encoding="utf-8"))
    paused = [k for k, v in (summary or {}).items() if isinstance(v, dict) and (v.get("fit") or {}).get("status") not in (None, "completed")]
    if paused:
        print(f"paused fits (time budget): {paused}; run the same command again to continue")
        return 3
    return 1 if pl.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
