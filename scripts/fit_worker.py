#!/usr/bin/env python
"""Stand-alone fit worker: runs (or resumes) ONE fit in a fresh process.

Used by the resume integration tests (hard kill + resume in a new process) and
usable by hand:

    python scripts/fit_worker.py --spec worker_spec.json

The JSON spec contains: {"dataset": {"kind": "synthetic", "scenario": ..., "seed": ...} |
{"kind": "real", "csv": ..., "data": {...overrides}}, "sampler": {...}, "model": {...},
"fit_name": ..., "fits_root": ..., "mode": "SMOKE", "resume": true, "force_refit": false}
"""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    if spec.get("sampler", {}).get("chain_method") == "parallel":
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={spec['sampler'].get('chains', 4)}"
    import jax
    jax.config.update("jax_enable_x64", True)
    from f1deg.config import load_config, ModelSpec, DataConfig
    from f1deg.sampler import ChunkedNUTS

    cfg = load_config(spec.get("config_yaml"), mode=spec.get("mode", "SMOKE"))
    if "seed" in spec:
        cfg.seed = int(spec["seed"])
    sampler = dataclasses.replace(cfg.sampler, **spec.get("sampler", {}))
    model = ModelSpec(**{**cfg.primary.to_dict(), **spec.get("model", {})})
    dsd = spec["dataset"]
    if dsd["kind"] == "synthetic":
        from f1deg import synthetic as S
        ds = S.generate(dsd.get("scenario", "baseline"), seed=int(dsd.get("seed", 0)), **dsd.get("kwargs", {}))
    else:
        from f1deg import data as D
        dcfg = dataclasses.replace(cfg.data, **dsd.get("data", {}))
        raw = D.load_raw(dsd["csv"])
        clean_df, _, _ = D.clean(raw, dcfg)
        ds = D.build_dataset(clean_df, raw, dcfg)
    runner = ChunkedNUTS(cfg, sampler, model, ds, spec["fit_name"], Path(spec["fits_root"]), log=lambda m: print(m, flush=True))
    status = runner.run(resume=spec.get("resume", True), force_refit=spec.get("force_refit", False), max_seconds=spec.get("max_seconds"))
    print("STATUS " + json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
