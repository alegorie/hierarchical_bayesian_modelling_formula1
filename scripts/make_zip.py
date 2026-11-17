#!/usr/bin/env python
"""Packages the deliverable ZIP (code, notebook, tests + logs, smoke/pilot outputs, reports).
The raw data CSV and large FULL posteriors are excluded on purpose."""
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "f1_tyre_degradation_revision.zip"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".fastf1_cache", "outputs/full/fits"}
EXCLUDE_FILES = {"f1_cache_combined_2022_2024.csv"}


def want(p: Path) -> bool:
    rel = p.relative_to(ROOT).as_posix()
    if any(part in EXCLUDE_DIRS for part in p.parts) or any(rel.startswith(d) for d in EXCLUDE_DIRS):
        return False
    # v1.1: the outputs root is configurable (config.yaml: outputs_root, now `outputs_local`), so the large FULL
    # posteriors are excluded under any top-level directory whose name starts with "outputs", not only under "outputs/"
    parts = rel.split("/")
    if len(parts) >= 3 and parts[0].startswith("outputs") and parts[1] == "full" and parts[2] == "fits":
        return False
    if p.name in EXCLUDE_FILES or p.suffix in (".pyc",):
        return False
    return True


def main():
    # top-level copies of the two deliverables the task asks for by name
    # v1.1: the reports of the FULL run are preferred (they describe the actual state of the experiments); the outputs
    # root may be `outputs` or `outputs_local`
    for name in ("reviewer_response_matrix.csv", "experiments_status.csv"):
        for src in (f"outputs_local/full/reports/{name}", f"outputs/full/reports/{name}", f"outputs_local/smoke/reports/{name}", f"outputs/smoke/reports/{name}"):
            if (ROOT / src).exists():
                shutil.copy(ROOT / src, ROOT / name)
                break
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "README.md").write_text("Place `f1_cache_combined_2022_2024.csv` here (not included in the ZIP). See README.md §1.\n", encoding="utf-8")
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and want(p):
                z.write(p, Path("f1_tyre_degradation_revision") / p.relative_to(ROOT))
                n += 1
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
