"""Stage registry: persisted status (pending/running/completed/failed/needs_attention),
fingerprints and dependencies, so that a repeated `Run All` skips completed,
compatible stages and recomputes exactly what changed.

A stage function returns a JSON-serialisable dict (paths to its artifacts,
small summaries).  The registry persists it; on a compatible rerun the stored
dict is returned without calling the function.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .checkpoint import atomic_write_json
from .config import sha256_of


class StageRegistry:
    def __init__(self, path: Path, log: Callable[[str], None] = print):
        self.path = Path(path)
        self.log = log
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                d = json.load(open(self.path, "r", encoding="utf-8"))
                self.stages = d.get("stages", {})
            except Exception:
                self.stages = {}

    def _save(self):
        atomic_write_json(self.path, {"stages": self.stages, "saved": time.time()})

    def status(self, name: str) -> str:
        return self.stages.get(name, {}).get("status", "pending")

    def result(self, name: str) -> Optional[Dict[str, Any]]:
        return self.stages.get(name, {}).get("result")

    def fingerprint_for(self, name: str) -> Optional[str]:
        return self.stages.get(name, {}).get("fingerprint")

    def set_status(self, name: str, status: str, note: str = ""):
        rec = self.stages.setdefault(name, {})
        rec["status"] = status
        if note:
            rec["note"] = note
        self._save()

    def run(self, name: str, fingerprint_fields: Dict[str, Any], fn: Callable[[], Dict[str, Any]], deps: List[str] = (),
            force: bool = False, outputs_exist: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Dict[str, Any]:
        fp = sha256_of({"fields": fingerprint_fields, "deps": {d: self.fingerprint_for(d) for d in deps}})
        rec = self.stages.get(name)
        if rec and rec.get("status") in ("completed", "needs_attention") and rec.get("fingerprint") == fp and not force:
            ok = True
            if outputs_exist is not None:
                try:
                    ok = bool(outputs_exist(rec.get("result", {})))
                except Exception:
                    ok = False
            if ok:
                self.log(f"[stage {name}] SKIPPED (completed, fingerprint {fp[:10]} unchanged)")
                self.events.append({"stage": name, "event": "skipped", "fingerprint": fp})
                return rec["result"]
        if rec and rec.get("fingerprint") not in (None, fp):
            self.log(f"[stage {name}] fingerprint changed ({str(rec.get('fingerprint'))[:10]} -> {fp[:10]}): recomputing")
        self.stages[name] = {"status": "running", "fingerprint": fp, "started": time.time(), "deps": list(deps)}
        self._save()
        self.log(f"[stage {name}] RUNNING")
        t0 = time.time()
        try:
            result = fn() or {}
        except Exception as e:  # noqa
            self.stages[name].update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc(), "finished": time.time()})
            self._save()
            self.log(f"[stage {name}] FAILED: {e!r}")
            raise
        status = result.pop("_status", "completed") if isinstance(result, dict) else "completed"
        self.stages[name].update({"status": status, "result": result, "finished": time.time(), "seconds": round(time.time() - t0, 1)})
        self._save()
        self.events.append({"stage": name, "event": status, "seconds": round(time.time() - t0, 1)})
        self.log(f"[stage {name}] {status.upper()} in {time.time() - t0:.1f}s")
        return result

    def table(self):
        import pandas as pd

        rows = []
        for k, v in self.stages.items():
            rows.append({"stage": k, "status": v.get("status"), "fingerprint": str(v.get("fingerprint", ""))[:10], "seconds": v.get("seconds"),
                         "note": v.get("note", v.get("error", ""))})
        return pd.DataFrame(rows)
