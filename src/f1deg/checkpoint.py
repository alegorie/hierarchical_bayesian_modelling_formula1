"""Transactional checkpoint store for chunked MCMC runs.

Layout of one fit directory  (fits/<fit_id>/):
    manifest.json                immutable description of the run (fingerprint, config, versions)
    lock.json                    writer lock with pid/host/heartbeat (stale-lock recovery)
    commit_000123.json           commit records; the highest *valid* one is the resume point
    chain_0/state_gen000123.npz  full sampler state of chain 0 at the committed iteration
    chain_0/draws_0000150_0000175.npz   immutable posterior chunk (iterations [150,175))
    chain_0/warmup_0000000_0000025.npz  immutable warm-up diagnostics chunk
    extensions.json              log of posterior-target extensions

Write protocol per checkpoint (all chains together, one generation):
    1. write every new chunk and state file to <name>.tmp on the same filesystem, fsync, os.replace
    2. fsync the chain directories
    3. write commit_<gen>.json.tmp (listing every file of the committed set with sha256), fsync, os.replace
    4. fsync the fit directory
A crash anywhere before step 3 leaves an orphan .tmp/.npz that is never referenced by a valid
commit and is ignored/removed on recovery.  A corrupted or truncated file fails its checksum and
the recovery falls back to the previous generation.  Only the last `keep_generations` state files
are kept; posterior chunks are kept forever (they are the result).
"""
from __future__ import annotations

import glob
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import canonical_json, sha256_file


class CheckpointError(RuntimeError):
    pass


class LockError(CheckpointError):
    pass


def _fsync_dir(path: Path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # some filesystems (Windows) do not support directory fsync


def atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: Any):
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True, default=str).encode("utf-8"))


def atomic_savez(path: Path, arrays: Dict[str, np.ndarray]) -> str:
    """Write an .npz atomically and return its sha256."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return sha256_file(path)


def crash_point(name: str):
    """Fault injection for the integration tests: if the environment variable
    F1DEG_CRASH_AT equals `name`, terminate the process immediately (like a
    power cut) at this point."""
    if os.environ.get("F1DEG_CRASH_AT") == name:
        os.write(2, f"[fault-injection] hard exit at {name}\n".encode())
        os._exit(137)


def pause_point(name: str, gen: int, marker_dir: Optional[Path] = None):
    """Fault injection for the integration tests: if F1DEG_PAUSE_AT == "<name>:<gen>:<seconds>",
    write a marker file and sleep, so that the test can SIGKILL the process at a
    well-defined point (e.g. after a chunk has been computed in memory but before
    anything was written)."""
    spec = os.environ.get("F1DEG_PAUSE_AT")
    if not spec:
        return
    try:
        n, g, secs = spec.split(":")
    except ValueError:
        return
    if n == name and int(g) == int(gen):
        if marker_dir is not None:
            try:
                (Path(marker_dir) / f"PAUSED_{name}_{gen}").write_text(str(os.getpid()))
            except OSError:
                pass
        os.write(2, f"[fault-injection] pausing {float(secs)}s at {name}:{gen}\n".encode())
        time.sleep(float(secs))


def _pid_alive(pid: int) -> bool:
    """Liveness check that is safe on Windows (os.kill(pid, 0) would TERMINATE the process there)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True   # unknown -> assume alive (the heartbeat age still decides staleness)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FitStore:
    """One fit directory.  Thread-unsafe by design (one writer process)."""

    def __init__(self, root: Path, fit_id: str, keep_generations: int = 2):
        self.dir = Path(root) / fit_id
        self.fit_id = fit_id
        self.keep_generations = keep_generations
        self._lock_held = False

    # ------------------------------------------------------------ manifest
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def read_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create(self, manifest: Dict[str, Any]):
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.exists():
            old = self.read_manifest()
            if old.get("fingerprint") != manifest.get("fingerprint"):
                raise CheckpointError(f"fit dir {self.dir} exists with a different fingerprint; refusing to reuse")
            return old
        manifest = dict(manifest)
        manifest["created"] = time.time()
        manifest["created_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(self.manifest_path, manifest)
        for c in range(int(manifest["chains"])):
            (self.dir / f"chain_{c}").mkdir(exist_ok=True)
        return manifest

    # ------------------------------------------------------------ lock
    @property
    def lock_path(self) -> Path:
        return self.dir / "lock.json"

    def acquire_lock(self, stale_seconds: float = 900.0, force: bool = False):
        """Refuse to start if another live writer holds the lock.  A lock is
        stale if its heartbeat is older than `stale_seconds` or if the owning
        pid on this host is dead."""
        if self.lock_path.exists():
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    lk = json.load(f)
            except Exception:
                lk = {}
            same_host = lk.get("host") == socket.gethostname()
            age = time.time() - float(lk.get("heartbeat", 0))
            alive = _pid_alive(int(lk.get("pid", -1))) if same_host else None
            stale = age > stale_seconds or (same_host and alive is False)
            if not stale and not force:
                raise LockError(f"fit {self.fit_id} is locked by pid {lk.get('pid')} on {lk.get('host')} (heartbeat {age:.0f}s ago); "
                                f"refusing to start a second writer. If that process is dead, wait {stale_seconds:.0f}s or remove {self.lock_path}.")
        atomic_write_json(self.lock_path, {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time(), "heartbeat": time.time()})
        self._lock_held = True

    def heartbeat(self):
        if self._lock_held:
            atomic_write_json(self.lock_path, {"pid": os.getpid(), "host": socket.gethostname(), "heartbeat": time.time()})

    def release_lock(self):
        if self._lock_held and self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                pass
        self._lock_held = False

    # ------------------------------------------------------------ commits
    def _commit_files(self) -> List[Tuple[int, Path]]:
        out = []
        for p in glob.glob(str(self.dir / "commit_*.json")):
            try:
                g = int(Path(p).stem.split("_")[1])
                out.append((g, Path(p)))
            except ValueError:
                continue
        return sorted(out)

    def _validate_commit(self, path: Path, verify_all: bool = False) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            return None
        if rec.get("fit_id") != self.fit_id or "files" not in rec:
            return None
        files = rec["files"]
        # always verify the state files and the files added by this commit; older chunks on request
        to_check = set(rec.get("new_files", list(files))) | {f for f in files if "/state_" in f}
        if verify_all:
            to_check = set(files)
        for rel in files:
            if not (self.dir / rel).exists():
                return None
        for rel in to_check:
            if sha256_file(self.dir / rel) != files[rel]:
                return None
        return rec

    def latest_commit(self, verify_all: bool = False) -> Optional[Dict[str, Any]]:
        """Highest valid commit record; corrupt/incomplete records are skipped."""
        for g, p in reversed(self._commit_files()):
            rec = self._validate_commit(p, verify_all=verify_all)
            if rec is not None:
                return rec
        return None

    def commit(self, gen: int, iteration: int, phase: str, new_files: Dict[str, str], prev: Optional[Dict[str, Any]], extra: Dict[str, Any]):
        files = dict(prev["files"]) if prev else {}
        # drop state files of previous generations from the committed set (they are deleted below)
        files = {k: v for k, v in files.items() if "/state_" not in k}
        files.update(new_files)
        rec = {"fit_id": self.fit_id, "gen": int(gen), "iteration": int(iteration), "phase": phase, "files": files,
               "new_files": sorted(new_files), "time": time.time(), **extra}
        crash_point("before_commit")
        atomic_write_json(self.dir / f"commit_{gen:06d}.json", rec)
        crash_point("after_commit")
        self._gc(gen)
        return rec

    def _gc(self, gen: int):
        """Keep the last `keep_generations` state files / commit records."""
        keep = {gen - k for k in range(self.keep_generations)}
        for g, p in self._commit_files():
            if g not in keep and g < gen:
                try:
                    p.unlink()
                except OSError:
                    pass
        for p in glob.glob(str(self.dir / "chain_*" / "state_gen*.npz")):
            try:
                g = int(Path(p).stem.split("gen")[1])
            except ValueError:
                continue
            if g not in keep and g < gen:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def cleanup_orphans(self, rec: Optional[Dict[str, Any]]):
        """Remove files not referenced by the valid commit (uncommitted tail, .tmp leftovers)."""
        referenced = set(rec["files"]) if rec else set()
        # state files of the kept previous generations are referenced by their own commit records
        for g, p in self._commit_files():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    referenced |= {k for k in json.load(f).get("files", {}) if "/state_" in k}
            except Exception:
                continue
        removed = []
        for p in glob.glob(str(self.dir / "chain_*" / "*")):
            rel = os.path.relpath(p, self.dir)
            if rel.endswith(".tmp") or (rel not in referenced and ("/draws_" in rel or "/warmup_" in rel or "/state_" in rel)):
                try:
                    os.remove(p)
                    removed.append(rel)
                except OSError:
                    pass
        for p in glob.glob(str(self.dir / "*.tmp")):
            try:
                os.remove(p)
            except OSError:
                pass
        return removed

    # ------------------------------------------------------------ file helpers
    def chunk_path(self, chain: int, kind: str, start: int, end: int) -> Path:
        return self.dir / f"chain_{chain}" / f"{kind}_{start:07d}_{end:07d}.npz"

    def state_path(self, chain: int, gen: int) -> Path:
        return self.dir / f"chain_{chain}" / f"state_gen{gen:06d}.npz"

    def rel(self, p: Path) -> str:
        return os.path.relpath(p, self.dir).replace(os.sep, "/")

    def list_chunks(self, rec: Dict[str, Any], chain: int, kind: str = "draws") -> List[Tuple[int, int, Path]]:
        out = []
        for rel in rec["files"]:
            if rel.startswith(f"chain_{chain}/{kind}_"):
                stem = Path(rel).stem
                _, s, e = stem.split("_")
                out.append((int(s), int(e), self.dir / rel))
        return sorted(out)

    def log_extension(self, entry: Dict[str, Any]):
        p = self.dir / "extensions.json"
        cur = []
        if p.exists():
            try:
                cur = json.load(open(p, "r", encoding="utf-8"))
            except Exception:
                cur = []
        cur.append(entry)
        atomic_write_json(p, cur)

    def write_status(self, status: Dict[str, Any]):
        atomic_write_json(self.dir / "status.json", status)

    def read_status(self) -> Dict[str, Any]:
        p = self.dir / "status.json"
        if not p.exists():
            return {}
        return json.load(open(p, "r", encoding="utf-8"))
