"""The arena: play games between players on this engine, and keep the record.

    python -m catanatron.arena.bench   --cand value --opp random --seeds 5 --table 1v1
    python -m catanatron.arena.register ladder
    python -m catanatron.arena.sweep   spec.json
    python -m catanatron.arena.study   --cand value --opp random --seeds 20

The arena never imports a player package. A player is named on the command
line and built by one of three routes, tried in order:

  1. a FACTORY given with --factory module:function (or CATAN_ARENA_FACTORY).
     It is called as factory(name, color, row) and returns a Player, or None
     to fall through. `row` is the register's agent row for that name (a dict
     with family / params) or None. This is how an outside repo plugs in its
     own players and their registered versions.
  2. a class path 'module:Class', built as Class(color).
  3. this engine's own bots: value, random, weighted, alphabeta, mcts.

Extra import roots for the factory's modules: --path DIR (repeatable) or
CATAN_ARENA_PATH (os.pathsep-separated); --setup MODULE imports a module first
(for a repo that arranges its own sys.path).

THE DATA lives outside the code, in one folder: the register database
(arena.sqlite) and the run, study and sweep files. --data DIR, else
CATAN_ARENA_DATA, else ./arena-data in the current directory. It is never
part of the package, so a pinned install cannot overwrite it.
"""
from __future__ import annotations

import os
import pathlib


def data_dir(explicit: str | os.PathLike | None = None) -> pathlib.Path:
    """The arena's data folder, created if missing."""
    d = pathlib.Path(explicit or os.environ.get("CATAN_ARENA_DATA") or (pathlib.Path.cwd() / "arena-data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def same_hash_seed() -> None:
    """Re-run this process with PYTHONHASHSEED=0 unless it already has it.

    A fixed game seed does not reproduce a game without it: something in a
    decision path iterates a set of strings, and the order follows the hash
    seed (1 run in 6 diverged, 8 of 8 agreed with the seed fixed). It must be
    set before the interpreter starts, so a process without it restarts itself.
    """
    import subprocess
    import sys
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ, PYTHONHASHSEED="0")
        raise SystemExit(subprocess.call([sys.executable] + sys.argv, env=env))


def engine_version() -> dict:
    """Which engine a run played on: version, and the git commit when pip knows it."""
    import json
    from importlib import metadata
    out = {"version": None, "commit": None, "source": None}
    try:
        dist = metadata.distribution("catanatron")
        out["version"] = dist.version
        raw = dist.read_text("direct_url.json")
        if raw:
            d = json.loads(raw)
            out["commit"] = (d.get("vcs_info") or {}).get("commit_id")
            out["source"] = d.get("url")
    except Exception:  # noqa: BLE001
        pass
    return out
