"""The arena register: which agents exist, which runs were played, and where each stands.

    python -m catanatron.arena.register agents          # every agent, its status and its record
    python -m catanatron.arena.register runs [--all]    # runs, newest first (running ones flagged)
    python -m catanatron.arena.register ladder          # agent x opponent per arena: games, wins, 95% CI
    python -m catanatron.arena.register games <run_id>  # one run's games
    python -m catanatron.arena.register set-status <agent> <status> [note]
    python -m catanatron.arena.register add-agent "thomas 1.2-a" --family thomas --params '{"thomas": {...}}' --desc "..."

One SQLite file, `arena.sqlite`, in the arena's data folder (see
catanatron.arena.data_dir: --data, CATAN_ARENA_DATA, else ./arena-data). It
is local and never part of the package or the repo. Written by bench, sweep
and study as they play; read by anyone -- a notebook, sqlite3 on the command
line. Small on purpose: three tables and
two views, and nothing a run's JSON does not already say. The views are the
contract; a reader that only ever opens `vw_ladder` and `vw_agent` will not
notice the tables changing underneath.

  agent   a brain that can sit at a table: our thomas / rules / value, their
          alpha / policy net / heuristics. `status` is ours to set: current,
          candidate, retired, reference (an outside bot we measure against).
  run     one measurement: an arena, a table shape, a candidate, an opponent,
          how many games, and the result once it is finished. `status` is
          running / finished / failed; a run that stopped mid-way stays as
          'failed' with the games it did play.
  game    one game of a run: the seed, the seat(s) the candidate held, who
          won, the candidate's points. The unit everything above is derived
          from -- win rates are never stored, always counted from here.

A VERSION is an agent row whose `params` say how it differs from the code as
shipped: {"thomas": {PARAMS overrides}, "setup": {setup_score.WEIGHTS
overrides}, "value": {value-function weights}}. bench.py builds the player
from the row, so "thomas 1.2-a" is a name the whole arena understands and the
tables it played with are on record beside its results.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def db_path(data=None):
    """The register file inside the arena's data folder."""
    from catanatron.arena import data_dir
    return data_dir(data) / "arena.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent (
    id          TEXT PRIMARY KEY,                -- 'thomas 1.1', 'value', 'alpha 8/96/300'
    family      TEXT NOT NULL,                   -- 'thomas', 'rules', 'value', 'rl-net', 'alpha', ...
    version     TEXT,                            -- '1.1', '3.2', '' for the plain ones
    source      TEXT NOT NULL,                   -- 'catan-player', 'catanatron', 'catan-rl'
    status      TEXT NOT NULL DEFAULT 'candidate',   -- current | candidate | retired | reference
    description TEXT,
    params      TEXT,                            -- JSON: the weights/settings this version plays with (thomas PARAMS, value weights)
    added_on    TEXT NOT NULL,                   -- ISO date
    note        TEXT
);
CREATE TABLE IF NOT EXISTS run (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',     -- running | finished | failed
    arena       TEXT NOT NULL,                   -- 'catanatron' | 'catan-rl'
    "table"     TEXT NOT NULL,                   -- '1v3' | '2v2' | '1v1'
    vps         INTEGER NOT NULL,
    candidate   TEXT NOT NULL REFERENCES agent(id),
    opponent    TEXT NOT NULL REFERENCES agent(id),
    settings    TEXT,                            -- JSON: ports, alpha config, seeds range, machine
    label       TEXT,                            -- free text: why this run
    note        TEXT
);
CREATE TABLE IF NOT EXISTS game (
    run_id      INTEGER NOT NULL REFERENCES run(id),
    seed        INTEGER NOT NULL,
    seats       TEXT NOT NULL,                   -- JSON list of the candidate's seat(s)
    won         INTEGER NOT NULL,                -- 1 if a candidate seat won
    winner_seat INTEGER,                         -- NULL = truncated, nobody won
    cand_vp     INTEGER,
    turns       INTEGER,
    decisions   INTEGER,
    fallbacks   INTEGER,
    PRIMARY KEY (run_id, seed, seats)
);
CREATE VIEW IF NOT EXISTS vw_run AS
    SELECT r.id, r.status, r.arena, r."table", r.vps, r.candidate, r.opponent,
           COUNT(g.run_id) AS games, COALESCE(SUM(g.won), 0) AS wins,
           ROUND(1.0 * COALESCE(SUM(g.won), 0) / MAX(COUNT(g.run_id), 1), 4) AS win_rate,
           ROUND(AVG(g.cand_vp), 2) AS avg_vp, ROUND(AVG(g.turns), 1) AS avg_turns,
           COALESCE(SUM(g.fallbacks), 0) AS fallbacks,
           r.started_at, r.finished_at, r.settings, r.label, r.note
    FROM run r LEFT JOIN game g ON g.run_id = r.id
    GROUP BY r.id;
CREATE VIEW IF NOT EXISTS vw_ladder AS
    SELECT r.arena, r."table", r.vps, r.candidate, r.opponent,
           COUNT(g.run_id) AS games, SUM(g.won) AS wins,
           ROUND(1.0 * SUM(g.won) / COUNT(g.run_id), 4) AS win_rate,
           ROUND(AVG(g.cand_vp), 2) AS avg_vp,
           COUNT(DISTINCT r.id) AS runs, MAX(r.finished_at) AS last_finished
    FROM run r JOIN game g ON g.run_id = r.id
    WHERE r.status = 'finished'
    GROUP BY r.arena, r."table", r.vps, r.candidate, r.opponent;
CREATE VIEW IF NOT EXISTS vw_agent AS
    SELECT a.id, a.family, a.version, a.source, a.status, a.added_on, a.params,
           (SELECT COUNT(*) FROM run r WHERE r.candidate = a.id AND r.status = 'finished') AS runs_as_candidate,
           (SELECT COUNT(*) FROM run r JOIN game g ON g.run_id = r.id
             WHERE r.candidate = a.id AND r.status = 'finished') AS games_as_candidate,
           (SELECT COUNT(*) FROM run r WHERE r.opponent = a.id AND r.status = 'finished') AS runs_as_opponent,
           a.description, a.note
    FROM agent a;
"""


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def connect(path=None):
    con = sqlite3.connect(path or db_path(), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


# ---- agents ------------------------------------------------------------------
def ensure_agent(con, agent_id, family=None, version=None, source=None, status="candidate", description=None, params=None):
    row = con.execute("SELECT id FROM agent WHERE id = ?", (agent_id,)).fetchone()
    if row:
        return agent_id
    if family is None:
        family, _, version = agent_id.partition(" ")
    con.execute("INSERT INTO agent (id, family, version, source, status, description, params, added_on) VALUES (?,?,?,?,?,?,?,?)",
                (agent_id, family, version or "", source or "?", status, description,
                 None if params is None else json.dumps(params, sort_keys=True), now()[:10]))
    con.commit()
    return agent_id


def get_agent(con, agent_id):
    row = con.execute("SELECT * FROM agent WHERE id = ?", (agent_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    return d


def set_status(con, agent_id, status, note=None):
    assert status in ("current", "candidate", "retired", "reference"), status
    con.execute("UPDATE agent SET status = ?, note = COALESCE(?, note) WHERE id = ?", (status, note, agent_id))
    con.commit()


# ---- runs and games ----------------------------------------------------------
def start_run(con, arena, table, vps, candidate, opponent, settings=None, label=None):
    cur = con.execute('INSERT INTO run (started_at, arena, "table", vps, candidate, opponent, settings, label) '
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (now(), arena, table, vps, candidate, opponent, json.dumps(settings or {}), label))
    con.commit()
    return cur.lastrowid


def add_game(con, run_id, seed, seats, won, winner_seat, cand_vp, turns, decisions=None, fallbacks=None):
    con.execute("INSERT OR REPLACE INTO game (run_id, seed, seats, won, winner_seat, cand_vp, turns, decisions, fallbacks) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, seed, json.dumps(sorted(seats)), int(won), winner_seat, cand_vp, turns, decisions, fallbacks))
    con.commit()


def finish_run(con, run_id, status="finished", note=None):
    con.execute("UPDATE run SET status = ?, finished_at = ?, note = COALESCE(?, note) WHERE id = ?",
                (status, now(), note, run_id))
    con.commit()


# ---- a run recorded from its JSON ------------------------------------------------
AGENT_IDS = {"thomas": "thomas 1.1", "rules": "rules 3.3", "value": "value", "rl": "rl-net 512",
             "random": "random", "alpha": "alpha 8/96/300", "rollout": "rollout 12/40",
             "heuristic": "heuristic v1", "heuristic_v2": "heuristic v2"}


def agent_id(kind, alpha=None):
    if kind == "alpha" and alpha:
        return "alpha %s" % "/".join(str(x) for x in alpha)
    return AGENT_IDS.get(kind, kind)


def import_json(con, path, label=None):
    d = json.load(open(path, encoding="utf-8"))
    s, games = d["summary"], d["games"]
    if "alpha" in s:                                   # rl_arena.py
        arena, table = "catan-rl", "1v3"
        cand, opp = agent_id(s["cand"]), agent_id(s["opp"], s.get("alpha"))
        settings = {"alpha": s.get("alpha"), "file": os.path.basename(path)}
    else:                                              # bench.py
        arena, table = "catanatron", s.get("table", "1v3")
        cand, opp = agent_id(s["cand"]), agent_id(s["opp"])
        settings = {"ports": s.get("ports"), "file": os.path.basename(path)}
    ensure_agent(con, cand); ensure_agent(con, opp)
    run_id = start_run(con, arena, table, int(s["vps"]), cand, opp, settings, label)
    for g in games:
        seats = g.get("seats") or [g.get("seat")]
        if "won" in g:
            won, winner = g["won"], (None if g["winner"] < 0 else g["winner"])
        else:
            won = g["winner"] is not None and g["winner"] in (g.get("cand_colors") or [g.get("cand_color")])
            winner = None
        add_game(con, run_id, g["seed"], seats, won, winner, g.get("cand_vp"), g.get("turns"),
                 g.get("decisions"), g.get("fallbacks"))
    finish_run(con, run_id, "finished")
    return run_id


# ---- the command line ----------------------------------------------------------
def _print_rows(rows, cols):
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r[c]) if r[c] is not None else "-") for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join((str(r[c]) if r[c] is not None else "-").ljust(widths[c]) for c in cols))


def main(argv):
    con = connect()
    cmd = argv[0] if argv else "agents"
    if cmd == "agents":
        rows = con.execute("SELECT * FROM vw_agent ORDER BY status, id").fetchall()
        _print_rows(rows, ["id", "status", "source", "runs_as_candidate", "games_as_candidate", "runs_as_opponent", "added_on"])
    elif cmd == "runs":
        where = "" if "--all" in argv else "WHERE status = 'running' OR finished_at >= date('now', '-14 days')"
        rows = con.execute(f"SELECT * FROM vw_run {where} ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            lo, hi = wilson(r["wins"], r["games"])
            d = dict(r)
            d["ci95"] = f"{lo:.1%}-{hi:.1%}" if r["games"] else "-"
            d["win_rate"] = f"{r['win_rate']:.1%}" if r["games"] else "-"
            out.append(d)
        _print_rows(out, ["id", "status", "arena", "table", "vps", "candidate", "opponent", "games", "wins", "win_rate", "ci95", "avg_vp", "fallbacks", "started_at"])
    elif cmd == "ladder":
        rows = con.execute("SELECT * FROM vw_ladder ORDER BY arena, opponent, win_rate DESC").fetchall()
        out = []
        for r in rows:
            lo, hi = wilson(r["wins"], r["games"])
            d = dict(r)
            d["ci95"] = f"{lo:.1%}-{hi:.1%}"
            d["win_rate"] = f"{r['win_rate']:.1%}"
            out.append(d)
        _print_rows(out, ["arena", "table", "vps", "candidate", "opponent", "games", "wins", "win_rate", "ci95", "avg_vp", "runs"])
    elif cmd == "games":
        rows = con.execute("SELECT * FROM game WHERE run_id = ? ORDER BY seed, seats", (int(argv[1]),)).fetchall()
        _print_rows(rows, ["seed", "seats", "won", "winner_seat", "cand_vp", "turns", "decisions", "fallbacks"])
    elif cmd == "import":
        # a run's JSON (bench.py / rl_arena.py --out), recorded after the fact
        run_id = import_json(con, argv[1], label=" ".join(argv[2:]) or None)
        print("run", run_id)
    elif cmd == "add-agent":
        import argparse
        ap = argparse.ArgumentParser(prog="register.py add-agent")
        ap.add_argument("id")
        ap.add_argument("--family", required=True, help="thomas | rules | value -- which class plays it")
        ap.add_argument("--params", default="{}", help='JSON: {"thomas": {...}, "setup": {...}, "value": {...}}')
        ap.add_argument("--desc", default=None)
        ap.add_argument("--status", default="candidate")
        ap.add_argument("--source", default="catan-player")
        o = ap.parse_args(argv[1:])
        if get_agent(con, o.id):
            print("exists:", o.id)
            return 1
        _, _, version = o.id.partition(" ")
        ensure_agent(con, o.id, o.family, version, o.source, o.status, o.desc, json.loads(o.params))
        print("added", o.id)
    elif cmd == "set-status":
        set_status(con, argv[1], argv[2], " ".join(argv[3:]) or None)
        print("ok")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
