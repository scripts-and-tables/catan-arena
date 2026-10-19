"""A sweep: several versions, several opponents, one night of CPU, one ladder.

    python -m catanatron.arena.sweep spec.json [--workers 6] [--dry] [--factory m:f --path DIR --setup M]

The spec is JSON:

    {
      "label": "thomas 1.2: the road award and the card points",
      "seeds": 250, "vps": 10, "table": "1v3",
      "opponents": ["thomas 1.1", "value"],
      "versions": {
        "thomas 1.2-roads": {"desc": "Longest Road worth its 2", "params": {"thomas": {"ROAD_AWARD_POINTS": 2.0}}},
        "thomas 1.2-cards": {"desc": "a card worth 0.55",        "params": {"thomas": {"CARD_POINTS": 0.55}}}
      }
    }

Every version is registered (family from the name's first word), then every
(version, opponent) pair is one bench run -- recorded in the register as
it plays, `--workers` of them at a time. A sweep that is interrupted leaves
its unfinished runs as 'running' in the register; `runs` shows them, and
the finished ones already count. The ladder at the end is the answer.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from catanatron.arena import register, data_dir, same_hash_seed
from catanatron.arena.bench import add_player_args, player_argv

PY = sys.executable


def main():
    same_hash_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="register the versions and print the runs, play nothing")
    add_player_args(ap)
    a = ap.parse_args()
    db = data_dir(a.data) / "arena.sqlite"
    spec = json.loads(pathlib.Path(a.spec).read_text(encoding="utf-8"))

    con = register.connect(db)
    for name, v in spec["versions"].items():
        family = name.split(" ")[0]
        if register.get_agent(con, name) is None:
            register.ensure_agent(con, name, family, name.partition(" ")[2], spec.get("source", "catan-player"), "candidate",
                                  v.get("desc"), v.get("params") or {})
            print("registered", name)
    for opp in spec["opponents"]:
        register.ensure_agent(con, register.agent_id(opp))
    con.close()

    # resume: a pair with a FINISHED run of the full length is not played again
    # (the harness kills background work when the machine runs low on memory;
    # a sweep is hours, and what is done stays done)
    con = register.connect(db)
    per = {"1v3": 4, "2v2": 6, "1v1": 2}[spec.get("table", "1v3")] * spec.get("seeds", 250)
    done = {(r[0], r[1]) for r in con.execute(
        "SELECT candidate, opponent FROM vw_run WHERE status='finished' AND \"table\"=? AND vps=? AND games>=?",
        (spec.get("table", "1v3"), spec.get("vps", 10), per))}
    con.close()
    jobs = []
    for name in spec["versions"]:
        for opp in spec["opponents"]:
            if (name, register.agent_id(opp)) in done:
                continue
            jobs.append([PY, "-m", "catanatron.arena.bench", "--cand", name, "--opp", opp,
                         "--seeds", str(spec.get("seeds", 250)), "--vps", str(spec.get("vps", 10)),
                         "--table", spec.get("table", "1v3"), "--quiet",
                         "--label", spec.get("label", pathlib.Path(a.spec).stem)] + player_argv(a))
    print(f"{len(jobs)} run(s) to play, {len(done)} already finished")
    for j in jobs:
        print(" ".join(x if " " not in x else f'"{x}"' for x in j[3:]))
    if a.dry:
        return 0

    t0 = time.time()

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        last = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        return cmd[4], cmd[6], (last[-1][:160] if last else f"FAILED rc={r.returncode}: {r.stderr[-300:]}")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for cand, opp, line in pool.map(run, jobs):
            print(f"[{time.time()-t0:6.0f}s] {cand} vs {opp}: {line}", flush=True)

    print()
    subprocess.run([PY, "-m", "catanatron.arena.register", "ladder"], env=dict(os.environ, CATAN_ARENA_DATA=str(db.parent)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
