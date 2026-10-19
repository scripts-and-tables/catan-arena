"""The bench: paired-seed games between two players, recorded in the register.

    python -m catanatron.arena.bench --cand value --opp random --seeds 50 --table 1v1
    python -m catanatron.arena.bench --cand "thomas 1.2-a" --opp value --vps 15 --friendly-robber \\
        --factory arena_players:make --path C:/code_repo/catan-player/src --setup paths

Every seed fixes the board and the dice stream, and the candidate plays it from
every seat, so board-and-seat luck cancels. Wilson 95% intervals; 1,000 games
resolve about 3 points. The controls are not optional: a mirror that is not at
its fair share (25% at 1v3, 50% at 1v1 or 2v2) means the harness, not the bot,
is being measured.

Players are built without importing any player package -- see the package
docstring (catanatron.arena) for the factory, class-path and built-in routes.
The process restarts itself with PYTHONHASHSEED=0 so a seed means a game.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import os
import random
import sys
import time
from itertools import combinations

from catanatron.arena import data_dir, engine_version, same_hash_seed

COLOR_NAMES = ["RED", "BLUE", "WHITE", "ORANGE"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def make_map(seed):
    """This engine seeds the GLOBAL random; the base map is drawn from it."""
    from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
    random.seed(seed)
    return CatanMap.from_template(BASE_MAP_TEMPLATE)


# ---- building players ------------------------------------------------------
def _builtin(kind, color):
    from catanatron.models.player import RandomPlayer
    if kind == "value":
        from catanatron.players.value import ValueFunctionPlayer
        return ValueFunctionPlayer(color)
    if kind == "random":
        return RandomPlayer(color)
    if kind == "weighted":
        from catanatron.players.weighted_random import WeightedRandomPlayer
        return WeightedRandomPlayer(color)
    if kind == "alphabeta":
        from catanatron.players.minimax import AlphaBetaPlayer
        return AlphaBetaPlayer(color)
    if kind == "mcts":
        from catanatron.players.mcts import MCTSPlayer
        return MCTSPlayer(color)
    return None


def _load(spec):
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected module:name, got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


class Builder:
    """Turns a player name into a Player: factory first, then module:Class, then built-ins."""

    def __init__(self, factory=None, paths=(), setup=None, reg=None):
        for p in reversed([str(x) for x in paths if x]):
            if p not in sys.path:
                sys.path.insert(0, p)
        if setup:
            importlib.import_module(setup)
        self.factory = _load(factory) if factory else None
        self.reg = reg

    def row(self, name):
        if self.reg is None:
            return None
        from catanatron.arena import register
        return register.get_agent(self.reg, name)

    def __call__(self, name, color):
        if self.factory is not None:
            player = self.factory(name, color, self.row(name))
            if player is not None:
                return player
        if ":" in name:
            return _load(name)(color)
        player = _builtin(name, color)
        if player is None:
            raise SystemExit(f"cannot build player {name!r}: no factory took it, it is not module:Class, "
                             f"and it is not a built-in (value, random, weighted, alphabeta, mcts)")
        return player


def add_player_args(ap):
    """The options every arena command shares for finding players and data."""
    ap.add_argument("--factory", default=os.environ.get("CATAN_ARENA_FACTORY"),
                    help="module:function building a player from (name, color, register_row)")
    ap.add_argument("--path", action="append", default=[p for p in os.environ.get("CATAN_ARENA_PATH", "").split(os.pathsep) if p],
                    help="extra import root for the factory's modules (repeatable)")
    ap.add_argument("--setup", default=os.environ.get("CATAN_ARENA_SETUP"),
                    help="a module imported first (e.g. one that arranges sys.path)")
    ap.add_argument("--data", default=None, help="the arena data folder (default CATAN_ARENA_DATA or ./arena-data)")


def player_argv(a):
    """The same options back as arguments, for a child process."""
    out = []
    if a.factory:
        out += ["--factory", a.factory]
    for p in a.path:
        out += ["--path", p]
    if a.setup:
        out += ["--setup", a.setup]
    if a.data:
        out += ["--data", a.data]
    return out


def layouts_for(table, colors):
    if table == "1v3":
        return [(colors, {c}) for c in colors]
    if table == "2v2":
        return [(colors, set(pair)) for pair in combinations(colors, 2)]
    return [(colors[:2], {c}) for c in colors[:2]]


# ---- the run -----------------------------------------------------------------
def main(argv=None):
    same_hash_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--vps", type=int, default=10)
    ap.add_argument("--friendly-robber", action="store_true",
                    help="the robber may not block a player under 3 victory points")
    ap.add_argument("--table", choices=["1v3", "2v2", "1v1"], default="1v3",
                    help="1v3: one candidate, three opponents (fair 25%%); 2v2: two of each, every seat pair "
                         "(fair 50%%); 1v1: a two-player game, both seats (fair 50%%)")
    ap.add_argument("--quiet", action="store_true", help="swallow the players' stderr chatter")
    ap.add_argument("--out", default=None, help="also write the games as JSON here")
    ap.add_argument("--label", default=None, help="why this run -- goes into the register")
    ap.add_argument("--no-register", action="store_true")
    add_player_args(ap)
    a = ap.parse_args(argv)

    from catanatron.game import Game
    from catanatron.models.player import Color
    from catanatron.arena import register

    data = data_dir(a.data)
    reg = register.connect(data / "arena.sqlite")
    build = Builder(a.factory, a.path, a.setup, reg)
    colors = [Color[n] for n in COLOR_NAMES]
    engine = engine_version()

    run_id = None
    if not a.no_register:
        cand_id, opp_id = register.agent_id(a.cand), register.agent_id(a.opp)
        register.ensure_agent(reg, cand_id)
        register.ensure_agent(reg, opp_id)
        run_id = register.start_run(reg, "catanatron", a.table, a.vps, cand_id, opp_id,
                                    {"ports": "base", "seeds": [a.start, a.start + a.seeds],
                                     "friendly_robber": a.friendly_robber, "engine": engine}, a.label)
        print(f"register: run {run_id}", flush=True)

    wins = games = vp_sum = turn_sum = truncated = 0
    t0 = time.time()
    rows = []
    for seed in range(a.start, a.start + a.seeds):
        for table, cand_colors in layouts_for(a.table, colors):
            cm = make_map(seed)
            players = [build(a.cand if c in cand_colors else a.opp, c) for c in table]
            game = Game(players, seed=seed, vps_to_win=a.vps, friendly_robber=a.friendly_robber, catan_map=cm)
            sink = io.StringIO() if a.quiet else sys.stderr
            with contextlib.redirect_stderr(sink):
                game.play()
            w = game.winning_color()
            s = game.state
            cand_vps = [s.player_state[f"P{s.color_to_index[c]}_ACTUAL_VICTORY_POINTS"] for c in cand_colors]
            games += 1
            wins += (w in cand_colors)
            truncated += (w is None)
            vp_sum += max(cand_vps)
            turn_sum += s.num_turns
            seats = sorted(s.color_to_index[c] for c in cand_colors)
            rows.append({"seed": seed, "cand_colors": sorted(c.value for c in cand_colors), "seats": seats,
                         "winner": None if w is None else w.value, "cand_vp": max(cand_vps), "turns": s.num_turns})
            if run_id is not None:
                register.add_game(reg, run_id, seed, seats, w in cand_colors,
                                  None if w is None else s.color_to_index[w], max(cand_vps), s.num_turns)
        if (seed - a.start + 1) % 5 == 0:
            lo, hi = wilson(wins, games)
            print(f"seeds {seed - a.start + 1:4d}  games {games:5d}  win {wins/games:6.1%}  "
                  f"CI [{lo:.1%}, {hi:.1%}]  avgVP {vp_sum/games:.2f}  {time.time()-t0:6.0f}s", flush=True)
    lo, hi = wilson(wins, games)
    summary = {"cand": a.cand, "opp": a.opp, "table": a.table, "vps": a.vps,
               "friendly_robber": a.friendly_robber, "engine": engine,
               "games": games, "wins": wins, "win_rate": wins / games if games else 0.0, "ci95": [lo, hi],
               "avg_vp": vp_sum / games if games else 0.0, "avg_turns": turn_sum / games if games else 0.0,
               "truncated": truncated, "seconds": time.time() - t0}
    print("RESULT", json.dumps(summary))
    if run_id is not None:
        register.finish_run(reg, run_id)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "games": rows}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
