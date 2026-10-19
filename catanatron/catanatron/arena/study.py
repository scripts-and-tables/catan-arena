"""A loss study: what did the winner have that our player did not?

    python -m catanatron.arena.study --cand value --opp random --seeds 100 [--table 1v3|1v1] [--vps 10]
    python -m catanatron.arena.study --report <data>/studies/<...>.json

Paired games as in the bench, but every game ends with a FACT SHEET per seat
-- settlements, cities, roads on the board, development cards bought and
played by kind, Longest Road / Largest Army, bank trades, points -- kept
in the data folder's studies/. The report splits our games into won and lost and
prints, for the lost ones, the winner's sheet beside ours: the lever a rule
should move is the column with the widest gap. Numbers, not impressions.
"""
import argparse
import collections
import contextlib
import io
import json
import pathlib
import statistics
import sys
import time

from catanatron.game import Game
from catanatron.models.enums import ActionType, SETTLEMENT, CITY, ROAD
from catanatron.models.player import Color
from catanatron.arena import bench, register, data_dir, same_hash_seed

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
DEV = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")


def sheet(game, color):
    """The facts of one seat at the end of a game."""
    s = game.state
    key = f"P{s.color_to_index[color]}"
    ps = s.player_state
    b = s.buildings_by_color.get(color, {}) or {}
    acts = [r.action for r in s.action_records if r.action.color == color]
    kinds = collections.Counter(a.action_type for a in acts)
    return {
        "vp": ps[f"{key}_ACTUAL_VICTORY_POINTS"],
        "settlements": len(b.get(SETTLEMENT, []) or []),
        "cities": len(b.get(CITY, []) or []),
        "roads": len(b.get(ROAD, []) or []),
        "longest_road_len": ps[f"{key}_LONGEST_ROAD_LENGTH"],
        "has_longest_road": int(ps[f"{key}_HAS_ROAD"]),
        "has_largest_army": int(ps[f"{key}_HAS_ARMY"]),
        "knights_played": ps[f"{key}_PLAYED_KNIGHT"],
        "cards_bought": kinds[ActionType.BUY_DEVELOPMENT_CARD],
        "vp_cards": ps[f"{key}_VICTORY_POINT_IN_HAND"],
        "bank_trades": kinds[ActionType.MARITIME_TRADE],
        "monopolies": kinds[ActionType.PLAY_MONOPOLY],
        "plenties": kinds[ActionType.PLAY_YEAR_OF_PLENTY],
        "road_buildings": kinds[ActionType.PLAY_ROAD_BUILDING],
        "cards_in_hand_end": sum(ps[f"{key}_{r}_IN_HAND"] for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")),
    }


def play(a):
    data = data_dir(a.data)
    reg = register.connect(data / "arena.sqlite")
    build = bench.Builder(a.factory, a.path, a.setup, reg)
    from itertools import combinations
    if a.table == "1v3":
        layouts = [(COLORS, {c}) for c in COLORS]
    elif a.table == "2v2":
        layouts = [(COLORS, set(pair)) for pair in combinations(COLORS, 2)]
    else:
        layouts = [(COLORS[:2], {c}) for c in COLORS[:2]]
    games = []
    t0 = time.time()
    for seed in range(a.start, a.start + a.seeds):
        for table, cand_colors in layouts:
            cm = bench.make_map(seed)
            players = [build(a.cand if c in cand_colors else a.opp, c) for c in table]
            game = Game(players, seed=seed, vps_to_win=a.vps, catan_map=cm)
            with contextlib.redirect_stderr(io.StringIO()):
                game.play()
            w = game.winning_color()
            games.append({
                "seed": seed, "turns": game.state.num_turns,
                "winner": None if w is None else w.value,
                "cand": sorted(c.value for c in cand_colors),
                "seats": {c.value: {"is_cand": c in cand_colors, **sheet(game, c)} for c in table},
            })
        if (seed - a.start + 1) % 10 == 0:
            won = sum(g["winner"] in g["cand"] for g in games)
            print(f"seeds {seed - a.start + 1:4d}  games {len(games):4d}  won {won/len(games):.1%}  {time.time()-t0:5.0f}s", flush=True)
    out = data / "studies" / f"{time.strftime('%Y-%m-%d-%H%M')}-{a.cand.replace(' ', '_')}-vs-{a.opp.replace(' ', '_')}-{a.table}-{a.vps}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cand": a.cand, "opp": a.opp, "table": a.table, "vps": a.vps, "games": games}, indent=0), encoding="utf-8")
    print("written", out)
    return out


def report(path):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    games = d["games"]
    cols = ["vp", "settlements", "cities", "roads", "longest_road_len", "has_longest_road", "has_largest_army",
            "knights_played", "cards_bought", "vp_cards", "bank_trades", "monopolies", "plenties", "road_buildings",
            "cards_in_hand_end"]
    ours_won, ours_lost, winners_over_us = [], [], []
    for g in games:
        for colour, s in g["seats"].items():
            if not s["is_cand"]:
                continue
            (ours_won if g["winner"] == colour else ours_lost).append(s)
            if g["winner"] is not None and g["winner"] not in g["cand"]:
                winners_over_us.append(g["seats"][g["winner"]])
    n = len(ours_won) + len(ours_lost)
    print(f"{d['cand']} vs {d['opp']}  table {d['table']}  first-to-{d['vps']}  games {len(games)}  "
          f"our seats {n}, won {len(ours_won)} ({len(ours_won)/max(n,1):.1%})  avg turns {statistics.mean(g['turns'] for g in games):.0f}")
    print()
    print(f"{'':22s} {'us: won':>9s} {'us: lost':>9s} {'winner over us':>15s}   gap (winner - us lost)")
    for c in cols:
        w = statistics.mean(s[c] for s in ours_won) if ours_won else float("nan")
        l = statistics.mean(s[c] for s in ours_lost) if ours_lost else float("nan")
        v = statistics.mean(s[c] for s in winners_over_us) if winners_over_us else float("nan")
        mark = " <--" if abs(v - l) >= 0.5 else ""
        print(f"{c:22s} {w:9.2f} {l:9.2f} {v:15.2f}   {v - l:+6.2f}{mark}")
    # how did the winners get their points, in the games we lost
    print()
    src = collections.Counter()
    for s in winners_over_us:
        src["settlements"] += s["settlements"]; src["cities x2"] += 2 * s["cities"]
        src["longest road x2"] += 2 * s["has_longest_road"]; src["largest army x2"] += 2 * s["has_largest_army"]
        src["vp cards"] += s["vp_cards"]
    tot = sum(src.values()) or 1
    print("where the winners' points came from, games we lost: " +
          ", ".join(f"{k} {v/len(winners_over_us):.2f} ({v/tot:.0%})" for k, v in src.most_common()))
    lost_by = collections.Counter(min(10, s["vp"]) for s in ours_lost)
    print("our points when we lost: " + ", ".join(f"{k}: {v}" for k, v in sorted(lost_by.items())))


def main():
    same_hash_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="rules")
    ap.add_argument("--opp", default="value")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--vps", type=int, default=10)
    ap.add_argument("--table", choices=["1v3", "2v2", "1v1"], default="1v3")
    ap.add_argument("--report", default=None, help="a study file to report on (no games played)")
    bench.add_player_args(ap)
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return 0
    out = play(a)
    print()
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
