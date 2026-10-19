# catan-arena

This repository is **the arena**: the Catan rules engine (a fork of
[bcollazo/catanatron](https://github.com/bcollazo/catanatron)) plus the gym that
plays games between players on it and keeps the record. Players live elsewhere
(catan-boss) and install this repo as a dependency pinned to one commit.

| what | where |
|---|---|
| rules engine | `catanatron/catanatron/` (board, longest road, awards, legality, cards) |
| the gym | `catanatron/catanatron/arena/`: `bench`, `register`, `sweep`, `study` |
| our engine changes | branch `pa/engine`, commits on top of upstream (see `git log`) |
| results | `arena-data/` next to this file: **local only, gitignored** |

Run from the player repo's environment, so the players are importable:

    python -m catanatron.arena.bench --cand value --opp random --seeds 50 --table 1v1
    python -m catanatron.arena.register ladder

A player is built by a factory the player repo supplies (`--factory
module:function`), by a class path (`module:Class`), or from this engine's
own bots (`value`, `random`, `weighted`, `alphabeta`, `mcts`). The arena never
imports a player package. Every command restarts itself with
`PYTHONHASHSEED=0`, so a seed reproduces a game, and every run records the
engine commit it played on.
