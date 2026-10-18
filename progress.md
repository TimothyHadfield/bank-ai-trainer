# BANK AI Trainer — progress

## START HERE
Last updated 2026-10-16. Read this file, then `CLAUDE.md` (file map, how to run and test, traps).

## Goal right now
None set by Tim. The project works: live browser trainer plus a local Python training stack.

## Status
| Area | State |
|---|---|
| Browser trainer (`index.html`, live) | Works. Train, Play vs AI, Statistics, About tabs. |
| Python stack (`main.py`, `src/`) | Works since 2026-10-16 (train, evaluate, save/load were crashing before). |
| Tests | `python test_installation.py` (Python only). Browser checked with phone-view scripts. |
| Handoff files | `CLAUDE.md` + this file, created 2026-10-16 by the project sweep. |
| Site icon (2026-10-17) | `icon.svg` (white die on the purple gradient) + `apple-touch-icon.png` 180 px, linked in `index.html`; replaced the empty favicon. |

## Authorized next steps
None.

## Standing instructions
- No project-specific Tim quotes yet. Global rules in `~/.claude/CLAUDE.md` apply.

## Traps
- See `CLAUDE.md` → Traps (game timers, pickle-safe Q-tables, `training=` kwarg, pickle5).
- No accounts or backend, so no `claude-test` account is needed.

## Decisions
- 2026-10-16 sweep: fixed Python crashes (training/evaluation TypeError, pickle save, pickle5 install)
  and browser defects (old game's AI timers leaking into a new game, lost forfeit message,
  empty inputs hanging the game/training or NaN-ing the Q-table, progress bar stopping short of 100%).
  Invalid inputs fall back to the input's default value (rounds 10, episodes 1000, LR 0.1, epsilon 0.5).

## NOT verified
- Real iPhone (only WebKit at iPhone size via phone-view).
- Python `play` mode (interactive CLI; not driven by a test).

## Rejected / parked
- (none)
