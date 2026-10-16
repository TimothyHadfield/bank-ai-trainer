# BANK AI Trainer — notes for Claude

Teach a Q-learning AI to play the push-your-luck dice game BANK in the browser, then play
against it. Tim's personal/showcase project (public repo, no accounts, no backend, no user data).
A sibling project, `Code Projects\AI & Machine Learning\Bank_Pro_AI` (AlphaZero-style v2 design
docs), is separate: don't mix them up.

- **Live:** https://timothyhadfield.github.io/bank-ai-trainer/ (GitHub Pages, legacy build from `main` /, about 1 min after a push)
- **Repo:** TimothyHadfield/bank-ai-trainer (public). Main checkout:
  `OneDrive\Desktop\my-website\Code Projects\Board and Card Games\bank_ai_project`
- Handoff state: `progress.md`.

## Two independent halves
1. **Browser trainer = `index.html`** (the live site). One file: CSS, HTML and a plain `<script>` with
   `QLearningAgent`, `RandomAgent`, `ThresholdAgent`, `BankGame`, `trainEpisode` (training loop on a
   1 ms `setInterval`), and the Play-vs-AI flow (`startGame` / `rollDice` / `playerBanks` /
   `aiTakeTurn` / `endRound` / `endGame`). Everything is in memory: saved AIs vanish on reload (by design).
2. **Python stack = `main.py` + `src/`** (local only, never deployed): `game_engine.py` (BankGame,
   `calculate_expected_value`), `agents.py` (Q-learning, opponent-modeling, random, threshold, EV),
   `training.py` (Trainer), `evaluation.py` (Evaluator + matplotlib/seaborn plots), `play_human.py` (CLI game).
   `models/` and `logs/` are git-ignored outputs.

The two halves use **different rules for when you may bank**: the browser only allows banking from
roll 3 on; the Python engine lets agents bank on any roll. Their state buckets and reward shaping
also differ. Don't "sync" them without Tim's say-so.

## Run locally
- Browser: `python -m http.server 8117` in the repo (background), open http://localhost:8117/.
- Python: needs numpy, matplotlib, seaborn, pandas, tqdm (`pip install -r requirements.txt`; use a venv).
  `python main.py train --episodes 300 --save-dir <dir> --log-dir <dir> --plot`, then
  `python main.py evaluate --model <dir>/agent_final.pkl --analyze --baseline`. Set `MPLBACKEND=Agg` headless.
  Real CLI flags are `--lr`, `--gamma`, `--eval-games`, `--rounds`, `--players` (README lists some old names).

## Test
- Python: `python test_installation.py` (plain functions returning True/False; exit code 1 on failure).
  It covers imports, a simulated game, agent creation, a 5-episode training run, save/load (pickle)
  and a quick evaluation. Training writes `models/agent_final.pkl` in the cwd (ignored by git).
- Browser: no test suite. Verify with `node ~/.claude/tools/phone-view.mjs <url> --eval "<js>"`.
  Page functions and `let` globals (`game`, `agent`, `startGame()`, `rollDice()`, `playerBanks()`)
  are reachable from `--eval`; stub `Math.random` for fixed dice.

## Traps
- Game-flow delays use `gameTimeout()`, not `setTimeout`: it drops callbacks from a game that was
  replaced by "Start New Game". Use it for any new game timer.
- Q-learning tables in `agents.py` must use module-level default factories, never lambdas, or
  `Agent.save` (pickle) breaks.
- Every agent's `get_action` must accept `training=` (Trainer and Evaluator pass it to all players).
- `pickle5` is only for Python < 3.8; on 3.11 it fails to build and breaks `pip install`.
- Buttons use the browser's default font (they don't inherit Segoe UI), so they look serif in
  Safari. That's the current look; don't change it uninvited.

## Standing instructions
- No Tim quotes recorded for this project yet. The global rules in `~/.claude/CLAUDE.md` apply.
