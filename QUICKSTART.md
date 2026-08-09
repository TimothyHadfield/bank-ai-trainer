# BANK AI - Quick Start Guide

Get up and running with the BANK AI training project in minutes!

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd "Bank Evolution/bank_ai_project"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train Your First Agent

Train a Q-learning agent with default settings (5000 episodes):

```bash
python main.py train
```

Train with custom settings:

```bash
python main.py train --episodes 10000 --opponents random threshold expected_value --plot
```

**What happens:**
- Agent learns by playing thousands of games
- Progress bar shows win rate and average score
- Models saved to `models/` directory
- Training curves saved to `logs/` if `--plot` is used

**Expected results after 5000 episodes:**
- Win rate: 30-40% (against 3 random/threshold opponents)
- Average score: 250-350 points

### 2. Evaluate a Trained Agent

Evaluate your trained agent:

```bash
python main.py evaluate --model models/agent_final.pkl --analyze --baseline
```

**What happens:**
- Agent plays 1000 evaluation games
- Generates banking pattern visualizations
- Compares performance to baseline agents
- Results saved to `logs/` directory

**Expected results:**
- Win rate: 35-45% in 4-player games
- Better performance vs Random agents
- Competitive with Threshold/Expected Value agents

### 3. Play Against Your AI

Test your skills against the trained agent:

```bash
python main.py play --model models/agent_final.pkl
```

Or play against a default agent without training:

```bash
python main.py play
```

**Interactive gameplay:**
- See game state (scores, bank total, roll count)
- Choose to [B]ank or [W]ait after each roll
- Compete head-to-head with the AI

## Common Commands

### Quick Training Run (for testing)
```bash
python main.py train --episodes 100
```

### Full Training Run (recommended)
```bash
python main.py train --episodes 10000 --opponents random threshold expected_value self --plot
```

### Advanced Training (custom hyperparameters)
```bash
python main.py train --episodes 15000 --lr 0.2 --gamma 0.99 --epsilon 0.5 --epsilon-decay 0.999 --plot
```

### Comprehensive Evaluation
```bash
python main.py evaluate --model models/agent_final.pkl --eval-games 2000 --analyze --baseline
```

## Understanding the Output

### During Training
```
Training: 45%|████▌     | 2250/5000 [01:23<01:36, 28.48it/s, win_rate=32.00%, avg_score=287.3, epsilon=0.124]
```
- `win_rate`: Percentage of games won (rolling 100-game average)
- `avg_score`: Average final score (rolling 100-game average)
- `epsilon`: Current exploration rate (decreases over time)

### After Evaluation
```
EVALUATION RESULTS
============================================================
Games Played: 1000
Win Rate: 38.20%
Average Score: 312.5 ± 89.3
Average Rank: 2.1

Rank Distribution:
  Rank 1: 38.2%
  Rank 2: 31.5%
  Rank 3: 20.8%
  Rank 4: 9.5%
============================================================
```

## File Structure

After training and evaluation, your project will look like:

```
bank_ai_project/
├── models/
│   ├── agent_episode_1000.pkl
│   ├── agent_episode_2000.pkl
│   └── agent_final.pkl          ← Your trained agent
├── logs/
│   ├── training_curves.png       ← Training progress
│   ├── banking_analysis.png      ← Banking decision heatmaps
│   └── score_progression.png     ← Score over rounds
└── src/
    └── ... (source code)
```

## Tips for Better Performance

1. **Train longer**: 10,000+ episodes generally produce better agents
2. **Diverse opponents**: Train against `random threshold expected_value self`
3. **Tune hyperparameters**: Try different learning rates (0.05-0.2) and epsilon decay
4. **Self-play**: Include 'self' in opponents for more challenging training

## Troubleshooting

**Problem:** Training is slow
**Solution:** Reduce episodes or players for faster testing

**Problem:** Win rate not improving
**Solution:** Train longer, adjust learning rate, or try opponent modeling agent

**Problem:** Import errors
**Solution:** Ensure you're in the `bank_ai_project` directory and dependencies are installed

## Next Steps

1. Experiment with different hyperparameters
2. Try the opponent modeling agent (`--agent-type opponent_modeling`)
3. Create custom opponent strategies in `src/agents.py`
4. Analyze banking patterns to understand agent behavior
5. Challenge your friends to beat your trained AI!

---

For detailed documentation, see [README.md](README.md)
