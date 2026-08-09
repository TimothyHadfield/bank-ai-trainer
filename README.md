# BANK Game AI - Reinforcement Learning Project

An AI trained to play the dice game BANK using Q-Learning and Deep Q-Networks (DQN).

🔗 **Live browser trainer:** https://timothyhadfield.github.io/bank-ai-trainer/

The repo has two halves that can be used independently: a self-contained browser trainer
([`index.html`](index.html) — no install, no backend) and the full Python training stack
under [`src/`](src/) driven by [`main.py`](main.py).

## Game Rules

**BANK** is a strategic dice game for 2+ players:

- Players compete over configurable rounds (10, 15, or 20)
- Each turn, players roll two dice, adding the sum to a shared BANK total
- **Doubles**:
  - First 3 rolls: face value only (e.g., 5-5 = 10 points)
  - Roll 4+: entire BANK total is doubled
- **Rolling a 7**:
  - First 3 rolls: worth 70 points (added to BANK)
  - Roll 4+: ends the round immediately, all unbanked points lost
- **Banking**: Any player can call "BANK" at any time to lock in the current total
- Round ends when a 7 is rolled (after roll 3) or all players bank
- Highest personal score after all rounds wins

## Project Structure

```
bank_ai_project/
├── src/
│   ├── game_engine.py      # Core game simulation
│   ├── agents.py            # AI agent implementations
│   ├── training.py          # Training loop and utilities
│   ├── evaluation.py        # Agent evaluation and analysis
│   └── play_human.py        # Interactive human vs AI mode
├── models/                  # Saved trained models
├── logs/                    # Training logs and metrics
├── main.py                  # Main entry point
└── requirements.txt         # Python dependencies
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train a New Agent

```bash
# Train with default parameters
python main.py train

# Train with custom parameters
python main.py train --episodes 100000 --learning-rate 0.01 --epsilon 0.2

# Train against different opponents
python main.py train --opponents random threshold self
```

### Evaluate a Trained Agent

```bash
# Evaluate against various opponents
python main.py evaluate --model models/best_agent.pkl --games 1000

# Detailed analysis with visualizations
python main.py evaluate --model models/best_agent.pkl --analyze
```

### Play Against the AI

```bash
# Interactive human vs AI mode
python main.py play --model models/best_agent.pkl
```

### Command-Line Arguments

**Training:**
- `--episodes`: Number of training episodes (default: 50000)
- `--learning-rate`: Q-learning alpha parameter (default: 0.1)
- `--discount`: Discount factor gamma (default: 0.95)
- `--epsilon`: Initial exploration rate (default: 0.3)
- `--epsilon-decay`: Epsilon decay rate (default: 0.9995)
- `--epsilon-min`: Minimum epsilon value (default: 0.01)
- `--opponents`: Opponent types to train against (default: random threshold)
- `--num-rounds`: Rounds per game (default: 10)
- `--num-players`: Number of players (default: 4)
- `--save-interval`: Episodes between saves (default: 1000)

**Evaluation:**
- `--model`: Path to trained model file
- `--games`: Number of evaluation games (default: 1000)
- `--analyze`: Generate detailed analytics and visualizations
- `--verbose`: Print game-by-game details

**Play:**
- `--model`: Path to trained model file
- `--num-rounds`: Rounds per game (default: 10)
- `--opponents`: Number of AI opponents (default: 3)

## Agent Types

1. **QLearningAgent**: Tabular Q-learning with state discretization
2. **DQNAgent**: Deep Q-Network for large state spaces (optional enhancement)
3. **RandomAgent**: Banks randomly for baseline comparison
4. **ThresholdAgent**: Banks when BANK total exceeds threshold
5. **ExpectedValueAgent**: Banks based on mathematical expected value

## State Representation

The agent observes:
- Current BANK total (bucketed: 0-50, 51-100, 101-200, 201-300, 300+)
- Score difference vs. leader (bucketed: behind by 100+, behind by 50-99, close ±49, ahead by 50-99, ahead by 100+)
- Current round (early: 1-3, mid: 4-7, late: 8+)
- Roll number within round (1-3: safe, 4-6: risky, 7+: very risky)
- Players already banked this round (0, 1, 2, all)
- Agent already banked (boolean)

## Training Features

- **Self-play**: Agents learn by competing against copies of themselves
- **Curriculum learning**: Gradually increase opponent difficulty
- **Experience replay**: Store and sample past experiences (DQN variant)
- **Opponent modeling**: Track opponent banking patterns
- **Progressive epsilon decay**: Balance exploration and exploitation

## Evaluation Metrics

- Win rate vs. different opponent types
- Average score per game
- Banking decision heatmaps (BANK total vs. score difference)
- Round-by-round banking patterns
- Comparison to expected value baseline

## Advanced Features

### Opponent Modeling
Track when opponents typically bank and use this information to make better decisions.

### Expected Value Calculation
Compare learned strategy against mathematically optimal expected value baseline.

### Visualization
- Training curves (win rate, average score over episodes)
- Banking decision heatmaps
- Score distribution histograms
- Round-by-round analysis

## Examples

### Quick Start
```bash
# Train for 10,000 episodes and save
python main.py train --episodes 10000 --save-interval 500

# Evaluate the trained agent
python main.py evaluate --model models/agent_episode_10000.pkl --games 1000

# Play against the AI
python main.py play --model models/agent_episode_10000.pkl
```

### Advanced Training
```bash
# Train with self-play and curriculum learning
python main.py train --episodes 50000 --opponents self --epsilon 0.5 --epsilon-decay 0.9998

# Train for tournament play (more players)
python main.py train --num-players 6 --num-rounds 15 --episodes 30000
```

## Tips for Best Results

1. **Start with random opponents** to learn basic banking strategies
2. **Gradually introduce tougher opponents** (threshold → self-play)
3. **Higher exploration initially** (epsilon ~0.3-0.5) with slow decay
4. **Train for sufficient episodes** (50,000+ for stable performance)
5. **Evaluate frequently** to track progress and prevent overfitting
6. **Adjust state bucketing** if agent doesn't converge

## Performance Expectations

A well-trained agent should:
- Beat random agents >80% of the time
- Beat threshold agents ~60-70% of the time
- Compete evenly with other trained agents in self-play
- Bank more aggressively when ahead
- Wait longer when behind and need points
- Recognize high-risk situations (roll 4+ with high BANK total)

## Troubleshooting

**Agent not learning:**
- Increase learning rate (0.1-0.3)
- Reduce state space (fewer buckets)
- Train longer (50,000+ episodes)

**Agent too conservative:**
- Reward taking risks when behind
- Penalize early banking in late rounds

**Agent too aggressive:**
- Increase penalty for busting
- Train against more conservative opponents

## Future Enhancements

- [ ] Multi-agent deep RL (Actor-Critic methods)
- [ ] Monte Carlo Tree Search (MCTS) for planning
- [ ] Neural network function approximation
- [ ] Tournament mode with multiple agents
- [ ] Web-based UI for human play
- [ ] Mobile app integration

## License

MIT License - feel free to use and modify for your projects!

## Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- Performance improvements
- New agent types
- Better visualizations
- Documentation improvements
