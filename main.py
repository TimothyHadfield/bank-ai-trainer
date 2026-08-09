#!/usr/bin/env python3
"""
Main entry point for BANK AI Training Project.

Provides command-line interface for training, evaluating, and playing
against AI agents.
"""

import argparse
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import QLearningAgent, OpponentModelingAgent, Agent
from training import Trainer
from evaluation import Evaluator
from play_human import play_against_ai


def train_agent(args):
    """Train a new agent."""
    print(f"Training {args.agent_type} agent...")
    print(f"Episodes: {args.episodes}")
    print(f"Players: {args.players}, Rounds: {args.rounds}")
    print(f"Opponents: {args.opponents}")
    print()

    # Create agent
    if args.agent_type == 'qlearning':
        agent = QLearningAgent(
            learning_rate=args.lr,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min
        )
    elif args.agent_type == 'opponent_modeling':
        agent = OpponentModelingAgent(
            learning_rate=args.lr,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min
        )
    else:
        print(f"Unknown agent type: {args.agent_type}")
        return

    # Create trainer
    trainer = Trainer(
        agent=agent,
        num_players=args.players,
        num_rounds=args.rounds,
        opponent_types=args.opponents,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )

    # Train
    results = trainer.train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        verbose=True
    )

    # Plot training curves
    if args.plot:
        plot_path = os.path.join(args.log_dir, 'training_curves.png')
        trainer.plot_training_curves(save_path=plot_path)

    print(f"\n✓ Training complete!")
    print(f"  Final Win Rate: {results['final_win_rate']:.2%}")
    print(f"  Final Avg Score: {results['final_avg_score']:.1f}")
    print(f"  Model saved to: {args.save_dir}/agent_final.pkl")


def evaluate_agent(args):
    """Evaluate a trained agent."""
    # Load agent
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    print(f"Loading agent from {args.model}...")
    agent = Agent.load(args.model)

    # Create evaluator
    evaluator = Evaluator(agent)

    # Run evaluation
    print(f"\nEvaluating agent over {args.eval_games} games...")
    results = evaluator.evaluate(
        num_games=args.eval_games,
        num_players=args.players,
        num_rounds=args.rounds,
        opponent_types=args.opponents,
        verbose=True
    )

    # Banking pattern analysis
    if args.analyze:
        print("\nAnalyzing banking patterns...")
        analysis_path = os.path.join(args.log_dir, 'banking_analysis.png')
        evaluator.analyze_banking_patterns(save_path=analysis_path)

        print("Plotting score progression...")
        progression_path = os.path.join(args.log_dir, 'score_progression.png')
        evaluator.plot_score_progression(save_path=progression_path)

    # Compare to baselines
    if args.baseline:
        print("\nComparing to baseline agents...")
        baseline_results = evaluator.compare_to_baseline(
            num_games=args.baseline_games,
            num_players=args.players,
            num_rounds=args.rounds
        )

    print("\n✓ Evaluation complete!")


def play_mode(args):
    """Play against a trained agent."""
    # Load agent
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model not found at {args.model}")
            return

        print(f"Loading agent from {args.model}...")
        agent = Agent.load(args.model)
    else:
        # Use default agent
        from agents import create_agent
        print("No model specified. Using Adaptive Threshold Agent.")
        agent = create_agent('threshold', threshold=150, adaptive=True)

    # Play game
    play_against_ai(agent, num_rounds=args.rounds, verbose=True)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='BANK AI Training Project - Train and evaluate RL agents for the BANK dice game',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a Q-learning agent
  python main.py train --episodes 5000 --opponents random threshold

  # Train with custom hyperparameters
  python main.py train --lr 0.2 --gamma 0.99 --epsilon 0.5

  # Evaluate a trained agent
  python main.py evaluate --model models/agent_final.pkl --analyze

  # Play against a trained agent
  python main.py play --model models/agent_final.pkl

  # Play against default agent
  python main.py play
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new agent')
    train_parser.add_argument(
        '--agent-type',
        type=str,
        default='qlearning',
        choices=['qlearning', 'opponent_modeling'],
        help='Type of agent to train (default: qlearning)'
    )
    train_parser.add_argument(
        '--episodes',
        type=int,
        default=5000,
        help='Number of training episodes (default: 5000)'
    )
    train_parser.add_argument(
        '--players',
        type=int,
        default=4,
        help='Number of players per game (default: 4)'
    )
    train_parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of rounds per game (default: 10)'
    )
    train_parser.add_argument(
        '--opponents',
        nargs='+',
        default=['random', 'threshold'],
        choices=['random', 'threshold', 'expected_value', 'self'],
        help='Opponent types to train against (default: random threshold)'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    train_parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='Discount factor (default: 0.95)'
    )
    train_parser.add_argument(
        '--epsilon',
        type=float,
        default=0.3,
        help='Initial exploration rate (default: 0.3)'
    )
    train_parser.add_argument(
        '--epsilon-decay',
        type=float,
        default=0.9995,
        help='Epsilon decay rate (default: 0.9995)'
    )
    train_parser.add_argument(
        '--epsilon-min',
        type=float,
        default=0.01,
        help='Minimum epsilon (default: 0.01)'
    )
    train_parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Save model every N episodes (default: 1000)'
    )
    train_parser.add_argument(
        '--eval-interval',
        type=int,
        default=500,
        help='Evaluate every N episodes (default: 500)'
    )
    train_parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    train_parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory to save logs and plots (default: logs)'
    )
    train_parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate training curve plots'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    eval_parser.add_argument(
        '--eval-games',
        type=int,
        default=1000,
        help='Number of evaluation games (default: 1000)'
    )
    eval_parser.add_argument(
        '--players',
        type=int,
        default=4,
        help='Number of players per game (default: 4)'
    )
    eval_parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of rounds per game (default: 10)'
    )
    eval_parser.add_argument(
        '--opponents',
        nargs='+',
        default=['random', 'threshold', 'expected_value'],
        choices=['random', 'threshold', 'expected_value'],
        help='Opponent types to evaluate against (default: random threshold expected_value)'
    )
    eval_parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate banking pattern analysis and visualizations'
    )
    eval_parser.add_argument(
        '--baseline',
        action='store_true',
        help='Compare to baseline agents'
    )
    eval_parser.add_argument(
        '--baseline-games',
        type=int,
        default=100,
        help='Number of games for baseline comparison (default: 100)'
    )
    eval_parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory to save analysis plots (default: logs)'
    )

    # Play command
    play_parser = subparsers.add_parser('play', help='Play against a trained agent')
    play_parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model file (optional, uses default agent if not provided)'
    )
    play_parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of rounds per game (default: 10)'
    )

    args = parser.parse_args()

    # Execute command
    if args.command == 'train':
        train_agent(args)
    elif args.command == 'evaluate':
        evaluate_agent(args)
    elif args.command == 'play':
        play_mode(args)
    else:
        parser.print_help()
        print("\nError: Please specify a command (train, evaluate, or play)")
        sys.exit(1)


if __name__ == "__main__":
    main()
