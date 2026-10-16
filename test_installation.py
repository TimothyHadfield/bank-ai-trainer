#!/usr/bin/env python3
"""
Quick test script to verify BANK AI installation.

Run this to ensure all dependencies are installed and modules work correctly.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False

    try:
        import seaborn as sns
        print("  ✓ seaborn")
    except ImportError as e:
        print(f"  ✗ seaborn: {e}")
        return False

    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False

    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False

    return True


def test_modules():
    """Test that all project modules can be imported."""
    print("\nTesting project modules...")

    try:
        from game_engine import BankGame, GamePhase, Player
        print("  ✓ game_engine")
    except ImportError as e:
        print(f"  ✗ game_engine: {e}")
        return False

    try:
        from agents import (
            Agent, RandomAgent, ThresholdAgent,
            ExpectedValueAgent, QLearningAgent,
            OpponentModelingAgent, create_agent
        )
        print("  ✓ agents")
    except ImportError as e:
        print(f"  ✗ agents: {e}")
        return False

    try:
        from training import Trainer
        print("  ✓ training")
    except ImportError as e:
        print(f"  ✗ training: {e}")
        return False

    try:
        from evaluation import Evaluator
        print("  ✓ evaluation")
    except ImportError as e:
        print(f"  ✗ evaluation: {e}")
        return False

    try:
        from play_human import HumanPlayer, play_against_ai
        print("  ✓ play_human")
    except ImportError as e:
        print(f"  ✗ play_human: {e}")
        return False

    return True


def test_game_simulation():
    """Test that a simple game can be simulated."""
    print("\nTesting game simulation...")

    try:
        from game_engine import BankGame
        from agents import create_agent

        # Create game
        game = BankGame(num_players=2, num_rounds=2, seed=42)

        # Add players with agents
        game.add_player("Agent1", agent=create_agent('random'))
        game.add_player("Agent2", agent=create_agent('threshold', threshold=100))

        # Play game
        result = game.play_game(verbose=False)

        # Verify result structure
        assert 'winners' in result
        assert 'final_scores' in result
        assert len(result['winners']) > 0

        print("  ✓ Game simulation works")
        print(f"    Winners: {result['winners']}")
        print(f"    Scores: {result['final_scores']}")

        return True

    except Exception as e:
        print(f"  ✗ Game simulation failed: {e}")
        return False


def test_agent_creation():
    """Test that all agent types can be created."""
    print("\nTesting agent creation...")

    try:
        from agents import create_agent, QLearningAgent, OpponentModelingAgent

        # Test factory function
        agent1 = create_agent('random')
        print("  ✓ RandomAgent")

        agent2 = create_agent('threshold', threshold=150)
        print("  ✓ ThresholdAgent")

        agent3 = create_agent('expected_value', risk_tolerance=1.0)
        print("  ✓ ExpectedValueAgent")

        agent4 = create_agent('qlearning')
        print("  ✓ QLearningAgent")

        agent5 = create_agent('opponent_modeling')
        print("  ✓ OpponentModelingAgent")

        # Test Q-learning functionality
        from game_engine import BankGame
        game = BankGame(num_players=2, num_rounds=1, seed=42)
        game.add_player("Q", agent=agent4)
        game.add_player("Random", agent=agent1)

        state = game.get_state(0)
        action = agent4.get_action(state)
        assert action in [0, 1], "Action must be 0 or 1"

        print("  ✓ Q-learning get_action works")

        return True

    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False


def test_training_basic():
    """Test that basic training works."""
    print("\nTesting training (quick run)...")

    try:
        from agents import QLearningAgent
        from training import Trainer

        # Create agent
        agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.3
        )

        # Create trainer
        trainer = Trainer(
            agent=agent,
            num_players=2,
            num_rounds=3,
            opponent_types=['random']
        )

        # Train for just a few episodes
        results = trainer.train(
            num_episodes=5,
            save_interval=100,
            eval_interval=100,
            verbose=False
        )

        assert results['episodes'] == 5
        assert 'final_win_rate' in results
        assert 'episode_scores' in results

        print("  ✓ Training works")
        print(f"    Episodes: {results['episodes']}")
        print(f"    Win rate: {results['final_win_rate']:.1%}")

        return True

    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test that trained agents can be saved and loaded (pickle)."""
    print("\nTesting save/load...")

    try:
        import tempfile
        from agents import Agent, QLearningAgent, OpponentModelingAgent

        for cls in (QLearningAgent, OpponentModelingAgent):
            agent = cls()
            agent.q_table[(0, 0, 0, 0, 0, 0)][1] = 5.0
            if isinstance(agent, OpponentModelingAgent):
                agent.record_opponent_bank(1, 120)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "agent.pkl")
                agent.save(path)
                loaded = Agent.load(path)
            assert loaded.q_table[(0, 0, 0, 0, 0, 0)][1] == 5.0
            assert loaded.q_table[(9, 9, 9, 9, 9, 9)] == [0.0, 0.0]
            print(f"  ✓ {cls.__name__} save/load")

        return True

    except Exception as e:
        print(f"  ✗ Save/load failed: {e}")
        return False


def test_evaluation_basic():
    """Test that evaluation runs against baseline opponents."""
    print("\nTesting evaluation (quick run)...")

    try:
        from agents import QLearningAgent
        from evaluation import Evaluator

        evaluator = Evaluator(QLearningAgent())
        results = evaluator.evaluate(
            num_games=3,
            num_players=4,
            num_rounds=2,
            opponent_types=['random', 'threshold', 'expected_value'],
            verbose=False
        )
        assert results['num_games'] == 3

        print("  ✓ Evaluation works")
        return True

    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("BANK AI - Installation Test")
    print("="*60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some dependencies are missing. Install with: pip install -r requirements.txt")

    # Test modules
    if not test_modules():
        all_passed = False
        print("\n⚠ Some project modules failed to import.")

    # Test functionality
    if not test_game_simulation():
        all_passed = False

    if not test_agent_creation():
        all_passed = False

    if not test_training_basic():
        all_passed = False

    if not test_save_load():
        all_passed = False

    if not test_evaluation_basic():
        all_passed = False

    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYour installation is working correctly.")
        print("\nNext steps:")
        print("  1. Train an agent: python main.py train --episodes 1000")
        print("  2. Evaluate it: python main.py evaluate --model models/agent_final.pkl")
        print("  3. Play against it: python main.py play --model models/agent_final.pkl")
        print("\nSee QUICKSTART.md for more examples.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
