"""
Human Play Mode - Interactive CLI to play BANK against AI agents.

Allows human players to compete against trained AI agents in a
text-based interface.
"""

import os
from typing import Optional
from game_engine import BankGame
from agents import Agent, create_agent


class HumanPlayer:
    """Human player interface for BANK game."""

    def __init__(self, name: str = "Human"):
        self.name = name

    def get_action(self, state: dict) -> int:
        """
        Get banking decision from human player.

        Args:
            state: Current game state

        Returns:
            0 = WAIT, 1 = BANK
        """
        if state['player_banked']:
            return 0  # Already banked

        print(f"\n{'='*60}")
        print(f"YOUR TURN - Round {state['current_round']}")
        print(f"{'='*60}")
        print(f"Your Score: {state['player_score']}")
        print(f"Opponent's Best Score: {state['max_other_score']}")
        print(f"Score Difference: {state['score_difference']:+d}")
        print(f"\nBank Total: {state['bank_total']}")
        print(f"Roll Count: {state['roll_count']}")
        print(f"Phase: {state['phase'].value}")
        print(f"Players who banked: {state['num_banked']}/{state['num_players']}")
        print(f"Rounds Remaining: {state['rounds_remaining']}")

        while True:
            choice = input("\n[B]ank or [W]ait? ").strip().lower()
            if choice in ['b', 'bank']:
                return 1
            elif choice in ['w', 'wait']:
                return 0
            else:
                print("Invalid choice. Please enter 'B' for Bank or 'W' for Wait.")


def play_against_ai(
    agent: Agent,
    num_rounds: int = 10,
    verbose: bool = True
):
    """
    Play a game of BANK against an AI agent.

    Args:
        agent: AI agent to play against
        num_rounds: Number of rounds in the game
        verbose: Print detailed game information
    """
    print("\n" + "="*60)
    print("WELCOME TO BANK - HUMAN vs AI")
    print("="*60)
    print("\nGame Rules:")
    print("- Roll dice to accumulate points in the 'bank total'")
    print("- First 3 rolls: 7 = +70 bonus, doubles = face value")
    print("- After 3 rolls: 7 = BUST (lose all unbanked points)")
    print("                  doubles = DOUBLE the bank total")
    print("- Decide to BANK (keep points) or WAIT (risk for more)")
    print("- Highest score after all rounds wins!")
    print("="*60)

    # Create game
    game = BankGame(num_players=2, num_rounds=num_rounds)

    # Add players
    human = HumanPlayer("You")
    game.add_player("You", agent=human)
    game.add_player("AI", agent=agent)

    # Play the game
    for round_num in range(1, num_rounds + 1):
        game.start_new_round()

        if verbose:
            print(f"\n\n{'#'*60}")
            print(f"# ROUND {round_num}/{num_rounds}")
            print(f"{'#'*60}")

        round_continues = True
        round_busted = False

        while round_continues and not game.all_players_banked():
            # Roll dice
            die1, die2 = game.roll_dice()
            round_continues, roll_msg = game.process_roll(die1, die2)

            if verbose:
                print(f"\n{roll_msg}")

            if not round_continues:
                round_busted = True
                print("\n💥 ROUND BUSTED! All unbanked points lost.")
                break

            # Each player decides whether to bank
            for player in game.players:
                if player.banked_this_round:
                    continue

                state = game.get_state(player.id)
                action = player.agent.get_action(state)

                if action == 1:  # Bank
                    bank_msg = game.player_banks(player.id)
                    if verbose:
                        if player.name == "You":
                            print(f"\n✓ {bank_msg}")
                        else:
                            print(f"\n🤖 {bank_msg}")

            # Check if all players banked
            if game.all_players_banked():
                if verbose:
                    print("\nAll players have banked. Round ends.")
                break

        game.end_round()

        # Show standings after round
        if verbose:
            print(f"\n--- Round {round_num} Complete ---")
            print("Current Standings:")
            for i, player in enumerate(game.get_rankings(), 1):
                emoji = "👤" if player.name == "You" else "🤖"
                print(f"  {i}. {emoji} {player.name}: {player.score}")

    # Game over
    game.game_over = True
    winners = game.get_winner()

    print("\n\n" + "="*60)
    print("GAME OVER!")
    print("="*60)
    print("\nFinal Standings:")
    for i, player in enumerate(game.get_rankings(), 1):
        emoji = "👤" if player.name == "You" else "🤖"
        print(f"  {i}. {emoji} {player.name}: {player.score}")

    if game.players[0] in winners:  # Human won
        if len(winners) == 1:
            print("\n🎉 CONGRATULATIONS! YOU WIN! 🎉")
        else:
            print("\n🤝 IT'S A TIE!")
    else:
        print("\n😔 AI WINS. Better luck next time!")

    print("="*60)

    return game


def main():
    """Main function for interactive play."""
    print("\n" + "="*60)
    print("BANK - Human vs AI Mode")
    print("="*60)

    # Choose opponent
    print("\nChoose your opponent:")
    print("1. Random Agent (Easy)")
    print("2. Threshold Agent (Medium)")
    print("3. Expected Value Agent (Hard)")
    print("4. Load trained Q-Learning Agent (Very Hard)")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

    if choice == '1':
        agent = create_agent('random', bank_probability=0.3)
        print("\n🎲 Playing against Random Agent")
    elif choice == '2':
        agent = create_agent('threshold', threshold=150, adaptive=True)
        print("\n📊 Playing against Adaptive Threshold Agent")
    elif choice == '3':
        agent = create_agent('expected_value', risk_tolerance=1.0)
        print("\n🧮 Playing against Expected Value Agent")
    else:
        # Load trained agent
        print("\nAvailable trained agents:")
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

        if not os.path.exists(models_dir):
            print("No models directory found. Training an agent first...")
            agent = create_agent('threshold', threshold=150, adaptive=True)
        else:
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

            if not model_files:
                print("No trained models found. Using Threshold Agent instead.")
                agent = create_agent('threshold', threshold=150, adaptive=True)
            else:
                print("\nAvailable models:")
                for i, model_file in enumerate(model_files, 1):
                    print(f"  {i}. {model_file}")

                while True:
                    model_choice = input(f"\nSelect model (1-{len(model_files)}): ").strip()
                    try:
                        idx = int(model_choice) - 1
                        if 0 <= idx < len(model_files):
                            break
                    except ValueError:
                        pass
                    print(f"Invalid choice. Please enter a number between 1 and {len(model_files)}.")

                model_path = os.path.join(models_dir, model_files[idx])
                agent = Agent.load(model_path)
                print(f"\n🧠 Loaded trained agent from {model_files[idx]}")

    # Choose number of rounds
    while True:
        rounds_input = input("\nNumber of rounds (default 10): ").strip()
        if not rounds_input:
            num_rounds = 10
            break
        try:
            num_rounds = int(rounds_input)
            if num_rounds > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Play the game
    play_against_ai(agent, num_rounds=num_rounds, verbose=True)

    # Play again?
    while True:
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again in ['y', 'yes']:
            main()
            break
        elif again in ['n', 'no']:
            print("\nThanks for playing BANK! Goodbye!")
            break
        else:
            print("Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()
