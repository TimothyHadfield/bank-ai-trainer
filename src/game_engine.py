"""
BANK Game Engine - Complete simulation of the dice game BANK.

Handles all game logic including dice rolling, doubles, sevens, banking,
and round/game end conditions.
"""

import random
from typing import List, Tuple, Dict, Optional
from enum import Enum


class GamePhase(Enum):
    """Tracks which phase of the round we're in."""
    SAFE_ROLLS = "safe_rolls"  # First 3 rolls
    RISKY_ROLLS = "risky_rolls"  # Roll 4+


class Player:
    """Represents a player in the game."""

    def __init__(self, player_id: int, name: str, agent=None):
        self.id = player_id
        self.name = name
        self.agent = agent  # AI agent controlling this player
        self.score = 0
        self.banked_this_round = False

    def reset_round(self):
        """Reset player state for a new round."""
        self.banked_this_round = False

    def bank(self, amount: int):
        """Add banked amount to player's score."""
        self.score += amount
        self.banked_this_round = True

    def __repr__(self):
        return f"Player({self.name}, score={self.score})"


class BankGame:
    """
    Complete BANK game simulation.

    Supports configurable rules, multiple players, and full game state tracking.
    """

    def __init__(
        self,
        num_players: int = 4,
        num_rounds: int = 10,
        safe_rolls: int = 3,
        seven_bonus: int = 70,
        seed: Optional[int] = None
    ):
        """
        Initialize a BANK game.

        Args:
            num_players: Number of players (2-8)
            num_rounds: Number of rounds to play (10, 15, or 20 typical)
            safe_rolls: Number of initial rolls where 7 = bonus and doubles = face value
            seven_bonus: Points awarded for rolling 7 during safe rolls
            seed: Random seed for reproducibility
        """
        if num_players < 2:
            raise ValueError("Need at least 2 players")
        if num_rounds < 1:
            raise ValueError("Need at least 1 round")

        self.num_players = num_players
        self.num_rounds = num_rounds
        self.safe_rolls = safe_rolls
        self.seven_bonus = seven_bonus

        if seed is not None:
            random.seed(seed)

        self.players: List[Player] = []
        self.current_round = 0
        self.roll_count = 0
        self.bank_total = 0
        self.game_over = False
        self.round_history = []  # Track what happened each round

    def add_player(self, name: str, agent=None) -> Player:
        """Add a player to the game."""
        if len(self.players) >= self.num_players:
            raise ValueError(f"Game already has {self.num_players} players")
        player = Player(len(self.players), name, agent)
        self.players.append(player)
        return player

    def get_state(self, player_id: int) -> Dict:
        """
        Get current game state from perspective of a specific player.

        Returns a dictionary with all relevant state information for decision-making.
        """
        player = self.players[player_id]
        other_scores = [p.score for p in self.players if p.id != player_id]
        max_other_score = max(other_scores) if other_scores else 0

        # Count how many players have banked
        num_banked = sum(1 for p in self.players if p.banked_this_round)

        # Determine game phase
        phase = GamePhase.SAFE_ROLLS if self.roll_count < self.safe_rolls else GamePhase.RISKY_ROLLS

        return {
            'player_id': player_id,
            'player_score': player.score,
            'player_banked': player.banked_this_round,
            'bank_total': self.bank_total,
            'current_round': self.current_round,
            'roll_count': self.roll_count,
            'phase': phase,
            'num_banked': num_banked,
            'num_players': self.num_players,
            'max_other_score': max_other_score,
            'score_difference': player.score - max_other_score,
            'rounds_remaining': self.num_rounds - self.current_round,
        }

    def roll_dice(self) -> Tuple[int, int]:
        """Roll two six-sided dice."""
        return random.randint(1, 6), random.randint(1, 6)

    def process_roll(self, die1: int, die2: int) -> Tuple[bool, str]:
        """
        Process a dice roll and update bank total.

        Returns:
            (round_continues, message) tuple
            - round_continues: False if round should end (7 rolled in risky phase)
            - message: Description of what happened
        """
        self.roll_count += 1
        total = die1 + die2
        is_double = (die1 == die2)
        is_seven = (total == 7)
        in_safe_phase = self.roll_count <= self.safe_rolls

        message = f"Roll #{self.roll_count}: {die1} + {die2} = {total}"

        # Handle seven
        if is_seven:
            if in_safe_phase:
                self.bank_total += self.seven_bonus
                message += f" (LUCKY SEVEN! +{self.seven_bonus})"
                return True, message
            else:
                message += " (BUSTED! Round ends, all unbanked points lost)"
                return False, message

        # Handle doubles
        if is_double:
            if in_safe_phase:
                self.bank_total += total
                message += f" (Doubles, +{total})"
            else:
                old_total = self.bank_total
                self.bank_total *= 2
                message += f" (DOUBLES! Bank doubled from {old_total} to {self.bank_total})"
        else:
            # Regular roll
            self.bank_total += total
            message += f" (Bank total: {self.bank_total})"

        return True, message

    def player_can_bank(self, player_id: int) -> bool:
        """Check if a player is eligible to bank."""
        return not self.players[player_id].banked_this_round

    def player_banks(self, player_id: int) -> str:
        """
        Process a player banking.

        Returns:
            Message describing the banking action
        """
        player = self.players[player_id]

        if player.banked_this_round:
            return f"{player.name} has already banked this round"

        player.bank(self.bank_total)
        return f"{player.name} banks {self.bank_total} points! (Total score: {player.score})"

    def all_players_banked(self) -> bool:
        """Check if all players have banked this round."""
        return all(p.banked_this_round for p in self.players)

    def start_new_round(self):
        """Initialize a new round."""
        self.current_round += 1
        self.roll_count = 0
        self.bank_total = 0

        # Reset all players
        for player in self.players:
            player.reset_round()

    def end_round(self):
        """
        Finalize the current round.

        Records round history for analysis.
        """
        round_data = {
            'round': self.current_round,
            'final_bank': self.bank_total,
            'rolls': self.roll_count,
            'player_scores': {p.name: p.score for p in self.players},
            'banked_players': [p.name for p in self.players if p.banked_this_round]
        }
        self.round_history.append(round_data)

    def get_winner(self) -> List[Player]:
        """
        Get the winner(s) of the game.

        Returns list in case of tie.
        """
        max_score = max(p.score for p in self.players)
        return [p for p in self.players if p.score == max_score]

    def get_rankings(self) -> List[Player]:
        """Get players sorted by score (descending)."""
        return sorted(self.players, key=lambda p: p.score, reverse=True)

    def play_round(self, verbose: bool = False) -> Dict:
        """
        Play one complete round of BANK.

        Returns:
            Dictionary with round results
        """
        self.start_new_round()

        if verbose:
            print(f"\n{'='*60}")
            print(f"ROUND {self.current_round}/{self.num_rounds}")
            print(f"{'='*60}")

        round_continues = True
        messages = []

        while round_continues and not self.all_players_banked():
            # Roll dice
            die1, die2 = self.roll_dice()
            round_continues, roll_msg = self.process_roll(die1, die2)
            messages.append(roll_msg)

            if verbose:
                print(roll_msg)

            if not round_continues:
                break

            # Give each player a chance to bank
            for player in self.players:
                if not player.banked_this_round and player.agent:
                    state = self.get_state(player.id)
                    action = player.agent.get_action(state)

                    if action == 1:  # Bank
                        bank_msg = self.player_banks(player.id)
                        messages.append(bank_msg)
                        if verbose:
                            print(f"  → {bank_msg}")

            # Check if all players banked
            if self.all_players_banked():
                if verbose:
                    print("All players have banked. Round ends.")
                break

        self.end_round()

        if verbose:
            print(f"\nRound {self.current_round} complete!")
            print("Current Standings:")
            for i, player in enumerate(self.get_rankings(), 1):
                print(f"  {i}. {player.name}: {player.score}")

        return {
            'round': self.current_round,
            'messages': messages,
            'final_scores': {p.name: p.score for p in self.players}
        }

    def play_game(self, verbose: bool = False) -> Dict:
        """
        Play a complete game of BANK.

        Returns:
            Dictionary with complete game results
        """
        if verbose:
            print("="*60)
            print("STARTING NEW BANK GAME")
            print(f"Players: {', '.join(p.name for p in self.players)}")
            print(f"Rounds: {self.num_rounds}")
            print("="*60)

        for _ in range(self.num_rounds):
            self.play_round(verbose=verbose)

        self.game_over = True
        winners = self.get_winner()

        if verbose:
            print("\n" + "="*60)
            print("GAME OVER!")
            print("="*60)
            print("\nFinal Standings:")
            for i, player in enumerate(self.get_rankings(), 1):
                print(f"  {i}. {player.name}: {player.score}")
            if len(winners) == 1:
                print(f"\n🏆 Winner: {winners[0].name} with {winners[0].score} points!")
            else:
                print(f"\n🏆 Tie between: {', '.join(w.name for w in winners)}")

        return {
            'winners': [w.name for w in winners],
            'final_scores': {p.name: p.score for p in self.players},
            'rankings': [p.name for p in self.get_rankings()],
            'round_history': self.round_history
        }


def calculate_expected_value(
    bank_total: int,
    roll_count: int,
    safe_rolls: int = 3,
    seven_bonus: int = 70
) -> Tuple[float, float]:
    """
    Calculate expected value of waiting vs. banking.

    This provides a mathematical baseline for comparison.

    Returns:
        (ev_wait, ev_bank) tuple
        - ev_wait: Expected value of waiting for another roll
        - ev_bank: Expected value of banking now (just the current total)
    """
    # Banking gives you the current total with certainty
    ev_bank = bank_total

    # Calculate probability and expected value of waiting
    in_safe_phase = roll_count < safe_rolls

    if in_safe_phase:
        # In safe phase: all rolls are beneficial
        # Expected dice sum = 7
        # 7 gives bonus, doubles give 2x their face value avg = 7
        # P(seven) = 6/36, P(doubles) = 6/36, P(other) = 24/36
        p_seven = 6/36
        p_doubles = 6/36
        p_normal = 24/36

        ev_wait = bank_total + (p_seven * seven_bonus + p_doubles * 7 + p_normal * 7)
    else:
        # In risky phase: 7 busts, doubles double the total
        p_seven = 6/36  # Bust!
        p_doubles = 6/36  # Doubles the total
        p_normal = 24/36  # Adds dice sum

        # Expected value considering bust risk
        ev_seven = 0  # Lose everything
        ev_doubles = bank_total * 2
        ev_normal = bank_total + 7

        ev_wait = p_seven * ev_seven + p_doubles * ev_doubles + p_normal * ev_normal

    return ev_wait, ev_bank


if __name__ == "__main__":
    # Test the game engine
    print("Testing BANK Game Engine\n")

    # Create a simple test game
    game = BankGame(num_players=2, num_rounds=2, seed=42)

    # Add players (without agents for now)
    game.add_player("Alice")
    game.add_player("Bob")

    # Manually play through some actions
    print("Starting Round 1...")
    game.start_new_round()

    # Simulate a few rolls
    for i in range(5):
        die1, die2 = game.roll_dice()
        continues, msg = game.process_roll(die1, die2)
        print(msg)

        if not continues:
            print("Round ended due to bust!")
            break

        # Bank after 3 rolls for testing
        if i == 2:
            print(game.player_banks(0))  # Alice banks

    game.end_round()

    print(f"\nRound complete. Alice's score: {game.players[0].score}")
    print(f"Bob's score: {game.players[1].score}")

    # Test expected value calculation
    print("\n\nTesting Expected Value Calculation:")
    for roll_num in [1, 3, 5]:
        for total in [50, 100, 200]:
            ev_wait, ev_bank = calculate_expected_value(total, roll_num)
            print(f"Roll {roll_num}, Bank={total}: EV(wait)={ev_wait:.1f}, EV(bank)={ev_bank:.1f}")
