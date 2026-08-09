"""
Evaluation module for trained BANK AI agents.

Provides comprehensive evaluation, analytics, and visualization
of agent performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm

from game_engine import BankGame
from agents import Agent, create_agent


class Evaluator:
    """Evaluates trained agents against various opponents."""

    def __init__(self, agent: Agent):
        """
        Initialize evaluator.

        Args:
            agent: Trained agent to evaluate
        """
        self.agent = agent

        # Evaluation metrics
        self.game_results = []
        self.banking_decisions = []  # Track when agent banks
        self.round_by_round_scores = defaultdict(list)

    def evaluate(
        self,
        num_games: int = 1000,
        num_players: int = 4,
        num_rounds: int = 10,
        opponent_types: List[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate agent over multiple games.

        Args:
            num_games: Number of games to play
            num_players: Players per game
            num_rounds: Rounds per game
            opponent_types: Types of opponents to face
            verbose: Print progress

        Returns:
            Dictionary with evaluation results
        """
        if opponent_types is None:
            opponent_types = ['random', 'threshold', 'expected_value']

        if verbose:
            print(f"Evaluating agent over {num_games} games")
            print(f"Opponents: {opponent_types}\n")

        wins = 0
        total_score = 0
        ranks = []

        pbar = tqdm(range(num_games), desc="Evaluating") if verbose else range(num_games)

        for game_num in pbar:
            # Create game
            game = BankGame(num_players=num_players, num_rounds=num_rounds)

            # Add our agent
            game.add_player("Agent", agent=self.agent)
            agent_player_id = 0

            # Add opponents
            for i in range(num_players - 1):
                opp_type = opponent_types[i % len(opponent_types)]
                if opp_type == 'random':
                    opp = create_agent('random', bank_probability=0.3)
                elif opp_type == 'threshold':
                    opp = create_agent('threshold', threshold=150, adaptive=True)
                elif opp_type == 'expected_value':
                    opp = create_agent('expected_value', risk_tolerance=1.0)
                else:
                    opp = create_agent('random')

                game.add_player(f"Opp_{i+1}_{opp_type[:3]}", agent=opp)

            # Play game and track decisions
            self._play_and_track(game, agent_player_id)

            # Record results
            agent_player = game.players[agent_player_id]
            winners = game.get_winner()
            won = agent_player in winners

            if won:
                wins += 1

            total_score += agent_player.score
            rankings = game.get_rankings()
            rank = rankings.index(agent_player) + 1
            ranks.append(rank)

            # Store game result
            self.game_results.append({
                'game_num': game_num,
                'won': won,
                'score': agent_player.score,
                'rank': rank,
                'opponents': opponent_types
            })

            # Update progress bar
            if verbose and game_num % 10 == 0:
                win_rate = wins / (game_num + 1)
                avg_score = total_score / (game_num + 1)
                pbar.set_postfix({
                    'win_rate': f'{win_rate:.2%}',
                    'avg_score': f'{avg_score:.1f}'
                })

        # Calculate final statistics
        win_rate = wins / num_games
        avg_score = total_score / num_games
        avg_rank = np.mean(ranks)

        results = {
            'num_games': num_games,
            'wins': wins,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_rank': avg_rank,
            'score_std': np.std([g['score'] for g in self.game_results]),
            'rank_distribution': {
                rank: ranks.count(rank) / num_games
                for rank in range(1, num_players + 1)
            }
        }

        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION RESULTS")
            print(f"{'='*60}")
            print(f"Games Played: {num_games}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Average Score: {avg_score:.1f} ± {results['score_std']:.1f}")
            print(f"Average Rank: {avg_rank:.2f}")
            print(f"\nRank Distribution:")
            for rank in sorted(results['rank_distribution'].keys()):
                pct = results['rank_distribution'][rank]
                print(f"  Rank {rank}: {pct:.1%}")
            print(f"{'='*60}")

        return results

    def _play_and_track(self, game: BankGame, agent_player_id: int):
        """
        Play a game and track agent's banking decisions.

        Args:
            game: Game instance to play
            agent_player_id: ID of agent's player
        """
        for round_num in range(1, game.num_rounds + 1):
            game.start_new_round()
            round_continues = True

            while round_continues and not game.all_players_banked():
                # Roll dice
                die1, die2 = game.roll_dice()
                round_continues, _ = game.process_roll(die1, die2)

                if not round_continues:
                    break

                # Each player decides whether to bank
                for player in game.players:
                    if player.banked_this_round:
                        continue

                    state = game.get_state(player.id)

                    # Use agent without exploration for evaluation
                    if hasattr(player.agent, 'get_action'):
                        action = player.agent.get_action(state, training=False)
                    else:
                        action = player.agent.get_action(state)

                    if action == 1:  # Bank
                        game.player_banks(player.id)

                        # Track agent's banking decisions
                        if player.id == agent_player_id:
                            self.banking_decisions.append({
                                'round': round_num,
                                'roll_count': game.roll_count,
                                'bank_total': game.bank_total,
                                'score_before': player.score,
                                'score_diff': state['score_difference']
                            })

                if game.all_players_banked():
                    break

            game.end_round()

            # Track round-by-round score
            if agent_player_id < len(game.players):
                self.round_by_round_scores[round_num].append(
                    game.players[agent_player_id].score
                )

        game.game_over = True

    def analyze_banking_patterns(self, save_path: Optional[str] = None):
        """
        Analyze and visualize when the agent banks.

        Args:
            save_path: If provided, save plot to this path
        """
        if not self.banking_decisions:
            print("No banking decisions to analyze. Run evaluate() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Banking by bank total
        bank_totals = [d['bank_total'] for d in self.banking_decisions]
        axes[0, 0].hist(bank_totals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(bank_totals), color='red', linestyle='--',
                          label=f'Mean: {np.mean(bank_totals):.0f}')
        axes[0, 0].set_title('Banking Decision by Bank Total')
        axes[0, 0].set_xlabel('Bank Total')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Banking by roll count
        roll_counts = [d['roll_count'] for d in self.banking_decisions]
        axes[0, 1].hist(roll_counts, bins=range(1, max(roll_counts) + 2),
                       edgecolor='black', alpha=0.7, align='left')
        axes[0, 1].set_title('Banking Decision by Roll Count')
        axes[0, 1].set_xlabel('Roll Number in Round')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Banking heatmap: bank total vs score difference
        # Create bins
        bank_bins = [0, 50, 100, 150, 200, 300, 1000]
        score_diff_bins = [-1000, -100, -50, 0, 50, 100, 1000]

        heatmap_data = np.zeros((len(score_diff_bins) - 1, len(bank_bins) - 1))

        for decision in self.banking_decisions:
            bank_idx = np.digitize([decision['bank_total']], bank_bins)[0] - 1
            diff_idx = np.digitize([decision['score_diff']], score_diff_bins)[0] - 1

            if 0 <= bank_idx < len(bank_bins) - 1 and 0 <= diff_idx < len(score_diff_bins) - 1:
                heatmap_data[diff_idx, bank_idx] += 1

        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'{bank_bins[i]}-{bank_bins[i+1]}' for i in range(len(bank_bins)-1)],
                   yticklabels=[f'{score_diff_bins[i]} to {score_diff_bins[i+1]}' for i in range(len(score_diff_bins)-1)],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Banking Heatmap: Bank Total vs Score Difference')
        axes[1, 0].set_xlabel('Bank Total')
        axes[1, 0].set_ylabel('Score Difference (vs Leader)')

        # 4. Banking by round
        rounds = [d['round'] for d in self.banking_decisions]
        axes[1, 1].hist(rounds, bins=range(1, max(rounds) + 2),
                       edgecolor='black', alpha=0.7, align='left')
        axes[1, 1].set_title('Banking Decision by Round Number')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved banking analysis to {save_path}")
        else:
            plt.show()

    def plot_score_progression(self, save_path: Optional[str] = None):
        """
        Plot how agent's score progresses round by round.

        Args:
            save_path: If provided, save plot to this path
        """
        if not self.round_by_round_scores:
            print("No round data to plot. Run evaluate() first.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        rounds = sorted(self.round_by_round_scores.keys())
        means = [np.mean(self.round_by_round_scores[r]) for r in rounds]
        stds = [np.std(self.round_by_round_scores[r]) for r in rounds]

        ax.plot(rounds, means, marker='o', linewidth=2, label='Mean Score')
        ax.fill_between(rounds,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, label='±1 Std Dev')

        ax.set_title('Score Progression by Round')
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Cumulative Score')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved score progression to {save_path}")
        else:
            plt.show()

    def compare_to_baseline(
        self,
        num_games: int = 100,
        num_players: int = 4,
        num_rounds: int = 10
    ) -> Dict:
        """
        Compare agent performance to baseline agents.

        Args:
            num_games: Number of games for comparison
            num_players: Players per game
            num_rounds: Rounds per game

        Returns:
            Dictionary with comparison results
        """
        print("Comparing to baseline agents...")

        baselines = {
            'Random': create_agent('random', bank_probability=0.3),
            'Threshold': create_agent('threshold', threshold=150),
            'Expected Value': create_agent('expected_value', risk_tolerance=1.0)
        }

        results = {}

        for baseline_name, baseline_agent in baselines.items():
            wins = 0
            agent_scores = []
            baseline_scores = []

            for _ in tqdm(range(num_games), desc=f"vs {baseline_name}"):
                game = BankGame(num_players=2, num_rounds=num_rounds)

                # Randomly assign positions
                if np.random.random() < 0.5:
                    game.add_player("Agent", agent=self.agent)
                    game.add_player(baseline_name, agent=baseline_agent)
                    agent_id = 0
                else:
                    game.add_player(baseline_name, agent=baseline_agent)
                    game.add_player("Agent", agent=self.agent)
                    agent_id = 1

                # Play game
                game.play_game(verbose=False)

                # Check who won
                winners = game.get_winner()
                if game.players[agent_id] in winners:
                    wins += 1

                agent_scores.append(game.players[agent_id].score)
                baseline_scores.append(game.players[1 - agent_id].score)

            win_rate = wins / num_games

            results[baseline_name] = {
                'win_rate': win_rate,
                'agent_avg_score': np.mean(agent_scores),
                'baseline_avg_score': np.mean(baseline_scores),
                'score_advantage': np.mean(agent_scores) - np.mean(baseline_scores)
            }

            print(f"\nvs {baseline_name}:")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Agent Avg Score: {results[baseline_name]['agent_avg_score']:.1f}")
            print(f"  {baseline_name} Avg Score: {results[baseline_name]['baseline_avg_score']:.1f}")
            print(f"  Score Advantage: {results[baseline_name]['score_advantage']:+.1f}")

        return results


if __name__ == "__main__":
    # Test evaluation
    print("Testing BANK Evaluation Module\n")

    # Load or create a simple agent for testing
    from agents import ThresholdAgent

    agent = ThresholdAgent(threshold=150, adaptive=True)

    # Create evaluator
    evaluator = Evaluator(agent)

    # Run evaluation
    results = evaluator.evaluate(
        num_games=100,
        num_players=4,
        num_rounds=10,
        opponent_types=['random', 'threshold'],
        verbose=True
    )

    # Analyze banking patterns
    evaluator.analyze_banking_patterns(save_path='../logs/banking_analysis.png')

    # Plot score progression
    evaluator.plot_score_progression(save_path='../logs/score_progression.png')

    # Compare to baselines
    baseline_results = evaluator.compare_to_baseline(num_games=50)
