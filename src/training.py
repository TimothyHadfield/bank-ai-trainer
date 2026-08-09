"""
Training module for BANK AI agents.

Handles training loops, opponent management, reward calculation,
and progress tracking.
"""

import os
import time
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from game_engine import BankGame
from agents import Agent, QLearningAgent, create_agent


class Trainer:
    """Manages the training process for BANK AI agents."""

    def __init__(
        self,
        agent: Agent,
        num_players: int = 4,
        num_rounds: int = 10,
        opponent_types: List[str] = None,
        save_dir: str = "models",
        log_dir: str = "logs"
    ):
        """
        Initialize trainer.

        Args:
            agent: The agent to train
            num_players: Number of players per game
            num_rounds: Rounds per game
            opponent_types: List of opponent agent types
            save_dir: Directory to save models
            log_dir: Directory to save training logs
        """
        self.agent = agent
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Default opponents if none specified
        if opponent_types is None:
            opponent_types = ['random', 'threshold']

        self.opponent_types = opponent_types

        # Training metrics
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_scores = []
        self.win_rate_history = []

    def create_opponents(self, num_opponents: int) -> List[Agent]:
        """
        Create opponent agents for training.

        Args:
            num_opponents: Number of opponents to create

        Returns:
            List of opponent agents
        """
        opponents = []

        for i in range(num_opponents):
            # Cycle through opponent types
            opp_type = self.opponent_types[i % len(self.opponent_types)]

            if opp_type == 'self':
                # Self-play: create a copy of the learning agent
                if isinstance(self.agent, QLearningAgent):
                    opp = QLearningAgent(
                        learning_rate=self.agent.learning_rate,
                        discount_factor=self.agent.discount_factor,
                        epsilon=0.0  # Opponents don't explore
                    )
                    # Share Q-table for true self-play
                    opp.q_table = self.agent.q_table
                else:
                    opp = create_agent('random')
            elif opp_type == 'random':
                opp = create_agent('random', bank_probability=np.random.uniform(0.2, 0.4))
            elif opp_type == 'threshold':
                threshold = np.random.randint(100, 200)
                opp = create_agent('threshold', threshold=threshold, adaptive=True)
            elif opp_type == 'expected_value':
                risk_tolerance = np.random.uniform(0.8, 1.2)
                opp = create_agent('expected_value', risk_tolerance=risk_tolerance)
            else:
                raise ValueError(f"Unknown opponent type: {opp_type}")

            opponents.append(opp)

        return opponents

    def calculate_reward(
        self,
        agent_player_id: int,
        game: BankGame,
        round_num: int,
        banked: bool,
        busted: bool = False
    ) -> float:
        """
        Calculate reward for the agent's action.

        Reward shaping is crucial for learning. We reward:
        - Winning the game (large bonus)
        - Banking successfully (proportional to amount)
        - Being in the lead (small ongoing reward)
        - Penalize busting (losing unbanked points)

        Args:
            agent_player_id: ID of the agent's player
            game: Current game state
            round_num: Current round number
            banked: Whether the agent banked this round
            busted: Whether the round busted (7 rolled)

        Returns:
            Reward value (can be negative)
        """
        player = game.players[agent_player_id]
        reward = 0.0

        # Game end bonus
        if game.game_over:
            winners = game.get_winner()
            if player in winners:
                reward += 100.0  # Large reward for winning
            else:
                # Penalty proportional to how far behind we are
                winner_score = max(p.score for p in winners)
                score_diff = winner_score - player.score
                reward -= score_diff * 0.1

        # Banking reward
        if banked:
            # Reward proportional to amount banked
            reward += game.bank_total * 0.1

            # Bonus for banking at good times
            if game.roll_count >= 4:  # Risky phase
                reward += 5.0  # Bonus for banking before bust

        # Bust penalty
        if busted and not banked:
            # Lost potential points
            reward -= game.bank_total * 0.15

        # Position-based reward
        rankings = game.get_rankings()
        player_rank = rankings.index(player)

        if player_rank == 0:
            reward += 2.0  # Small bonus for being in the lead
        elif player_rank == len(rankings) - 1:
            reward -= 1.0  # Small penalty for being last

        return reward

    def train_episode(self) -> Dict:
        """
        Run one training episode (one complete game).

        Returns:
            Dictionary with episode statistics
        """
        # Create game
        game = BankGame(
            num_players=self.num_players,
            num_rounds=self.num_rounds
        )

        # Add our learning agent
        game.add_player("Agent", agent=self.agent)
        agent_player_id = 0

        # Add opponents
        opponents = self.create_opponents(self.num_players - 1)
        for i, opp in enumerate(opponents):
            game.add_player(f"Opponent_{i+1}", agent=opp)

        # Track states and actions for learning
        episode_data = []

        # Play the game
        for round_num in range(1, self.num_rounds + 1):
            game.start_new_round()
            round_continues = True
            round_busted = False

            while round_continues and not game.all_players_banked():
                # Get state before roll
                state_before = game.get_state(agent_player_id)

                # Roll dice
                die1, die2 = game.roll_dice()
                round_continues, _ = game.process_roll(die1, die2)

                if not round_continues:
                    round_busted = True
                    break

                # Get state after roll
                state_after = game.get_state(agent_player_id)

                # Each player (including agent) decides whether to bank
                for player in game.players:
                    if player.banked_this_round:
                        continue

                    state = game.get_state(player.id)
                    action = player.agent.get_action(state, training=True)

                    if action == 1:  # Bank
                        game.player_banks(player.id)

                        # Record for agent if it's our agent
                        if player.id == agent_player_id:
                            episode_data.append({
                                'state': state_before,
                                'action': action,
                                'next_state': state_after,
                                'banked': True,
                                'round': round_num
                            })

                # Check if all players banked
                if game.all_players_banked():
                    break

            # Calculate reward for this round
            agent_player = game.players[agent_player_id]
            reward = self.calculate_reward(
                agent_player_id,
                game,
                round_num,
                agent_player.banked_this_round,
                round_busted
            )

            # Update agent for this round's decisions
            for data in episode_data:
                if data['round'] == round_num:
                    done = (round_num == self.num_rounds)
                    self.agent.update(
                        data['state'],
                        data['action'],
                        reward,
                        data['next_state'],
                        done
                    )

            game.end_round()

        # Game finished
        game.game_over = True
        final_reward = self.calculate_reward(
            agent_player_id,
            game,
            self.num_rounds,
            False,
            False
        )

        # Get results
        agent_player = game.players[agent_player_id]
        winners = game.get_winner()
        won = agent_player in winners

        return {
            'won': won,
            'score': agent_player.score,
            'reward': final_reward,
            'rankings': [p.name for p in game.get_rankings()],
            'agent_rank': game.get_rankings().index(agent_player)
        }

    def train(
        self,
        num_episodes: int,
        save_interval: int = 1000,
        eval_interval: int = 500,
        verbose: bool = True
    ) -> Dict:
        """
        Train the agent for multiple episodes.

        Args:
            num_episodes: Number of training episodes
            save_interval: Save model every N episodes
            eval_interval: Evaluate agent every N episodes
            verbose: Print progress

        Returns:
            Dictionary with training statistics
        """
        if verbose:
            print(f"Starting training for {num_episodes} episodes")
            print(f"Players: {self.num_players}, Rounds: {self.num_rounds}")
            print(f"Opponents: {self.opponent_types}")
            print(f"Agent: {type(self.agent).__name__}")
            print()

        start_time = time.time()

        # Use deque for rolling averages
        recent_wins = deque(maxlen=100)
        recent_scores = deque(maxlen=100)

        # Progress bar
        pbar = tqdm(range(num_episodes), desc="Training") if verbose else range(num_episodes)

        for episode in pbar:
            # Run episode
            result = self.train_episode()

            # Record metrics
            self.episode_rewards.append(result['reward'])
            self.episode_wins.append(1 if result['won'] else 0)
            self.episode_scores.append(result['score'])

            recent_wins.append(1 if result['won'] else 0)
            recent_scores.append(result['score'])

            # Decay epsilon (if applicable)
            if isinstance(self.agent, QLearningAgent):
                self.agent.decay_epsilon()

            # Update progress bar
            if verbose and episode % 10 == 0:
                win_rate = np.mean(recent_wins) if recent_wins else 0
                avg_score = np.mean(recent_scores) if recent_scores else 0
                epsilon = self.agent.epsilon if isinstance(self.agent, QLearningAgent) else 0

                pbar.set_postfix({
                    'win_rate': f'{win_rate:.2%}',
                    'avg_score': f'{avg_score:.1f}',
                    'epsilon': f'{epsilon:.3f}'
                })

            # Evaluate periodically
            if (episode + 1) % eval_interval == 0:
                win_rate = np.mean(recent_wins) if recent_wins else 0
                self.win_rate_history.append(win_rate)

            # Save periodically
            if (episode + 1) % save_interval == 0:
                save_path = os.path.join(self.save_dir, f"agent_episode_{episode+1}.pkl")
                self.agent.save(save_path)
                if verbose:
                    print(f"\nSaved model to {save_path}")

        # Final save
        final_save_path = os.path.join(self.save_dir, "agent_final.pkl")
        self.agent.save(final_save_path)

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\n\nTraining complete!")
            print(f"Time elapsed: {elapsed_time:.1f} seconds")
            print(f"Final win rate (last 100): {np.mean(recent_wins):.2%}")
            print(f"Final avg score (last 100): {np.mean(recent_scores):.1f}")

            if isinstance(self.agent, QLearningAgent):
                stats = self.agent.get_statistics()
                print(f"\nAgent statistics:")
                print(f"  Episodes: {stats['episodes']}")
                print(f"  Q-table size: {stats['q_table_size']}")
                print(f"  Final epsilon: {stats['epsilon']:.4f}")

        return {
            'episodes': num_episodes,
            'final_win_rate': np.mean(recent_wins) if recent_wins else 0,
            'final_avg_score': np.mean(recent_scores) if recent_scores else 0,
            'training_time': elapsed_time,
            'episode_rewards': self.episode_rewards,
            'episode_wins': self.episode_wins,
            'episode_scores': self.episode_scores,
            'win_rate_history': self.win_rate_history
        }

    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.

        Args:
            save_path: If provided, save plot to this path
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Win rate over time
        window = 100
        if len(self.episode_wins) >= window:
            win_rate = np.convolve(self.episode_wins, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(win_rate)
            axes[0, 0].set_title(f'Win Rate (rolling {window}-episode average)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Win Rate')
            axes[0, 0].grid(True)

        # Average score over time
        if len(self.episode_scores) >= window:
            avg_score = np.convolve(self.episode_scores, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(avg_score)
            axes[0, 1].set_title(f'Average Score (rolling {window}-episode average)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True)

        # Reward distribution
        axes[1, 0].hist(self.episode_rewards[-1000:], bins=50, edgecolor='black')
        axes[1, 0].set_title('Reward Distribution (last 1000 episodes)')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)

        # Score distribution
        axes[1, 1].hist(self.episode_scores[-1000:], bins=50, edgecolor='black')
        axes[1, 1].set_title('Score Distribution (last 1000 episodes)')
        axes[1, 1].set_xlabel('Final Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Test training
    print("Testing BANK Training Module\n")

    # Create a Q-learning agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.999,
        epsilon_min=0.01
    )

    # Create trainer
    trainer = Trainer(
        agent=agent,
        num_players=4,
        num_rounds=10,
        opponent_types=['random', 'threshold'],
        save_dir='../models',
        log_dir='../logs'
    )

    # Run short training
    results = trainer.train(
        num_episodes=100,
        save_interval=50,
        eval_interval=25,
        verbose=True
    )

    print(f"\nTraining Results:")
    print(f"Final Win Rate: {results['final_win_rate']:.2%}")
    print(f"Final Avg Score: {results['final_avg_score']:.1f}")
    print(f"Training Time: {results['training_time']:.1f}s")

    # Plot training curves
    trainer.plot_training_curves(save_path='../logs/training_test.png')
