"""
AI Agents for BANK Game.

Includes Q-Learning agent, baseline agents (Random, Threshold, Expected Value),
and support for opponent modeling.
"""

import random
import pickle
from typing import Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from game_engine import calculate_expected_value, GamePhase


class Agent:
    """Base class for all agents."""

    def get_action(self, state: Dict) -> int:
        """
        Decide whether to bank or wait.

        Args:
            state: Current game state dictionary

        Returns:
            0 = WAIT, 1 = BANK
        """
        raise NotImplementedError

    def update(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Update agent after taking action (for learning agents)."""
        pass

    def save(self, filepath: str):
        """Save agent to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class RandomAgent(Agent):
    """Agent that banks randomly with given probability."""

    def __init__(self, bank_probability: float = 0.3):
        self.bank_probability = bank_probability

    def get_action(self, state: Dict) -> int:
        if state['player_banked']:
            return 0  # Already banked, can't bank again

        return 1 if random.random() < self.bank_probability else 0


class ThresholdAgent(Agent):
    """Agent that banks when bank total exceeds a threshold."""

    def __init__(self, threshold: int = 150, adaptive: bool = False):
        """
        Args:
            threshold: Bank when total exceeds this
            adaptive: If True, adjust threshold based on score difference
        """
        self.threshold = threshold
        self.adaptive = adaptive

    def get_action(self, state: Dict) -> int:
        if state['player_banked']:
            return 0

        threshold = self.threshold

        # Adaptive threshold based on game situation
        if self.adaptive:
            # If behind, wait for more points
            if state['score_difference'] < -50:
                threshold += 50
            # If ahead, bank earlier to protect lead
            elif state['score_difference'] > 50:
                threshold -= 30

            # Higher threshold early in game, lower threshold late
            if state['current_round'] <= 3:
                threshold += 20
            elif state['current_round'] >= 8:
                threshold -= 20

        return 1 if state['bank_total'] >= threshold else 0


class ExpectedValueAgent(Agent):
    """Agent that banks based on expected value calculation."""

    def __init__(self, risk_tolerance: float = 1.0):
        """
        Args:
            risk_tolerance: Multiplier for EV comparison
                          < 1.0 = conservative (bank early)
                          > 1.0 = aggressive (wait longer)
        """
        self.risk_tolerance = risk_tolerance

    def get_action(self, state: Dict) -> int:
        if state['player_banked']:
            return 0

        ev_wait, ev_bank = calculate_expected_value(
            state['bank_total'],
            state['roll_count']
        )

        # Bank if expected value of banking is better (adjusted by risk tolerance)
        return 1 if ev_bank >= ev_wait * self.risk_tolerance else 0


class QLearningAgent(Agent):
    """
    Q-Learning agent with state discretization.

    Uses tabular Q-learning with epsilon-greedy exploration.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01
    ):
        """
        Args:
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for future reward discounting
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps (state, action) -> Q-value
        self.q_table = defaultdict(lambda: [0.0, 0.0])  # [wait, bank]

        # Track learning statistics
        self.episodes = 0
        self.total_reward = 0.0

    def discretize_state(self, state: Dict) -> Tuple:
        """
        Convert continuous state to discrete tuple for Q-table lookup.

        This is crucial for tabular Q-learning - we bucket continuous values
        into discrete categories to keep the state space manageable.
        """
        # Bucket bank total
        if state['bank_total'] <= 50:
            bank_bucket = 0
        elif state['bank_total'] <= 100:
            bank_bucket = 1
        elif state['bank_total'] <= 150:
            bank_bucket = 2
        elif state['bank_total'] <= 200:
            bank_bucket = 3
        elif state['bank_total'] <= 300:
            bank_bucket = 4
        else:
            bank_bucket = 5

        # Bucket score difference
        diff = state['score_difference']
        if diff <= -100:
            diff_bucket = 0  # Far behind
        elif diff <= -50:
            diff_bucket = 1  # Behind
        elif diff <= 50:
            diff_bucket = 2  # Close
        elif diff <= 100:
            diff_bucket = 3  # Ahead
        else:
            diff_bucket = 4  # Far ahead

        # Bucket round number
        round_num = state['current_round']
        if round_num <= 3:
            round_bucket = 0  # Early game
        elif round_num <= 7:
            round_bucket = 1  # Mid game
        else:
            round_bucket = 2  # Late game

        # Bucket roll count
        roll_count = state['roll_count']
        if roll_count <= 3:
            roll_bucket = 0  # Safe rolls
        elif roll_count <= 6:
            roll_bucket = 1  # Moderately risky
        else:
            roll_bucket = 2  # Very risky

        # Number of players who have banked
        banked_bucket = min(state['num_banked'], 3)  # 0, 1, 2, or 3+

        # Have we already banked?
        already_banked = 1 if state['player_banked'] else 0

        return (
            bank_bucket,
            diff_bucket,
            round_bucket,
            roll_bucket,
            banked_bucket,
            already_banked
        )

    def get_action(self, state: Dict, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current game state
            training: If True, use epsilon-greedy; if False, always use greedy

        Returns:
            0 = WAIT, 1 = BANK
        """
        # Can't bank if already banked
        if state['player_banked']:
            return 0

        discrete_state = self.discretize_state(state)

        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, 1)
        else:
            # Exploit: choose best action
            q_values = self.q_table[discrete_state]
            return int(np.argmax(q_values))

    def update(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """
        Update Q-values using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        current_q = self.q_table[discrete_state][action]

        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: include discounted future reward
            max_next_q = max(self.q_table[discrete_next_state])
            target = reward + self.discount_factor * max_next_q

        # Q-learning update
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[discrete_state][action] = new_q

        self.total_reward += reward

    def decay_epsilon(self):
        """Decay exploration rate after episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes += 1

    def get_q_value(self, state: Dict, action: int) -> float:
        """Get Q-value for state-action pair."""
        discrete_state = self.discretize_state(state)
        return self.q_table[discrete_state][action]

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'episodes': self.episodes,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.episodes),
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
        }


class OpponentModelingAgent(QLearningAgent):
    """
    Enhanced Q-Learning agent that models opponent behavior.

    Tracks when opponents typically bank and uses this as additional state information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track opponent banking patterns
        # Format: {opponent_id: {bank_total: count}}
        self.opponent_banking_history = defaultdict(lambda: defaultdict(int))
        self.opponent_total_banks = defaultdict(int)

    def record_opponent_bank(self, opponent_id: int, bank_total: int):
        """Record when an opponent banks."""
        # Bucket the bank total
        bucket = bank_total // 50  # 0-49, 50-99, 100-149, etc.
        self.opponent_banking_history[opponent_id][bucket] += 1
        self.opponent_total_banks[opponent_id] += 1

    def get_opponent_bank_probability(self, opponent_id: int, bank_total: int) -> float:
        """
        Estimate probability that opponent will bank at this total.

        Returns value between 0 and 1.
        """
        if self.opponent_total_banks[opponent_id] == 0:
            return 0.3  # Default estimate

        bucket = bank_total // 50
        bucket_count = self.opponent_banking_history[opponent_id][bucket]
        total_count = self.opponent_total_banks[opponent_id]

        return bucket_count / total_count

    def discretize_state(self, state: Dict) -> Tuple:
        """
        Extended state representation including opponent model.
        """
        base_state = super().discretize_state(state)

        # Add opponent aggressiveness estimate
        # This is a simplified version - could be extended
        if state['num_banked'] > 0:
            # Opponents are banking early
            opp_aggressiveness = 0  # Conservative
        else:
            opp_aggressiveness = 1  # Aggressive (waiting)

        return base_state + (opp_aggressiveness,)


def create_agent(agent_type: str, **kwargs) -> Agent:
    """
    Factory function to create agents by type name.

    Args:
        agent_type: One of 'random', 'threshold', 'expected_value', 'qlearning', 'opponent_modeling'
        **kwargs: Agent-specific parameters

    Returns:
        Agent instance
    """
    agent_types = {
        'random': RandomAgent,
        'threshold': ThresholdAgent,
        'expected_value': ExpectedValueAgent,
        'qlearning': QLearningAgent,
        'opponent_modeling': OpponentModelingAgent,
    }

    if agent_type not in agent_types:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(agent_types.keys())}")

    return agent_types[agent_type](**kwargs)


if __name__ == "__main__":
    # Test agents
    print("Testing BANK Agents\n")

    # Create sample state
    test_state = {
        'player_id': 0,
        'player_score': 100,
        'player_banked': False,
        'bank_total': 150,
        'current_round': 5,
        'roll_count': 4,
        'phase': GamePhase.RISKY_ROLLS,
        'num_banked': 1,
        'num_players': 4,
        'max_other_score': 120,
        'score_difference': -20,
        'rounds_remaining': 5,
    }

    # Test each agent type
    agents = {
        'Random': RandomAgent(),
        'Threshold': ThresholdAgent(threshold=150),
        'Expected Value': ExpectedValueAgent(),
        'Q-Learning': QLearningAgent(),
    }

    print(f"Test State: Bank={test_state['bank_total']}, "
          f"Score={test_state['player_score']}, Round={test_state['current_round']}")
    print()

    for name, agent in agents.items():
        action = agent.get_action(test_state)
        action_str = "BANK" if action == 1 else "WAIT"
        print(f"{name:20s}: {action_str}")

    # Test Q-learning update
    print("\n\nTesting Q-Learning Update:")
    q_agent = QLearningAgent()

    print(f"Initial Q-values: {q_agent.get_q_value(test_state, 0):.3f} (WAIT), "
          f"{q_agent.get_q_value(test_state, 1):.3f} (BANK)")

    # Simulate getting a reward for banking
    q_agent.update(test_state, action=1, reward=10.0, next_state=test_state, done=True)

    print(f"After update:     {q_agent.get_q_value(test_state, 0):.3f} (WAIT), "
          f"{q_agent.get_q_value(test_state, 1):.3f} (BANK)")

    print(f"\nQ-Learning Stats: {q_agent.get_statistics()}")
