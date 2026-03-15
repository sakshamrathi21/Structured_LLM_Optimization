import numpy as np
import pickle
from collections import deque
from typing import Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class StateAction:
    batch_size: int
    chunks_before: int
    chunks_after: int
    requests_waiting: int
    action: bool # True: add request ; False: don't add

    def to_tuple(self):
        return (self.batch_size, self.chunks_before, self.chunks_after, self.requests_waiting, self.action)
  

class QLearningScheduler:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 1000,
        state_discretization: dict = None
    ):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        
        if state_discretization is None:
            state_discretization = {
                'batch_size': [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                'chunk_diff': [-20, -10, -5, -2, -1, 0, 1, 2],  # chunks_before - chunks_after
                'requests_waiting': [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            }
        
        self.state_discretization = state_discretization
        
        # Q-table: (discretized_state, action) -> Q-value
        self.q_table = {}
        
        # Track episodes and steps
        self.step_count = 0
        self.episode_count = 0
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Metrics tracking
        self.metrics = {
            'rewards': [],
            'epsilons': [],
            'q_values': [],
            'decisions': {'add': 0, 'skip': 0}
        }
        
    def discretize_state(self, batch_size: int, chunks_before: int, 
                        chunks_after: int, requests_waiting: int) -> Tuple:
        """Convert continuous state to discrete buckets"""
        
        # Discretize batch size
        batch_bucket = self._find_bucket(batch_size, self.state_discretization['batch_size'])
        
        # Discretize chunk difference (this is the key trade-off metric)
        chunk_diff = chunks_before - chunks_after
        chunk_diff_bucket = self._find_bucket(chunk_diff, self.state_discretization['chunk_diff'])
        
        # Discretize requests waiting
        requests_bucket = self._find_bucket(requests_waiting, 
                                           self.state_discretization['requests_waiting'])
        
        return (batch_bucket, chunk_diff_bucket, requests_bucket)
    
    def _find_bucket(self, value: float, buckets: list) -> int:
        """Find which bucket a value falls into"""
        for i, threshold in enumerate(buckets):
            if value <= threshold:
                return i
        return len(buckets)
    
    def get_q_value(self, state: Tuple, action: bool) -> float:
        """Get Q-value for a state-action pair"""
        key = (state, action)
        return self.q_table.get(key, 0.0)
    
    def should_add_request(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
        exploration: bool = True
    ) -> bool:
        """
        Decide whether to add the next request to the batch.
        
        Args:
            batch_size: Current number of requests in batch
            chunks_before: Shared chunks before adding request
            chunks_after: Shared chunks after adding request
            requests_waiting: Number of requests at same prefix level
            exploration: Whether to use epsilon-greedy exploration
            
        Returns:
            True if should add request, False otherwise
        """
        # Always add if batch is empty
        if batch_size == 0:
            return True
        
        # Discretize state
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        
        # Epsilon-greedy action selection
        if exploration and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.random() < 0.5
            self.metrics['decisions']['add' if action else 'skip'] += 1
            return action
        else:
            # Exploit: choose action with highest Q-value
            q_add = self.get_q_value(state, True)
            q_skip = self.get_q_value(state, False)
            
            action = q_add > q_skip
            self.metrics['decisions']['add' if action else 'skip'] += 1
            
            # Track Q-values for monitoring
            self.metrics['q_values'].append(max(q_add, q_skip))
            
            return action
    
    def update(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
        action: bool,
        reward: float,
        next_batch_size: Optional[int] = None,
        next_chunks_before: Optional[int] = None,
        next_chunks_after: Optional[int] = None,
        next_requests_waiting: Optional[int] = None
    ):
        """
        Update Q-values based on observed reward.
        
        Args:
            batch_size, chunks_before, chunks_after, requests_waiting: Current state
            action: Action taken
            reward: Observed reward (negative execution time)
            next_*: Next state (optional, for TD learning)
        """
        # Discretize current state
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Compute TD target
        if next_batch_size is not None:
            # Have next state: use TD learning
            next_state = self.discretize_state(next_batch_size, next_chunks_before,
                                              next_chunks_after, next_requests_waiting)
            max_next_q = max(self.get_q_value(next_state, True),
                           self.get_q_value(next_state, False))
            td_target = reward + self.gamma * max_next_q
        else:
            # Terminal state or no next state: just use reward
            td_target = reward
        
        # Q-learning update
        new_q = current_q + self.alpha * (td_target - current_q)
        
        # Update Q-table
        key = (state, action)
        self.q_table[key] = new_q
        
        # Update step count and epsilon
        self.step_count += 1
        self._update_epsilon()
        
        # Track metrics
        self.metrics['rewards'].append(reward)
        self.metrics['epsilons'].append(self.epsilon)
        
        # Store experience in replay buffer
        self.replay_buffer.append({
            'state': (batch_size, chunks_before, chunks_after, requests_waiting),
            'action': action,
            'reward': reward,
            'next_state': (next_batch_size, next_chunks_before, 
                          next_chunks_after, next_requests_waiting) if next_batch_size is not None else None
        })
    
    def _update_epsilon(self):
        """Decay epsilon over time (linear decay)"""
        if self.step_count < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.step_count / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end
    
    def save(self, filepath: str):
        """Save the Q-table and model state"""
        state = {
            'q_table': self.q_table,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'metrics': self.metrics,
            'state_discretization': self.state_discretization
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load the Q-table and model state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.q_table = state['q_table']
        self.step_count = state['step_count']
        self.episode_count = state['episode_count']
        self.epsilon = state['epsilon']
        self.metrics = state['metrics']
        self.state_discretization = state['state_discretization']
    
    def get_statistics(self) -> dict:
        """Get training statistics"""
        recent_rewards = self.metrics['rewards'][-100:] if self.metrics['rewards'] else []
        recent_q_values = self.metrics['q_values'][-100:] if self.metrics['q_values'] else []
        
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'avg_recent_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_recent_q_value': np.mean(recent_q_values) if recent_q_values else 0,
            'total_decisions': sum(self.metrics['decisions'].values()),
            'add_ratio': self.metrics['decisions']['add'] / max(1, sum(self.metrics['decisions'].values()))
        }


class ContextualBanditScheduler:
    """
    Simpler contextual bandit approach using Upper Confidence Bound (UCB).
    """
    
    def __init__(self, exploration_factor: float = 2.0):
        """
        Args:
            exploration_factor: UCB exploration parameter (higher = more exploration)
        """
        self.c = exploration_factor
        
        # Track statistics for each discretized state-action pair
        self.action_counts = {}  # (state, action) -> count
        self.action_rewards = {}  # (state, action) -> sum of rewards
        
        self.total_steps = 0
        
    def discretize_state(self, batch_size: int, chunks_before: int,
                        chunks_after: int, requests_waiting: int) -> str:
        """Simple state discretization into a string key"""
        chunk_diff = chunks_before - chunks_after
        
        # Bin the values
        batch_bin = min(batch_size // 4, 10)  # 0-3, 4-7, 8-11, ..., 40+
        chunk_diff_bin = max(-5, min(5, chunk_diff))  # -5 to 5
        requests_bin = min(requests_waiting // 4, 10)  # Similar binning
        
        return f"{batch_bin}_{chunk_diff_bin}_{requests_bin}"
    
    def get_ucb_score(self, state: str, action: bool) -> float:
        """Calculate UCB score for a state-action pair"""
        key = (state, action)
        
        if key not in self.action_counts or self.action_counts[key] == 0:
            return float('inf')  # Encourage trying unexplored actions
        
        count = self.action_counts[key]
        avg_reward = self.action_rewards[key] / count
        
        # UCB formula: avg_reward + c * sqrt(log(total_steps) / count)
        exploration_bonus = self.c * np.sqrt(np.log(self.total_steps + 1) / count)
        
        return avg_reward + exploration_bonus
    
    def should_add_request(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int
    ) -> bool:
        """Decide whether to add request using UCB"""
        if batch_size == 0:
            return True
        
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        
        # Calculate UCB scores for both actions
        ucb_add = self.get_ucb_score(state, True)
        ucb_skip = self.get_ucb_score(state, False)
        
        return ucb_add > ucb_skip
    
    def update(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
        action: bool,
        reward: float
    ):
        """Update statistics based on observed reward"""
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        key = (state, action)
        
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        self.action_rewards[key] = self.action_rewards.get(key, 0.0) + reward
        
        self.total_steps += 1
    
    def save(self, filepath: str):
        """Save model state"""
        state = {
            'action_counts': self.action_counts,
            'action_rewards': self.action_rewards,
            'total_steps': self.total_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load model state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.action_counts = state['action_counts']
        self.action_rewards = state['action_rewards']
        self.total_steps = state['total_steps']


class HeuristicScheduler:
    """
    Simple heuristic-based scheduler as baseline.
    Uses hand-crafted rules based on domain knowledge.
    """
    
    def __init__(
        self,
        max_chunk_loss: int = 2,
        min_batch_size: int = 10,
    ):
        """
        Args:
            max_chunk_loss: Maximum acceptable loss in shared chunks
            min_batch_size: Minimum batch size before being conservative
            max_batch_size: Maximum batch size
        """
        self.max_chunk_loss = max_chunk_loss
        self.min_batch_size = min_batch_size
    
    def should_add_request(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int
    ) -> bool:
        """Hand-crafted decision rule"""
        if batch_size == 0:
            return True
        
        chunk_loss = chunks_before - chunks_after
        
        # If we're maintaining or improving chunk sharing, always add
        if chunk_loss <= 0:
            return True
        
        # If batch is small, be more aggressive
        if batch_size < self.min_batch_size:
            return chunk_loss <= self.max_chunk_loss + 2
        
        # For larger batches, be more conservative about chunk loss
        if chunk_loss <= self.max_chunk_loss:
            return True
        
        # If many requests are waiting at this level, might be worth the chunk loss
        if requests_waiting >= 10 and chunk_loss <= self.max_chunk_loss + 1:
            return True
        
        return False