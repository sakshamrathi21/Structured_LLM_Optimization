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
    action: bool

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
                'chunk_diff': [-20, -10, -5, -2, -1, 0, 1, 2],
                'requests_waiting': [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            }
        
        self.state_discretization = state_discretization
        self.q_table = {}
        self.step_count = 0
        self.episode_count = 0
        self.replay_buffer = deque(maxlen=10000)
        
        self.metrics = {
            'rewards': [],
            'epsilons': [],
            'q_values': [],
            'decisions': {'add': 0, 'skip': 0}
        }
        
    def discretize_state(self, batch_size: int, chunks_before: int, 
                        chunks_after: int, requests_waiting: int) -> Tuple:
        batch_bucket = self._find_bucket(batch_size, self.state_discretization['batch_size'])
        chunk_diff = chunks_before - chunks_after
        chunk_diff_bucket = self._find_bucket(chunk_diff, self.state_discretization['chunk_diff'])
        requests_bucket = self._find_bucket(requests_waiting, self.state_discretization['requests_waiting'])
        return (batch_bucket, chunk_diff_bucket, requests_bucket)
    
    def _find_bucket(self, value: float, buckets: list) -> int:
        for i, threshold in enumerate(buckets):
            if value <= threshold:
                return i
        return len(buckets)
    
    def get_q_value(self, state: Tuple, action: bool) -> float:
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
        if batch_size == 0:
            return True

        if chunks_before == chunks_after and chunks_before > 0:
            return True
        
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        chunk_diff = chunks_before - chunks_after
        q_add  = self.get_q_value(state, True)
        q_skip = self.get_q_value(state, False)

        if exploration and np.random.random() < self.epsilon:
            action = np.random.random() < 0.5
            decision_type = "EXPLORE"
        else:
            action = q_add > q_skip
            decision_type = "EXPLOIT"
            self.metrics['q_values'].append(max(q_add, q_skip))

        self.metrics['decisions']['add' if action else 'skip'] += 1

        print(
            f"[QL | DECISION | {decision_type}] "
            f"state={state} bs={batch_size} chunk_diff={chunk_diff} rw={requests_waiting} | "
            f"q_add={q_add:.4f} q_skip={q_skip:.4f} | "
            f"action={'ADD' if action else 'SKIP'} eps={self.epsilon:.4f} step={self.step_count}"
        )

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
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        current_q = self.get_q_value(state, action)
        chunk_diff = chunks_before - chunks_after

        if next_batch_size is not None:
            next_state = self.discretize_state(next_batch_size, next_chunks_before,
                                              next_chunks_after, next_requests_waiting)
            max_next_q = max(self.get_q_value(next_state, True),
                           self.get_q_value(next_state, False))
            td_target = reward + self.gamma * max_next_q
        else:
            next_state = None
            max_next_q = 0.0
            td_target = reward

        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error

        self.q_table[(state, action)] = new_q
        self.step_count += 1
        self._update_epsilon()
        
        self.metrics['rewards'].append(reward)
        self.metrics['epsilons'].append(self.epsilon)
        
        self.replay_buffer.append({
            'state': (batch_size, chunks_before, chunks_after, requests_waiting),
            'action': action,
            'reward': reward,
            'next_state': (next_batch_size, next_chunks_before, 
                          next_chunks_after, next_requests_waiting) if next_batch_size is not None else None
        })

        print(
            f"[QL | UPDATE] "
            f"state={state} bs={batch_size} chunk_diff={chunk_diff} rw={requests_waiting} | "
            f"action={'ADD' if action else 'SKIP'} reward={reward:.4f} | "
            f"old_q={current_q:.4f} td_error={td_error:.4f} new_q={new_q:.4f} | "
            f"next_state={next_state} max_next_q={max_next_q:.4f} eps={self.epsilon:.4f} step={self.step_count}"
        )
    
    def _update_epsilon(self):
        if self.step_count < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.step_count / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end
    
    def save(self, filepath: str):
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
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.q_table = state['q_table']
        self.step_count = state['step_count']
        self.episode_count = state['episode_count']
        self.epsilon = state['epsilon']
        self.metrics = state['metrics']
        self.state_discretization = state['state_discretization']
    
    def get_statistics(self) -> dict:
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
    def __init__(self, exploration_factor: float = 200.0):
        self.c = exploration_factor
        self.action_counts = {}
        self.action_rewards = {}
        self.total_steps = 0
        
    def discretize_state(self, batch_size: int, chunks_before: int,
                        chunks_after: int, requests_waiting: int) -> str:
        chunk_diff = chunks_before - chunks_after
        batch_bin = min(batch_size // 4, 10)
        chunk_diff_bin = max(-5, min(5, chunk_diff))
        requests_bin = min(requests_waiting // 4, 10)
        return f"{batch_bin}_{chunk_diff_bin}_{requests_bin}"
    
    def get_ucb_score(self, state: str, action: bool) -> float:
        key = (state, action)
        if key not in self.action_counts or self.action_counts[key] == 0:
            return float('inf')
        count = self.action_counts[key]
        avg_reward = self.action_rewards[key] / count
        exploration_bonus = self.c * np.sqrt(np.log(self.total_steps + 1) / count)
        return avg_reward + exploration_bonus
    
    def should_add_request(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int
    ) -> bool:
        if batch_size == 0:
            return True

        if chunks_before == chunks_after and chunks_before > 0:
            return True
        
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        chunk_diff = chunks_before - chunks_after

        ucb_add  = self.get_ucb_score(state, True)
        ucb_skip = self.get_ucb_score(state, False)
        action   = ucb_add >= ucb_skip

        count_add  = self.action_counts.get((state, True),  0)
        count_skip = self.action_counts.get((state, False), 0)
        avg_add  = (self.action_rewards.get((state, True),  0.0) / count_add)  if count_add  else float('nan')
        avg_skip = (self.action_rewards.get((state, False), 0.0) / count_skip) if count_skip else float('nan')

        print(
            f"[CB | DECISION] "
            f"state={state} bs={batch_size} chunk_diff={chunk_diff} rw={requests_waiting} | "
            f"ucb_add={'inf' if ucb_add == float('inf') else f'{ucb_add:.4f}'} "
            f"ucb_skip={'inf' if ucb_skip == float('inf') else f'{ucb_skip:.4f}'} | "
            f"avg_r_add={avg_add:.4f} avg_r_skip={avg_skip:.4f} n_add={count_add} n_skip={count_skip} | "
            f"action={'ADD' if action else 'SKIP'} steps={self.total_steps}"
        )

        return action
    
    def update(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
        action: bool,
        reward: float
    ):
        state = self.discretize_state(batch_size, chunks_before, chunks_after, requests_waiting)
        key = (state, action)
        chunk_diff = chunks_before - chunks_after

        self.action_counts[key]  = self.action_counts.get(key, 0) + 1
        self.action_rewards[key] = self.action_rewards.get(key, 0.0) + reward
        self.total_steps += 1

        new_avg = self.action_rewards[key] / self.action_counts[key]

        print(
            f"[CB | UPDATE] "
            f"state={state} bs={batch_size} chunk_diff={chunk_diff} rw={requests_waiting} | "
            f"action={'ADD' if action else 'SKIP'} reward={reward:.4f} | "
            f"new_avg_r={new_avg:.4f} n={self.action_counts[key]} steps={self.total_steps}"
        )
    
    def save(self, filepath: str):
        state = {
            'action_counts': self.action_counts,
            'action_rewards': self.action_rewards,
            'total_steps': self.total_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.action_counts = state['action_counts']
        self.action_rewards = state['action_rewards']
        self.total_steps = state['total_steps']


class HeuristicScheduler:
    def __init__(
        self,
        max_chunk_loss: int = 2,
        min_batch_size: int = 10,
    ):
        self.max_chunk_loss = max_chunk_loss
        self.min_batch_size = min_batch_size
    
    def should_add_request(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int
    ) -> bool:
        if batch_size == 0:
            return True
        
        chunk_loss = chunks_before - chunks_after
        
        if chunk_loss <= 0:
            action = True
        elif batch_size < self.min_batch_size:
            action = chunk_loss <= self.max_chunk_loss + 2
        elif chunk_loss <= self.max_chunk_loss:
            action = True
        elif requests_waiting >= 10 and chunk_loss <= self.max_chunk_loss + 1:
            action = True
        else:
            action = False
        
        return action