import time
from typing import Optional, Iterator
from .rl import QLearningScheduler, ContextualBanditScheduler, HeuristicScheduler
import sglang.srt.managers.schedule_batch as schedule_batch

try:
    from chunked_hash_tree import ChunkedHashTree_RL
except ImportError:
    ChunkedHashTree_RL = None


class DynamicBatchHelper:
    def __init__(
        self,
        scheduler_type: str = 'heuristic',  # 'qlearning', 'bandit', or 'heuristic'
        scheduler_params: dict = None,
        model_save_path: Optional[str] = None
    ):
        if ChunkedHashTree_RL is None:
            raise ImportError("chunked_hash_tree C++ module is not installed. Please compile it first.")

        self._radix = ChunkedHashTree_RL(chunk_size=500)
        self._id_map = {}  # dict[str, int]
        self._id_to_request = {}  # dict[int, str]
        self._next_id = 1
        
        # Initialize scheduler
        if scheduler_params is None:
            scheduler_params = {}
        
        if scheduler_type == 'qlearning':
            self.scheduler = QLearningScheduler(**scheduler_params)
        elif scheduler_type == 'bandit':
            self.scheduler = ContextualBanditScheduler(**scheduler_params)
        elif scheduler_type == 'heuristic':
            self.scheduler = HeuristicScheduler(**scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.scheduler_type = scheduler_type
        self.model_save_path = model_save_path
        
        if model_save_path and hasattr(self.scheduler, 'load'):
            try:
                self.scheduler.load(model_save_path)
            except FileNotFoundError:
                pass
        
        self.current_batch_info = None
        self.last_batch_start_time = None
        self.batch_count = 0
        self.decisions_made = []

    def _intern_id(self, request_id: str) -> int:
        rid = self._id_map.get(request_id)
        if rid is None:
            rid = self._next_id
            self._next_id += 1
            self._id_map[request_id] = rid
            self._id_to_request[rid] = request_id
        return rid

    def _release_id(self, request_id: str) -> None:
        rid = self._id_map.pop(request_id, None)
        if rid is not None:
            self._id_to_request.pop(rid, None)

    def add_request(self, req) -> None:
        rid = self._intern_id(req.rid)
        # SGLang uses req.origin_input_ids for all token ids.
        self._radix.insert(rid, list(req.origin_input_ids))

    def remove_request(self, req) -> None:
        rid = self._id_map.get(req.rid)
        if rid is not None:
            self._radix.remove(rid)
            self._release_id(req.rid)

    def finish_request(self, req) -> None:
        rid = self._id_map.get(req.rid)
        if rid is not None:
            self._radix.finish_request(rid)
            self._release_id(req.rid)
            
    def clear(self) -> None:
        # Re-initialize the tree
        self._radix = ChunkedHashTree_RL(chunk_size=500)
        self._id_map = {}
        self._id_to_request = {}
        self._next_id = 1

    def find_best_request(self) -> Optional[str]:
        rid, cost, chunks_before, chunks_after = self._radix.find_best_request()
        if rid == 0:
            return None
        self._radix.activate_request(rid)
        return self._id_to_request[rid]

    def should_add_more_to_batch(self, **kwargs) -> bool:
        current_batch_size = kwargs.get("current_batch_size", 0)
        last_batch_time = kwargs.get("last_batch_time", 0.0)
        current_batch_id = kwargs.get("current_batch_id", 0)
        
        if current_batch_size == 0:
            self.last_batch_start_time = time.time()
            self.current_batch_info = None
            return True
        
        rid, chunks_before, chunks_after, requests_waiting = self._radix.find_best_request()
        
        if rid == 0:
            return False
            
        if (
            last_batch_time > 0
            and self.current_batch_info is not None
            and current_batch_id != self.current_batch_info.get('batch_id')
            and self.scheduler_type != "heuristic"
        ):
            self._provide_feedback(last_batch_time)
            
        decision = self.scheduler.should_add_request(
            batch_size=current_batch_size,
            chunks_before=chunks_before,
            chunks_after=chunks_after,
            requests_waiting=requests_waiting
        )
        
        if (
            self.current_batch_info is None
            or current_batch_id != self.current_batch_info.get('batch_id')
        ):
            self.current_batch_info = {
                'batch_size': current_batch_size,
                'chunks_before': chunks_before,
                'chunks_after': chunks_after,
                'requests_waiting': requests_waiting,
                'action': decision,
                'batch_id': current_batch_id
            }
            
        return decision
        
    def _provide_feedback(self, batch_execution_time: float):
        if self.current_batch_info is None:
            return
            
        reward = -batch_execution_time
        rid, chunks_before, chunks_after, requests_waiting = self._radix.find_best_request()
        
        next_state = None
        if rid != 0:
            next_state = (
                self.current_batch_info['batch_size'],
                chunks_before,
                chunks_after,
                requests_waiting
            )
            
        if self.scheduler_type == 'qlearning':
            self.scheduler.update(
                batch_size=self.current_batch_info['batch_size'],
                chunks_before=self.current_batch_info['chunks_before'],
                chunks_after=self.current_batch_info['chunks_after'],
                requests_waiting=self.current_batch_info['requests_waiting'],
                action=self.current_batch_info['action'],
                reward=reward,
                next_batch_size=next_state[0] if next_state else None,
                next_chunks_before=next_state[1] if next_state else None,
                next_chunks_after=next_state[2] if next_state else None,
                next_requests_waiting=next_state[3] if next_state else None
            )
        elif self.scheduler_type == 'bandit':
            self.scheduler.update(
                batch_size=self.current_batch_info['batch_size'],
                chunks_before=self.current_batch_info['chunks_before'],
                chunks_after=self.current_batch_info['chunks_after'],
                requests_waiting=self.current_batch_info['requests_waiting'],
                action=self.current_batch_info['action'],
                reward=reward
            )

