# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
import time
import math

from vllm.v1.request import Request
from vllm.v1.core.sched.radix_tree import TokenRadixTree, SyncRadixTreeClient
from vllm.v1.core.sched.rl import QLearningScheduler, ContextualBanditScheduler, HeuristicScheduler, StateAction
from vllm.v1.core.sched.build.chunked_hash_tree import ChunkedHashTree

# import sys
# sys.path.append("/home/saksham/CS496/code/vllm/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/build")

# from chunked_hash_tree import ChunkedHashTree


import time
import functools
import collections

LOG_EVERY = 100

_function_stats = collections.defaultdict(
    lambda: {"calls": 0, "total_ns": 0}
)

def log_cycles(fn):
    name = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter_ns() - start

        stats = _function_stats[name]
        stats["calls"] += 1
        stats["total_ns"] += elapsed

        if stats["calls"] % LOG_EVERY == 0:
            avg = stats["total_ns"] / stats["calls"]
            print(
                f"[PROFILE] {name} | "
                f"calls={stats['calls']} | "
                f"total={stats['total_ns']} ns | "
                f"avg={avg:.1f} ns"
            )

        return result
    return wrapper




class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    RADIX_COST = "radix_cost"
    FAMILY_LOAD = "family_load"
    GREEDY_L2 = "greedy_l2"
    DYNAMIC_BATCHING = "dynamic_batching"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass

    def free_request(self, request: Request) -> None:
        pass

    def should_add_more_to_batch(self, **kwargs) -> bool:
        """Check if more requests can be added to the current batch."""
        return True


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)
        return request

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


class RadixCostRequestQueue(RequestQueue):

    def __init__(self):
        # C++ radix structure (integer IDs only)
        self._radix = ChunkedHashTree(chunk_size=500)

        # vLLM-visible storage
        self._pending_requests: dict[str, Request] = {}

        # String <-> int ID interning
        self._id_map: dict[str, int] = {}
        self._id_to_request: dict[int, str] = {}
        self._next_id: int = 1

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _intern_id(self, request_id: str) -> int:
        """Map vLLM string request_id -> compact int ID."""
        rid = self._id_map.get(request_id)
        if rid is None:
            rid = self._next_id
            self._next_id += 1
            self._id_map[request_id] = rid
            self._id_to_request[rid] = request_id
        return rid

    def _release_id(self, request_id: str) -> None:
        """Remove ID mapping when request is fully gone."""
        rid = self._id_map.pop(request_id, None)
        if rid is not None:
            self._id_to_request.pop(rid, None)

    # -----------------------------
    # Required RequestQueue API
    # -----------------------------
    def add_request(self, request: Request) -> None:
        rid = self._intern_id(request.request_id)
        self._radix.insert(rid, request.all_token_ids)
        self._pending_requests[request.request_id] = request

    def pop_request(self) -> Request:
        rid, cost = self._radix.find_best_request()
        if rid == 0:
            raise IndexError("pop from empty radix queue")

        request_id = self._id_to_request[rid]
        request = self._pending_requests.pop(request_id)

        self._radix.activate_request(rid)
        return request

    def peek_request(self) -> Request:
        rid, _ = self._radix.find_best_request()
        if rid == 0:
            raise IndexError("peek from empty radix queue")

        request_id = self._id_to_request[rid]
        return self._pending_requests[request_id]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        request_id = request.request_id
        if request_id in self._pending_requests:
            rid = self._id_map.get(request_id)
            if rid is not None:
                self._radix.remove(rid)
            del self._pending_requests[request_id]
            self._release_id(request_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        has_pending = bool(self._pending_requests)
        rid, _ = self._radix.find_best_request()
        has_waiting = rid != 0
        return has_waiting

    def __len__(self) -> int:
        return len(self._pending_requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._pending_requests.values())

    def __reversed__(self) -> Iterator[Request]:
        for request_id in reversed(list(self._pending_requests.keys())):
            yield self._pending_requests[request_id]

    def free_request(self, request: Request) -> None:
        request_id = request.request_id
        rid = self._id_map.get(request_id)
        if rid is not None:
            self._radix.finish_request(rid)
            self._release_id(request_id)

    def should_add_more_to_batch(self, **kwargs) -> bool:
        current_batch_size = kwargs.get("current_batch_size", 0)
        if current_batch_size == 0:
            return True
        rid, cost = self._radix.find_best_request()
        if rid == 0:  # No requests available
            return False
        return cost <= 1
    
    

class RadixCostRequestQueue_RL(RequestQueue):

    def __init__(
        self,
        scheduler_type: str = 'heuristic',  # 'qlearning', 'bandit', or 'heuristic'
        scheduler_params: dict = None,
        model_save_path: Optional[str] = None
    ):
        
        self._radix = ChunkedHashTree(chunk_size=500)
        self._pending_requests = {}  # dict[str, Request]
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
        
        # Load existing model if available
        if model_save_path and hasattr(self.scheduler, 'load'):
            try:
                self.scheduler.load(model_save_path)
                print(f"Loaded RL model from {model_save_path}")
            except FileNotFoundError:
                print(f"No existing model found at {model_save_path}, starting fresh")
        
        # Track current batch info for reward computation
        self.current_batch_info = None
        self.last_batch_start_time = None
        self.batch_count = 0
        
        # Performance tracking
        self.batch_execution_times = []
        self.batch_sizes = []
        self.decisions_made = []

    def _intern_id(self, request_id: str) -> int:
        """Map vLLM string request_id -> compact int ID."""
        rid = self._id_map.get(request_id)
        if rid is None:
            rid = self._next_id
            self._next_id += 1
            self._id_map[request_id] = rid
            self._id_to_request[rid] = request_id
        return rid

    def _release_id(self, request_id: str) -> None:
        """Remove ID mapping when request is fully gone."""
        rid = self._id_map.pop(request_id, None)
        if rid is not None:
            self._id_to_request.pop(rid, None)

    def add_request(self, request: Request) -> None:
        rid = self._intern_id(request.request_id)
        self._radix.insert(rid, request.all_token_ids)
        self._pending_requests[request.request_id] = request

    def pop_request(self) -> Request:
        rid, _, _, _ = self._radix.find_best_request()
        if rid == 0:
            raise IndexError("pop from empty radix queue")

        request_id = self._id_to_request[rid]
        request = self._pending_requests.pop(request_id)

        self._radix.activate_request(rid)
        return request

    def peek_request(self) -> Request:
        rid, _, _, _ = self._radix.find_best_request()
        if rid == 0:
            raise IndexError("peek from empty radix queue")

        request_id = self._id_to_request[rid]
        return self._pending_requests[request_id]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        request_id = request.request_id
        if request_id in self._pending_requests:
            rid = self._id_map.get(request_id)
            if rid is not None:
                self._radix.remove(rid)
            del self._pending_requests[request_id]
            self._release_id(request_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        has_pending = bool(self._pending_requests)
        rid, _, _, _ = self._radix.find_best_request()
        has_waiting = rid != 0
        return has_waiting

    def __len__(self) -> int:
        return len(self._pending_requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._pending_requests.values())

    def __reversed__(self) -> Iterator[Request]:
        for request_id in reversed(list(self._pending_requests.keys())):
            yield self._pending_requests[request_id]

    def free_request(self, request: Request) -> None:
        request_id = request.request_id
        rid = self._id_map.get(request_id)
        if rid is not None:
            self._radix.finish_request(rid)
            self._release_id(request_id)

    def should_add_more_to_batch(self, **kwargs) -> bool:
        """
        RL-based decision on whether to add more requests to the batch.
        
        This is the core method where the RL agent makes decisions.
        """
        current_batch_size = kwargs.get("current_batch_size", 0)
        last_batch_time = kwargs.get("last_batch_time", 0.0)
        current_batch_id = kwargs.get("current_batch_id", 0)
        
        # Always add to empty batch
        if current_batch_size == 0:
            self.last_batch_start_time = time.time()
            self.current_batch_info = None
            return True
        
        # Get state information from radix tree
        rid, chunks_before, chunks_after, requests_waiting = self._radix.find_best_request()
        
        if rid == 0:
            # No requests available
            return False
        
        # Provide feedback to RL model from previous batch — only once per batch ID transition.
        # should_add_more_to_batch is called in a tight loop (once per candidate request),
        # so we gate on batch_id changing to avoid redundant updates.
        if (
            last_batch_time > 0
            and self.current_batch_info is not None
            and current_batch_id != self.current_batch_info.get('batch_id')
            and self.scheduler_type != "heuristic"
        ):
            self._provide_feedback(last_batch_time)
        
        # Make RL decision
        decision = self.scheduler.should_add_request(
            batch_size=current_batch_size,
            chunks_before=chunks_before,
            chunks_after=chunks_after,
            requests_waiting=requests_waiting
        )
        
        # Capture state snapshot once per batch (on the first call for this batch_id).
        # Subsequent calls within the same batch reuse the same snapshot for the RL update.
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
        
        # Log decision (you can remove this in production)
        print(f"[RL Decision] batch_size={current_batch_size}, "
              f"chunks_before={chunks_before}, chunks_after={chunks_after}, "
              f"requests_waiting={requests_waiting}, decision={'ADD' if decision else 'SKIP'}, "
              f"batch_id={current_batch_id}")
        
        if hasattr(self.scheduler, 'epsilon'):
            print(f"[RL Stats] epsilon={self.scheduler.epsilon:.3f}, "
                  f"steps={self.scheduler.step_count}")
        
        self.decisions_made.append({
            'batch_size': current_batch_size,
            'chunks_before': chunks_before,
            'chunks_after': chunks_after,
            'requests_waiting': requests_waiting,
            'decision': decision,
            'batch_id': current_batch_id
        })
        
        return decision
    
    def _provide_feedback(self, batch_execution_time: float):
        """
        Provide reward feedback to the RL model based on batch execution time.
        
        Args:
            batch_execution_time: Time taken for the last batch decode step
        """
        if self.current_batch_info is None:
            return
        
        # Reward is negative execution time (we want to minimize latency)
        reward = -batch_execution_time
        
        # Get current state for TD learning (if applicable)
        rid, chunks_before, chunks_after, requests_waiting = self._radix.find_best_request()
        
        next_state = None
        if rid != 0:
            next_state = (
                self.current_batch_info['batch_size'],  # Batch size hasn't changed yet
                chunks_before,
                chunks_after,
                requests_waiting
            )
        
        # Update the model
        if self.scheduler_type == 'qlearning' and next_state:
            self.scheduler.update(
                batch_size=self.current_batch_info['batch_size'],
                chunks_before=self.current_batch_info['chunks_before'],
                chunks_after=self.current_batch_info['chunks_after'],
                requests_waiting=self.current_batch_info['requests_waiting'],
                action=self.current_batch_info['action'],
                reward=reward,
                next_batch_size=next_state[0],
                next_chunks_before=next_state[1],
                next_chunks_after=next_state[2],
                next_requests_waiting=next_state[3]
            )
        else:
            # For bandit or terminal states
            self.scheduler.update(
                batch_size=self.current_batch_info['batch_size'],
                chunks_before=self.current_batch_info['chunks_before'],
                chunks_after=self.current_batch_info['chunks_after'],
                requests_waiting=self.current_batch_info['requests_waiting'],
                action=self.current_batch_info['action'],
                reward=reward
            )
        
        # Track performance
        self.batch_execution_times.append(batch_execution_time)
        self.batch_sizes.append(self.current_batch_info['batch_size'])
        self.batch_count += 1
        
        # Periodically save model
        if self.model_save_path and self.batch_count % 100 == 0:
            if hasattr(self.scheduler, 'save'):
                self.scheduler.save(self.model_save_path)
                print(f"Saved RL model to {self.model_save_path} (batch {self.batch_count})")
    
    def get_statistics(self) -> dict:
        """Get training and performance statistics"""
        stats = {
            'batch_count': self.batch_count,
            'avg_batch_time': sum(self.batch_execution_times[-100:]) / len(self.batch_execution_times[-100:])
                             if self.batch_execution_times else 0,
            'avg_batch_size': sum(self.batch_sizes[-100:]) / len(self.batch_sizes[-100:])
                             if self.batch_sizes else 0,
        }
        
        # Add scheduler-specific stats
        if hasattr(self.scheduler, 'get_statistics'):
            stats.update(self.scheduler.get_statistics())
        
        return stats
    
    def print_statistics(self):
        """Print performance statistics"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("RL Batch Scheduler Statistics")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.4f}")
            else:
                print(f"{key:30s}: {value}")
        print("="*60 + "\n")
    




class FamilyLoadRequestQueue(RequestQueue):
    def __init__(self):
        self.family_requests: dict[int, deque[Request]] = {}
        self.last_served_family: int = 0

    @log_cycles
    def add_request(self, request: Request) -> None:
        # start_time = time.time()
        family_id = request.all_token_ids[26]
        if family_id not in self.family_requests:
            self.family_requests[family_id] = deque()
        self.family_requests[family_id].append(request)
        # end_time = time.time()

    @log_cycles
    def pop_request(self) -> Request:
        if self.last_served_family not in self.family_requests or not self.family_requests[self.last_served_family]:
            available_families = [fid for fid in self.family_requests if self.family_requests[fid]]
            self.last_served_family = min(available_families)
        request = self.family_requests[self.last_served_family].popleft()
        return request

    @log_cycles
    def peek_request(self) -> Request:
        # start_time = time.time()
        if self.last_served_family not in self.family_requests or not self.family_requests[self.last_served_family]:
            available_families = [fid for fid in self.family_requests if self.family_requests[fid]]
            self.last_served_family = min(available_families)
        # end_time = time.time()
        return self.family_requests[self.last_served_family][0]

    @log_cycles
    def prepend_request(self, request: Request) -> None:
        family_id = request.all_token_ids[26]
        if family_id not in self.family_requests:
            self.family_requests[family_id] = deque()
        self.family_requests[family_id].appendleft(request)
    
    @log_cycles
    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)
    
    @log_cycles
    def remove_request(self, request: Request) -> None:
        family_id = request.all_token_ids[3]
        if family_id in self.family_requests:
            self.family_requests[family_id].remove(request)
    
    @log_cycles
    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)
    
    def __bool__(self) -> bool:
        return any(self.family_requests[fid] for fid in self.family_requests)
    
    def __len__(self) -> int:
        return sum(len(self.family_requests[fid]) for fid in self.family_requests)

    def __iter__(self) -> Iterator[Request]:
        for fid in sorted(self.family_requests.keys()):
            for request in self.family_requests[fid]:
                yield request

    def __reversed__(self) -> Iterator[Request]:
        for fid in sorted(self.family_requests.keys(), reverse=True):
            for request in reversed(self.family_requests[fid]):
                yield request

    @log_cycles
    def should_add_more_to_batch(self, **kwargs) -> bool:
        current_batch_size = kwargs.get("current_batch_size", 0)
        if current_batch_size == 0:
            return True
        if self.last_served_family not in self.family_requests or not self.family_requests[self.last_served_family]:
            return False
        return True


class GreedyRequestQueue(RequestQueue):
    """
    Strategy: Pick requests in FCFC order, until our working set size increases beyond a point.
    """

    def __init__(self):
        self._queue = FCFSRequestQueue()
        self._working_set: set[Request] = set()

    def add_request(self, request: Request) -> None:
        self._queue.add_request(request)

    def pop_request(self) -> Request:
        request = self._queue.pop_request()
        self._working_set.add(request)
        return request
    
    def peek_request(self) -> Request:
        return self._queue.peek_request()
    
    def prepend_request(self, request: Request) -> None:
        self._queue.prepend_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self._queue.prepend_requests(requests)

    def remove_request(self, request: Request) -> None:
        self._queue.remove_request(request)
        self._working_set.discard(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        self._queue.remove_requests(requests)
        for request in requests:
            self._working_set.discard(request)

    def __bool__(self) -> bool:
        return bool(self._queue)
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def __iter__(self) -> Iterator[Request]:
        return iter(self._queue)
    
    def __reversed__(self) -> Iterator[Request]:
        return reversed(self._queue)
    
    def free_request(self, request: Request) -> None:
        self._working_set.discard(request)

    def should_add_more_to_batch(self, **kwargs) -> bool:
        current_batch_size = kwargs.get("current_batch_size", 0)
        if current_batch_size == 0:
            return True
        # we iterate over the working set, to find the request with max overlap
        max_overlap = 0
        for req in self._working_set:
            # we need to find the longest prefix match between req and the next request
            overlap_count = 0
            for t1, t2 in zip(req.all_token_ids, self.peek_request().all_token_ids):
                if t1 == t2:
                    overlap_count += 1
                else:
                    break
            max_overlap = max(max_overlap, overlap_count)
        tokens_not_in_working_set = len(self.peek_request().all_token_ids) - max_overlap
        if tokens_not_in_working_set < 50:
            return True
        return False


class RadixCostRequestQueue_SeparateProcess(RequestQueue):
    """
    Request queue using the multiprocess radix tree.
    All methods are synchronous - compatible with existing code.
    """
    
    def __init__(self):
        self._radix = SyncRadixTreeClient()
        self._radix.start()
        self._pending_requests: dict[str, 'Request'] = {}
    
    def add_request(self, request: 'Request') -> None:
        self._radix.insert(request.request_id, request.all_token_ids)
        self._pending_requests[request.request_id] = request
    
    def pop_request(self) -> 'Request':
        req_id, cost = self._radix.get_best_request()
        if req_id is None:
            raise IndexError("pop from empty radix queue")
        request = self._pending_requests.pop(req_id)
        self._radix.activate_request(req_id)
        return request
    
    def peek_request(self) -> 'Request':
        """Synchronous peek - uses cached best request."""
        req_id, _ = self._radix.get_best_request()
        while req_id is None:
            req_id, _ = self._radix.get_best_request()
        if req_id is None:
            raise IndexError("peek from empty radix queue")
        return self._pending_requests[req_id]
    
    def prepend_request(self, request: 'Request') -> None:
        self.add_request(request)
    
    def prepend_requests(self, requests: 'RequestQueue') -> None:
        for request in requests:
            self.add_request(request)
    
    def remove_request(self, request: 'Request') -> None:
        if request.request_id in self._pending_requests:
            self._radix.remove(request.request_id)
            del self._pending_requests[request.request_id]
    
    def remove_requests(self, requests: Iterable['Request']) -> None:
        for request in requests:
            self.remove_request(request)
    
    def __bool__(self) -> bool:
        return bool(self._pending_requests)
    
    def __len__(self) -> int:
        return len(self._pending_requests)
    
    def __iter__(self) -> Iterator['Request']:
        for req_id in self._pending_requests:
            yield self._pending_requests[req_id]
    
    def __reversed__(self) -> Iterator['Request']:
        for req_id in reversed(list(self._pending_requests.keys())):
            yield self._pending_requests[req_id]
    
    def free_request(self, request: 'Request') -> None:
        self._radix.finish_request(request.request_id)
    
    def should_add_more_to_batch(self, **kwargs) -> bool:
        """Synchronous check using cached best request."""
        current_batch_size = kwargs.get("current_batch_size", 0)
        if current_batch_size == 0:
            return True
        req_id, cost = self._radix.get_best_request()
        req = self._pending_requests.get(req_id, None)
        if cost < 50:
            return True
        return False
    
    def shutdown(self):
        """Cleanup method to shutdown the worker process."""
        self._radix.shutdown()
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.shutdown()
        except:
            pass


class DynamicBatchingRequestQueue(FCFSRequestQueue):
    # source: https://github.com/KevinLee1110/dynamic-batching
    MEMORY_FACTOR = 0.95
    BATCH_SIZE_LOWER = 100
    BATCH_SIZE_UPPER = 1000
    def should_add_more_to_batch(self, **kwargs) -> bool:
        current_batch_size = kwargs.get("current_batch_size", 0)
        if current_batch_size == 0:
            return True
        num_total_gpu_blocks = kwargs.get("num_total_gpu_blocks", 0)
        num_total_free_gpu_blocks = kwargs.get("num_total_free_gpu_blocks", 0)
        used_blocks = num_total_gpu_blocks - num_total_free_gpu_blocks
        blocks_per_seq = used_blocks / current_batch_size
        new_max_num_seqs = math.floor(num_total_gpu_blocks * self.MEMORY_FACTOR / blocks_per_seq)
        new_max_num_seqs = max(new_max_num_seqs, current_batch_size)
        new_max_num_seqs = min(max(new_max_num_seqs, self.BATCH_SIZE_LOWER), self.BATCH_SIZE_UPPER)
        if current_batch_size >= new_max_num_seqs:
            return False
        return True



def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.RADIX_COST:
        return RadixCostRequestQueue_RL()
    elif policy == SchedulingPolicy.FAMILY_LOAD:
        return FamilyLoadRequestQueue()
    elif policy == SchedulingPolicy.GREEDY_L2:
        return GreedyRequestQueue()
    elif policy == SchedulingPolicy.DYNAMIC_BATCHING:
        return DynamicBatchingRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
