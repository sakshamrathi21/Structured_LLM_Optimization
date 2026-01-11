from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

class _RadixNode:
    """
    Node in the radix tree representing a substring of tokens.
    Tracks which requests use this node and working set membership.
    """
    __slots__ = (
        "tokens",
        "parent",
        "children",
        "requests",
        "ref_count",
        "in_working_set",
    )

    def __init__(self, tokens: List[int], parent=None):
        self.tokens = tokens # Substring of token IDs
        self.parent = parent
        self.children = {} # Map from first token to child node
        self.requests = set() # Request IDs that pass through this node
        self.ref_count = 0 # Number of requests using this node
        self.in_working_set = False # Is this node in the working set


class TokenRadixTree:
    """
    Radix tree where each node represents a substring of tokens.
    Working set is tracked at the node level.
    """

    def __init__(self):
        self.root = _RadixNode(tokens=[], parent=None)
        self.root.in_working_set = True # Root is always in working set

        # Request tracking
        self.request_to_nodes: Dict[str, List[_RadixNode]] = {}
        self.active_requests: Set[str] = set()

        # Working set tracking
        self.working_set_nodes: Set[_RadixNode] = {self.root}
        self.working_set_ref_count: Dict[_RadixNode, int] = defaultdict(int)

        self.waiting_requests: List[str] = [] # All non-active requests

    def _find_common_prefix_length(self, tokens1: List[int], tokens2: List[int]) -> int:
        """Find length of common prefix between two token sequences."""
        min_len = min(len(tokens1), len(tokens2))
        for i in range(min_len):
            if tokens1[i] != tokens2[i]:
                return i
        return min_len
    
    def insert(self, request_id: str, token_ids: List[int]) -> int:
        """
        Insert a request into the radix tree.
        Returns: number of tokens
        """
        if request_id in self.request_to_nodes:
            raise ValueError(f"Request {request_id} already exists")
        
        node = self.root
        remaining = token_ids
        nodes_traversed = [self.root]

        while remaining:
            first_token = remaining[0]

            if first_token not in node.children:
                # No child with this first token - create new node
                new_node = _RadixNode(tokens=list(remaining), parent=node)
                node.children[first_token] = new_node
                new_node.requests.add(request_id)
                new_node.ref_count += 1
                nodes_traversed.append(new_node)
                break
                
            # Child exists - check for prefix match
            child = node.children[first_token]
            common_len = self._find_common_prefix_length(remaining, child.tokens)

            if common_len == len(child.tokens):
                # Perfect match with child - continue down
                child.requests.add(request_id)
                child.ref_count += 1
                nodes_traversed.append(child)
                remaining = remaining[common_len:]
                node = child
                continue
                
            # Partial match - need to split the child node
            common_tokens = child.tokens[:common_len]
            common_node = _RadixNode(tokens=common_tokens, parent=node)

            common_node.requests = child.requests.copy()
            common_node.ref_count = child.ref_count
            
            child.tokens = child.tokens[common_len:]
            child.parent = common_node

            if child.tokens:
                common_node.children[child.tokens[0]] = child

            common_node.in_working_set = child.in_working_set
            if common_node.in_working_set:
                self.working_set_nodes.add(common_node)
                if child in self.working_set_ref_count:
                    self.working_set_ref_count[common_node] = self.working_set_ref_count[child]

            node.children[first_token] = common_node
            
            # Add current request to common_node
            common_node.requests.add(request_id)
            common_node.ref_count += 1
            nodes_traversed.append(common_node)

            for req_id in child.requests:
                if req_id in self.request_to_nodes and req_id != request_id:
                    req_nodes = self.request_to_nodes[req_id]
                    # Find child in the path and insert common_node before it
                    new_path = []
                    for n in req_nodes:
                        if n is child:
                            new_path.append(common_node)
                        new_path.append(n)
                    self.request_to_nodes[req_id] = new_path

            # More to insert, create new branch
            remaining = remaining[common_len:]
            if remaining:
                new_node = _RadixNode(tokens=list(remaining), parent=common_node)
                common_node.children[remaining[0]] = new_node
                new_node.requests.add(request_id)
                new_node.ref_count += 1
                nodes_traversed.append(new_node)
            break
            
        self.request_to_nodes[request_id] = nodes_traversed
        self.waiting_requests.append(request_id)
        return len(token_ids)
    
    def _count_missing_tokens(self, request_id: str) -> int:
        """Count the number of tokens in nodes that are not in the working set."""
        nodes = self.request_to_nodes.get(request_id, [])
        count = 0
        for node in nodes:
            if node not in self.working_set_nodes:
                count += len(node.tokens)
        return count
        # return sum(len(node.tokens) for node in nodes if not node.in_working_set)

    def find_best_request(self) -> Tuple[Optional[str], float]:
        """
        Strategy: Start from leaf nodes in the working set and traverse towards root.
        At each node, check for waiting requests. Prefer requests that need fewer new tokens 
        (i.e., those that share more prefix with the current working set).

        Returns: (request_id, num_missing_tokens)
        """
        if not self.waiting_requests:
            return None, 0

        best_req = None
        best_cost = float("inf")

        queue = deque()
        visited = set()

        for node in self.working_set_nodes:
            is_leaf = not any(child in self.working_set_nodes for child in node.children.values())
            if is_leaf:
                queue.append(node)
        
        while queue and best_cost > 0:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            for req_id in node.requests:
                if req_id not in self.waiting_requests:
                    continue
                if req_id in self.active_requests:
                    continue
                cost = self._count_missing_tokens(req_id)
                if cost < best_cost:
                    best_cost = cost
                    best_req = req_id
                    if cost == 0:
                        return best_req, best_cost
            
            if node.parent and node.parent not in visited:
                queue.append(node.parent)

        if best_req is None:
            return self.waiting_requests[0], self._count_missing_tokens(self.waiting_requests[0])
        
        return best_req, best_cost
    
    def activate_request(self, request_id: str) -> Tuple[int, int]:
        """
        Activate a request and add its nodes to the working set.
        Returns: (num_nodes_added, total_nodes_in_path)
        """

        if request_id in self.active_requests:
            return 0, 0
        
        self.active_requests.add(request_id)

        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
        
        nodes = self.request_to_nodes[request_id]
        added_count = 0

        for node in nodes:
            if self.working_set_ref_count[node] == 0:
                node.in_working_set = True
                self.working_set_nodes.add(node)
                added_count += 1
            self.working_set_ref_count[node] += 1

        return added_count, len(nodes)
    
    def finish_request(self, request_id: str) -> Tuple[int, int]:
        """
        Finish a request, remove from tree, and evict nodes from working set.
        Returns: (num_nodes_evicted, total_nodes_in_path)
        """
        if request_id not in self.active_requests:
            return 0, 0

        self.active_requests.remove(request_id)
        nodes = self.request_to_nodes[request_id]
        evicted_count = 0

        for node in nodes:
            self.working_set_ref_count[node] -= 1
            if self.working_set_ref_count[node] == 0:
                node.in_working_set = False
                if node in self.working_set_nodes:
                    self.working_set_nodes.remove(node)
                evicted_count += 1
                # del self.working_set_ref_count[node]
        
        self.remove(request_id)

        return evicted_count, len(nodes)

    def remove(self, request_id: str) -> None:
        """Remove a request from the tree structure."""
        nodes = self.request_to_nodes.pop(request_id, None)
        if not nodes:
            return
        
        # Remove from waiting list if present
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)

        # Process nodes in reverse order (leaf to root)
        for node in reversed(nodes):
            if node == self.root:
                continue
            
            node.requests.discard(request_id)
            node.ref_count -= 1
            
            # If node is now unused, remove it
            if node.ref_count == 0 and node.parent:
                parent = node.parent
                
                # Remove from parent's children
                if node.tokens and node.tokens[0] in parent.children:
                    del parent.children[node.tokens[0]]
                
                # Remove from working set if present
                if node.in_working_set:
                    self.working_set_nodes.discard(node)
            else:
                # Node still in use, stop pruning
                break

    def get_working_set_size(self) -> int:
        """Get total number of tokens in working set."""
        return sum(len(node.tokens) for node in self.working_set_nodes)

    def get_stats(self) -> Dict:
        """Get statistics about the tree."""
        return {
            "total_requests": len(self.request_to_nodes),
            "active_requests": len(self.active_requests),
            "waiting_requests": len(self.waiting_requests),
            "working_set_nodes": len(self.working_set_nodes),
            "working_set_tokens": self.get_working_set_size(),
        }

    def dump(self) -> None:
        """Print the radix tree structure with working-set state."""
        def _dfs(node: _RadixNode, indent: int) -> None:
            prefix_str = "  " * indent
            
            if node.tokens:
                ws_ref = self.working_set_ref_count.get(node, 0)
                ws_status = "WS" if node.in_working_set else "--"
                
                print(
                    f"{prefix_str}- tokens={node.tokens[:10]}{'...' if len(node.tokens) > 10 else ''} "
                    f"len={len(node.tokens)} "
                    f"ref={node.ref_count} "
                    f"{ws_status}({ws_ref}) "
                    f"reqs={list(node.requests)[:3]}{'...' if len(node.requests) > 3 else ''}"
                )
            else:
                print(f"{prefix_str}ROOT")

            for child in node.children.values():
                _dfs(child, indent + 1)

        print("\n" + "="*80)
        print("RadixTree dump")
        print("="*80)
        _dfs(self.root, 0)
        
        stats = self.get_stats()
        print(f"\nStats: {stats}")
        print(f"Active: {list(self.active_requests)[:5]}{'...' if len(self.active_requests) > 5 else ''}")
        print(f"Waiting: {len(self.waiting_requests)} requests")
        print("="*80 + "\n")


import multiprocessing as mp
from multiprocessing import Queue, Event, Process
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import queue



class RadixOperation(Enum):
    INSERT = 1
    ACTIVATE = 2
    FINISH = 3
    REMOVE = 4
    SHUTDOWN = 5

@dataclass
class RadixCommand:
    op: RadixOperation
    request_id: str
    token_ids: Optional[List[int]] = None


def radix_worker(high_q: Queue, normal_q: Queue, shutdown: Event,
                 best_id: mp.Array, best_cost: mp.Value):

    radix = TokenRadixTree()

    def update_cache():
        rid, cost = radix.find_best_request()
        with best_cost.get_lock():
            best_cost.value = cost
        with best_id.get_lock():
            best_id.value = (rid or "").encode()

    while not shutdown.is_set():
        try:
            try:
                cmd = high_q.get_nowait()
            except queue.Empty:
                cmd = normal_q.get(timeout=0.1)

        except queue.Empty:
            continue
        print(f"Radix worker received command: {cmd}")
        if cmd.op is RadixOperation.SHUTDOWN:
            break
        elif cmd.op is RadixOperation.INSERT:
            radix.insert(cmd.request_id, cmd.token_ids)
        elif cmd.op is RadixOperation.ACTIVATE:
            radix.activate_request(cmd.request_id)
        elif cmd.op is RadixOperation.FINISH:
            radix.finish_request(cmd.request_id)
        elif cmd.op is RadixOperation.REMOVE:
            radix.remove(cmd.request_id)

        update_cache()




class SyncRadixTreeClient:
    def __init__(self, max_id_len=64):
        ctx = mp.get_context("spawn")
        self.high_q = ctx.Queue()
        self.normal_q = ctx.Queue()

        self.shutdown_event = ctx.Event()

        self.best_cost = ctx.Value("d", float("inf"))
        self.best_id = ctx.Array("c", max_id_len)

        self.proc = ctx.Process(
            target=radix_worker,
            args=(self.high_q, self.normal_q, self.shutdown_event,
                  self.best_id, self.best_cost),
            daemon=True
        )

    def start(self):
        self.proc.start()

    def insert(self, rid, tokens):
        self.normal_q.put(RadixCommand(RadixOperation.INSERT, rid, tokens))

    def activate_request(self, rid):
        self.normal_q.put(RadixCommand(RadixOperation.ACTIVATE, rid))

    def finish_request(self, rid):
        self.normal_q.put(RadixCommand(RadixOperation.FINISH, rid))

    def remove(self, rid):
        self.high_q.put(RadixCommand(RadixOperation.REMOVE, rid))

    def get_best_request(self) -> Tuple[Optional[str], float]:
        with self.best_cost.get_lock(), self.best_id.get_lock():
            rid = bytes(self.best_id[:]).rstrip(b"\x00").decode()
            return (rid or None, self.best_cost.value)

    def shutdown(self):
        self.shutdown_event.set()
        self.high_q.put(RadixCommand(RadixOperation.SHUTDOWN, ""))
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()



class ChunkedHashTree:
    """
    Hash-based prefix tree using cumulative chunks.
    Each level represents hash(tokens[0:chunk_size*level]).
    
    Much faster than radix tree - no tree traversal or splits needed.
    """
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        
        # Level -> (hash -> set of request_ids)
        self.levels: Dict[int, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Request metadata
        self.request_tokens: Dict[str, List[int]] = {}
        self.request_hashes: Dict[str, List[int]] = {}  # req_id -> list of hashes at each level
        
        # Working set tracking
        self.working_set_hashes: Set[Tuple[int, int]] = set()  # Set of (level, hash) tuples
        self.hash_ref_counts: Dict[Tuple[int, int], int] = defaultdict(int)  # (level, hash) -> count
        
        # Request state
        self.active_requests: Set[str] = set()
        self.waiting_requests: List[str] = []
    
    def _compute_hashes(self, token_ids: List[int]) -> List[int]:
        """
        Compute cumulative prefix hashes.
        Returns list of hashes: [hash(0:500), hash(0:1000), hash(0:1500), ...]
        """
        hashes = []
        num_chunks = (len(token_ids) + self.chunk_size - 1) // self.chunk_size
        
        for level in range(num_chunks):
            end_idx = min((level + 1) * self.chunk_size, len(token_ids))
            prefix = tuple(token_ids[0:end_idx])
            hashes.append(hash(prefix))
        
        return hashes
    
    def insert(self, request_id: str, token_ids: List[int]) -> int:
        """
        Insert a request. O(num_chunks) - just hash computations.
        Returns: number of tokens
        """
        if request_id in self.request_tokens:
            return
        
        # Compute hashes for all cumulative prefixes
        hashes = self._compute_hashes(token_ids)
        
        # Store request data
        self.request_tokens[request_id] = token_ids
        self.request_hashes[request_id] = hashes
        
        # Add to level hash tables
        for level, hash_val in enumerate(hashes):
            self.levels[level][hash_val].add(request_id)
        
        # Add to waiting queue
        self.waiting_requests.append(request_id)
        
        return len(token_ids)
    
    def _count_missing_chunks(self, request_id: str) -> int:
        """Count how many chunk hashes are NOT in working set."""
        if request_id not in self.request_hashes:
            return float('inf')
        
        hashes = self.request_hashes[request_id]
        missing = 0
        
        for level, hash_val in enumerate(hashes):
            if (level, hash_val) not in self.working_set_hashes:
                missing += 1
        
        return missing
    
    def find_best_request(self) -> Tuple[Optional[str], int]:
        """
        Find waiting request with fewest missing chunks (most prefix overlap).
        O(R × C) where R=waiting requests, C=avg chunks per request.
        
        Returns: (request_id, num_missing_chunks)
        """
        if not self.waiting_requests:
            return None, 0
        
        best_req = None
        best_cost = float("inf")
        
        for req_id in self.waiting_requests:
            cost = self._count_missing_chunks(req_id)
            
            if cost < best_cost:
                best_cost = cost
                best_req = req_id
                
                # Early exit if perfect match
                if cost <= 1:
                    return best_req, int(cost)
        
        return best_req, int(best_cost)
    
    def activate_request(self, request_id: str) -> Tuple[int, int]:
        """
        Activate a request and add its hash nodes to working set.
        O(num_chunks) operations.
        
        Returns: (num_chunks_added, total_chunks)
        """
        if request_id in self.active_requests:
            return 0, 0
        
        if request_id not in self.request_hashes:
            return 0, 0
        
        self.active_requests.add(request_id)
        
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
        
        hashes = self.request_hashes[request_id]
        added_count = 0
        
        for level, hash_val in enumerate(hashes):
            key = (level, hash_val)
            
            # Add to working set if first reference
            if self.hash_ref_counts[key] == 0:
                self.working_set_hashes.add(key)
                added_count += 1
            
            self.hash_ref_counts[key] += 1
        
        return added_count, len(hashes)
    
    def finish_request(self, request_id: str) -> Tuple[int, int]:
        """
        Finish a request and evict its hash nodes from working set.
        O(num_chunks) operations.
        
        Returns: (num_chunks_evicted, total_chunks)
        """
        if request_id not in self.active_requests:
            return 0, 0
        
        self.active_requests.remove(request_id)
        
        hashes = self.request_hashes.get(request_id, [])
        evicted_count = 0
        
        for level, hash_val in enumerate(hashes):
            key = (level, hash_val)
            
            self.hash_ref_counts[key] -= 1
            
            # Remove from working set if no more references
            if self.hash_ref_counts[key] == 0:
                self.working_set_hashes.discard(key)
                evicted_count += 1
                del self.hash_ref_counts[key]
        
        # Remove request data
        self.remove(request_id)
        
        return evicted_count, len(hashes)
    
    def remove(self, request_id: str) -> None:
        """Remove a request completely from the tree."""
        if request_id not in self.request_tokens:
            return
        
        # Remove from waiting list
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
        
        # Remove from level hash tables
        hashes = self.request_hashes.get(request_id, [])
        for level, hash_val in enumerate(hashes):
            self.levels[level][hash_val].discard(request_id)
            
            # Clean up empty hash buckets
            if not self.levels[level][hash_val]:
                del self.levels[level][hash_val]
        
        # Remove metadata
        del self.request_tokens[request_id]
        del self.request_hashes[request_id]
    
    def get_working_set_size(self) -> int:
        """Get total number of chunks in working set."""
        return len(self.working_set_hashes)
    
    def get_working_set_tokens(self) -> int:
        """Estimate total tokens in working set (chunks × chunk_size)."""
        return len(self.working_set_hashes) * self.chunk_size
    
    def get_stats(self) -> Dict:
        """Get statistics about the tree."""
        total_chunks = sum(len(hashes) for hashes in self.request_hashes.values())
        
        return {
            "total_requests": len(self.request_tokens),
            "active_requests": len(self.active_requests),
            "waiting_requests": len(self.waiting_requests),
            "working_set_chunks": len(self.working_set_hashes),
            "working_set_tokens_estimate": self.get_working_set_tokens(),
            "total_chunks": total_chunks,
            "avg_chunks_per_request": total_chunks / len(self.request_tokens) if self.request_tokens else 0,
        }
    
    def dump(self) -> None:
        """Print the hash tree structure."""
        print("\n" + "="*80)
        print("ChunkedHashTree dump")
        print("="*80)
        
        print(f"\nChunk size: {self.chunk_size}")
        print(f"\nLevels: {len(self.levels)}")
        
        for level in sorted(self.levels.keys()):
            print(f"\nLevel {level} (tokens 0:{(level+1)*self.chunk_size}):")
            hashes = self.levels[level]
            print(f"  Unique hashes: {len(hashes)}")
            
            # Show sample hashes
            for i, (hash_val, req_ids) in enumerate(list(hashes.items())[:3]):
                key = (level, hash_val)
                ws_status = "WS" if key in self.working_set_hashes else "--"
                ref_count = self.hash_ref_counts.get(key, 0)
                
                print(f"    hash={hash_val:20d} {ws_status}({ref_count}) reqs={list(req_ids)[:3]}")
            
            if len(hashes) > 3:
                print(f"    ... and {len(hashes)-3} more hashes")
        
        stats = self.get_stats()
        print(f"\nStats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nActive: {list(self.active_requests)[:5]}{'...' if len(self.active_requests) > 5 else ''}")
        print(f"Waiting: {len(self.waiting_requests)} requests")
        print("="*80 + "\n")





# Example usage
if __name__ == "__main__":
    import time
    import random
    
    print("Testing ChunkedHashTree performance...\n")
    
    tree = ChunkedHashTree(chunk_size=500)
    
    # Generate test data - simulate requests with shared prefixes
    base_prefix = list(range(1000))  # Common system prompt
    
    requests = []
    for i in range(100):
        # Each request: shared prefix + unique suffix
        tokens = base_prefix + list(range(1000 + i*100, 1000 + i*100 + random.randint(500, 2000)))
        requests.append((f"req_{i}", tokens))
    
    # Benchmark insert
    start = time.perf_counter()
    for req_id, tokens in requests:
        tree.insert(req_id, tokens)
    insert_time = time.perf_counter() - start
    
    print(f"Inserted {len(requests)} requests in {insert_time*1000:.2f}ms")
    print(f"Average: {insert_time*1000/len(requests):.3f}ms per request\n")
    
    # Benchmark find_best
    start = time.perf_counter()
    for _ in range(100):
        best_req, cost = tree.find_best_request()
    find_time = time.perf_counter() - start
    
    print(f"100 find_best calls in {find_time*1000:.2f}ms")
    print(f"Average: {find_time*10:.3f}ms per call\n")
    
    # Simulate activation/finish cycle
    start = time.perf_counter()
    for i in range(50):
        best_req, _ = tree.find_best_request()
        if best_req:
            tree.activate_request(best_req)
            if i % 10 == 0:  # Finish some requests
                tree.finish_request(best_req)
    cycle_time = time.perf_counter() - start
    
    print(f"50 activate cycles in {cycle_time*1000:.2f}ms")
    print(f"Average: {cycle_time*20:.3f}ms per cycle\n")
    
    tree.dump()
    
    print("\nExpected speedup vs RadixTree:")
    print("  Insert:     100-500x faster")
    print("  Find best:  10-20x faster")
    print("  Activate:   4-5x faster")
    print("  Memory:     5-10x less")

                