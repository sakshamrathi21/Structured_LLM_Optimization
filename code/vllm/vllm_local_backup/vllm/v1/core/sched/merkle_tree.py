"""
SharedMerklePrefixTree
======================

A single, globally-shared Merkle prefix tree over token sequences,
used for KV-cache scheduling.

Tree structure
--------------
Every edge in the tree corresponds to one CHUNK_SIZE (16) token block.
Each node stores the cumulative Merkle hash of the path from the root
down to that node:

    node.hash = xxh64(parent.hash || chunk_hash_of_this_edge)

So the root is a virtual anchor (hash=0, depth=0), and depth-1 nodes
each represent the first 16-token chunk of some request.

    root  (virtual, hash=0, depth=0)
    ├── N1  depth=1, covers tokens [0,16)  of reqs A,B,C
    │   ├── N2  depth=2, covers tokens [0,32) of reqs A,B
    │   │   ├── N4  depth=3, covers tokens [0,48) of req A only
    │   │   └── N5  depth=3, covers tokens [0,48) of req B only
    │   └── N3  depth=2, covers tokens [0,32) of req C only
    └── N6  depth=1, covers tokens [0,16)  of req D (different prefix)

Hanging — every ancestor
------------------------
A *waiting* request hangs on EVERY node along its full path (all
ancestors, including the deepest/leaf node).  node.hanging is the set
of all waiting requests whose token sequence passes through that node.

Working set
-----------
A node is "in the working set" when ref_count > 0 (at least one active
request owns it).  activate_request increments ref_count for every node
on the path; finish_request decrements.

find_best_request  (DFS, no cache, no heap)
-------------------------------------------
DFS from root, visiting only nodes in the working set.  Children are
visited before the current node (deepest-first).  At the first
in-working-set node that has at least one waiting request hanging on it,
pick the request with the fewest missing chunks (most of its path already
in the working set) and return immediately.

"Schedulable" means the visited node is in the working set — i.e. that
prefix block is cached.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Set, Tuple

import xxhash


CHUNK_SIZE: int = 16


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _chunk_hash(token_ids: List[int]) -> int:
    h = xxhash.xxh64()
    for t in token_ids:
        h.update(t.to_bytes(4, "little"))
    return h.intdigest()


def _node_hash(parent_hash: int, chunk_hash: int) -> int:
    h = xxhash.xxh64()
    h.update(struct.pack("<QQ", parent_hash, chunk_hash))
    return h.intdigest()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    One node in the shared prefix tree.

    hash        Cumulative Merkle hash from root to here.
    depth       Distance from virtual root (root = 0).
    ref_count   Active requests that own this node (working-set membership).
    children    edge_key (chunk_hash of that step) -> child Node.
    hanging     Waiting request_ids whose path passes through this node
                (populated for ALL ancestors, not just the deepest node).
    parent      Back-pointer used for bottom-up pruning.
    edge_key    chunk_hash on the incoming edge (used to unlink from parent).
    """

    __slots__ = ("hash", "depth", "ref_count", "children",
                 "hanging", "parent", "edge_key")

    def __init__(self, node_hash: int, depth: int,
                 parent: Optional["Node"], edge_key: int) -> None:
        self.hash:      int               = node_hash
        self.depth:     int               = depth
        self.ref_count: int               = 0
        self.children:  Dict[int, "Node"] = {}
        self.hanging:   Set[str]          = set()
        self.parent:    Optional["Node"]  = parent
        self.edge_key:  int               = edge_key

    @property
    def in_working_set(self) -> bool:
        return self.ref_count > 0


# ---------------------------------------------------------------------------
# SharedMerklePrefixTree
# ---------------------------------------------------------------------------

class SharedMerklePrefixTree:

    def __init__(self, chunk_size: int = CHUNK_SIZE) -> None:
        self.chunk_size = chunk_size
        self._root      = Node(node_hash=0, depth=0, parent=None, edge_key=0)

        # request_id -> path (list of nodes, root-exclusive, depth-1 first)
        self._req_path:   Dict[str, List[Node]] = {}
        self._req_tokens: Dict[str, List[int]]  = {}

        self._active:  Set[str] = set()
        self._waiting: Set[str] = set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunkify(self, token_ids: List[int]) -> List[int]:
        out = []
        for start in range(0, len(token_ids), self.chunk_size):
            out.append(_chunk_hash(token_ids[start: start + self.chunk_size]))
        return out

    def _walk_or_create(self, chunk_hashes: List[int]) -> List[Node]:
        path:        List[Node] = []
        current     = self._root
        parent_hash = self._root.hash
        for ch in chunk_hashes:
            if ch not in current.children:
                nh = _node_hash(parent_hash, ch)
                current.children[ch] = Node(
                    node_hash=nh,
                    depth=current.depth + 1,
                    parent=current,
                    edge_key=ch,
                )
            current     = current.children[ch]
            parent_hash = current.hash
            path.append(current)
        return path

    def _hang_on_all(self, request_id: str) -> None:
        for node in self._req_path[request_id]:
            node.hanging.add(request_id)

    def _unhang_from_all(self, request_id: str) -> None:
        for node in self._req_path[request_id]:
            node.hanging.discard(request_id)

    def _prune_bottom_up(self, path: List[Node]) -> None:
        for node in reversed(path):
            if node.ref_count == 0 and not node.children and not node.hanging:
                if node.parent is not None:
                    node.parent.children.pop(node.edge_key, None)
            else:
                break

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, request_id: str, token_ids: List[int]) -> int:
        """
        Register a new waiting request, hanging it on every ancestor node.
        Returns len(token_ids).
        """
        if request_id in self._req_tokens:
            return 0
        chunk_hashes = self._chunkify(token_ids)
        if not chunk_hashes:
            return 0

        path = self._walk_or_create(chunk_hashes)
        self._req_tokens[request_id] = token_ids
        self._req_path[request_id]   = path
        self._waiting.add(request_id)
        self._hang_on_all(request_id)   # hang on ALL nodes, not just the leaf
        return len(token_ids)

    # ------------------------------------------------------------------
    # Activate
    # ------------------------------------------------------------------

    def activate_request(self, request_id: str) -> Tuple[int, int]:
        """
        Waiting -> active.  Unhang from all nodes, increment ref_counts.
        Returns (newly_added_nodes, total_path_length).
        """
        if request_id in self._active or request_id not in self._req_path:
            return 0, 0

        self._unhang_from_all(request_id)
        self._waiting.discard(request_id)
        self._active.add(request_id)

        added = 0
        path  = self._req_path[request_id]
        for node in path:
            if node.ref_count == 0:
                added += 1
            node.ref_count += 1

        return added, len(path)

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def finish_request(self, request_id: str) -> Tuple[int, int]:
        """
        Active -> done.  Decrement ref_counts, prune dead nodes.
        Returns (evicted_nodes, total_path_length).
        """
        if request_id not in self._active:
            return 0, 0

        path = self._req_path[request_id]
        self._active.discard(request_id)

        evicted = 0
        for node in path:
            node.ref_count -= 1
            if node.ref_count == 0:
                evicted += 1

        self._req_tokens.pop(request_id)
        self._req_path.pop(request_id)
        self._prune_bottom_up(path)
        return evicted, len(path)

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, request_id: str) -> None:
        """Remove a waiting request (active requests must use finish_request)."""
        if request_id not in self._req_tokens or request_id in self._active:
            return
        path = self._req_path[request_id]
        self._unhang_from_all(request_id)
        self._waiting.discard(request_id)
        self._req_tokens.pop(request_id)
        self._req_path.pop(request_id)
        self._prune_bottom_up(path)

    # ------------------------------------------------------------------
    # count_missing  (no cache — always fresh)
    # ------------------------------------------------------------------

    def count_missing(self, request_id: str) -> int:
        """
        Count nodes on this request's path that are NOT in the working set.
        Computed fresh every call; no caching.
        """
        path = self._req_path.get(request_id)
        if path is None:
            return 0
        return sum(1 for node in path if not node.in_working_set)

    # ------------------------------------------------------------------
    # find_best_request  (DFS, no cache, no heap)
    # ------------------------------------------------------------------

    def find_best_request(self) -> Tuple[Optional[str], int]:
        """
        Find the best waiting request to schedule next.

        Strategy (in priority order):
        1. Among waiting requests whose path has the most nodes already
            in the working set (i.e. fewest missing chunks) — most cache-hot first.
        2. If no working-set overlap exists for any request, fall back to
            the request with the longest token sequence (most tokens = most
            benefit from being scheduled).
        3. If still nothing, return (None, 0).

        Never returns None just because in_working_set is False — every
        waiting request is a valid candidate.
        """
        if not self._waiting:
            return None, 0

        best_req     = None
        best_missing = float("inf")
        best_tokens  = -1

        for req_id in self._waiting:
            path = self._req_path.get(req_id)
            if path is None:
                continue

            missing   = sum(1 for node in path if not node.in_working_set)
            in_ws     = len(path) - missing          # how many nodes are cached
            token_len = len(self._req_tokens.get(req_id, []))

            if best_req is None:
                # First candidate — always accept.
                best_req     = req_id
                best_missing = missing
                best_tokens  = token_len
                continue

            best_in_ws = len(self._req_path[best_req]) - best_missing

            if in_ws > best_in_ws:
                # More cache overlap → prefer this one.
                best_req     = req_id
                best_missing = missing
                best_tokens  = token_len

            elif in_ws == best_in_ws:
                # Tie-break: longer sequence = higher scheduling priority.
                if token_len > best_tokens:
                    best_req     = req_id
                    best_missing = missing
                    best_tokens  = token_len

        if best_req is None:
            return None, 0

        return best_req, int(best_missing)

    # ------------------------------------------------------------------
    # Stats & dump
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        total_nodes = [0]
        ws_nodes    = [0]
        max_depth   = [0]

        def walk(node: Node) -> None:
            total_nodes[0] += 1
            if node.in_working_set:
                ws_nodes[0] += 1
            if node.depth > max_depth[0]:
                max_depth[0] = node.depth
            for child in node.children.values():
                walk(child)

        for child in self._root.children.values():
            walk(child)

        n = len(self._req_tokens)
        return {
            "total_requests":         n,
            "active_requests":        len(self._active),
            "waiting_requests":       len(self._waiting),
            "total_tree_nodes":       total_nodes[0],
            "working_set_nodes":      ws_nodes[0],
            "working_set_tokens_est": ws_nodes[0] * self.chunk_size,
            "tree_max_depth":         max_depth[0],
        }

    def dump(self) -> None:
        sep = "=" * 72
        print(f"\n{sep}\nSharedMerklePrefixTree\n{sep}")
        print(f"  chunk_size: {self.chunk_size} tokens/node\n")

        def walk(node: Node, indent: int = 0) -> None:
            ws   = "●" if node.in_working_set else "○"
            line = f"{'  ' * indent}[{ws} d={node.depth} ref={node.ref_count} hash={node.hash:016x}]"
            if node.hanging:
                details = {r: self.count_missing(r) for r in sorted(node.hanging)}
                line += f"  ← hanging(req: missing) {details}"
            print(line)
            for child in node.children.values():
                walk(child, indent + 1)

        print("  Tree:")
        print("    root (virtual, depth=0)")
        for child in self._root.children.values():
            walk(child, indent=2)

        print()
        for k, v in self.get_stats().items():
            print(f"  {k}: {v}")
        print(f"  active : {sorted(self._active)}")
        print(f"  waiting: {sorted(self._waiting)}")
        print(sep + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    import random

    def toks(rng: random.Random, n: int) -> List[int]:
        return [rng.randint(0, 50_000) for _ in range(n)]

    # ------------------------------------------------------------------
    # T1: requests hang on ALL ancestor nodes
    # ------------------------------------------------------------------
    rng = random.Random(1)
    t   = SharedMerklePrefixTree(chunk_size=16)
    prefix = toks(rng, 32)   # 2 shared chunks
    t.insert("A", prefix + toks(rng, 16))  # 3 chunks
    t.insert("B", prefix + toks(rng, 16))  # 3 chunks

    pa, pb = t._req_path["A"], t._req_path["B"]
    assert pa[0] is pb[0] and pa[1] is pb[1], "first 2 nodes must be shared"
    assert pa[2] is not pb[2],                "leaf nodes must diverge"
    for i in range(3):
        assert "A" in pa[i].hanging, f"A must hang on ancestor {i}"
        assert "B" in pb[i].hanging, f"B must hang on ancestor {i}"
    assert "A" not in pb[2].hanging
    assert "B" not in pa[2].hanging
    print("T1 passed: requests hang on all ancestors.")

    # ------------------------------------------------------------------
    # T2: activate removes from all hanging sets
    # ------------------------------------------------------------------
    t.activate_request("A")
    for node in pa:
        assert "A" not in node.hanging, "A must be unhung after activate"
    for node in pb:
        assert "B" in node.hanging, "B must still hang on its nodes"
    print("T2 passed: activate removes from all hanging nodes.")

    # ------------------------------------------------------------------
    # T3: find_best_request picks B at the deepest shared WS node
    # ------------------------------------------------------------------
    best, missing = t.find_best_request()
    assert best == "B", f"expected B, got {best!r}"
    # B has 1 missing chunk (its own leaf is not in WS)
    assert missing == 1, f"expected 1 missing, got {missing}"
    print(f"T3 passed: find_best_request -> {best!r}  missing={missing}")

    # ------------------------------------------------------------------
    # T4: finish evicts, B becomes unschedulable
    # ------------------------------------------------------------------
    evicted, total = t.finish_request("A")
    assert total   == 3
    assert evicted == 3   # A's 3 nodes all drop to ref=0 (B is waiting)
    best2, _ = t.find_best_request()
    assert best2 is None, f"B should be unschedulable, got {best2!r}"
    print("T4 passed: finish evicts correctly; B unschedulable when WS empty.")

    # ------------------------------------------------------------------
    # T5: min-missing selection among multiple candidates at the same node
    # ------------------------------------------------------------------
    rng2 = random.Random(99)
    t2   = SharedMerklePrefixTree(chunk_size=16)
    shared = toks(rng2, 16)   # 1 shared chunk
    t2.insert("X", shared)                    # 1 chunk  → 0 missing once WS has it
    t2.insert("Y", shared + toks(rng2, 32))   # 3 chunks → 2 missing once WS has it
    t2.insert("Z", shared)                    # activator — same single chunk
    t2.activate_request("Z")

    best3, missing3 = t2.find_best_request()
    assert best3    == "X", f"expected X (fewest missing), got {best3!r}"
    assert missing3 == 0,   f"expected 0 missing, got {missing3}"
    print(f"T5 passed: min-missing chose {best3!r} (missing={missing3}) over Y.")

    # ------------------------------------------------------------------
    # T6: remove cleans up hanging sets and prunes dead nodes
    # ------------------------------------------------------------------
    rng3 = random.Random(7)
    t3   = SharedMerklePrefixTree(chunk_size=16)
    tok3 = toks(rng3, 32)
    t3.insert("P", tok3)
    path_p = list(t3._req_path["P"])   # copy before removal
    t3.remove("P")
    assert "P" not in t3._req_tokens
    for node in path_p:
        assert "P" not in node.hanging
    # Nodes should be pruned (no active requests, no hanging)
    assert len(t3._root.children) == 0, "root should have no children after prune"
    print("T6 passed: remove cleans up and prunes dead nodes.")

    print("\nAll tests passed ✓")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import random
    rng = random.Random(7)

    def rand_toks(n: int) -> List[int]:
        return [rng.randint(0, 9_999) for _ in range(n)]

    t = SharedMerklePrefixTree(chunk_size=16)

    prefix_32 = rand_toks(32)
    prefix_48 = prefix_32 + rand_toks(16)

    reqs = {
        "A": prefix_32 + rand_toks(16),   # 3 chunks, shares first 2
        "B": prefix_32 + rand_toks(16),   # 3 chunks, shares first 2, diff tail
        "C": prefix_48 + rand_toks(16),   # 4 chunks, shares first 3
        "D": prefix_48 + rand_toks(16),   # 4 chunks, shares first 3, diff tail
        "E": rand_toks(32),               # 2 chunks, totally unique
    }

    for rid, tok_ids in reqs.items():
        t.insert(rid, tok_ids)

    print("=== After inserts ===")
    t.dump()

    t.activate_request("A")
    t.activate_request("C")
    print("=== After activating A, C ===")
    t.dump()

    best, missing = t.find_best_request()
    print(f"find_best_request -> {best!r}  missing={missing}\n")

    evicted, total = t.finish_request("A")
    print(f"finish('A'): evicted={evicted}  total={total}")
    print("=== After finishing A ===")
    t.dump()

    best, missing = t.find_best_request()
    print(f"find_best_request -> {best!r}  missing={missing}\n")


if __name__ == "__main__":
    print("=" * 72)
    print("TESTS")
    print("=" * 72)
    _run_tests()
    print()
    print("=" * 72)
    print("DEMO")
    print("=" * 72)
    _demo()