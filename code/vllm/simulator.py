import random
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimConfig:
    num_families: int = 1
    requests_per_family: int = 500
    blocks_per_request: int = 401
    shared_prefix_blocks: int = 400
    decode_iterations: int = 100
    l2_cache_size: int = 25  # in blocks
    prefetch_distance: int = 5  # blocks to prefetch ahead
    num_sms: int = 200  # Number of SMs
    stride_history_size: int = 3
    confidence_threshold: int = 2
    scheduling_policy: str = "random"  # "random" or "round_robin"

    # Prefetch strategy
    enable_stride_prefetch: bool = True
    enable_next_line_prefetch: bool = False
    enable_region_prefetch: bool = True


class StridePredictor:
    def __init__(self, history_size: int = 3, confidence_threshold: int = 2):
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.access_history = deque(maxlen=history_size)
        self.last_stride = None
        self.stride_count = 0

    def record_access(self, block_id: str) -> Optional[int]:
        block_num = self._extract_block_number(block_id)
        if block_num is None:
            self.access_history.clear()
            return None

        self.access_history.append((block_id, block_num))

        if len(self.access_history) < 2:
            return None

        current_stride = self.access_history[-1][1] - self.access_history[-2][1]

        if self.last_stride == current_stride:
            self.stride_count += 1
        else:
            self.last_stride = current_stride
            self.stride_count = 1

        if self.stride_count >= self.confidence_threshold:
            return current_stride

        return None

    def _extract_block_number(self, block_id: str) -> Optional[int]:
        try:
            parts = block_id.split("_")
            return int(parts[-1])
        except (ValueError, IndexError):
            return None

    def predict_next_blocks(
        self, last_block_id: str, stride: int, count: int
    ) -> List[str]:
        predicted_blocks = []
        parts = last_block_id.split("_")
        if len(parts) < 2:
            return []

        try:
            base_num = int(parts[-1])
            base_pattern = "_".join(parts[:-1])

            for i in range(1, count + 1):
                next_num = base_num + (stride * i)
                predicted_blocks.append(f"{base_pattern}_{next_num}")
        except (ValueError, IndexError):
            pass

        return predicted_blocks

    def reset(self):
        self.access_history.clear()
        self.last_stride = None
        self.stride_count = 0


class SpatialRegionPredictor:

    def __init__(self, region_size: int = 4):
        self.region_size = region_size
        self.recent_accesses = deque(maxlen=10)

    def get_nearby_blocks(self, block_id: str, distance: int) -> List[str]:

        nearby = []

        # Extract pattern and block number
        parts = block_id.split("_")
        if len(parts) < 2:
            return []

        try:
            base_num = int(parts[-1])
            base_pattern = "_".join(parts[:-1])

            # Prefetch blocks within a spatial window
            for offset in range(1, distance + 1):
                nearby.append(f"{base_pattern}_{base_num + offset}")

        except (ValueError, IndexError):
            pass

        return nearby


class L2Cache:
    def __init__(
        self,
        size: int,
        prefetch_distance: int = 0,
        stride_history_size: int = 3,
        confidence_threshold: int = 2,
        enable_stride_prefetch: bool = True,
        enable_next_line_prefetch: bool = True,
        enable_region_prefetch: bool = True,
    ):
        self.size = size
        self.prefetch_distance = prefetch_distance
        self.cache = OrderedDict()  # LRU cache

        self.enable_stride_prefetch = enable_stride_prefetch
        self.enable_next_line_prefetch = enable_next_line_prefetch
        self.enable_region_prefetch = enable_region_prefetch

        # Statistics
        self.demand_hits = 0
        self.demand_misses = 0
        self.dram_accesses = 0
        self.useful_prefetches = 0
        self.wasted_prefetches = 0

        self.prefetched_blocks = set()
        self.accessed_blocks = set()

        # Predictors
        self.stride_predictor = StridePredictor(
            stride_history_size, confidence_threshold
        )
        self.region_predictor = SpatialRegionPredictor()

        # Prefetch statistics
        self.stride_prefetches = 0
        self.region_prefetches = 0
        self.next_line_prefetches = 0
        self.confident_prefetches = 0
        self.total_prefetch_opportunities = 0

    def access(self, block_id: str, timestamp: int, next_block_ids: List[str] = []):
        # Track if this was a useful prefetch
        if block_id in self.prefetched_blocks:
            self.useful_prefetches += 1
            self.prefetched_blocks.discard(block_id)

        # Handle demand access
        if block_id in self.cache:
            self.demand_hits += 1
            self.cache.move_to_end(block_id)
        else:
            self.demand_misses += 1
            self.dram_accesses += 1
            self._insert(block_id, timestamp)

        self.accessed_blocks.add(block_id)

        if self.prefetch_distance > 0:
            self.total_prefetch_opportunities += 1
            self._do_prefetching(block_id, timestamp, next_block_ids)

    def _do_prefetching(self, block_id: str, timestamp: int, next_block_ids: List[str]):
        blocks_to_prefetch = set()

        # Strategy 1: Next-line prefetching (simple sequential from known next blocks)
        if self.enable_next_line_prefetch and next_block_ids:
            for next_id in next_block_ids[: self.prefetch_distance]:
                blocks_to_prefetch.add(next_id)
                self.next_line_prefetches += 1

        # Strategy 2: Stride-based confidence prefetching
        if self.enable_stride_prefetch:
            predicted_stride = self.stride_predictor.record_access(block_id)
            if predicted_stride is not None:
                self.confident_prefetches += 1
                predicted_blocks = self.stride_predictor.predict_next_blocks(
                    block_id, predicted_stride, self.prefetch_distance
                )
                for pred_id in predicted_blocks:
                    blocks_to_prefetch.add(pred_id)
                    self.stride_prefetches += 1

        # Strategy 3: Spatial region prefetching (nearby blocks in address space)
        if self.enable_region_prefetch:
            nearby_blocks = self.region_predictor.get_nearby_blocks(
                block_id, min(2, self.prefetch_distance)
            )
            for nearby_id in nearby_blocks:
                blocks_to_prefetch.add(nearby_id)
                self.region_prefetches += 1

        # Execute prefetches
        for prefetch_id in blocks_to_prefetch:
            if prefetch_id not in self.cache:
                self.dram_accesses += 1
                self.prefetched_blocks.add(prefetch_id)
                self._insert(prefetch_id, timestamp)

    def _insert(self, block_id: str, timestamp: int):
        if len(self.cache) >= self.size:
            evicted_id, _ = self.cache.popitem(last=False)

            if evicted_id in self.prefetched_blocks:
                self.wasted_prefetches += 1
                self.prefetched_blocks.discard(evicted_id)

        self.cache[block_id] = timestamp

    def get_stats(self):
        total_accesses = self.demand_hits + self.demand_misses
        hit_rate = (
            (self.demand_hits / total_accesses * 100) if total_accesses > 0 else 0
        )

        total_prefetches = self.useful_prefetches + self.wasted_prefetches
        prefetch_accuracy = (
            (self.useful_prefetches / total_prefetches * 100)
            if total_prefetches > 0
            else 0
        )

        confidence_rate = (
            (self.confident_prefetches / self.total_prefetch_opportunities * 100)
            if self.total_prefetch_opportunities > 0
            else 0
        )

        return {
            "demand_hits": self.demand_hits,
            "demand_misses": self.demand_misses,
            "total_accesses": total_accesses,
            "hit_rate": hit_rate,
            "dram_accesses": self.dram_accesses,
            "useful_prefetches": self.useful_prefetches,
            "wasted_prefetches": self.wasted_prefetches,
            "prefetch_accuracy": prefetch_accuracy,
            "confident_prefetches": self.confident_prefetches,
            "confidence_rate": confidence_rate,
            "stride_prefetches": self.stride_prefetches,
            "region_prefetches": self.region_prefetches,
            "next_line_prefetches": self.next_line_prefetches,
        }


class Request:
    def __init__(self, req_id: str, family: int, blocks: List[str]):
        self.id = req_id
        self.family = family
        self.blocks = blocks
        self.current_block_idx = 0

    def get_next_access(self):
        if self.current_block_idx < len(self.blocks):
            block_id = self.blocks[self.current_block_idx]
            next_blocks = self.blocks[self.current_block_idx + 1 :]
            self.current_block_idx += 1
            return block_id, next_blocks
        return None, []

    def reset(self):
        self.current_block_idx = 0

    def is_complete(self):
        return self.current_block_idx >= len(self.blocks)


def create_requests(config: SimConfig) -> List[Request]:
    requests = []

    for fam in range(config.num_families):
        for req in range(config.requests_per_family):
            blocks = []

            # Shared prefix (e.g., system prompt + user prompt)
            for i in range(config.shared_prefix_blocks):
                blocks.append(f"fam{fam}_shared_{i}")

            # Unique generated tokens per request
            for i in range(config.shared_prefix_blocks, config.blocks_per_request):
                blocks.append(f"fam{fam}_req{req}_block{i}")

            requests.append(
                Request(req_id=f"fam{fam}_req{req}", family=fam, blocks=blocks)
            )

    return requests


def schedule_requests(
    requests: List[Request], num_sms: int, policy: str
) -> List[Request]:
    if policy == "round_robin":
        # Simple round-robin: take first N requests
        return requests[: min(num_sms, len(requests))]
    else:
        return random.sample(requests, min(num_sms, len(requests)))


def run_simulation(config: SimConfig, enable_prefetch: bool):
    requests = create_requests(config)

    if enable_prefetch:
        cache = L2Cache(
            config.l2_cache_size,
            config.prefetch_distance,
            config.stride_history_size,
            config.confidence_threshold,
            config.enable_stride_prefetch,
            config.enable_next_line_prefetch,
            config.enable_region_prefetch,
        )
    else:
        cache = L2Cache(config.l2_cache_size, 0)

    timestamp = 0
    hit_rate_history = []

    for _ in range(config.decode_iterations):
        # Reset all requests for new decode iteration
        for req in requests:
            req.reset()

        active_requests = list(requests)

        while active_requests:
            # Schedule requests based on policy
            executing_requests = schedule_requests(
                active_requests, config.num_sms, config.scheduling_policy
            )

            for req in executing_requests:
                block_id, next_blocks = req.get_next_access()

                if block_id:
                    next_blocks_limited = next_blocks[: config.prefetch_distance]
                    cache.access(block_id, timestamp, next_blocks_limited)
                    timestamp += 1

                if req.is_complete():
                    active_requests.remove(req)

        stats = cache.get_stats()
        hit_rate_history.append(stats["hit_rate"])

    return cache.get_stats(), hit_rate_history


def print_detailed_stats(stats_with_no_pf, stats_with_pf, config: SimConfig):
    """Print detailed comparison statistics"""
    print("\n" + "=" * 70)
    print("   SIMULATION CONFIGURATION:")
    print("-" * 70)
    print(f"  Number of SMs:         {config.num_sms}")
    print(f"  Request Families:      {config.num_families}")
    print(f"  Requests per Family:   {config.requests_per_family}")
    print(
        f"  Total Requests:        {config.num_families * config.requests_per_family}"
    )
    print(f"  Blocks per Request:    {config.blocks_per_request}")
    print(f"  Shared Prefix Blocks:  {config.shared_prefix_blocks}")
    print(f"  L2 Cache Size:         {config.l2_cache_size} blocks")
    print(f"  Decode Iterations:     {config.decode_iterations}")
    print(f"  Scheduling Policy:     {config.scheduling_policy}")
    print(f"  Prefetch Distance:     {config.prefetch_distance} blocks")

    print("\n  WITHOUT PREFETCHING:")
    print("-" * 70)
    print(f"  Cache Hit Rate:        {stats_with_no_pf['hit_rate']:.2f}%")
    print(f"  DRAM Accesses:         {stats_with_no_pf['dram_accesses']:,}")
    print(f"  Demand Hits:         {stats_with_no_pf['demand_hits']:,}")
    print(f"  Demand Misses:         {stats_with_no_pf['demand_misses']:,}")
    print(f"  Total Accesses:         {stats_with_no_pf['total_accesses']:,}")

    print("\n  WITH PREFETCHING:")
    print("-" * 70)
    print(f"  Cache Hit Rate:        {stats_with_pf['hit_rate']:.2f}%")
    print(f"  DRAM Accesses:         {stats_with_pf['dram_accesses']:,}")
    print(f"  Demand Hits:         {stats_with_pf['demand_hits']:,}")
    print(f"  Demand Misses:         {stats_with_pf['demand_misses']:,}")
    print(f"  Total Accesses:         {stats_with_pf['total_accesses']:,}")
    print(f"  Useful Prefetches:     {stats_with_pf['useful_prefetches']:,}")
    print(f"  Wasted Prefetches:     {stats_with_pf['wasted_prefetches']:,}")
    print(f"  Prefetch Accuracy:     {stats_with_pf['prefetch_accuracy']:.2f}%")
    print(f"  Stride Prefetches:     {stats_with_pf['stride_prefetches']:,}")
    print(f"  Region Prefetches:     {stats_with_pf['region_prefetches']:,}")
    print(f"  Next-Line Prefetches:  {stats_with_pf['next_line_prefetches']:,}")


if __name__ == "__main__":
    config = SimConfig()

    print("=" * 70)
    print("  Running simulations...")

    print("  Without prefetching...")
    stats_with_no_pf, _ = run_simulation(config, enable_prefetch=False)

    print("  With prefetching...")
    stats_with_pf, _ = run_simulation(config, enable_prefetch=True)

    print_detailed_stats(stats_with_no_pf, stats_with_pf, config)

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
