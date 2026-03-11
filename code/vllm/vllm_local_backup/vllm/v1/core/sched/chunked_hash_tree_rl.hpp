#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <xxhash.h>
#include <algorithm>
#include <iostream>

class ChunkedHashTree_RL {
private:
    uint32_t chunk_size;

    // request_id -> list of chunk-level hashes
    std::unordered_map<uint32_t, std::vector<uint64_t>> request_hashes;

    // requests currently being processed
    std::unordered_set<uint32_t> active_requests;

    // requests inserted but not yet activated (waiting)
    std::unordered_set<uint32_t> waiting_requests;

    // ref-count per packed key (level, hash)
    std::unordered_map<uint64_t, uint32_t> hash_ref_counts;

    // packed key -> set of waiting request_ids that need it
    std::unordered_map<uint64_t, std::unordered_set<uint32_t>> hash_to_waiting;

    // ---- shared prefix chain (mirrors ChunkedHashTree_RL_RL) ----
    std::vector<uint64_t> shared_prefix_chain;
    uint32_t             shared_chain_length;

    // ---- cache for find_best_request ----
    bool cache_valid;
    // (request_id, chunks_before, chunks_after, waiters_at_level)
    std::optional<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> cached_best_request;

    // helpers
    uint64_t pack_key(uint32_t level, uint64_t hash) const;
    std::vector<uint64_t> compute_hashes(const std::vector<uint32_t>& tokens);
    void invalidate_cache();

    // shared-prefix maintenance
    void recompute_shared_chain();
    void update_chain_on_activate(uint32_t request_id);
    void update_chain_on_finish(uint32_t request_id);

    // new cost helpers
    uint32_t compute_chain_length_with_request(uint32_t request_id) const;
    uint32_t count_waiters_at_level(uint32_t level) const;

public:
    explicit ChunkedHashTree_RL(uint32_t chunk);

    // Insert a new request (not yet active).
    // Returns total token count.
    uint32_t insert(uint32_t request_id, const std::vector<uint32_t>& tokens);

    // Find the best waiting request to schedule next.
    // Returns (request_id, chunks_before, chunks_after, waiters_at_chosen_level).
    // request_id == 0 means no waiting request exists.
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> find_best_request();

    // Move a waiting request into the active set.
    // Returns (newly_added_keys, total_keys).
    std::pair<uint32_t, uint32_t> activate_request(uint32_t request_id);

    // Mark an active request as finished; evict its keys if ref-count hits 0.
    // Returns (evicted_keys, total_keys).
    std::pair<uint32_t, uint32_t> finish_request(uint32_t request_id);

    // Remove a waiting (not active) request entirely.
    void remove(uint32_t request_id);
};