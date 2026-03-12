#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <optional>
#include <tuple>
#include <cstdint>

struct HeapEntry {
    int missing;
    uint32_t request_id;
    
    bool operator<(const HeapEntry& other) const {
        if (missing != other.missing) return missing > other.missing;
        return request_id > other.request_id;
    }
};

class ChunkedHashTree_RL {
public:
    ChunkedHashTree_RL(uint32_t chunk);
    
    uint32_t insert(uint32_t request_id, const std::vector<uint32_t>& tokens);
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> find_best_request();
    std::pair<uint32_t, uint32_t> activate_request(uint32_t request_id);
    std::pair<uint32_t, uint32_t> finish_request(uint32_t request_id);
    void remove(uint32_t request_id);
    
private:
    uint32_t chunk_size;
    
    // Core data structures
    std::unordered_map<uint32_t, std::vector<uint64_t>> request_hashes;
    std::unordered_set<uint64_t> working_set;
    std::unordered_set<uint32_t> active_requests;
    
    // Tracking structures
    std::unordered_map<uint64_t, std::unordered_set<uint32_t>> hash_to_waiting;
    std::unordered_map<uint32_t, int> missing_cache;
    std::unordered_map<uint64_t, uint32_t> hash_ref_counts;
    
    // Priority queue for finding best request
    std::priority_queue<HeapEntry> waiting_heap;
    
    // Cache for find_best_request
    bool cache_valid;
    std::optional<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> cached_best_request;
    
    // Shared prefix chain among active requests
    std::vector<uint64_t> shared_prefix_chain; // The hashes that form the shared prefix
    uint32_t shared_chain_length; // Length of the shared prefix chain
    
    // Helper functions
    uint64_t pack_key(uint32_t level, uint64_t hash);
    std::vector<uint64_t> compute_hashes(const std::vector<uint32_t>& tokens);
    void invalidate_cache();
    void recompute_shared_chain();
    void update_chain_on_activate(uint32_t request_id);
    void update_chain_on_finish(uint32_t request_id);
    uint32_t compute_chain_length_with_request(uint32_t request_id);
    uint32_t count_requests_sharing_prefix(uint32_t prefix_length);
};