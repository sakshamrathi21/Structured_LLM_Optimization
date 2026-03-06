#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <optional>
#include <xxhash.h>
#include <algorithm>
#include <iostream>

class ChunkedHashTree {
private:
    struct HeapEntry {
        int missing;
        uint32_t request_id;
        
        bool operator<(const HeapEntry& other) const {
            if (missing != other.missing) return missing > other.missing;
            return request_id > other.request_id;
        }
    };

    uint32_t chunk_size;
    
    std::unordered_map<uint32_t, std::vector<uint64_t>> request_hashes;
    std::unordered_map<uint32_t, int> missing_cache;
    std::unordered_set<uint32_t> active_requests;
    std::unordered_set<uint64_t> working_set;
    std::unordered_map<uint64_t, uint32_t> hash_ref_counts;
    std::unordered_map<uint64_t, std::unordered_set<uint32_t>> hash_to_waiting;
    
    std::priority_queue<HeapEntry> waiting_heap;
    
    bool cache_valid;
    std::optional<std::pair<uint32_t, int>> cached_best_request;
    
    uint64_t pack_key(uint32_t level, uint64_t hash);
    std::vector<uint64_t> compute_hashes(const std::vector<uint32_t>& tokens);
    void invalidate_cache();

public:
    ChunkedHashTree(uint32_t chunk);
    
    uint32_t insert(uint32_t request_id, const std::vector<uint32_t>& tokens);
    std::pair<uint32_t, int> find_best_request();
    std::pair<uint32_t, uint32_t> activate_request(uint32_t request_id);
    std::pair<uint32_t, uint32_t> finish_request(uint32_t request_id);
    void remove(uint32_t request_id);
};