#include "chunked_hash_tree.hpp"
#include <xxhash.h>
#include <algorithm>
#include <iostream>

ChunkedHashTree::ChunkedHashTree(uint32_t chunk) 
    : chunk_size(chunk), cache_valid(false) 
{
    // Reserve capacity for common use cases
    request_hashes.reserve(1024);
    missing_cache.reserve(1024);
    hash_to_waiting.reserve(4096);
    working_set.reserve(4096);
    hash_ref_counts.reserve(4096);
}

uint64_t ChunkedHashTree::pack_key(uint32_t level, uint64_t hash) 
{
    return (uint64_t(level) << 56) | (hash & 0x00FFFFFFFFFFFFFFULL);
}

void ChunkedHashTree::invalidate_cache()
{
    cache_valid = false;
    cached_best_request.reset();
}

std::vector<uint64_t>
ChunkedHashTree::compute_hashes(const std::vector<uint32_t>& tokens) 
{
    std::vector<uint64_t> hashes;
    size_t num_chunks = (tokens.size() + chunk_size - 1) / chunk_size;
    hashes.reserve(num_chunks);
    
    XXH64_state_t* state = XXH64_createState();
    XXH64_reset(state, 0);
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        XXH64_update(state, &tokens[i], sizeof(uint32_t));
        if ((i + 1) % chunk_size == 0 || i + 1 == tokens.size()) {
            hashes.push_back(XXH64_digest(state));
        }
    }
    
    XXH64_freeState(state);
    return hashes;
}

uint32_t ChunkedHashTree::insert(uint32_t request_id, const std::vector<uint32_t>& tokens)
{
    if (request_hashes.count(request_id)) {
        return 0;
    }
    
    auto hashes = compute_hashes(tokens);
    request_hashes[request_id] = hashes;
    
    int missing = 0;
    for (uint32_t level = 0; level < hashes.size(); ++level) 
    {
        uint64_t key = pack_key(level, hashes[level]);
        if (!working_set.count(key)) missing++;
        
        auto& waiting_set = hash_to_waiting[key];
        if (waiting_set.empty()) {
            waiting_set.reserve(16);
        }
        waiting_set.insert(request_id);
    }
    
    missing_cache[request_id] = missing;
    waiting_heap.push({missing, request_id});
    
    invalidate_cache();
    
    return tokens.size();
}

std::pair<uint32_t, int> ChunkedHashTree::find_best_request() {
    if (cache_valid && cached_best_request.has_value()) {
        return cached_best_request.value();
    }
    
    while (!waiting_heap.empty())
    {
        auto top = waiting_heap.top();
        waiting_heap.pop();
        
        if (active_requests.count(top.request_id)) {
            continue;
        }
        
        // Skip if request no longer exists
        auto it = missing_cache.find(top.request_id);
        if (it == missing_cache.end()) {
            continue;
        }
        
        // Skip if this heap entry is stale (missing count has changed)
        if (it->second != top.missing) {
            continue;
        }
        
        // Found the best request - cache it and return
        cached_best_request = {top.request_id, top.missing};
        cache_valid = true;
        
        // Put it back in the heap since we're not activating it yet
        waiting_heap.push(top);
        
        return cached_best_request.value();
    }
    
    // No valid request found
    cached_best_request = {0, 0};
    cache_valid = true;
    return {0, 0};
}

std::pair<uint32_t, uint32_t>
ChunkedHashTree::activate_request(uint32_t request_id) 
{
    if (active_requests.count(request_id)) {
        return {0, 0};
    }
    
    if (!request_hashes.count(request_id)) {
        return {0, 0};
    }
    
    active_requests.insert(request_id);
    auto& hashes = request_hashes[request_id];
    uint32_t added = 0;
    
    for (uint32_t level = 0; level < hashes.size(); ++level)
    {
        uint64_t key = pack_key(level, hashes[level]);
        if (hash_ref_counts[key]++ == 0) {
            working_set.insert(key);
            
            // Update all waiting requests that are affected by this new hash
            auto waiting_it = hash_to_waiting.find(key);
            if (waiting_it != hash_to_waiting.end()) {
                for (auto w : waiting_it->second) {
                    if (active_requests.count(w) || !missing_cache.count(w)) continue;
                    if (--missing_cache[w] >= 0) {
                        waiting_heap.push({missing_cache[w], w});
                    }
                }
            }
            added++;
        }
    }
    
    missing_cache.erase(request_id);
    
    // Invalidate cache since working set changed
    invalidate_cache();

    return {added, (uint32_t)hashes.size()};
}

void ChunkedHashTree::remove(uint32_t request_id) {
    auto it = request_hashes.find(request_id);
    if (it != request_hashes.end()) {
        auto& hashes = it->second;
        for (uint32_t level = 0; level < hashes.size(); ++level) {
            uint64_t key = pack_key(level, hashes[level]);
            auto map_it = hash_to_waiting.find(key);
            if (map_it != hash_to_waiting.end()) {
                // O(1) erase with unordered_set instead of O(n) vector scan
                map_it->second.erase(request_id);
                
                // Clean up empty sets
                if (map_it->second.empty()) {
                    hash_to_waiting.erase(map_it);
                }
            }
        }
    }
    
    request_hashes.erase(request_id);
    missing_cache.erase(request_id);
    
    // Invalidate cache since we removed a request
    invalidate_cache();
}

std::pair<uint32_t, uint32_t>
ChunkedHashTree::finish_request(uint32_t request_id) {
    if (!active_requests.erase(request_id)) {
        return {0, 0};
    }
    
    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) {
        return {0, 0};
    }
    
    auto& hashes = it->second;
    uint32_t evicted = 0;
    
    for (uint32_t level = 0; level < hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        if (--hash_ref_counts[key] == 0) {
            working_set.erase(key);
            hash_ref_counts.erase(key);
            
            // Update all waiting requests that are affected by this eviction
            auto waiting_it = hash_to_waiting.find(key);
            if (waiting_it != hash_to_waiting.end()) {
                for (auto w : waiting_it->second) {
                    if (active_requests.count(w) || !missing_cache.count(w)) continue;
                    missing_cache[w]++;
                    waiting_heap.push({missing_cache[w], w});
                }
            }
            evicted++;
        }
    }
    
    remove(request_id);
    
    // Invalidate cache since working set changed
    invalidate_cache();
    
    return {evicted, (uint32_t)hashes.size()};
}