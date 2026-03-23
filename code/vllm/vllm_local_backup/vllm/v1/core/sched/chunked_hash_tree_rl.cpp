#include "chunked_hash_tree_rl.hpp"
#include <xxhash.h>
#include <algorithm>
#include <iostream>
#include <chrono>

// ── Timing infrastructure ────────────────────────────────────────────────────

struct FunctionStats {
    uint64_t total_ns   = 0;
    uint64_t call_count = 0;
};

struct ChunkedHashTree_RL_Stats {
    FunctionStats insert;
    FunctionStats find_best_request;
    FunctionStats activate;
    FunctionStats finish;
    // Add more as you instrument other functions

    uint64_t update_count = 0;   // total "updates" across all functions

    void log() const {
        std::cout << "[ChunkedHashTree_RL] === Stats after "
                  << update_count << " updates ===\n";

        auto report = [](const char* name, const FunctionStats& s) {
            if (s.call_count == 0) return;
            std::cout << "  " << name
                      << "  calls=" << s.call_count
                      << "  total_ns=" << s.total_ns
                      << "  avg_ns=" << s.total_ns / s.call_count
                      << '\n';
        };

        report("insert              ", insert);
        report("find_best_request      ", find_best_request);
        report("activate", activate);
        report("finish", finish);
        std::cout << std::flush;
    }
};

// One instance per ChunkedHashTree_RL object (or make it static/global if preferred)
static ChunkedHashTree_RL_Stats g_stats;

// RAII timer: records elapsed ns into a FunctionStats slot and bumps update_count
struct ScopedTimer {
    using Clock = std::chrono::high_resolution_clock;

    FunctionStats&          slot;
    ChunkedHashTree_RL_Stats& stats;
    Clock::time_point       start;

    ScopedTimer(FunctionStats& slot_, ChunkedHashTree_RL_Stats& stats_)
        : slot(slot_), stats(stats_), start(Clock::now()) {}

    ~ScopedTimer() {
        uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start).count());
        slot.total_ns   += ns;
        slot.call_count += 1;
        stats.update_count += 1;
        // if (stats.update_count % 100 == 0)
        //     stats.log();
    }
};

// Convenience macro so call sites stay one-liners
#define TIME_FN(slot_name) \
    ScopedTimer _timer_(g_stats.slot_name, g_stats)

ChunkedHashTree_RL::ChunkedHashTree_RL(uint32_t chunk) 
    : chunk_size(chunk), cache_valid(false), shared_chain_length(0)
{
    // Reserve capacity for common use cases
    request_hashes.reserve(1024);
    missing_cache.reserve(1024);
    hash_to_waiting.reserve(4096);
    working_set.reserve(4096);
    hash_ref_counts.reserve(4096);
    shared_prefix_chain.reserve(256); // Reserve space for shared chain
}

uint64_t ChunkedHashTree_RL::pack_key(uint32_t level, uint64_t hash) 
{
    return (uint64_t(level) << 56) | (hash & 0x00FFFFFFFFFFFFFFULL);
}

void ChunkedHashTree_RL::invalidate_cache()
{
    cache_valid = false;
    cached_best_request.reset();
}

void ChunkedHashTree_RL::recompute_shared_chain() {
    // Recompute the entire shared prefix chain from scratch
    shared_prefix_chain.clear();
    shared_chain_length = 0;
    
    if (active_requests.empty()) {
        return;
    }
    
    // Find the maximum possible chain length
    uint32_t max_len = UINT32_MAX;
    for (auto req_id : active_requests) {
        auto it = request_hashes.find(req_id);
        if (it != request_hashes.end()) {
            max_len = std::min(max_len, (uint32_t)it->second.size());
        }
    }
    
    if (max_len == 0 || max_len == UINT32_MAX) {
        return;
    }
    
    // Build the shared chain level by level
    for (uint32_t level = 0; level < max_len; ++level) {
        uint64_t common_hash = 0;
        bool first = true;
        bool all_match = true;
        
        for (auto req_id : active_requests) {
            auto it = request_hashes.find(req_id);
            if (it != request_hashes.end() && level < it->second.size()) {
                uint64_t hash = it->second[level];
                if (first) {
                    common_hash = hash;
                    first = false;
                } else if (hash != common_hash) {
                    all_match = false;
                    break;
                }
            } else {
                all_match = false;
                break;
            }
        }
        
        if (all_match) {
            shared_prefix_chain.push_back(common_hash);
            shared_chain_length++;
        } else {
            break; // Chain ends here
        }
    }
}

void ChunkedHashTree_RL::update_chain_on_activate(uint32_t request_id) {
    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) {
        return;
    }

    const auto& hashes = it->second;

    if (active_requests.size() == 1) {
        // First active request - the entire request becomes the chain
        shared_prefix_chain = hashes;
        shared_chain_length = hashes.size();
        return;
    }

    // With cumulative hashes, matching is monotone: if level k matches,
    // all levels 0..k-1 match too. So we binary search for the new chain length.
    uint32_t lo = 0;
    uint32_t hi = std::min(shared_chain_length, (uint32_t)hashes.size());

    // Binary search for the longest prefix that still matches
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2; // upper-mid to avoid infinite loop
        if (hashes[mid - 1] == shared_prefix_chain[mid - 1]) {
            lo = mid; // level mid-1 matches, so at least mid levels match
        } else {
            hi = mid - 1; // level mid-1 doesn't match, chain ends before here
        }
    }

    shared_chain_length = lo;
    shared_prefix_chain.resize(shared_chain_length);
}

void ChunkedHashTree_RL::update_chain_on_finish(uint32_t request_id) {
    if (active_requests.empty()) {
        shared_prefix_chain.clear();
        shared_chain_length = 0;
        return;
    }
    
    if (active_requests.size() == 1) {
        // Only one request left - recompute chain as entire request
        recompute_shared_chain();
        return;
    }
    
    // Chain length won't decrease, but might extend
    // Check if we can extend the chain beyond current length
    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) {
        return;
    }
    
    const auto& finished_hashes = it->second;
    
    // Find max possible extension length
    uint32_t max_possible = UINT32_MAX;
    for (auto req_id : active_requests) {
        auto req_it = request_hashes.find(req_id);
        if (req_it != request_hashes.end()) {
            max_possible = std::min(max_possible, (uint32_t)req_it->second.size());
        }
    }
    
    // Try to extend the chain
    for (uint32_t level = shared_chain_length; level < max_possible; ++level) {
        uint64_t common_hash = 0;
        bool first = true;
        bool all_match = true;
        
        for (auto req_id : active_requests) {
            auto req_it = request_hashes.find(req_id);
            if (req_it != request_hashes.end() && level < req_it->second.size()) {
                uint64_t hash = req_it->second[level];
                if (first) {
                    common_hash = hash;
                    first = false;
                } else if (hash != common_hash) {
                    all_match = false;
                    break;
                }
            } else {
                all_match = false;
                break;
            }
        }
        
        if (all_match) {
            shared_prefix_chain.push_back(common_hash);
            shared_chain_length++;
        } else {
            break; // Can't extend further
        }
    }
}

uint32_t ChunkedHashTree_RL::compute_chain_length_with_request(uint32_t request_id) {
    // Compute what the chain length would be if we added this request
    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) {
        return shared_chain_length;
    }
    
    const auto& hashes = it->second;
    
    if (active_requests.empty()) {
        // Would become the first request - entire request is the chain
        return hashes.size();
    }
    
    // Check how far the new request matches the current chain
    uint32_t matching_length = 0;
    for (uint32_t level = 0; level < shared_chain_length && level < hashes.size(); ++level) {
        if (hashes[level] == shared_prefix_chain[level]) {
            matching_length++;
        } else {
            break;
        }
    }
    
    return matching_length;
}

uint32_t ChunkedHashTree_RL::count_requests_sharing_prefix(uint32_t prefix_length) {
    // Count how many waiting requests share the prefix up to prefix_length
    // We only need to check the LAST level since all previous levels are dependent on it
    if (prefix_length == 0 || prefix_length > shared_chain_length) {
        return 0;
    }
    
    // Check only the last level of the prefix (prefix_length - 1)
    uint32_t last_level = prefix_length - 1;
    uint64_t key = pack_key(last_level, shared_prefix_chain[last_level]);
    
    auto it = hash_to_waiting.find(key);
    if (it == hash_to_waiting.end()) {
        return 0; // No requests have this hash at this level
    }
    
    // Count only waiting (non-active) requests
    uint32_t count = 0;
    for (auto req_id : it->second) {
        if (!active_requests.count(req_id)) {
            count++;
        }
    }
    
    return count;
}

std::vector<uint64_t>
ChunkedHashTree_RL::compute_hashes(const std::vector<uint32_t>& tokens) 
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

uint32_t ChunkedHashTree_RL::insert(uint32_t request_id, const std::vector<uint32_t>& tokens)
{
    // TIME_FN(insert);
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

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> ChunkedHashTree_RL::find_best_request() {
    // std::cout << "heap size: " << waiting_heap.size()
    //       << " missing_cache size: " << missing_cache.size()
    //       << std::endl;
    // std::cout << "inside chunked_hash_tree_rl --> find_best_request()" << std::endl;
    // TIME_FN(find_best_request);
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
        
        // Found the best request
        // Current chain length is chunks_before
        uint32_t chunks_before = shared_chain_length;
        
        // Compute what chain length would be after adding this request
        uint32_t chunks_after = compute_chain_length_with_request(top.request_id);
        
        // Count how many waiting requests share the prefix up to chunks_after level
        uint32_t requests_at_same_level = count_requests_sharing_prefix(chunks_after);
        
        // Cache the result
        cached_best_request = std::make_tuple(top.request_id, chunks_before, chunks_after, requests_at_same_level);
        cache_valid = true;
        
        // Put it back in the heap since we're not activating it yet
        waiting_heap.push(top);
        
        return cached_best_request.value();
    }
    
    // No valid request found
    cached_best_request = std::make_tuple(0, 0, 0, 0);
    cache_valid = true;
    return std::make_tuple(0, 0, 0, 0);
}

std::pair<uint32_t, uint32_t>
ChunkedHashTree_RL::activate_request(uint32_t request_id) 
{
    // TIME_FN(activate);
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
                    auto &m = missing_cache[w];
                    if (m > 0) {
                        m--;
                        waiting_heap.push({m, w});
                    }
                }
            }
            added++;
        }
    }
    
    missing_cache.erase(request_id);
    
    // Update the shared prefix chain
    update_chain_on_activate(request_id);
    
    // Invalidate cache since working set changed
    invalidate_cache();

    return {added, (uint32_t)hashes.size()};
}

void ChunkedHashTree_RL::remove(uint32_t request_id) {
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
ChunkedHashTree_RL::finish_request(uint32_t request_id) {
    // TIME_FN(finish);
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
    
    // Update the shared prefix chain (may extend)
    update_chain_on_finish(request_id);
    
    remove(request_id);
    
    // Invalidate cache since working set changed
    invalidate_cache();
    
    return {evicted, (uint32_t)hashes.size()};
}