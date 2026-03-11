#include "chunked_hash_tree_rl.hpp"

ChunkedHashTree_RL::ChunkedHashTree_RL(uint32_t chunk)
    : chunk_size(chunk), shared_chain_length(0), cache_valid(false)
{
    request_hashes.reserve(8192);
    hash_to_waiting.reserve(8192);
    hash_ref_counts.reserve(8192);
    shared_prefix_chain.reserve(256);
}

uint64_t ChunkedHashTree_RL::pack_key(uint32_t level, uint64_t hash) const
{
    return (uint64_t(level) << 56) | (hash & 0x00FFFFFFFFFFFFFFULL);
}

void ChunkedHashTree_RL::invalidate_cache()
{
    cache_valid = false;
    cached_best_request.reset();
}

void ChunkedHashTree_RL::recompute_shared_chain()
{
    shared_prefix_chain.clear();
    shared_chain_length = 0;

    if (active_requests.empty()) return;

    uint32_t max_len = UINT32_MAX;
    for (auto req_id : active_requests) {
        auto it = request_hashes.find(req_id);
        if (it != request_hashes.end()) {
            max_len = std::min(max_len, (uint32_t)it->second.size());
        }
    }

    if (max_len == 0 || max_len == UINT32_MAX) return;

    for (uint32_t level = 0; level < max_len; ++level) {
        uint64_t common_hash = 0;
        bool first = true;
        bool all_match = true;

        for (auto req_id : active_requests) {
            auto it = request_hashes.find(req_id);
            if (it != request_hashes.end() && level < it->second.size()) {
                uint64_t h = it->second[level];
                if (first) { common_hash = h; first = false; }
                else if (h != common_hash) { all_match = false; break; }
            } 
            else {
                all_match = false;
                break;
            }
        }

        if (all_match) {
            shared_prefix_chain.push_back(common_hash);
            shared_chain_length++;
        } 
        else {
            break;
        }
    }
}

void ChunkedHashTree_RL::update_chain_on_activate(uint32_t request_id)
{
    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) return;

    const auto& hashes = it->second;

    if (active_requests.size() == 1) {
        shared_prefix_chain = hashes;
        shared_chain_length = (uint32_t)hashes.size();
        return;
    }

    uint32_t new_len = 0;
    for (uint32_t level = 0; level < shared_chain_length && level < hashes.size(); ++level) {
        if (hashes[level] == shared_prefix_chain[level]) {
            new_len++;
        } else {
            break;
        }
    }

    shared_chain_length = new_len;
    shared_prefix_chain.resize(shared_chain_length);
}

void ChunkedHashTree_RL::update_chain_on_finish(uint32_t request_id)
{
    if (active_requests.empty()) {
        shared_prefix_chain.clear();
        shared_chain_length = 0;
        return;
    }

    if (active_requests.size() == 1) {
        recompute_shared_chain();
        return;
    }

    uint32_t max_possible = UINT32_MAX;
    for (auto req_id : active_requests) {
        auto req_it = request_hashes.find(req_id);
        if (req_it != request_hashes.end()) {
            max_possible = std::min(max_possible, (uint32_t)req_it->second.size());
        }
    }

    for (uint32_t level = shared_chain_length; level < max_possible; ++level) {
        uint64_t common_hash = 0;
        bool first = true;
        bool all_match = true;

        for (auto req_id : active_requests) {
            auto req_it = request_hashes.find(req_id);
            if (req_it != request_hashes.end() && level < req_it->second.size()) {
                uint64_t h = req_it->second[level];
                if (first) { common_hash = h; first = false; }
                else if (h != common_hash) { all_match = false; break; }
            } else {
                all_match = false;
                break;
            }
        }

        if (all_match) {
            shared_prefix_chain.push_back(common_hash);
            shared_chain_length++;
        } 
        else {
            break;
        }
    }
}

uint32_t ChunkedHashTree_RL::compute_chain_length_with_request(uint32_t request_id) const
{
    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) {
        return shared_chain_length;
    }

    const auto& hashes = it->second;

    if (active_requests.empty()) return (uint32_t)hashes.size();

    uint32_t match = 0;
    for (uint32_t level = 0; level < shared_chain_length && level < hashes.size(); ++level) {
        if (hashes[level] == shared_prefix_chain[level]) {
            match++;
        } 
        else {
            break;
        }
    }

    return match;
}

uint32_t ChunkedHashTree_RL::count_waiters_at_level(uint32_t level) const
{
    if (level >= shared_chain_length) return 0;

    uint64_t key = pack_key(level, shared_prefix_chain[level]);
    auto it = hash_to_waiting.find(key);
    if (it == hash_to_waiting.end()) return 0;

    uint32_t count = 0;
    for (auto req_id : it->second) {
        if (!active_requests.count(req_id)) {
            count++;
        }
    }

    return count;
}

uint32_t ChunkedHashTree_RL::insert(uint32_t request_id, const std::vector<uint32_t>& tokens)
{
    if (request_hashes.count(request_id)) return 0;

    auto hashes = compute_hashes(tokens);

    request_hashes[request_id] = hashes;
    waiting_requests.insert(request_id);

    for (uint32_t level = 0; level < (uint32_t)hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        auto& ws = hash_to_waiting[key];
        if (ws.empty()) ws.reserve(16);
        ws.insert(request_id);
    }

    invalidate_cache();
    return (uint32_t)tokens.size();
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> ChunkedHashTree_RL::find_best_request()
{
    if (cache_valid && cached_best_request.has_value()) return cached_best_request.value();

    if (waiting_requests.empty()) {
        cached_best_request = {0u, 0u, 0u, 0u};
        cache_valid = true;
        return cached_best_request.value();
    }

    uint32_t chunks_before = shared_chain_length;

    for (int32_t level = (int32_t)shared_chain_length - 1; level >= 0; --level) {
        uint64_t key = pack_key((uint32_t)level, shared_prefix_chain[(uint32_t)level]);

        auto it = hash_to_waiting.find(key);
        if (it == hash_to_waiting.end()) continue;

        uint32_t chosen = 0;
        for (auto req_id : it->second) {
            if (!active_requests.count(req_id) && waiting_requests.count(req_id)) {
                chosen = req_id;
                break;
            }
        }

        if (chosen == 0) continue;

        uint32_t chunks_after = compute_chain_length_with_request(chosen);

        uint32_t waiters = count_waiters_at_level((uint32_t)level);

        cached_best_request = {chosen, chunks_before, chunks_after, waiters};
        cache_valid = true;
        return cached_best_request.value();
    }

    uint32_t fallback = *waiting_requests.begin();
    uint32_t chunks_after = compute_chain_length_with_request(fallback);
    uint32_t waiters = 0;

    cached_best_request = {fallback, chunks_before, chunks_after, waiters};
    cache_valid = true;
    return cached_best_request.value();
}

std::pair<uint32_t, uint32_t> ChunkedHashTree_RL::activate_request(uint32_t request_id)
{
    if (active_requests.count(request_id)) return {0, 0};

    if (!request_hashes.count(request_id)) return {0, 0};

    waiting_requests.erase(request_id);
    active_requests.insert(request_id);

    auto& hashes = request_hashes[request_id];
    uint32_t added = 0;

    for (uint32_t level = 0; level < (uint32_t)hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        if (hash_ref_counts[key]++ == 0) {
            added++;
        }
    }

    update_chain_on_activate(request_id);
    invalidate_cache();

    return {added, (uint32_t)hashes.size()};
}

std::pair<uint32_t, uint32_t> ChunkedHashTree_RL::finish_request(uint32_t request_id)
{
    if (!active_requests.erase(request_id)) return {0, 0};

    auto it = request_hashes.find(request_id);
    if (it == request_hashes.end()) return {0, 0};

    auto& hashes = it->second;
    uint32_t evicted = 0;

    for (uint32_t level = 0; level < (uint32_t)hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        auto rc_it = hash_ref_counts.find(key);
        if (rc_it != hash_ref_counts.end() && --rc_it->second == 0) {
            hash_ref_counts.erase(rc_it);
            evicted++;
        }
    }

    update_chain_on_finish(request_id);

    for (uint32_t level = 0; level < (uint32_t)hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        auto map_it = hash_to_waiting.find(key);
        if (map_it != hash_to_waiting.end()) {
            map_it->second.erase(request_id);
            if (map_it->second.empty()) {
                hash_to_waiting.erase(map_it);
            }
        }
    }

    request_hashes.erase(request_id);
    invalidate_cache();

    return {evicted, (uint32_t)hashes.size()};
}

void ChunkedHashTree_RL::remove(uint32_t request_id)
{
    waiting_requests.erase(request_id);

    auto it = request_hashes.find(request_id);
    if (it != request_hashes.end()) {
        auto& hashes = it->second;
        for (uint32_t level = 0; level < (uint32_t)hashes.size(); ++level) {
            uint64_t key = pack_key(level, hashes[level]);
            auto map_it = hash_to_waiting.find(key);
            if (map_it != hash_to_waiting.end()) {
                map_it->second.erase(request_id);
                if (map_it->second.empty()) {
                    hash_to_waiting.erase(map_it);
                }
            }
        }
        request_hashes.erase(it);
    }

    invalidate_cache();
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
            uint64_t h = XXH64_digest(state);
            hashes.push_back(h);
        }
    }

    XXH64_freeState(state);
    return hashes;
}