#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>
#include <iostream>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Configuration for Llama 3.1 8B
constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 32;
constexpr int NUM_KV_HEADS = 8; // GQA: 4 query heads per KV head

// Kernel parameters
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Tile sizes for blocking
constexpr int Br = 16; // Query tile size
constexpr int Bc = 64; // KV tile size

/**
 * Batch descriptor for requests sharing the same prefix
 */
struct PrefixBatch {
    int prefix_start_token;    // Start position of shared prefix in global KV cache
    int prefix_length;         // Length of shared prefix
    int num_requests;          // Number of requests in this family
    int* request_ids;          // Array of request IDs
    int* unique_start_tokens;  // Per-request unique token start positions
    int* unique_lengths;       // Per-request unique token lengths
};

/**
 * Online softmax computation (numerically stable)
 */
struct SoftmaxState {
    float max_val;
    float sum_exp;
    
    __device__ SoftmaxState() : max_val(-INFINITY), sum_exp(0.0f) {}
    
    __device__ void update(float new_val) {
        float old_max = max_val;
        max_val = fmaxf(max_val, new_val);
        sum_exp = sum_exp * expf(old_max - max_val) + expf(new_val - max_val);
    }
    
    __device__ void merge(const SoftmaxState& other) {
        float old_max = max_val;
        max_val = fmaxf(max_val, other.max_val);
        sum_exp = sum_exp * expf(old_max - max_val) + 
                  other.sum_exp * expf(other.max_val - max_val);
    }
};

/**
 * Warp-level reduction for softmax statistics
 */
__device__ void warp_reduce_softmax(SoftmaxState& state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, state.max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, state.sum_exp, offset);
        
        SoftmaxState other;
        other.max_val = other_max;
        other.sum_exp = other_sum;
        state.merge(other);
    }
}

/**
 * Load query tile into shared memory
 * Q: [batch, num_heads, query_len, head_dim]
 */
__device__ void load_query_tile(
    const half* __restrict__ Q,
    half* __restrict__ Q_shared,
    int batch_idx,
    int head_idx,
    int query_start,
    int query_len,
    int head_dim,
    int tid
) {
    const int num_elements = Br * head_dim;
    
    for (int i = tid; i < num_elements; i += BLOCK_SIZE) {
        int q_row = i / head_dim;
        int q_col = i % head_dim;
        int actual_q_row = query_start + q_row;
        
        if (actual_q_row < query_len && q_col < head_dim) {
            int q_idx = batch_idx * NUM_HEADS * query_len * head_dim +
                       head_idx * query_len * head_dim +
                       actual_q_row * head_dim +
                       q_col;
            Q_shared[i] = Q[q_idx];
        } else {
            Q_shared[i] = __float2half(0.0f);
        }
    }
}

/**
 * Load KV tile from global memory
 * K/V: [batch, num_kv_heads, kv_len, head_dim]
 */
__device__ void load_kv_tile(
    const half* __restrict__ KV,
    half* __restrict__ KV_shared,
    int batch_idx,
    int kv_head_idx,
    int kv_start,
    int kv_len,
    int head_dim,
    int tid
) {
    const int num_elements = Bc * head_dim;
    
    for (int i = tid; i < num_elements; i += BLOCK_SIZE) {
        int kv_row = i / head_dim;
        int kv_col = i % head_dim;
        int actual_kv_row = kv_start + kv_row;
        
        if (actual_kv_row < kv_len && kv_col < head_dim) {
            int kv_idx = batch_idx * NUM_KV_HEADS * kv_len * head_dim +
                        kv_head_idx * kv_len * head_dim +
                        actual_kv_row * head_dim +
                        kv_col;
            KV_shared[i] = KV[kv_idx];
        } else {
            KV_shared[i] = __float2half(0.0f);
        }
    }
}

/**
 * Compute attention scores: QK^T
 * Uses warp-level computation
 */
__device__ void compute_qk_scores(
    const half* __restrict__ Q_shared,
    const half* __restrict__ K_shared,
    float* __restrict__ scores,
    int q_tile_size,
    int kv_tile_size,
    int head_dim,
    float scale,
    int lane_id,
    int warp_id
) {
    // Each warp computes a portion of the Q x K^T matrix
    const int q_rows_per_warp = (Br + NUM_WARPS - 1) / NUM_WARPS;
    
    int warp_q_start = warp_id * q_rows_per_warp;
    
    for (int q_local = 0; q_local < q_rows_per_warp; q_local++) {
        int q_row = warp_q_start + q_local;
        if (q_row >= q_tile_size) break;
        
        for (int kv_col = lane_id; kv_col < kv_tile_size; kv_col += WARP_SIZE) {
            float score = 0.0f;
            
            // Dot product across head dimension
            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                float q_val = __half2float(Q_shared[q_row * head_dim + d]);
                float k_val = __half2float(K_shared[kv_col * head_dim + d]);
                score += q_val * k_val;
            }
            
            scores[q_row * Bc + kv_col] = score * scale;
        }
    }
}

/**
 * Apply causal mask and compute softmax
 */
__device__ void masked_softmax(
    float* __restrict__ scores,
    SoftmaxState* __restrict__ softmax_state,
    int q_tile_start,
    int kv_tile_start,
    int q_tile_size,
    int kv_tile_size,
    int query_pos,
    int lane_id,
    int warp_id
) {
    const int q_rows_per_warp = (Br + NUM_WARPS - 1) / NUM_WARPS;
    int warp_q_start = warp_id * q_rows_per_warp;
    
    for (int q_local = 0; q_local < q_rows_per_warp; q_local++) {
        int q_row = warp_q_start + q_local;
        if (q_row >= q_tile_size) break;
        
        int q_abs_pos = query_pos + q_tile_start + q_row;
        SoftmaxState local_state;
        
        for (int kv_col = lane_id; kv_col < kv_tile_size; kv_col += WARP_SIZE) {
            int kv_abs_pos = kv_tile_start + kv_col;
            
            float score = scores[q_row * Bc + kv_col];
            
            // Apply causal mask (query can only attend to past KV)
            if (kv_abs_pos <= q_abs_pos) {
                local_state.update(score);
                scores[q_row * Bc + kv_col] = score;
            } else {
                scores[q_row * Bc + kv_col] = -INFINITY;
            }
        }
        
        // Reduce within warp
        warp_reduce_softmax(local_state);
        
        // Lane 0 writes back
        if (lane_id == 0) {
            softmax_state[q_row].merge(local_state);
        }
    }
}

/**
 * Compute attention output: softmax(QK^T)V
 * Accumulates into shared memory buffer
 */
__device__ void compute_attention_output(
    const float* __restrict__ scores,
    const half* __restrict__ V_shared,
    float* __restrict__ O_shared,
    const SoftmaxState* __restrict__ softmax_state,
    int q_tile_size,
    int kv_tile_size,
    int head_dim,
    int lane_id,
    int warp_id
) {
    const int q_rows_per_warp = (Br + NUM_WARPS - 1) / NUM_WARPS;
    int warp_q_start = warp_id * q_rows_per_warp;
    
    for (int q_local = 0; q_local < q_rows_per_warp; q_local++) {
        int q_row = warp_q_start + q_local;
        if (q_row >= q_tile_size) break;
        
        // Normalize scores
        float norm = softmax_state[q_row].sum_exp;
        float max_val = softmax_state[q_row].max_val;
        
        if (norm == 0.0f) continue; // Skip if no valid attention
        
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float output = 0.0f;
            
            #pragma unroll 4
            for (int kv_col = 0; kv_col < kv_tile_size; kv_col++) {
                float score = scores[q_row * Bc + kv_col];
                if (score > -INFINITY) {
                    float attn_weight = expf(score - max_val) / norm;
                    float v_val = __half2float(V_shared[kv_col * head_dim + d]);
                    output += attn_weight * v_val;
                }
            }
            
            // Use atomic add for accumulation across KV tiles
            atomicAdd(&O_shared[q_row * head_dim + d], output);
        }
    }
}

/**
 * Main kernel for prefix-shared attention
 * Processes one family of requests sharing a common prefix
 */
__global__ void prefix_attention_kernel(
    const half* __restrict__ Q,          // [total_requests, num_heads, query_len, head_dim]
    const half* __restrict__ K,          // [total_requests, num_kv_heads, kv_len, head_dim]
    const half* __restrict__ V,          // [total_requests, num_kv_heads, kv_len, head_dim]
    half* __restrict__ O,                // [total_requests, num_heads, query_len, head_dim]
    const PrefixBatch* __restrict__ batch_info,
    int query_len,
    int batch_family_idx
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Determine which request this block handles
    const int request_idx_in_family = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    const PrefixBatch& batch = batch_info[batch_family_idx];
    
    if (request_idx_in_family >= batch.num_requests) return;
    
    const int global_request_id = batch.request_ids[request_idx_in_family];
    const int kv_head_idx = head_idx / (NUM_HEADS / NUM_KV_HEADS); // GQA mapping
    
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    // Shared memory for tiles
    extern __shared__ char shared_mem[];
    half* Q_shared = (half*)shared_mem;
    half* K_shared = Q_shared + Br * HEAD_DIM;
    half* V_shared = K_shared + Bc * HEAD_DIM;
    float* scores = (float*)(V_shared + Bc * HEAD_DIM);
    SoftmaxState* softmax_states = (SoftmaxState*)(scores + Br * Bc);
    float* O_shared = (float*)(softmax_states + Br);
    
    // Process query in tiles
    for (int q_tile_start = 0; q_tile_start < query_len; q_tile_start += Br) {
        int q_tile_size = min(Br, query_len - q_tile_start);
        
        // Initialize output accumulator for this query tile
        for (int i = tid; i < Br * HEAD_DIM; i += BLOCK_SIZE) {
            O_shared[i] = 0.0f;
        }
        __syncthreads();
        
        // Load query tile
        load_query_tile(Q, Q_shared, global_request_id, head_idx, 
                       q_tile_start, query_len, HEAD_DIM, tid);
        __syncthreads();
        
        // Initialize softmax states for this query tile
        if (tid < Br) {
            softmax_states[tid] = SoftmaxState();
        }
        __syncthreads();
        
        // Total KV length = prefix + unique
        int total_kv_len = batch.prefix_length + batch.unique_lengths[request_idx_in_family];
        
        // Process KV in tiles (prefix + unique portions)
        for (int kv_tile_start = 0; kv_tile_start < total_kv_len; kv_tile_start += Bc) {
            int kv_tile_size = min(Bc, total_kv_len - kv_tile_start);
            
            // Load K tile
            load_kv_tile(K, K_shared, global_request_id, kv_head_idx,
                        kv_tile_start, total_kv_len, HEAD_DIM, tid);
            __syncthreads();
            
            // Compute QK^T scores
            compute_qk_scores(Q_shared, K_shared, scores, q_tile_size, 
                            kv_tile_size, HEAD_DIM, scale, lane_id, warp_id);
            __syncthreads();
            
            // Apply mask and update softmax statistics
            masked_softmax(scores, softmax_states, q_tile_start, kv_tile_start,
                          q_tile_size, kv_tile_size, 0, lane_id, warp_id);
            __syncthreads();
            
            // Load V tile
            load_kv_tile(V, V_shared, global_request_id, kv_head_idx,
                        kv_tile_start, total_kv_len, HEAD_DIM, tid);
            __syncthreads();
            
            // Compute partial attention output
            compute_attention_output(scores, V_shared, O_shared, softmax_states,
                                    q_tile_size, kv_tile_size, HEAD_DIM, 
                                    lane_id, warp_id);
            __syncthreads();
        }
        
        // Write output for this query tile
        for (int i = tid; i < q_tile_size * HEAD_DIM; i += BLOCK_SIZE) {
            int q_row = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int actual_q_row = q_tile_start + q_row;
            
            int o_idx = global_request_id * NUM_HEADS * query_len * HEAD_DIM +
                        head_idx * query_len * HEAD_DIM +
                        actual_q_row * HEAD_DIM + d;
            O[o_idx] = __float2half(O_shared[i]);
        }
        __syncthreads();
    }
}

/**
 * Host-side launcher - takes array of host-side batch info and max requests per batch
 */
void launch_prefix_attention(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    const PrefixBatch* d_batch_info,  // Device pointer to batch array
    int num_families,
    int query_len,
    cudaStream_t stream,
    const int* h_num_requests_per_family  // Host array with num_requests for each family
) {
    const size_t shared_mem_size = 
        sizeof(half) * Br * HEAD_DIM +        // Q_shared
        sizeof(half) * Bc * HEAD_DIM +        // K_shared
        sizeof(half) * Bc * HEAD_DIM +        // V_shared
        sizeof(float) * Br * Bc +             // scores
        sizeof(SoftmaxState) * Br +           // softmax_states
        sizeof(float) * Br * HEAD_DIM;        // O_shared
        CUDA_CHECK(cudaFuncSetAttribute(
        prefix_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    ));
    for (int family_idx = 0; family_idx < num_families; family_idx++) {
        int num_requests = h_num_requests_per_family[family_idx];
        
        dim3 grid(num_requests, NUM_HEADS);
        dim3 block(BLOCK_SIZE);
        
        prefix_attention_kernel<<<grid, block, shared_mem_size, stream>>>(
            Q, K, V, O, d_batch_info, query_len, family_idx
        );
    }
}
