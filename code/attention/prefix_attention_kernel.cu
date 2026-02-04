#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)


constexpr int HEAD_DIM = 256;
constexpr int NUM_HEADS = 256;
constexpr int NUM_KV_HEADS = 64; 

constexpr int BLOCK_SIZE = 256;

constexpr int Br = 16;
constexpr int Bc = 64;

struct RequestDescriptor {
    int request_id;               
    const half* Q;              
    int query_len;
    
    const half* K_prefix;         
    const half* V_prefix;         
    int prefix_len;           
    
    const half* K_unique;       
    const half* V_unique;      
    int unique_len;           
    
    half* O;                   
};

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
    
    __device__ void rescale(float old_max) {
        sum_exp = sum_exp * expf(old_max - max_val);
    }
};

__device__ void block_reduce_softmax(
    SoftmaxState& state,
    SoftmaxState* __restrict__ shared_states,
    int tid
) {
    shared_states[tid] = state;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_states[tid].merge(shared_states[tid + stride]);
        }
        __syncthreads();
    }
    
    state = shared_states[0];
    __syncthreads();
}

__device__ void load_query_tile(
    const half* __restrict__ Q,
    half* __restrict__ Q_shared,
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
            int q_idx = head_idx * query_len * head_dim +
                       actual_q_row * head_dim +
                       q_col;
            Q_shared[i] = Q[q_idx];
        } else {
            Q_shared[i] = __float2half(0.0f);
        }
    }
}

__device__ void load_kv_tile(
    const half* __restrict__ KV,
    half* __restrict__ KV_shared,
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
            int kv_idx = kv_head_idx * kv_len * head_dim +
                        actual_kv_row * head_dim +
                        kv_col;
            KV_shared[i] = KV[kv_idx];
        } else {
            KV_shared[i] = __float2half(0.0f);
        }
    }
}

__device__ void compute_qk_scores_masked(
    const half* __restrict__ Q_shared,
    const half* __restrict__ K_shared,
    float* __restrict__ scores,
    int q_tile_start,
    int kv_tile_start,
    int q_tile_size,
    int kv_tile_size,
    int head_dim,
    float scale,
    int tid
) {
    const int total_elements = Br * Bc;
    
    for (int idx = tid; idx < total_elements; idx += BLOCK_SIZE) {
        int q_row = idx / Bc;
        int kv_col = idx % Bc;
        
        if (q_row < q_tile_size && kv_col < kv_tile_size) {
            int q_abs_pos = q_tile_start + q_row;
            int kv_abs_pos = kv_tile_start + kv_col;
        
            float score = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                float q_val = __half2float(Q_shared[q_row * head_dim + d]);
                float k_val = __half2float(K_shared[kv_col * head_dim + d]);
                score += q_val * k_val;
            }
            
            if (kv_abs_pos <= q_abs_pos) {
                scores[idx] = score * scale;
            } else {
                scores[idx] = -INFINITY;
            }
        } else {
            scores[idx] = -INFINITY;
        }
    }
}

__device__ void compute_tile_softmax_stats(
    const float* __restrict__ scores,
    SoftmaxState* __restrict__ row_states,
    int q_tile_size,
    int tid
) {
    for (int q_row = 0; q_row < q_tile_size; q_row++) {
        SoftmaxState local_state;
        
        for (int kv_col = tid; kv_col < Bc; kv_col += BLOCK_SIZE) {
            float score = scores[q_row * Bc + kv_col];
            if (score > -INFINITY) {
                local_state.update(score);
            }
        }
        
        __shared__ SoftmaxState reduction_buffer[BLOCK_SIZE];
        reduction_buffer[tid] = local_state;
        __syncthreads();
        
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction_buffer[tid].merge(reduction_buffer[tid + stride]);
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            row_states[q_row].merge(reduction_buffer[0]);
        }
        __syncthreads();
    }
}


__device__ void compute_tile_output(
    const float* __restrict__ scores,
    const half* __restrict__ V_shared,
    float* __restrict__ O_accum,
    const SoftmaxState* __restrict__ old_states,
    const SoftmaxState* __restrict__ new_states,
    int q_tile_size,
    int head_dim,
    int tid
) {
    for (int q_row = 0; q_row < q_tile_size; q_row++) {
        float old_max = old_states[q_row].max_val;
        float new_max = new_states[q_row].max_val;
        
        float rescale_factor = expf(old_max - new_max);
        
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            int o_idx = q_row * head_dim + d;
            
            float old_output = O_accum[o_idx] * rescale_factor;
            
            float new_contrib = 0.0f;
            #pragma unroll 4
            for (int kv_col = 0; kv_col < Bc; kv_col++) {
                float score = scores[q_row * Bc + kv_col];
                if (score > -INFINITY) {
                    float prob = expf(score - new_max);
                    float v_val = __half2float(V_shared[kv_col * head_dim + d]);
                    new_contrib += prob * v_val;
                }
            }
            O_accum[o_idx] = old_output + new_contrib;
        }
    }
    __syncthreads();
}

__global__ void per_request_attention_kernel(
    const RequestDescriptor* __restrict__ request_desc
) {
    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    
    const RequestDescriptor& req = *request_desc;
    const int kv_head_idx = head_idx / (NUM_HEADS / NUM_KV_HEADS); // GQA mapping
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    extern __shared__ char shared_mem[];
    half* Q_shared = (half*)shared_mem;
    half* K_shared = Q_shared + Br * HEAD_DIM;
    half* V_shared = K_shared + Bc * HEAD_DIM;
    float* scores = (float*)(V_shared + Bc * HEAD_DIM);
    SoftmaxState* softmax_reduction = (SoftmaxState*)(scores + Br * Bc);
    float* O_accum = (float*)(softmax_reduction + BLOCK_SIZE);
    SoftmaxState* row_states_old = (SoftmaxState*)(O_accum + Br * HEAD_DIM);
    SoftmaxState* row_states_new = row_states_old + Br;
    
    for (int q_tile_start = 0; q_tile_start < req.query_len; q_tile_start += Br) {
        int q_tile_size = min(Br, req.query_len - q_tile_start);
        
        for (int i = tid; i < Br * HEAD_DIM; i += BLOCK_SIZE) {
            O_accum[i] = 0.0f;
        }
        if (tid < Br) {
            row_states_old[tid] = SoftmaxState();
            row_states_new[tid] = SoftmaxState();
        }
        __syncthreads();
        
        load_query_tile(req.Q, Q_shared, head_idx, 
                       q_tile_start, req.query_len, HEAD_DIM, tid);
        __syncthreads();
        
        if (req.prefix_len > 0) {
            for (int kv_tile_start = 0; kv_tile_start < req.prefix_len; kv_tile_start += Bc) {
                int kv_tile_size = min(Bc, req.prefix_len - kv_tile_start);
                
                load_kv_tile(req.K_prefix, K_shared, kv_head_idx,
                            kv_tile_start, req.prefix_len, HEAD_DIM, tid);
                __syncthreads();
                
                compute_qk_scores_masked(Q_shared, K_shared, scores,
                                        q_tile_start, kv_tile_start,
                                        q_tile_size, kv_tile_size,
                                        HEAD_DIM, scale, tid);
                __syncthreads();
                
                if (tid < Br) {
                    row_states_old[tid] = row_states_new[tid];
                }
                __syncthreads();
                
                compute_tile_softmax_stats(scores, row_states_new, q_tile_size, tid);
                
                load_kv_tile(req.V_prefix, V_shared, kv_head_idx,
                            kv_tile_start, req.prefix_len, HEAD_DIM, tid);
                __syncthreads();
                
                compute_tile_output(scores, V_shared, O_accum,
                                  row_states_old, row_states_new,
                                  q_tile_size, HEAD_DIM, tid);
            }
        }
        
        if (req.unique_len > 0) {
            for (int kv_tile_start = 0; kv_tile_start < req.unique_len; kv_tile_start += Bc) {
                int kv_tile_size = min(Bc, req.unique_len - kv_tile_start);
                int global_kv_pos = req.prefix_len + kv_tile_start;
                
                load_kv_tile(req.K_unique, K_shared, kv_head_idx,
                            kv_tile_start, req.unique_len, HEAD_DIM, tid);
                __syncthreads();
                
                compute_qk_scores_masked(Q_shared, K_shared, scores,
                                        q_tile_start, global_kv_pos,
                                        q_tile_size, kv_tile_size,
                                        HEAD_DIM, scale, tid);
                __syncthreads();

                if (tid < Br) {
                    row_states_old[tid] = row_states_new[tid];
                }
                __syncthreads();
                
                compute_tile_softmax_stats(scores, row_states_new, q_tile_size, tid);
                
                load_kv_tile(req.V_unique, V_shared, kv_head_idx,
                            kv_tile_start, req.unique_len, HEAD_DIM, tid);
                __syncthreads();
                
                compute_tile_output(scores, V_shared, O_accum,
                                  row_states_old, row_states_new,
                                  q_tile_size, HEAD_DIM, tid);
            }
        }
        
        for (int q_row = 0; q_row < q_tile_size; q_row++) {
            float final_sum = row_states_new[q_row].sum_exp;
            
            if (final_sum > 0.0f) {
                for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
                    int o_idx = q_row * HEAD_DIM + d;
                    int actual_q_row = q_tile_start + q_row;
                    
                    int global_o_idx = head_idx * req.query_len * HEAD_DIM +
                                      actual_q_row * HEAD_DIM + d;
                    
                    req.O[global_o_idx] = __float2half(O_accum[o_idx] / final_sum);
                }
            }
        }
        __syncthreads();
    }
}

void launch_per_request_attention(
    const RequestDescriptor* h_requests,
    int num_requests,
    int num_decode_iterations,  
    cudaStream_t* streams = nullptr
) {
    const size_t shared_mem_size = 
        sizeof(half) * Br * HEAD_DIM +       
        sizeof(half) * Bc * HEAD_DIM +      
        sizeof(half) * Bc * HEAD_DIM +       
        sizeof(float) * Br * Bc +            
        sizeof(SoftmaxState) * BLOCK_SIZE +   
        sizeof(float) * Br * HEAD_DIM +      
        sizeof(SoftmaxState) * Br * 2;       
    
    CUDA_CHECK(cudaFuncSetAttribute(
        per_request_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    ));
    
    RequestDescriptor* d_requests;
    CUDA_CHECK(cudaMalloc(&d_requests, num_requests * sizeof(RequestDescriptor)));
    CUDA_CHECK(cudaMemcpy(d_requests, h_requests, 
                         num_requests * sizeof(RequestDescriptor),
                         cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(NUM_HEADS);
    
    for (int iter = 0; iter < num_decode_iterations; iter++) {
        for (int req_idx = 0; req_idx < num_requests; req_idx++) {
            cudaStream_t stream = (streams != nullptr) ? streams[req_idx] : 0;
            
            per_request_attention_kernel<<<grid, block, shared_mem_size, stream>>>(
                &d_requests[req_idx]
            );
            
            CUDA_CHECK(cudaGetLastError());
        }
        
        if (streams != nullptr) {
            for (int req_idx = 0; req_idx < num_requests; req_idx++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[req_idx]));
            }
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    CUDA_CHECK(cudaFree(d_requests));
}


RequestDescriptor create_request_with_prefix(
    int request_id,
    const half* Q, int query_len,
    const half* K_prefix, const half* V_prefix, int prefix_len,
    const half* K_unique, const half* V_unique, int unique_len,
    half* O
) {
    RequestDescriptor desc;
    desc.request_id = request_id;
    desc.Q = Q;
    desc.query_len = query_len;
    desc.K_prefix = K_prefix;
    desc.V_prefix = V_prefix;
    desc.prefix_len = prefix_len;
    desc.K_unique = K_unique;
    desc.V_unique = V_unique;
    desc.unique_len = unique_len;
    desc.O = O;
    return desc;
}

RequestDescriptor create_request_no_prefix(
    int request_id,
    const half* Q, int query_len,
    const half* K, const half* V, int kv_len,
    half* O
) {
    RequestDescriptor desc;
    desc.request_id = request_id;
    desc.Q = Q;
    desc.query_len = query_len;
    desc.K_prefix = nullptr;
    desc.V_prefix = nullptr;
    desc.prefix_len = 0;
    desc.K_unique = K;
    desc.V_unique = V;
    desc.unique_len = kv_len;
    desc.O = O;
    return desc;
}

void example_usage() {
    CUDA_CHECK(cudaSetDevice(0));
    constexpr int NUM_FAMILIES = 1;
    constexpr int REQUESTS_PER_FAMILY = 600;
    constexpr int NUM_DECODE_ITERATIONS = 100;

    constexpr int PREFIX_LEN  = 12000;
    constexpr int SUFFIX_LEN  = 10;
    constexpr int QUERY_LEN   = 1; 

    const int total_requests = NUM_FAMILIES * REQUESTS_PER_FAMILY;

    size_t q_elems = (size_t)total_requests * NUM_HEADS * QUERY_LEN * HEAD_DIM;
    half* d_Q = nullptr;
    half* d_O = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, q_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, q_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_Q, 0, q_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_O, 0, q_elems * sizeof(half)));


    int total_kv_tokens = NUM_FAMILIES * PREFIX_LEN + total_requests * SUFFIX_LEN;
    size_t kv_elems = (size_t)total_kv_tokens * NUM_KV_HEADS * HEAD_DIM;
    
    half* d_K_cache = nullptr;
    half* d_V_cache = nullptr;
    CUDA_CHECK(cudaMalloc(&d_K_cache, kv_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_cache, kv_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_K_cache, 0, kv_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_V_cache, 0, kv_elems * sizeof(half)));

    std::vector<RequestDescriptor> requests;
    requests.reserve(total_requests);

    int kv_cursor = 0;

    for (int fam = 0; fam < NUM_FAMILIES; ++fam) {
        int prefix_start = kv_cursor;
        int prefix_len = PREFIX_LEN;
        
        half* K_prefix = d_K_cache + prefix_start * NUM_KV_HEADS * HEAD_DIM;
        half* V_prefix = d_V_cache + prefix_start * NUM_KV_HEADS * HEAD_DIM;
        
        kv_cursor += PREFIX_LEN;

        for (int r = 0; r < REQUESTS_PER_FAMILY; ++r) {
            int request_id = fam * REQUESTS_PER_FAMILY + r;
            
            int unique_start = kv_cursor;
            int unique_len = SUFFIX_LEN;
            
            half* K_unique = d_K_cache + unique_start * NUM_KV_HEADS * HEAD_DIM;
            half* V_unique = d_V_cache + unique_start * NUM_KV_HEADS * HEAD_DIM;
            
            kv_cursor += SUFFIX_LEN;

            half* Q_req = d_Q + request_id * NUM_HEADS * QUERY_LEN * HEAD_DIM;
            half* O_req = d_O + request_id * NUM_HEADS * QUERY_LEN * HEAD_DIM;

            RequestDescriptor desc = create_request_with_prefix(
                request_id,
                Q_req, QUERY_LEN,
                K_prefix, V_prefix, prefix_len,
                K_unique, V_unique, unique_len,
                O_req
            );
            
            requests.push_back(desc);
        }
    }

    std::vector<cudaStream_t> streams(total_requests);
    for (int i = 0; i < total_requests; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    launch_per_request_attention(
        requests.data(), 
        total_requests, 
        NUM_DECODE_ITERATIONS, 
        streams.data()
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    for (int i = 0; i < total_requests; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_K_cache));
    CUDA_CHECK(cudaFree(d_V_cache));

    std::cout << " Prefix attention completed\n";
    std::cout << "  Families            : " << NUM_FAMILIES << "\n";
    std::cout << "  Requests / family   : " << REQUESTS_PER_FAMILY << "\n";
    std::cout << "  Total requests      : " << total_requests << "\n";
    std::cout << "  Decode iterations   : " << NUM_DECODE_ITERATIONS << "\n";
    std::cout << "  Total KV tokens     : " << kv_cursor << "\n";
    std::cout << "  Time (ms)           : " << duration.count() << "\n";
}

int main() {
    example_usage();
    return 0;
}
