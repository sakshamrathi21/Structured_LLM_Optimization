#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <map>
#include <iostream>
#include <stdexcept>
#include <algorithm>

/* ============================================================
 *  CUDA ERROR CHECKING
 * ============================================================ */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(err) << std::endl;   \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

/* ============================================================
 *  KERNEL INTERFACE
 * ============================================================ */
#include "prefix_attention_kernel.cu"

/* ============================================================
 *  REQUEST + PREFIX FAMILY DESCRIPTORS
 * ============================================================ */

struct Request {
    int request_id;
    int prefix_id;

    int prefix_start;   // absolute KV token index
    int prefix_len;

    int unique_start;   // absolute KV token index
    int unique_len;
};

struct PrefixFamily {
    int prefix_start;
    int prefix_len;

    std::vector<int> request_ids;
    std::vector<int> unique_starts;
    std::vector<int> unique_lens;
};

/* ============================================================
 *  PREFIX ATTENTION MANAGER
 * ============================================================ */

class PrefixAttentionManager {
public:
    PrefixAttentionManager(
        int max_batch_size,
        int max_kv_length
    )
        : max_batch_size_(max_batch_size),
          max_kv_length_(max_kv_length)
    {
        size_t kv_elems =
            (size_t)max_batch_size_
            * NUM_KV_HEADS
            * max_kv_length_
            * HEAD_DIM;

        CUDA_CHECK(cudaMalloc(&d_K_cache_, kv_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_V_cache_, kv_elems * sizeof(half)));

        CUDA_CHECK(cudaMemset(d_K_cache_, 0, kv_elems * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_V_cache_, 0, kv_elems * sizeof(half)));
    }

    ~PrefixAttentionManager() {
        cudaFree(d_K_cache_);
        cudaFree(d_V_cache_);
    }

    void run_attention(
        const std::vector<Request>& requests,
        half* d_Q,
        half* d_O,
        int query_len,
        cudaStream_t stream
    ) {
        if (requests.empty()) {
            std::cerr << "No requests provided\n";
            return;
        }

        auto families = group_by_prefix(requests);

        std::vector<PrefixBatch> h_batches(families.size());
        PrefixBatch* d_batches = nullptr;

        CUDA_CHECK(cudaMalloc(
            &d_batches,
            families.size() * sizeof(PrefixBatch)
        ));

        std::vector<int> h_num_requests(families.size());

        for (size_t i = 0; i < families.size(); ++i) {
            const PrefixFamily& fam = families[i];

            PrefixBatch& batch = h_batches[i];
            batch.prefix_start_token = fam.prefix_start;
            batch.prefix_length      = fam.prefix_len;
            batch.num_requests       = (int)fam.request_ids.size();

            h_num_requests[i] = batch.num_requests;

            CUDA_CHECK(cudaMalloc(
                &batch.request_ids,
                batch.num_requests * sizeof(int)
            ));
            CUDA_CHECK(cudaMalloc(
                &batch.unique_start_tokens,
                batch.num_requests * sizeof(int)
            ));
            CUDA_CHECK(cudaMalloc(
                &batch.unique_lengths,
                batch.num_requests * sizeof(int)
            ));

            CUDA_CHECK(cudaMemcpyAsync(
                batch.request_ids,
                fam.request_ids.data(),
                batch.num_requests * sizeof(int),
                cudaMemcpyHostToDevice,
                stream
            ));
            CUDA_CHECK(cudaMemcpyAsync(
                batch.unique_start_tokens,
                fam.unique_starts.data(),
                batch.num_requests * sizeof(int),
                cudaMemcpyHostToDevice,
                stream
            ));
            CUDA_CHECK(cudaMemcpyAsync(
                batch.unique_lengths,
                fam.unique_lens.data(),
                batch.num_requests * sizeof(int),
                cudaMemcpyHostToDevice,
                stream
            ));
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_batches,
            h_batches.data(),
            h_batches.size() * sizeof(PrefixBatch),
            cudaMemcpyHostToDevice,
            stream
        ));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_prefix_attention(
            d_Q,
            d_K_cache_,
            d_V_cache_,
            d_O,
            d_batches,
            (int)families.size(),
            query_len,
            stream,
            h_num_requests.data()
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_batches));
        for (auto& b : h_batches) {
            CUDA_CHECK(cudaFree(b.request_ids));
            CUDA_CHECK(cudaFree(b.unique_start_tokens));
            CUDA_CHECK(cudaFree(b.unique_lengths));
        }
    }

private:
    std::vector<PrefixFamily>
    group_by_prefix(const std::vector<Request>& requests) {
        std::map<int, PrefixFamily> map;

        for (const auto& r : requests) {
            auto& fam = map[r.prefix_id];

            if (fam.request_ids.empty()) {
                fam.prefix_start = r.prefix_start;
                fam.prefix_len   = r.prefix_len;
            } else {
                if (fam.prefix_start != r.prefix_start ||
                    fam.prefix_len   != r.prefix_len) {
                    throw std::runtime_error(
                        "Inconsistent prefix metadata within family"
                    );
                }
            }

            fam.request_ids.push_back(r.request_id);
            fam.unique_starts.push_back(r.unique_start);
            fam.unique_lens.push_back(r.unique_len);
        }

        std::vector<PrefixFamily> families;
        for (auto& kv : map) {
            families.push_back(std::move(kv.second));
        }
        return families;
    }

private:
    int max_batch_size_;
    int max_kv_length_;
    half* d_K_cache_;
    half* d_V_cache_;
};

/* ============================================================
 *  MAIN
 * ============================================================ */

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* ===================== CONFIG ===================== */

    constexpr int NUM_FAMILIES = 2;
    constexpr int REQUESTS_PER_FAMILY = 100;

    constexpr int PREFIX_LEN  = 6000;
    constexpr int SUFFIX_LEN  = 10;
    constexpr int QUERY_LEN   = 20;

    /* ================================================== */

    std::vector<Request> requests;
    requests.reserve(NUM_FAMILIES * REQUESTS_PER_FAMILY);

    int request_id = 0;
    int kv_cursor  = 0;

    for (int fam = 0; fam < NUM_FAMILIES; ++fam) {
        int prefix_start = kv_cursor;
        kv_cursor += PREFIX_LEN;

        for (int r = 0; r < REQUESTS_PER_FAMILY; ++r) {
            requests.push_back(Request{
                request_id++,
                fam,
                prefix_start,
                PREFIX_LEN,
                kv_cursor,
                SUFFIX_LEN
            });
            kv_cursor += SUFFIX_LEN;
        }
    }

    const int total_requests = (int)requests.size();

    size_t q_elems =
        (size_t)total_requests
        * NUM_HEADS
        * QUERY_LEN
        * HEAD_DIM;

    half* d_Q = nullptr;
    half* d_O = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Q, q_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, q_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_Q, 0, q_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_O, 0, q_elems * sizeof(half)));

    PrefixAttentionManager manager(
        /* max_batch_size */ total_requests,
        /* max_kv_length  */ kv_cursor
    );

    manager.run_attention(
        requests,
        d_Q,
        d_O,
        QUERY_LEN,
        stream
    );

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "✓ Prefix attention completed\n";
    std::cout << "  Families            : " << NUM_FAMILIES << "\n";
    std::cout << "  Requests / family   : " << REQUESTS_PER_FAMILY << "\n";
    std::cout << "  Total requests      : " << total_requests << "\n";
    std::cout << "  Total KV tokens     : " << kv_cursor << "\n";

    return 0;
}
