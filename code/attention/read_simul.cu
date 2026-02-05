#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define CHECK(x) do {                                  \
    cudaError_t err = x;                               \
    if (err != cudaSuccess) {                          \
        printf("CUDA error %s:%d: %s\n",               \
               __FILE__, __LINE__,                     \
               cudaGetErrorString(err));               \
        exit(1);                                       \
    }                                                  \
} while (0)

__global__ void read_kernel_multi(
    const float* base,
    size_t prefix_elems,
    size_t suffix_elems,
    size_t family_elems,
    int requests,
    int blocks_per_request,
    float* sink)
{
    int block = blockIdx.x;
    int request_id = block / blocks_per_request;

    if (request_id >= requests)
        return;

    const float* req_base =
        base + request_id * suffix_elems;

    float acc = 0.f;

    #pragma unroll 4
    for (size_t i = threadIdx.x;
         i + 4 * blockDim.x < prefix_elems;
         i += 4 * blockDim.x)
    {
        acc += req_base[i];
        acc += req_base[i + blockDim.x];
        acc += req_base[i + 2 * blockDim.x];
        acc += req_base[i + 3 * blockDim.x];
    }

    const float* suffix = req_base + prefix_elems;
    for (size_t i = threadIdx.x; i < suffix_elems; i += blockDim.x)
        acc += suffix[i];

    if (threadIdx.x == 0)
        sink[blockIdx.x] = acc;
}

int main(int argc, char** argv)
{
    int families = 5;
    int requests = 100;
    int prefix_mb = 1800;
    int suffix_mb = 3;
    int iters = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--families")) families = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--requests")) requests = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefix_mb")) prefix_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--suffix_mb")) suffix_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    }

    size_t prefix_elems =
        (size_t)prefix_mb * 1024 * 1024 / sizeof(float);
    size_t suffix_elems =
        (size_t)suffix_mb * 1024 * 1024 / sizeof(float);

    size_t family_elems =
        prefix_elems + (size_t)requests * suffix_elems;
    size_t total_elems =
        (size_t)families * family_elems;

    printf("Total memory: %.2f GB\n",
           total_elems * sizeof(float) / 1e9);

    float* d_mem;
    CHECK(cudaMalloc(&d_mem, total_elems * sizeof(float)));
    CHECK(cudaMemset(d_mem, 1, total_elems * sizeof(float)));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, 1024 * 1024 * sizeof(float)));

    int sm_count;
    CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, 0));

    dim3 block(1024);

    int blocks_per_request = sm_count * 2;
    int total_blocks = requests * blocks_per_request;
    dim3 grid(total_blocks);

    cudaStream_t capture_stream;
    CHECK(cudaStreamCreate(&capture_stream));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeGlobal));

    for (int f = 0; f < families; f++) {
        const float* family_base =
            d_mem + f * family_elems;

        read_kernel_multi<<<grid, block, 0, capture_stream>>>(
            family_base,
            prefix_elems,
            suffix_elems,
            family_elems,
            requests,
            blocks_per_request,
            d_sink);
    }

    CHECK(cudaStreamEndCapture(capture_stream, &graph));
    CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++) {
        CHECK(cudaGraphLaunch(graph_exec, 0));
    }
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    size_t bytes_per_iter = families * (
        prefix_elems * sizeof(float) * requests + 
        suffix_elems * sizeof(float) * requests  
    );
    size_t total_bytes = bytes_per_iter * iters;
    float time_s = ms / 1000.0f;
    float bandwidth_GBps = (total_bytes / 1e9) / time_s;

    printf("\n=== BANDWIDTH ANALYSIS ===\n");
    printf("Data per iteration: %.2f GB\n", bytes_per_iter / 1e9);
    printf("Total data transferred: %.2f GB\n", total_bytes / 1e9);
    printf("Time: %.2f ms (%.4f s)\n", ms, time_s);
    printf("Achieved DRAM Bandwidth: %.2f GB/s\n", bandwidth_GBps);

    CHECK(cudaDeviceSynchronize());

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
    cudaStreamDestroy(capture_stream);
    cudaFree(d_mem);
    cudaFree(d_sink);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float memory_clock_khz = prop.memoryClockRate;  
    int memory_bus_width = prop.memoryBusWidth;   
    float peak_bandwidth_GBps = 2.0 * memory_clock_khz * (memory_bus_width / 8) / 1e6;

    printf("Peak DRAM Bandwidth: %.2f GB/s\n", peak_bandwidth_GBps);
    printf("Bandwidth Utilization: %.1f%%\n", 
       100.0 * bandwidth_GBps / peak_bandwidth_GBps);

    printf("Done.\n");
    return 0;
}
