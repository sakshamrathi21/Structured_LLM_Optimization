/*
  pcie_bw.cu
  Measure PCIe (GPU<->CPU) bandwidth both directions.

  Build:
    nvcc -o pcie_bw pcie_bw.cu -ccbin g++-11 -std=c++14

  Run:
    ./pcie_bw

  Notes:
    - Uses cudaEvent timing (GPU timers).
    - Tests pageable vs pinned host memory, sync vs async, and simultaneous transfers.
    - Some OS/hardware may restrict paging/pinning; cudaMallocHost may fail if not enough pinned memory.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <cmath>
#include <getopt.h>

#define CHECK(call)                                                   \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

using u64 = unsigned long long;

struct Options {
  size_t min_size = 4 * 1024;          // 4 KB
  size_t max_size = 256 * 1024 * 1024; // 256 MB
  int iters = 50;
  int skip = 5;
} opts;

void parse_args(int argc, char** argv) {
  const struct option longopts[] = {
    {"min-size", required_argument, 0, 'm'},
    {"max-size", required_argument, 0, 'M'},
    {"iters", required_argument, 0, 'i'},
    {"skip", required_argument, 0, 's'},
    {0,0,0,0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "m:M:i:s:", longopts, NULL)) != -1) {
    switch (opt) {
      case 'm': opts.min_size = std::stoull(optarg); break;
      case 'M': opts.max_size = std::stoull(optarg); break;
      case 'i': opts.iters = std::stoi(optarg); break;
      case 's': opts.skip = std::stoi(optarg); break;
      default: break;
    }
  }
}

std::vector<size_t> generate_sizes(size_t min_sz, size_t max_sz) {
  std::vector<size_t> sizes;
  size_t s = min_sz;
  if (s == 0) s = 1;
  while (s <= max_sz) {
    sizes.push_back(s);
    if (s < 1024) s *= 2;
    else if (s < 1024*1024) s *= 2;
    else s *= 2;
    // prevent infinite loop on overflow
    if (s == 0) break;
  }
  return sizes;
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t end) {
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, end));
  return static_cast<double>(ms);
}

void fill_buf(char* buf, size_t n, char val=0x5A) {
  memset(buf, val, n);
}

void print_header() {
  std::cout << "# PCIe bandwidth measurement (GPU <-> CPU)\n";
  std::cout << "# Columns: size(bytes), pageable_H2D_GBps, pageable_D2H_GBps, pinned_H2D_GBps, pinned_D2H_GBps, async_pinned_H2D_GBps, async_pinned_D2H_GBps, concurrent_bidir_GBps\n";
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  int dev = 0;
  CHECK(cudaSetDevice(dev));

  // print device summary
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Device: " << prop.name << " pciBusID: " << prop.pciBusID << " (multiProcessorCount=" << prop.multiProcessorCount << ")\n";
  std::cout << "PCIe bandwidth test - iterations: " << opts.iters << " skip: " << opts.skip << "\n\n";

  auto sizes = generate_sizes(opts.min_size, opts.max_size);
  print_header();

  for (size_t sz : sizes) {
    // allocate device buffer
    char* d_buf = nullptr;
    CHECK(cudaMalloc(&d_buf, sz));

    // pageable host memory
    char* h_pageable = (char*)malloc(sz);
    if (!h_pageable) {
      std::cerr << "malloc failed for size " << sz << "\n";
      return 1;
    }
    fill_buf(h_pageable, sz, 0xA5);

    // pinned host memory
    char* h_pinned = nullptr;
    cudaError_t perr = cudaMallocHost((void**)&h_pinned, sz); // pinned
    bool pinned_ok = (perr == cudaSuccess);
    if (pinned_ok) fill_buf(h_pinned, sz, 0xA5);
    else {
      // fallback: use pageable and mark pinned_ok false
      h_pinned = h_pageable;
    }

    // device source data for D2H tests
    // fill device buffer with a pattern using a short kernel? We'll cudaMemset to pattern (works for bytes 0/255). For variety, use cudaMemcpy from host pinned.
    if (pinned_ok) {
      CHECK(cudaMemcpy(d_buf, h_pinned, sz, cudaMemcpyHostToDevice));
    } else {
      CHECK(cudaMemcpy(d_buf, h_pageable, sz, cudaMemcpyHostToDevice));
    }

    // Events for timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto measure_sync = [&](void* hptr, cudaMemcpyKind kind, int iters, int skip) -> double {
      // synchronous cudaMemcpy measured with events
      // warm-ups included in skip
      double total_ms = 0.0;
      for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpy(d_buf, hptr, sz, kind)); // if kind is DeviceToHost, order args reversed below; but we'll call properly
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        double ms = elapsed_ms(start, stop);
        if (i >= skip) total_ms += ms;
      }
      int measured = iters - skip;
      if (measured <= 0) measured = 1;
      double avg_ms = total_ms / measured;
      double gb = (double)sz / (1<<30);
      double gbps = gb / (avg_ms / 1000.0);
      return gbps;
    };

    // For correct argument order, define wrappers:
    auto measure_sync_H2D = [&](void* hptr)->double{
      // H2D: host -> device
      double total_ms = 0.0;
      for (int i=0;i<opts.iters;++i) {
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpy(d_buf, hptr, sz, cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        double ms = elapsed_ms(start, stop);
        if (i >= opts.skip) total_ms += ms;
      }
      int measured = opts.iters - opts.skip;
      if (measured<=0) measured=1;
      double avg_ms = total_ms / measured;
      double gb = (double)sz / (1<<30);
      return gb / (avg_ms/1000.0);
    };
    auto measure_sync_D2H = [&](void* hptr)->double{
      // D2H: device -> host
      double total_ms = 0.0;
      for (int i=0;i<opts.iters;++i) {
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpy(hptr, d_buf, sz, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        double ms = elapsed_ms(start, stop);
        if (i >= opts.skip) total_ms += ms;
      }
      int measured = opts.iters - opts.skip;
      if (measured<=0) measured=1;
      double avg_ms = total_ms / measured;
      double gb = (double)sz / (1<<30);
      return gb / (avg_ms/1000.0);
    };

    // Measure pageable synchronous (use h_pageable)
    double page_H2D_gbps = measure_sync_H2D(h_pageable);
    double page_D2H_gbps = measure_sync_D2H(h_pageable);

    // Measure pinned synchronous (use h_pinned if available, else same as pageable)
    double pinned_H2D_gbps = page_H2D_gbps;
    double pinned_D2H_gbps = page_D2H_gbps;
    if (pinned_ok) {
      pinned_H2D_gbps = measure_sync_H2D(h_pinned);
      pinned_D2H_gbps = measure_sync_D2H(h_pinned);
    }

    // Asynchronous single-stream pinned (H2D and D2H separately)
    double async_H2D_gbps = 0.0, async_D2H_gbps = 0.0;
    if (pinned_ok) {
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));
      for (int i = 0; i < opts.iters; ++i) {
        CHECK(cudaEventRecord(start, stream));
        CHECK(cudaMemcpyAsync(d_buf, h_pinned, sz, cudaMemcpyHostToDevice, stream));
        CHECK(cudaEventRecord(stop, stream));
        CHECK(cudaEventSynchronize(stop));
        double ms = elapsed_ms(start, stop);
        if (i >= opts.skip) async_H2D_gbps += ms;
      }
      async_H2D_gbps = async_H2D_gbps / std::max(1, opts.iters - opts.skip);
      async_H2D_gbps = ((double)sz / (1<<30)) / (async_H2D_gbps / 1000.0);

      // D2H async
      for (int i = 0; i < opts.iters; ++i) {
        CHECK(cudaEventRecord(start, stream));
        CHECK(cudaMemcpyAsync(h_pinned, d_buf, sz, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaEventRecord(stop, stream));
        CHECK(cudaEventSynchronize(stop));
        double ms = elapsed_ms(start, stop);
        if (i >= opts.skip) async_D2H_gbps += ms;
      }
      async_D2H_gbps = async_D2H_gbps / std::max(1, opts.iters - opts.skip);
      async_D2H_gbps = ((double)sz / (1<<30)) / (async_D2H_gbps / 1000.0);

      CHECK(cudaStreamDestroy(stream));
    }

    // Concurrent bi-directional: two streams, two pinned buffers
    double bidi_gbps = 0.0;
    if (pinned_ok) {
      // allocate another device buffer and pinned host buffer for concurrency
      char* d_buf2 = nullptr;
      CHECK(cudaMalloc(&d_buf2, sz));
      char* h_pinned2 = nullptr;
      CHECK(cudaMallocHost((void**)&h_pinned2, sz));
      fill_buf(h_pinned2, sz, 0x3C);
      // create streams
      cudaStream_t s1, s2;
      CHECK(cudaStreamCreate(&s1));
      CHECK(cudaStreamCreate(&s2));
      // warm-up: copy d_buf <- h_pinned and h_pinned2 <- d_buf2
      CHECK(cudaMemcpyAsync(d_buf, h_pinned, sz, cudaMemcpyHostToDevice, s1));
      CHECK(cudaMemcpyAsync(h_pinned2, d_buf2, sz, cudaMemcpyDeviceToHost, s2));
      CHECK(cudaStreamSynchronize(s1));
      CHECK(cudaStreamSynchronize(s2));

      double total_ms = 0.0;
      for (int i = 0; i < opts.iters; ++i) {
        CHECK(cudaEventRecord(start, 0)); // record on default stream for global time
        CHECK(cudaMemcpyAsync(d_buf, h_pinned, sz, cudaMemcpyHostToDevice, s1));
        CHECK(cudaMemcpyAsync(h_pinned2, d_buf2, sz, cudaMemcpyDeviceToHost, s2));
        // synchronize both streams
        CHECK(cudaStreamSynchronize(s1));
        CHECK(cudaStreamSynchronize(s2));
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        double ms = elapsed_ms(start, stop);
        if (i >= opts.skip) total_ms += ms;
      }
      int measured = opts.iters - opts.skip; if (measured <= 0) measured = 1;
      double avg_ms = total_ms / measured;
      // combined transferred bytes = 2 * sz per iteration (one in each direction)
      double gb = (double)(2*sz) / (1<<30);
      bidi_gbps = gb / (avg_ms / 1000.0);

      CHECK(cudaStreamDestroy(s1));
      CHECK(cudaStreamDestroy(s2));
      CHECK(cudaFree(d_buf2));
      CHECK(cudaFreeHost(h_pinned2));
    }

    // print row
    std::cout << sz << ", "
              << page_H2D_gbps << ", "
              << page_D2H_gbps << ", "
              << pinned_H2D_gbps << ", "
              << pinned_D2H_gbps << ", "
              << async_H2D_gbps << ", "
              << async_D2H_gbps << ", "
              << bidi_gbps << "\n";

    // cleanup
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    free(h_pageable);
    if (pinned_ok) CHECK(cudaFreeHost(h_pinned));
    CHECK(cudaFree(d_buf));
    // small sleep? not necessary
  } // sizes loop

  std::cout << "\n# Done\n";
  return 0;
}
