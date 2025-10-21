#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvToolsExt.h>   // NVTX for timeline ranges

#define BLOCK_SIZE 16      // threads per block dimension
#define TILE_SIZE 8192      // host tiling (adjust based on GPU memory)

const int N = TILE_SIZE;

// CUDA kernel with shared memory tiling
__global__ void matmul_tiled(float *A, float *B, float *C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        if (row < n && k * BLOCK_SIZE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + k * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < n && k * BLOCK_SIZE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int t = 0; t < BLOCK_SIZE; t++)
            sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] += sum;
}

// Copy a TILE_SIZE × TILE_SIZE submatrix from host to device
void copySubMatrixToDevice(float *dst, float *src,
                           int row, int col, int N, int tileSize) {
    for (int i = 0; i < tileSize; i++) {
        if (row + i >= N) break;
        cudaMemcpy(dst + i * tileSize,
                   src + (row + i) * N + col,
                   tileSize * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
}

// Copy back submatrix from device to host
void copySubMatrixToHost(float *dst, float *src,
                         int row, int col, int N, int tileSize) {
    for (int i = 0; i < tileSize; i++) {
        if (row + i >= N) break;
        cudaMemcpy(dst + (row + i) * N + col,
                   src + i * tileSize,
                   tileSize * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
}

int main() {
    
    size_t size = N * N * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_B[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_C[i] = 0.0f;
    }

    // Process in tiles
    for (int i = 0; i < N; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            // Allocate device memory for sub-block of C
            float *d_subC;
            cudaMalloc(&d_subC, TILE_SIZE * TILE_SIZE * sizeof(float));
            cudaMemset(d_subC, 0, TILE_SIZE * TILE_SIZE * sizeof(float));

            for (int k = 0; k < N; k += TILE_SIZE) {
                // Allocate sub-blocks
                float *d_subA, *d_subB;
                cudaMalloc(&d_subA, TILE_SIZE * TILE_SIZE * sizeof(float));
                cudaMalloc(&d_subB, TILE_SIZE * TILE_SIZE * sizeof(float));

                // ---- H2D copy ----
                nvtxRangePush("H2D Copy");
                copySubMatrixToDevice(d_subA, h_A, i, k, N, TILE_SIZE);
                copySubMatrixToDevice(d_subB, h_B, k, j, N, TILE_SIZE);
                nvtxRangePop();

                // ---- Kernel compute ----
                nvtxRangePush("Kernel Compute");
                dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
                dim3 blocks((TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                            (TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
                matmul_tiled<<<blocks, threads>>>(d_subA, d_subB, d_subC, TILE_SIZE);
                cudaDeviceSynchronize();
                nvtxRangePop();

                cudaFree(d_subA);
                cudaFree(d_subB);
            }

            nvtxRangePush("D2H Copy");
            copySubMatrixToHost(h_C, d_subC, i, j, N, TILE_SIZE);
            nvtxRangePop();

            cudaFree(d_subC);
        }
    }
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "C[N-1][N-1] = " << h_C[N*N-1] << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
