# Makefile for Prefix Attention Kernel

CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# GPU Architecture - adjust based on your GPU
# SM_80: A100, SM_86: RTX 3090, SM_89: RTX 4090, SM_90: H100
GPU_ARCH ?= sm_80

# Compiler flags
NVCC_FLAGS := -O3 \
              -std=c++17 \
              -arch=$(GPU_ARCH) \
              -use_fast_math \
              -lineinfo \
              --ptxas-options=-v \
              -Xcompiler -fopenmp,-O3,-march=native -ccbin=/usr/bin/gcc-11 -lcudart -lstdc++ -lnvToolsExt -lcublas

# Debugging flags (uncomment for debug builds)
# NVCC_FLAGS += -G -g -DDEBUG

# Include paths
INCLUDES := -I$(CUDA_PATH)/include

# Libraries
LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcublas

KERNEL_SRC := prefix_attention_kernel.cu

EXAMPLE_BIN := prefix_attention_example

# Targets
.PHONY: all clean example benchmark profile

all: example

example: $(KERNEL_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(KERNEL_SRC) -o $(EXAMPLE_BIN) $(LIBS)
	@echo "Build complete: $(EXAMPLE_BIN)"
	@echo "Run with: ./$(EXAMPLE_BIN)"

# Run with profiler
profile: example
	ncu --set full --target-processes all ./$(EXAMPLE_BIN)

# Check compilation details
verbose: NVCC_FLAGS += -v
verbose: example

# Clean build artifacts
clean:
	rm -f $(EXAMPLE_BIN) *.o *.ptx

# Print configuration
info:
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "GPU Architecture: $(GPU_ARCH)"
	@echo "Compiler: $(NVCC)"
	@echo "Flags: $(NVCC_FLAGS)"

# Help
help:
	@echo "Available targets:"
	@echo "  make test      - Build simple compilation test"
	@echo "  make example   - Build the full example program"
	@echo "  make profile   - Build and profile with Nsight Compute"
	@echo "  make verbose   - Build with verbose output"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make info      - Show build configuration"
	@echo ""
	@echo "Configuration:"
	@echo "  GPU_ARCH=sm_XX - Set GPU architecture (default: sm_80)"
	@echo "  Example: make test GPU_ARCH=sm_90"
