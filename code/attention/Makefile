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

# Source files
KERNEL_SRC := prefix_attention_kernel.cu
EXAMPLE_SRC := example_usage.cu
TEST_SRC := test_compile.cu
MINIMAL_TEST_SRC := minimal_test.cu

# Output binaries
EXAMPLE_BIN := prefix_attention_example
TEST_BIN := test_compile
MINIMAL_BIN := minimal_test

# Targets
.PHONY: all clean example test minimal benchmark profile

all: test minimal

example: $(EXAMPLE_SRC) $(KERNEL_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $(EXAMPLE_BIN) $(EXAMPLE_SRC) $(LIBS)
	@echo "Build complete: $(EXAMPLE_BIN)"
	@echo "Run with: ./$(EXAMPLE_BIN)"

test: $(TEST_SRC) $(KERNEL_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $(TEST_BIN) $(TEST_SRC) $(LIBS)
	@echo "Build complete: $(TEST_BIN)"
	@echo "Run with: ./$(TEST_BIN)"

minimal: $(MINIMAL_TEST_SRC) $(KERNEL_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $(MINIMAL_BIN) $(MINIMAL_TEST_SRC) $(LIBS)
	@echo "Build complete: $(MINIMAL_BIN)"
	@echo "Run with: ./$(MINIMAL_BIN)"

# Run with profiler
profile: example
	ncu --set full --target-processes all ./$(EXAMPLE_BIN)

# Check compilation details
verbose: NVCC_FLAGS += -v
verbose: example

# Clean build artifacts
clean:
	rm -f $(EXAMPLE_BIN) $(TEST_BIN) $(MINIMAL_BIN) *.o *.ptx

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
