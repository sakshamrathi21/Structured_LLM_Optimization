# Important Commands

NVIDIA-SMI

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -lms 50
```

---

CUDA-compilation

```bash
nvcc -ccbin=/usr/bin/gcc-11 -std=c++17 -o matmul matrixmul.cu -lcudart -lstdc++ -lnvToolsExt
```

---

DCGMI

```bash
sudo dcgmi dmon -e 1002,1003,1004 -d 10
```


---

NCU

```bash
nv-nsight-cu-cli --target-processes all --section MemoryWorkloadAnalysis --launch-skip 200 --launch-count 50 --csv python vllm_multiple_families.py > out.csv
```