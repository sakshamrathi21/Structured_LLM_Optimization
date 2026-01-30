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
dcgmi profile -l -i 0
sudo dcgmi dmon -e 1002,1003,1004,1005 -d 1
```


---

NCU

```bash
nv-nsight-cu-cli --target-processes all --section MemoryWorkloadAnalysis --launch-skip 200 --launch-count 50 --csv python vllm_multiple_families.py > out.csv
```

--

cp for vllm local backup


```bash
cp -r /home/saksham/CS496/code/sglang/.venv/lib/python3.12/site-packages/vllm/v1/core/sched /home/saksham/CS496/code/vllm/vllm_local_backup/vllm/v1/core/
```