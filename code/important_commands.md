# Important Commands

---

```bash
python3 examples/offline.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/
```

---

```bash
python3 examples/online.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/
```

---

```bash
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/
```

---

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --mem-fraction-static 0.7 --disable-cuda-graph
```

---

```bash
python3 -m sglang.launch_server   --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0   --host 0.0.0.0
```

---

For NEO:

```bash
python examples/example.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --model-name llama2_7b
```

---

NSYS:
```bash
nsys profile --trace=cuda --cuda-memory-usage=true --gpu-metrics-device=all -o llm_mem_bw python vllm_without.py
```

---

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
ncu --set full --metrics l1tex__t_sector_hit_rate.pct,lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct python vllm_multiple_families.py
```

---

NV-NSIGHT-CU-CLI
```bash
nv-nsight-cu-cli --metrics l2_read_transactions,l2_write_transactions,l2_read_bytes,l2_write_bytes,l2_read_transactions_miss,l2_write_transactions_miss --csv python vllm_multiple_families.py 
```


```bash
nv-nsight-cu-cli --target-processes all --kernel-name "regex:.*gemm.*|.*flash_attn.*" --launch-skip 200 --launch-count 50 --metrics l1tex__t_sectors_hit.sum,l1tex__t_sectors_miss.sum,lts__t_sectors_hit.sum,lts__t_sectors_miss.sum python vllm_multiple_families.py
```


```bash
nv-nsight-cu-cli --target-processes all --section MemoryWorkloadAnalysis --launch-skip 200 --launch-count 50 --csv python vllm_multiple_families.py > out.csv
```