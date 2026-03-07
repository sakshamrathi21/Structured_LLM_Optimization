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
cp -r /home/saksham/CS496/code/vllm/.venv/lib/python3.12/site-packages/vllm/v1/core/sched /home/saksham/CS496/code/vllm/vllm_local_backup/vllm/v1/core/
```


```bash
cd /home/saksham/CS496/code/vllm/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/build ; rm -rf * ; cmake -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. ; make -j ; cd -
```


## NSYS

```bash
nsys profile --trace=cuda --gpu-metrics-device=all ./read
nsys export --t sqlite -o report1.sqlite report1.nsys-rep
sqlite3 report1.sqlite "
SELECT 
    AVG(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) AS avg_dram_read,
    MAX(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) AS peak_dram_read
FROM GENERIC_EVENTS
WHERE json_extract(data, '$.\"DRAM Read Throughput\"') IS NOT NULL
  AND (CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL) > 0 
       OR CAST(json_extract(data, '$.\"DRAM Write Throughput\"') AS REAL) > 0);

SELECT 
    SUM(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) AS total_dram_read
FROM GENERIC_EVENTS
WHERE json_extract(data, '$.\"DRAM Read Throughput\"') IS NOT NULL
  AND (CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL) > 0 
       OR CAST(json_extract(data, '$.\"DRAM Write Throughput\"') AS REAL) > 0);
"
rm report1.sqlite report1.nsys-rep
```