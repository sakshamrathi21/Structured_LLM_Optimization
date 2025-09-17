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
nsys profile --sample=process-tree --trace=cuda,osrt --gpu-metrics-device=all --cuda-memory-usage true --gpu-metrics-frequency=100 --force-overwrite=true -o llm_mem_bw python vllm_without.py
```