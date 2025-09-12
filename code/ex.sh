cd swiftLLM
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 5 > ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 10 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 15 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 20 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 25 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 30 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 35 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 40 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 45 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 50 >> ../vllm_ex_5.txt
sleep 120
python3 examples/try.py --model-path ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/ --qps 100 >> ../vllm_ex_5.txt