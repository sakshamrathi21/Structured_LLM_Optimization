# python vllm_multiple_families_varying_workloads.py --num_families 5 --req_per_sec 20 --policy chunked_hash_tree_bandit --num_requests 5000 2>&1 | tee -a vllm_multiple_families_varying_workloads_5_families_20_rps_chunked_hash_tree_bandit.log

# python vllm_multiple_families_varying_workloads.py --num_families 5 --req_per_sec 20 --policy chunked_hash_tree_qlearning --num_requests 5000 2>&1 | tee -a vllm_multiple_families_varying_workloads_5_families_20_rps_chunked_hash_tree_qlearning.log

python vllm_multiple_families_varying_workloads.py --num_families 5 --req_per_sec 500 --policy chunked_hash_tree_bandit --num_requests 5000 2>&1 | tee -a vllm_multiple_families_varying_workloads_5_families_500_rps_chunked_hash_tree_bandit.log

python vllm_multiple_families_varying_workloads.py --num_families 5 --req_per_sec 500 --policy chunked_hash_tree_qlearning --num_requests 5000 2>&1 | tee -a vllm_multiple_families_varying_workloads_5_families_500_rps_chunked_hash_tree_qlearning.log