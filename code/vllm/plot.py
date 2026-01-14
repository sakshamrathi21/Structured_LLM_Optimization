import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Length factors
# -----------------------------
len_factors = [1, 2, 5, 10]

# -----------------------------
# SGLang results (max_batch=500)
# -----------------------------
# First = LPM, Second = FCFS
sglang_lpm = {
    1: 35.1254,
    2: 28.3909,
    5: 19.7881,
    10: 10.4652
}

sglang_fcfs = {
    1: 34.4680,
    2: 28.6956,
    5: 19.7313,
    10: 10.4794
}

# Fill missing length factors with NaN (so matplotlib skips them)
sglang_lpm_vals = [sglang_lpm.get(l, np.nan) for l in len_factors]
sglang_fcfs_vals = [sglang_fcfs.get(l, np.nan) for l in len_factors]

# -----------------------------
# vLLM results
# -----------------------------
# batch=1000
vllm_radix = {
    1: 43.5445, 2: 39.7591, 5: 30.1317, 10: 18.9529
}

vllm_lpm = {
    1: 42.6260, 2: 42.1705, 5: 37.4262, 10: 24.2105
}

# batch=500
vllm_fcfs = {
    1: 19.1136, 2: 12.5005, 5: 6.0260, 10: 3.3082
}

vllm_radix_vals = [vllm_radix[l] for l in len_factors]
vllm_lpm_vals = [vllm_lpm[l] for l in len_factors]
vllm_fcfs_vals = [vllm_fcfs[l] for l in len_factors]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(11, 7))

# SGLang
plt.plot(len_factors, sglang_lpm_vals, 'o--', linewidth=2, label='SGLang LPM')
plt.plot(len_factors, sglang_fcfs_vals, 's--', linewidth=2, label='SGLang FCFS')

# vLLM
plt.plot(len_factors, vllm_radix_vals, 'o-', linewidth=2, label='vLLM Radix-Cost')
plt.plot(len_factors, vllm_lpm_vals, 's-', linewidth=2, label='vLLM Family-Load')
plt.plot(len_factors, vllm_fcfs_vals, '^-', linewidth=2, label='vLLM FCFS')

plt.xlabel('Length Factor (multiples of 1300 tokens)', fontsize=12)
plt.ylabel('Throughput (req/s)', fontsize=12)
plt.title('Throughput vs Seuence Length Length\nSGLang vs vLLM (Llama-3.1-8B-Instruct)', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("throughput_vllm_sglang_comparison_scheduling_policies.png")
