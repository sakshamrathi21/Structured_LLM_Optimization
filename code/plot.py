import matplotlib.pyplot as plt

# QPS values
qps = [5, 10, 15, 20, 25, 30, 35, 40, 50, 100]

# --- vLLM Without Automatic Prefix Caching ---
avg_latency_no_cache = [0.351, 0.450, 0.576, 0.827, 1.106, 1.265, 1.421, 1.684, 1.659, 2.255]
p95_latency_no_cache = [0.357, 0.503, 0.631, 0.972, 1.286, 1.512, 1.908, 2.016, 2.026, 2.640]
throughput_no_cache = [4.95, 9.40, 13.81, 16.68, 18.55, 19.29, 19.28, 19.41, 20.70, 21.94]

# --- vLLM With Automatic Prefix Caching ---
avg_latency_cache = [0.516, 0.558, 1.023, 1.214, 3.304, 1.892, 3.438, 2.358, 2.248, 2.101]
p95_latency_cache = [0.954, 1.718, 1.669, 2.343, 4.188, 2.428, 4.245, 2.379, 2.281, 2.687]
throughput_cache = [4.63, 8.58, 11.59, 17.24, 12.74, 18.05, 12.51, 19.60, 20.46, 20.61]

# --- SGLang Results ---
avg_latency_sglang = [0.528, 0.635, 0.742, 0.895, 0.996, 1.117, 1.088, 1.190, 1.271, 1.388]
p95_latency_sglang = [0.578, 0.689, 0.841, 0.961, 1.107, 1.253, 1.278, 1.361, 1.376, 1.446]
throughput_sglang = [4.85, 9.10, 13.17, 15.84, 19.08, 20.18, 22.30, 25.43, 26.87, 31.19]

# --- SwiftLLM Results ---
avg_latency_swift = [5.608, 5.546, 7.634, 7.824, 8.139, 13.792, 13.873, 13.546, 13.038, 14.121]
p95_latency_swift = [6.871, 6.223, 9.260, 9.900, 9.083, 13.892, 13.964, 13.639, 13.121, 14.212]
throughput_swift = [3.85, 5.44, 5.72, 5.92, 6.39, 5.63, 5.29, 5.57, 5.58, 5.18]

# --- Plot Avg Latency ---
plt.figure(figsize=(8, 5))
plt.plot(qps, avg_latency_no_cache, marker='o', linestyle='-', label="vllm - No Cache")
plt.plot(qps, avg_latency_cache, marker='s', linestyle='--', label="vllm - With Cache")
plt.plot(qps, avg_latency_sglang, marker='^', linestyle='-.', label="SGLang")
plt.plot(qps, avg_latency_swift, marker='d', linestyle=':', label="SwiftLLM")
plt.xlabel("Queries Per Second")
plt.ylabel("Avg Latency (s)")
plt.title("Average Latency vs QPS")
plt.grid(True)
plt.legend()
plt.savefig("avg_latency_vs_qps.png")

# --- Plot P95 Latency ---
plt.figure(figsize=(8, 5))
plt.plot(qps, p95_latency_no_cache, marker='o', linestyle='-', color='orange', label="vllm - No Cache")
plt.plot(qps, p95_latency_cache, marker='s', linestyle='--', color='red', label="vllm - With Cache")
plt.plot(qps, p95_latency_sglang, marker='^', linestyle='-.', color='blue', label="SGLang")
plt.plot(qps, p95_latency_swift, marker='d', linestyle=':', color='green', label="SwiftLLM")
plt.xlabel("Queries Per Second")
plt.ylabel("P95 Latency (s)")
plt.title("P95 Latency vs QPS")
plt.grid(True)
plt.legend()
plt.savefig("p95_latency_vs_qps.png")

# --- Plot Throughput ---
plt.figure(figsize=(8, 5))
plt.plot(qps, throughput_no_cache, marker='o', linestyle='-', color='green', label="vllm - No Cache")
plt.plot(qps, throughput_cache, marker='s', linestyle='--', color='purple', label="vllm - With Cache")
plt.plot(qps, throughput_sglang, marker='^', linestyle='-.', color='black', label="SGLang")
plt.plot(qps, throughput_swift, marker='d', linestyle=':', color='orange', label="SwiftLLM")
plt.xlabel("Queries Per Second")
plt.ylabel("Throughput (req/s)")
plt.title("Throughput vs QPS")
plt.grid(True)
plt.legend()
plt.savefig("throughput_vs_qps.png")
