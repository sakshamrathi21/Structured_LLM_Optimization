import matplotlib.pyplot as plt

# X-axis: request rates
req_sec = [10, 100, 200, 500, 1000, 2000, 5000, 10000]

# Throughput values
radix_cost = [
    9.16,
    32.35663690486607,
    40.36479061288881,
    42.21662840658566,
    42.10114677289437,
    42.1386184755942,
    42.0401697764121,
    42.17005833420642,
]

family_load = [
    9.20,
    34.17528806058268,
    40.222898031731674,
    42.11460893653297,
    42.07490540425598,
    42.26738324567834,
    42.15866076559627,
    42.450330405363815,
]

fcfs = [
    9.20,
    18.120485455874917,
    18.481390153445936,
    18.590202062206405,
    18.741692642513357,
    19.061368466940785,
    19.000823734732364,
    18.79672128381361,
]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(req_sec, radix_cost, marker="o", label="radix_cost")
plt.plot(req_sec, family_load, marker="s", label="family_load")
plt.plot(req_sec, fcfs, marker="^", label="fcfs")

plt.xlabel("Request Rate (req/sec)")
plt.ylabel("Throughput")
plt.title("Throughput vs Request Rate")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("../../results/throughput_vs_req_per_sec_poisson.png")
