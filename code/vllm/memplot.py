import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv("memcpy.csv")

# Convert timestamps from ns to ms for easier readability
df["start_ms"] = (df["start"] - df["start"].min()) / 1e6
df["end_ms"] = (df["end"] - df["start"].min()) / 1e6
df["duration_ms"] = df["duration"] / 1e6

print(df.head())

# Plot bytes transferred vs time
plt.figure(figsize=(10, 5))
plt.scatter(
    df["start_ms"], df["bytes"], c="blue", label="Bytes per transfer", alpha=0.7
)
plt.xlabel("Time (ms since start)")
plt.ylabel("Bytes transferred")
plt.title("GPU Memory Transfers over Time")
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plot bandwidth = bytes / duration
df["bandwidth_GBps"] = df["bytes"] / df["duration"] * 1e9 / (1024**3)

plt.figure(figsize=(10, 5))
plt.plot(df["start_ms"], df["bandwidth_GBps"], marker="o", linestyle="none", alpha=0.7)
plt.xlabel("Time (ms since start)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("GPU Memory Transfer Bandwidth")
plt.grid(True)
plt.savefig("memcpy_bandwidth.png")
