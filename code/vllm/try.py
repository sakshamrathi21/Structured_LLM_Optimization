import re
import matplotlib.pyplot as plt

# Read file
with open("out_infinite", "r") as f:
    lines = f.readlines()

numbers = []
for line in lines:
    match = re.search(r"Number of outputs in this step:\s+(\d+)", line)
    if match:
        numbers.append(int(match.group(1)))

# Plot
plt.plot(numbers, marker='o')
plt.xlabel("Step")
plt.ylabel("Number of outputs")
plt.title("Outputs per Step")
plt.grid(True)
plt.savefig("vllm_batch_outputs_chunked_prefill.png")
