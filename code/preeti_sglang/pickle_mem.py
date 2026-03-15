import pickle
import matplotlib.pyplot as plt

# --- Load the data (same as before) ---
filename = '/home/preeti/sglang/prof1/1758551299.0762882/1758551300.159891-TP-0-memory.pickle'
try:
    with open(filename, 'rb') as f:
        memory_data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    exit()

# --- Plot the data ---
# We assume the data is a dictionary based on sglang's typical output
if isinstance(memory_data, dict) and 'time' in memory_data and 'mem_mb' in memory_data:
    timestamps = memory_data['time']
    memory_usage_mb = memory_data['mem_mb']

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, memory_usage_mb, label='Memory Usage')
    
    # Make the plot pretty
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    

else:
    print("Could not find expected keys ('time', 'mem_mb') in the data.")
    print("Please inspect the 'memory_data' object to understand its structure.")
    print(str(memory_data)[:500])