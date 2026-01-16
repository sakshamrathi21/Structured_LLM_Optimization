import numpy as np
import matplotlib.pyplot as plt


def read_drama(filename):
    drama = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip headers / empty lines
            if not line or line.startswith("#") or line.startswith("ID"):
                continue

            parts = line.split()

            # DRAMA is last column
            if len(parts) < 6:
                continue

            try:
                drama.append(float(parts[-1]))
            except ValueError:
                continue

    drama = np.array(drama)

    # Trim leading & trailing zeros only
    nz = np.nonzero(drama)[0]
    if len(nz) == 0:
        raise ValueError(f"{filename}: DRAMA column is all zeros")

    return drama[nz[0] : nz[-1] + 1]


def plot_cdf(data, label):
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    plt.plot(data_sorted, cdf, label=label)


def print_stats(data, label):
    print(f"\n=== {label} ===")
    print(f"Sum      : {np.sum(data):.6f}")
    print(f"Count    : {len(data)}")
    print(f"Mean     : {np.mean(data):.6f}")
    print(f"Median   : {np.median(data):.6f}")

    for p in [50, 75, 90, 95, 99]:
        print(f"P{p:<2}      : {np.percentile(data, p):.6f}")


# -------------------------------
# Read both files
# -------------------------------
drama1 = read_drama(".out_fam")
drama2 = read_drama(".out_fcfs")

# -------------------------------
# Print statistics
# -------------------------------
print_stats(drama1, "User gives out the family IDs")
print_stats(drama2, "First come first serve")

# -------------------------------
# Plot both CDFs
# -------------------------------
plt.figure()
plot_cdf(drama1, "User gives out the family IDs")
plot_cdf(drama2, "First come first serve")

plt.xlabel("DRAMA")
plt.ylabel("CDF")
plt.title("CDF of DRAMA")
plt.grid(True)
plt.legend()
plt.savefig("../../results/drama_family_fcfs.png")
