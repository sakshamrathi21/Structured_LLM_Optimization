import numpy as np
import matplotlib.pyplot as plt

def remove_bursts(data, window=50, k=1.5, min_burst_len=20):
    """
    Identify and remove contiguous 'bursts' where the rolling average
    is significantly higher than baseline.

    Parameters:
        data : np.array
        window : size of rolling window (samples)
        k : IQR multiplier for threshold
        min_burst_len : minimum contiguous samples to call a burst

    Returns:
        filtered_data, mask
    """

    if len(data) < window:
        return data, np.ones_like(data, dtype=bool)

    # Rolling mean
    kernel = np.ones(window) / window
    rolling = np.convolve(data, kernel, mode="same")

    # Robust threshold based on distribution
    q1 = np.percentile(rolling, 25)
    q3 = np.percentile(rolling, 75)
    iqr = q3 - q1

    threshold = 0.35

    # Identify potential burst points
    high = rolling > threshold

    # Group contiguous regions
    mask = np.ones_like(data, dtype=bool)

    i = 0
    n = len(data)

    while i < n:
        if high[i]:
            start = i
            while i < n and high[i]:
                i += 1
            end = i

            if (end - start) >= min_burst_len:
                mask[start:end] = False
        else:
            i += 1

    return data[mask], mask


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
                # if (float(parts[-1])) > 0.35:
                #     continue
                drama.append(float(parts[-1]))
            except ValueError:
                continue

    drama = np.array(drama)

    # Trim leading & trailing zeros only
    nz = np.nonzero(drama)[0]
    if len(nz) == 0:
        raise ValueError(f"{filename}: DRAMA column is all zeros")

    # return drama
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
    # print(f"Max      : {np.max(data):.6f}")

    for p in [75, 90, 95, 99]:
        print(f"P{p:<2}      : {np.percentile(data, p):.6f}")


# -------------------------------
# Read both files
# -------------------------------
drama1 = read_drama(".out_1")
drama2 = read_drama(".out_5")
# filtered1, mask1 = remove_bursts(drama1)
# filtered2, mask2 = remove_bursts(drama2)
print_stats(drama1, "Single Family ; 500 Requests")
print_stats(drama2, "5 Families ; 100 Requests Each")

# plt.figure()
# plt.plot(drama1, label="User gives out the family IDs")
# plt.plot(drama2, label="First come first serve")

# plt.xlabel("Index")
# plt.ylabel("DRAMA value")
# plt.title("Raw DRAMA Values")
# plt.grid(True)
# plt.legend()
# # plt.savefig("try.png")
# # -------------------------------
# # Plot both CDFs
# # -------------------------------
# plt.figure()
# plot_cdf(drama1, "User gives out the family IDs")
# plot_cdf(drama2, "First come first serve")

# plt.xlabel("DRAMA")
# plt.ylabel("CDF")
# plt.title("CDF of DRAMA")
# plt.grid(True)
# plt.legend()
# plt.savefig("../../results/drama_family_fcfs.png")
