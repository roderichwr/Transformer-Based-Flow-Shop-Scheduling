

import numpy as np
import matplotlib.pyplot as plt
import os

iteration = 1  # change this to match your file
Xf = f"permutations_with_workers_{iteration}.npy"

# Load the array
X = np.load(Xf, allow_pickle=True)

# data_dir = "results"
# prefix_lens = [4,5,6,7,8,9,10,11,12,13,14,15,16]

# for p in prefix_lens:
#     data = np.load(os.path.join(data_dir, f"makespan_prefix{p}.npz"))

#     tf = data["transformer"]
#     heur = data["heuristic"]
#     milp = data["milp"]
#     rand = data["rand_ms"]
#     ga = data["ga_ms"]
#     all_vals = np.concatenate([tf, heur, milp,rand,ga])
#     x_min, x_max = all_vals.min(), all_vals.max()

#     fig, axes = plt.subplots(1, 5, figsize=(12, 4), sharey=True)

#     for ax, vals, title in zip(
#         axes,
#         [tf, heur, milp,rand,ga],
#         ["Transformer", "Heuristic (NEH)", "MILP", "Random", "GA"]
#     ):
#         ax.hist(vals, bins=50, density=True)
#         ax.set_xlim(x_min, x_max)
#         ax.set_title(title)
#         ax.set_xlabel("Makespan")

#         mean = np.mean(vals)
#         median = np.median(vals)

#         ax.axvline(mean, linestyle="--", linewidth=1, label=f"Mean: {mean:.1f}")
#         ax.axvline(median, linestyle=":", linewidth=1, label=f"Median: {median:.1f}")
#         ax.legend(fontsize=8)

#     axes[0].set_ylabel("Density")
#     #fig.suptitle(f"Makespan Distributions (prefix_len = {p})")
#     plt.tight_layout()
#     #plt.show()

# Example prefix lengths
prefix_lens = [6,7,8,9,10,11,12,13,14,15,16]
# Add IG method
methods = {
    "Transformer": "transformer",
    "IG": "ig_ms",
    "NEH": "heuristic",
    "GA": "ga_ms",
    "MILP": "milp",
    "Random": "rand_ms"
}

markers = {
    "Transformer": "P",
    "IG": "o",
    "NEH": "s",
    "GA": "D",
    "MILP": "^",
    "Random": "v"  
}

# Original data_dir for MILP etc.
data_dir = "results_MILP_600s"
ig_data_dir = "results_ig"  # if separate folder for IG method is used (legacy) otherwise, online data_dir is needed. Adjust code accoringly

# Collect mean and median across prefixes
means = {m: [] for m in methods}
medians = {m: [] for m in methods}
valid_prefixes = []

for p in prefix_lens:
    filepath = os.path.join(data_dir, f"makespan_prefix{p}.npz")
    
    if not os.path.exists(filepath):
        print(f"Missing file for prefix {p}")
        continue

    data = np.load(filepath)

    # Load IG data separately
    ig_filepath = os.path.join(ig_data_dir, f"makespan_prefix{p}.npz")
    if os.path.exists(ig_filepath):
        ig_data = np.load(ig_filepath)
    else:
        ig_data = {"ig_ms": np.array([])}  # empty if missing

    for method_name, key in methods.items():
        if method_name == "IG":
            vals = ig_data.get("ig_ms", np.array([]))
        else:
            vals = data[key]
        means[method_name].append(np.mean(vals))
        medians[method_name].append(np.median(vals))

    valid_prefixes.append(p)

# ---------------------------------------
# Plot side-by-side
# ---------------------------------------
plt.rcParams.update({
    "font.size": 12,          # base size
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# ---------- MEAN ----------
for method, vals in means.items():
    axes[0].scatter(
        valid_prefixes,
        vals,
        marker=markers[method],
        facecolors='none',
        edgecolors='black',
        linewidths=1,
        s=60,
        label=method
    )

axes[0].set_title("Mean")
axes[0].set_xlabel("Prefix length")
axes[0].set_ylabel("Makespan")
axes[0].grid(True, linewidth=0.5, alpha=0.5)

# ---------- MEDIAN ----------
for method, vals in medians.items():
    axes[1].scatter(
        valid_prefixes,
        vals,
        marker=markers[method],
        facecolors='none',
        edgecolors='black',
        linewidths=1,
        s=60
    )

axes[1].set_title("Median")
axes[1].set_xlabel("Prefix length")
axes[1].grid(True, linewidth=0.5, alpha=0.5)


# ---------- Shared legend below ----------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),  # move legend upward/downward
    ncol=len(labels),
    frameon=True
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("makespan_summary.pdf", bbox_inches="tight")
plt.show()

# ---------------------------------------
# Average % gap: Transformer vs NEH & IG
# ---------------------------------------

target_prefixes = [6,7,8,9,10]

mean_gaps_neh = []
mean_gaps_ig = []

median_gaps_neh = []
median_gaps_ig = []

for p in target_prefixes:
    if p not in valid_prefixes:
        continue

    i = valid_prefixes.index(p)

    # --- Mean gaps ---
    tf_mean = means["Transformer"][i]
    neh_mean = means["NEH"][i]
    ig_mean = means["IG"][i]

    gap_mean_neh = (tf_mean - neh_mean) / neh_mean * 100
    gap_mean_ig = (tf_mean - ig_mean) / ig_mean * 100

    mean_gaps_neh.append(gap_mean_neh)
    mean_gaps_ig.append(gap_mean_ig)

    # --- Median gaps ---
    tf_med = medians["Transformer"][i]
    neh_med = medians["NEH"][i]
    ig_med = medians["IG"][i]

    gap_med_neh = (tf_med - neh_med) / neh_med * 100
    gap_med_ig = (tf_med - ig_med) / ig_med * 100

    median_gaps_neh.append(gap_med_neh)
    median_gaps_ig.append(gap_med_ig)

# Averages
avg_mean_gap_neh = np.mean(mean_gaps_neh)
avg_mean_gap_ig = np.mean(mean_gaps_ig)

avg_median_gap_neh = np.mean(median_gaps_neh)
avg_median_gap_ig = np.mean(median_gaps_ig)

print("\n--- Transformer Gap (Prefix 6–10) ---")
print(f"Mean vs NEH: {avg_mean_gap_neh:.2f}%")
print(f"Mean vs IG:  {avg_mean_gap_ig:.2f}%")
print(f"Median vs NEH: {avg_median_gap_neh:.2f}%")
print(f"Median vs IG:  {avg_median_gap_ig:.2f}%")
