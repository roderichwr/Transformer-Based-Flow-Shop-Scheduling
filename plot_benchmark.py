import numpy as np
import matplotlib.pyplot as plt
import os

iteration = 2  # change this to match your file
Xf = f"permutations_with_workers_{iteration}.npy"

# Load the array
X = np.load(Xf, allow_pickle=True)


# Example prefix lengths
prefix_lens = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
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

# Symbol-based (black/white, print-safe) vs color-based boxplot coding
USE_SYMBOL_CODING = False

hatches = {
    "Transformer": "///",
    "IG": "\\\\\\",
    "NEH": "xxx",
    "GA": "...",
    "MILP": "+++",
    "Random": "ooo"
}

# Original data_dir for MILP etc.
data_dir = "results"

# Runtime arrays stored alongside the makespans in the npz files
time_keys = {
    "Transformer": "time_transformer",
    "IG": "time_ig",
    "NEH": "time_heuristic",
    "GA": "time_ga",
    "MILP": "time_milp",
    "Random": "time_rand"
}

# Collect mean and median across prefixes + raw distributions per prefix
means = {m: [] for m in methods}
medians = {m: [] for m in methods}
raw = {m: {} for m in methods}        # raw[method][prefix] = makespans of the 50 instances
raw_time = {m: {} for m in methods}   # raw_time[method][prefix] = runtimes of the 50 instances
valid_prefixes = []

for p in prefix_lens:
    filepath = os.path.join(data_dir, f"makespan_prefix{p}.npz")
    
    if not os.path.exists(filepath):
        print(f"Missing file for prefix {p}")
        continue

    data = np.load(filepath)

    for method_name, key in methods.items():
        vals = np.asarray(data[key], dtype=float)
        vals = vals[~np.isnan(vals)]  # MILP entries can be NaN (no incumbent)
        raw[method_name][p] = vals
        means[method_name].append(np.mean(vals))
        medians[method_name].append(np.median(vals))

        tkey = time_keys[method_name]
        if tkey in data.files:
            raw_time[method_name][p] = np.asarray(data[tkey], dtype=float)

    valid_prefixes.append(p)


# ---------------------------------------
# Grouped boxplots: distributions across the 50 repetitions
# ---------------------------------------
def grouped_boxplot(raw_dict, ylabel, title, outfile, logy=False):
    """
    Two stacked panels (prefix lengths 1-8 and 9-16) so that at \\textwidth the
    individual boxes remain wide enough to read.

    USE_SYMBOL_CODING = True  -> black/white, print-safe coding: light gray
        boxes with a distinct hatch pattern per method AND the method's marker
        symbol (same symbols as in the summary scatter plot) drawn on the
        median. The marker keeps methods identifiable even where a tight
        distribution collapses the box to a line (e.g. runtimes, log axis).
    USE_SYMBOL_CODING = False -> color coding (method-colored box faces,
        edges, whiskers, caps, and fliers).
    """
    method_names = list(methods.keys())
    n_m = len(method_names)
    group_width = n_m + 1.6
    colors = plt.get_cmap("tab10")

    rows = [valid_prefixes[:len(valid_prefixes)//2],
            valid_prefixes[len(valid_prefixes)//2:]]
    rows = [r for r in rows if r]

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 4.0 * len(rows)),
                             sharey=True)
    if len(rows) == 1:
        axes = [axes]

    for ax, row_prefixes in zip(axes, rows):
        for mi, m in enumerate(method_names):
            c = "black" if USE_SYMBOL_CODING else colors(mi)
            positions, series = [], []
            for pi, p in enumerate(row_prefixes):
                if p in raw_dict[m] and len(raw_dict[m][p]) > 0:
                    positions.append(pi * group_width + mi)
                    series.append(raw_dict[m][p])
            if not series:
                continue
            bp = ax.boxplot(
                series,
                positions=positions,
                widths=0.8,
                patch_artist=True,
                showfliers=False,
                flierprops=dict(marker=".", markersize=3.5,
                                markerfacecolor=c, markeredgecolor=c, alpha=0.6),
                medianprops=dict(color="black", linewidth=1.4),
                whiskerprops=dict(color=c, linewidth=1.1),
                capprops=dict(color=c, linewidth=1.1),
                boxprops=dict(edgecolor=c, linewidth=1.1)
            )
            for box in bp["boxes"]:
                if USE_SYMBOL_CODING:
                    box.set_facecolor("0.92")
                    box.set_hatch(hatches[m])
                else:
                    box.set_facecolor(colors(mi))
                    box.set_alpha(0.45)

            if USE_SYMBOL_CODING:
                med = [np.median(s) for s in series]
                ax.plot(positions, med, linestyle="none",
                        marker=markers[m], markersize=6,
                        markerfacecolor="white", markeredgecolor="black",
                        markeredgewidth=1.1, zorder=5)

        centers = [pi * group_width + (n_m - 1) / 2 for pi in range(len(row_prefixes))]
        ax.set_xticks(centers)
        ax.set_xticklabels([str(p) for p in row_prefixes])
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="both", labelsize=12)

        for pi in range(1, len(row_prefixes)):
            ax.axvline(pi * group_width - 1.3, color="gray",
                       linewidth=0.5, alpha=0.45)
        ax.set_xlim(-1.3, (len(row_prefixes) - 1) * group_width + n_m + 0.3)

    axes[-1].set_xlabel("Prefix length", fontsize=13)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    if USE_SYMBOL_CODING:
        handles = [
            (Patch(facecolor="0.92", edgecolor="black", hatch=hatches[m]),
             Line2D([], [], linestyle="none", marker=markers[m], markersize=6,
                    markerfacecolor="white", markeredgecolor="black"))
            for m in method_names
        ]
        fig.legend(handles, method_names, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=n_m, frameon=True,
                   fontsize=12, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        handles = [Patch(facecolor=colors(mi), alpha=0.45, edgecolor=colors(mi),
                         linewidth=1.4, label=m)
                   for mi, m in enumerate(method_names)]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=n_m, frameon=True, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


grouped_boxplot(
    raw,
    ylabel="Makespan",
    title="Makespan distribution across the 50 instances per prefix length",
    outfile="makespan_boxplots.pdf"
)

if any(raw_time[m] for m in methods):
    grouped_boxplot(
        raw_time,
        ylabel="Runtime [s]",
        title="Runtime distribution across the 50 instances per prefix length",
        outfile="runtime_boxplots.pdf",
        logy=True
    )
else:
    print("No time_* arrays found in the result files — runtime plot skipped.")

# ---------------------------------------
# Plot mean only
# ---------------------------------------
plt.rcParams.update({
    "font.size": 12,          # base size
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

fig, ax = plt.subplots(figsize=(6, 6))

# ---------- MEAN ----------
for method, vals in means.items():
    ax.scatter(
        valid_prefixes,
        vals,
        marker=markers[method],
        facecolors='none',
        edgecolors='black',
        linewidths=1,
        s=60,
        label=method
    )

ax.set_xlabel("Prefix length")
ax.set_ylabel("Mean makespan")
ax.grid(True, linewidth=0.5, alpha=0.5)

# ---------- Legend below ----------
handles, labels = ax.get_legend_handles_labels()

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
# Average % gap to MILP: Transformer & GA
# ---------------------------------------

target_prefixes = [9,10,11,12,13,14,15,16]  # only the larger prefixes

mean_gaps_tf = []
mean_gaps_ga = []

for p in target_prefixes:
    if p not in valid_prefixes:
        continue

    i = valid_prefixes.index(p)

    milp_mean = means["MILP"][i]
    tf_mean = means["Transformer"][i]
    ga_mean = means["NEH"][i]

    mean_gaps_tf.append((tf_mean - milp_mean) / milp_mean * 100)
    mean_gaps_ga.append((ga_mean - milp_mean) / milp_mean * 100)

avg_mean_gap_tf = np.mean(mean_gaps_tf)
avg_mean_gap_ga = np.mean(mean_gaps_ga)

print("\n--- Average mean gap to MILP (Prefix 1-8) ---")
print(f"Transformer vs MILP: {avg_mean_gap_tf:+.2f}%")
print(f"GA vs MILP:          {avg_mean_gap_ga:+.2f}%")

# ---------------------------------------
# Average % gap: Transformer vs Random (all prefix lengths)
# ---------------------------------------

target_prefixes = valid_prefixes  # all available prefix lengths

mean_gaps_tf_rand = []

for p in target_prefixes:
    if p not in valid_prefixes:
        continue

    i = valid_prefixes.index(p)

    tf_mean = means["Transformer"][i]
    rand_mean = means["Random"][i]

    mean_gaps_tf_rand.append((tf_mean - rand_mean) / rand_mean * 100)

avg_mean_gap_tf_rand = np.mean(mean_gaps_tf_rand)

print("\n--- Average mean gap: Transformer vs Random (all prefix lengths) ---")
print(f"Transformer vs Random: {avg_mean_gap_tf_rand:+.2f}%")


# ---------------------------------------
# Average runtime advantage: Transformer vs MILP (all prefix lengths)
# ---------------------------------------

tf_times_all = []
milp_times_all = []

for p in valid_prefixes:
    filepath = os.path.join(data_dir, f"makespan_prefix{p}.npz")
    data = np.load(filepath)

    tf_times_all.append(data[time_keys["Transformer"]])
    milp_times_all.append(data[time_keys["MILP"]])

tf_times_all = np.concatenate(tf_times_all)
milp_times_all = np.concatenate(milp_times_all)

avg_tf_time = np.mean(tf_times_all)
avg_milp_time = np.mean(milp_times_all)
speedup = avg_milp_time / avg_tf_time

print("\n--- Average runtime: Transformer vs MILP (all prefix lengths & instances) ---")
print(f"Transformer mean runtime: {avg_tf_time:.2f} s")
print(f"MILP mean runtime:        {avg_milp_time:.2f} s")
print(f"MILP / Transformer speedup factor: {speedup:.2f}x")
print(f"Absolute time saved per instance:  {avg_milp_time - avg_tf_time:.2f} s")