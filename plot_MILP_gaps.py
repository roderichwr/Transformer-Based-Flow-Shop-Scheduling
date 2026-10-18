import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# ---------------------------------------
# SETTINGS
# ---------------------------------------

prefix_lens = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
data_dir = "results"

# ---------------------------------------
# LOAD COMPLETION GAPS (LEFT PLOT)
# ---------------------------------------

completion_gap_data = []
valid_prefixes = []

for p in prefix_lens:

    filepath = os.path.join(data_dir, f"makespan_prefix{p}.npz")

    if not os.path.exists(filepath):
        continue

    data = np.load(filepath)

    ub = data["milp"]
    lb = data["milp_lb"]

    valid_mask = ub > 0
    gaps = np.zeros_like(ub, dtype=float)
    gaps[valid_mask] = (ub[valid_mask] - lb[valid_mask]) / ub[valid_mask] * 100

    completion_gap_data.append(gaps)
    valid_prefixes.append(p)


# ---------------------------------------
# LOAD TRAINING DATA GAPS (RIGHT PLOT)
# ---------------------------------------

train_makespan = np.load("makespans_milp_3.npy")
train_bounds   = np.load("bounds_milp_3.npy")

valid_mask = train_makespan > 0
train_gaps = np.zeros_like(train_makespan, dtype=float)
train_gaps[valid_mask] = (
    (train_makespan[valid_mask] - train_bounds[valid_mask])
    / train_makespan[valid_mask] * 100
)


# ---------------------------------------
# PLOT
# ---------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---------------------------------------
# LEFT: COMPLETION TASK (BOXPLOTS)
# ---------------------------------------

ax = axes[1]

positions = np.arange(len(valid_prefixes))
box_width = 0.35

box = ax.boxplot(
    completion_gap_data,
    positions=positions,
    widths=box_width,
    patch_artist=False,
    showfliers=False
)

for element in ['boxes', 'whiskers', 'caps', 'medians']:
    for item in box[element]:
        item.set(color='black', linewidth=1.2)

# Mean lines
for i, gaps in enumerate(completion_gap_data):
    mean_val = np.mean(gaps)

    ax.hlines(
        mean_val,
        positions[i] - box_width / 2,
        positions[i] + box_width / 2,
        colors='black',
        
        linewidth=1.5,
        linestyles=(0, (1.5, 2)),
    )

# Scatter points
for i, gaps in enumerate(completion_gap_data):
    jitter = np.random.normal(0, 0.03, size=len(gaps))
    ax.scatter(
        np.full_like(gaps, positions[i]) + jitter,
        gaps,
        s=15,
        facecolors='none',
        edgecolors='black',
        linewidths=0.6,
        alpha=0.6
    )

ax.set_xticks(positions)
ax.set_xticklabels(valid_prefixes)
ax.set_title("MILP completion")
ax.set_xlabel("Prefix length")
ax.set_ylabel("Optimality gap (%)")
ax.set_ylim(bottom=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)


# ---------------------------------------
# RIGHT: TRAINING DATA (VIOLIN)
# ---------------------------------------

ax = axes[0]

positions = [0]

violin = ax.violinplot(
    [train_gaps],
    positions=positions,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

# Style violin (black outline, no fill)
for body in violin['bodies']:
    body.set_facecolor('none')
    body.set_edgecolor('black')
    body.set_linewidth(1.2)
    body.set_alpha(1)

# Add median (solid line)
median_val = np.median(train_gaps)
ax.hlines(
    median_val,
    positions[0] - 0.1,
    positions[0] + 0.1,
    colors='black',
    linewidth=1.5
)

# Add mean (dashed line)
mean_val = np.mean(train_gaps)
ax.hlines(
    mean_val,
    positions[0] - 0.1,
    positions[0] + 0.1,
    colors='black',
    linestyle='--',
    linewidth=1.5
)

ax.set_xticks(positions)
ax.set_xticklabels(["Full-length schedule"])
ax.set_title("MILP training")
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(bottom=0)

# --- Add datapoints (light, subtle) ---
jitter = np.random.normal(0, 0.015, size=len(train_gaps))

ax.scatter(
    np.full_like(train_gaps, positions[0]) + jitter,
    train_gaps,
    s=10,
    facecolors='none',
    edgecolors='black',
    linewidths=0.5,
    alpha=0.25  # <-- key: make them subtle
)

# ---------------------------------------
# SHARED LEGEND
# ---------------------------------------

legend_elements = [
    Line2D([0], [0], color='black', lw=1.5, label='Median'),
    Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Mean'),
    Line2D([0], [0], marker='o', color='black',
           markerfacecolor='white', markersize=5,
           linestyle='None', label='Instances')
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=True
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("milp_gap_comparison_violin.pdf", bbox_inches="tight")
plt.show()

# ---------------------------------------
# TRAINING DATA: SHARE OF INSTANCES WITHIN GAP THRESHOLDS + LATEX TABLE
# ---------------------------------------
GAP_THRESHOLDS = [1, 10]   # in percent


def tex_int(n):
    return f"{n:,}".replace(",", "{,}")   # 40000 -> 40{,}000


train_ms = np.load("makespans_milp_3.npy")
train_lb = np.load("bounds_milp_3.npy")

has_incumbent = train_ms > 0
n_total = len(train_ms)
train_gap = np.full(n_total, np.inf)   # no incumbent -> counted as outside every threshold
train_gap[has_incumbent] = (train_ms[has_incumbent] - train_lb[has_incumbent]) / train_ms[has_incumbent] * 100

rows = []
for thr in GAP_THRESHOLDS:
    n_within = int(np.sum(train_gap <= thr))
    rows.append((thr, n_within, 100 * n_within / n_total))

print(f"\n--- Training data: optimality gap thresholds (n = {n_total}) ---")
if (~has_incumbent).any():
    print(f"{(~has_incumbent).sum()} instances without incumbent (counted as outside all thresholds)")
for thr, n_within, share in rows:
    print(f"Gap <= {thr:>3}%: {n_within:>6} instances ({share:.1f}%)")

print("\n% --- LATEX TABLE ---")
print("\\begin{table}[htbp]")
print("\\centering")
print(f"\\caption{{Share of the {tex_int(n_total)} MILP-generated training schedules whose "
      f"optimality gap at termination does not exceed the given threshold.}}")
print("\\label{tab:training_gap_thresholds}")
print("\\begin{tabular}{lrr}")
print("\\toprule")
print("Optimality gap & Instances & Share (\\%) \\\\")
print("\\midrule")
for thr, n_within, share in rows:
    print(f"$\\leq {thr}\\,\\%$ & {tex_int(n_within)} & {share:.1f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")