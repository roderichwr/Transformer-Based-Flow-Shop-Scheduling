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


# ==============================================================================
# COMPARISON: results vs results_it2
# Two side-by-side plots:
#   Left:  MILP optimality gap distributions (pooled across all prefix lengths)
#   Right: Upper-bound (primal makespan) difference per instance
#
# Narrative: wide gap differences between the two runs are driven by weak dual
# bounds (different LB from different solves), while the primal solutions
# (the actual schedules found) are close — illustrating that gap is an
# unreliable quality indicator when dual bounds are loose.
# ==============================================================================

data_dir_2 = "results_it2"

gaps_it1, gaps_it2 = [], []
ub_it1_all, ub_it2_all = [], []
ub_diffs = []          # it2_ub - it1_ub per matched instance

common_prefixes = []

for p in prefix_lens:
    f1 = os.path.join(data_dir,   f"makespan_prefix{p}.npz")
    f2 = os.path.join(data_dir_2, f"makespan_prefix{p}.npz")
    if not os.path.exists(f1) or not os.path.exists(f2):
        continue

    d1 = np.load(f1)
    d2 = np.load(f2)

    ub1, lb1 = np.asarray(d1["milp"], float), np.asarray(d1["milp_lb"], float)
    ub2, lb2 = np.asarray(d2["milp"], float), np.asarray(d2["milp_lb"], float)

    # Only use instances where both runs found a valid incumbent
    valid = (ub1 > 0) & (ub2 > 0) & np.isfinite(ub1) & np.isfinite(ub2)

    if valid.sum() == 0:
        continue

    g1 = (ub1[valid] - lb1[valid]) / ub1[valid] * 100
    g2 = (ub2[valid] - lb2[valid]) / ub2[valid] * 100

    gaps_it1.extend(g1)
    gaps_it2.extend(g2)
    ub_diffs.extend(ub2[valid] - ub1[valid])   # positive = it2 primal is worse
    ub_it1_all.extend(ub1[valid])
    ub_it2_all.extend(ub2[valid])
    common_prefixes.append(p)

gaps_it1  = np.array(gaps_it1)
gaps_it2  = np.array(gaps_it2)
ub_diffs  = np.array(ub_diffs)
ub_it1_all = np.array(ub_it1_all)
ub_it2_all = np.array(ub_it2_all)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

# -------------------------------------------------------
# LEFT: Optimality gap distributions (it1 vs it2)
# Side-by-side violins
# -------------------------------------------------------
ax = axes2[0]

vp1 = ax.violinplot([gaps_it1], positions=[0],
                    showmeans=False, showmedians=False, showextrema=False)
vp2 = ax.violinplot([gaps_it2], positions=[1],
                    showmeans=False, showmedians=False, showextrema=False)

for vp, ls in [(vp1, '-'), (vp2, '--')]:
    for body in vp['bodies']:
        body.set_facecolor('none')
        body.set_edgecolor('black')
        body.set_linewidth(1.2)
        body.set_linestyle(ls)
        body.set_alpha(1)

# Median and mean markers
for pos, gaps, ls in [(0, gaps_it1, '-'), (1, gaps_it2, '--')]:
    ax.hlines(np.median(gaps), pos - 0.12, pos + 0.12, colors='black',
              linewidth=1.5, linestyles=ls)
    ax.hlines(np.mean(gaps),   pos - 0.12, pos + 0.12, colors='black',
              linewidth=1.5, linestyles=(0, (1.5, 2)))

# jittered scatter (subtle)
rng = np.random.default_rng(0)
for pos, gaps in [(0, gaps_it1), (1, gaps_it2)]:
    jitter = rng.normal(0, 0.025, size=len(gaps))
    ax.scatter(pos + jitter, gaps, s=8, facecolors='none',
               edgecolors='black', linewidths=0.4, alpha=0.2)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Run 1", "Run 2"])
ax.set_ylabel("Optimality gap (%)")
ax.set_title("MILP optimality gap distribution\n(pooled across all prefix lengths)")
ax.set_ylim(bottom=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# annotation: median gap values
for pos, gaps, ha in [(0, gaps_it1, 'right'), (1, gaps_it2, 'left')]:
    ax.annotate(f"median {np.median(gaps):.1f}%\nmean {np.mean(gaps):.1f}%",
                xy=(pos, ax.get_ylim()[1] * 0.97),
                ha=ha, va='top', fontsize=10,
                xytext=(-8 if ha == 'right' else 8, 0),
                textcoords='offset points')

# -------------------------------------------------------
# RIGHT: Upper-bound (primal) difference it2 - it1
# Violin + scatter; zero line = identical primal solutions
# -------------------------------------------------------
ax = axes2[1]

vp = ax.violinplot([ub_diffs], positions=[0],
                   showmeans=False, showmedians=False, showextrema=False)
for body in vp['bodies']:
    body.set_facecolor('none')
    body.set_edgecolor('black')
    body.set_linewidth(1.2)
    body.set_alpha(1)

ax.hlines(np.median(ub_diffs), -0.12, 0.12, colors='black',
          linewidth=1.5, linestyles='-')
ax.hlines(np.mean(ub_diffs),   -0.12, 0.12, colors='black',
          linewidth=1.5, linestyles=(0, (1.5, 2)))

jitter = rng.normal(0, 0.02, size=len(ub_diffs))
ax.scatter(jitter, ub_diffs, s=10, facecolors='none',
           edgecolors='black', linewidths=0.5, alpha=0.25)

# zero line for reference
ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
ax.annotate("identical primal", xy=(0.52, 0), xycoords=('axes fraction', 'data'),
            va='bottom', ha='left', fontsize=9, color='gray')

ax.set_xticks([0])
ax.set_xticklabels(["Run 2 − Run 1"])
ax.set_ylabel("Makespan difference (Run 2 − Run 1)")
ax.set_title("Primal solution difference\n(upper bounds, matched instances)")
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

pct_close = np.mean(np.abs(ub_diffs) / ub_it1_all * 100)
ax.annotate(f"mean |Δ| / UB$_1$ = {pct_close:.1f}%",
            xy=(0.05, 0.97), xycoords='axes fraction',
            va='top', ha='left', fontsize=10)

# -------------------------------------------------------
# Shared legend for the comparison figure
# -------------------------------------------------------
legend_elements2 = [
    Line2D([0], [0], color='black', lw=1.5, linestyle='-',  label='Median'),
    Line2D([0], [0], color='black', lw=1.5, linestyle=(0, (1.5, 2)), label='Mean'),
    Line2D([0], [0], color='black', lw=1.2, linestyle='-',  label='Run 1 (violin)'),
    Line2D([0], [0], color='black', lw=1.2, linestyle='--', label='Run 2 (violin)'),
    Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
           markersize=5, linestyle='None', label='Instances'),
]

fig2.legend(handles=legend_elements2, loc="lower center",
            bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("milp_gap_comparison_runs.pdf", bbox_inches="tight")
plt.show()

# Print summary
n = len(ub_diffs)
print(f"\n--- Run comparison ({n} matched instances across prefixes {common_prefixes[0]}–{common_prefixes[-1]}) ---")
print(f"Gap  Run 1: median {np.median(gaps_it1):.1f}%  mean {np.mean(gaps_it1):.1f}%")
print(f"Gap  Run 2: median {np.median(gaps_it2):.1f}%  mean {np.mean(gaps_it2):.1f}%")
print(f"Primal diff (UB2-UB1): median {np.median(ub_diffs):+.1f}  mean {np.mean(ub_diffs):+.1f}")
print(f"Mean |primal diff| as % of Run 1 UB: {pct_close:.2f}%")
print("(Large gap difference with small primal difference confirms weak dual bounds.)")