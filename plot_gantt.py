import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_flowshop_with_workers(processing_times, order, C, group_assign=None,
                               worker_capacities=None, z=None, time_scale=1.0):
    """
    Plots:
      1. Flow shop Gantt chart (job-machine schedule)
      2. Optional activity heatmap for z[j, t] (job activity over time)

    Args:
        processing_times: ndarray (num_jobs, num_machines)
        order: list[int] – job indices in scheduled order
        C: dict {(j,m): completion time} or ndarray [job,m]
        group_assign: ndarray (num_jobs, num_machines) – worker group per job-machine
        worker_capacities: list[int] – capacity of each group
        z: ndarray or dict (optional) – binary activity variable z[j,m,t]
        time_scale: float – scaling for time unit (optional)
    """

    num_jobs, num_machines = processing_times.shape
    fig, axes = plt.subplots(2 if z is not None else 1, 1, figsize=(12, 8 if z is not None else 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]

    colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    get_C = (lambda j, m: C[j, m]) if isinstance(C, dict) else (lambda j, m: C[j, m])

    # --- Define group colors if provided ---
    if group_assign is not None:
        unique_groups = np.unique(group_assign[~np.isnan(group_assign)]) if np.any(group_assign) else []
        unique_groups = unique_groups.astype(int)
        group_colors = plt.cm.Set2(np.linspace(0, 1, max(unique_groups) + 1)) if len(unique_groups) > 0 else None
    else:
        group_colors = None

    # --- GANTT CHART ---
    for m in range(num_machines):
        for idx, job_idx in enumerate(order):
            # Start and finish times
            if m == 0:
                start_time = get_C(job_idx, 0) - processing_times[job_idx, 0]
            else:
                start_time = get_C(job_idx, m - 1)
            finish_time = get_C(job_idx, m)

            # Determine group for this job-machine
            g_label = None
            bar_color = colors[job_idx % len(colors)]
            if group_assign is not None:
                if group_assign.ndim == 2:
                    g_label = int(group_assign[job_idx, m])
                    if group_colors is not None:
                        bar_color = group_colors[g_label % len(group_colors)]
                elif group_assign.ndim == 1:
                    g_label = int(group_assign[job_idx])
                    if group_colors is not None:
                        bar_color = group_colors[g_label % len(group_colors)]

            # Draw bar
            ax.barh(
                m,
                finish_time - start_time,
                left=start_time,
                color=bar_color,
                edgecolor="black",
                height=0.6,
            )

            # Label inside the bar
            label = f"J{job_idx}" + (f"(G{g_label})" if g_label is not None else "")
            ax.text(
                start_time + (finish_time - start_time) / 2,
                m,
                label,
                va="center",
                ha="center",
                fontsize=8,
                color="black",
            )

    # --- Axes and formatting ---
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Flow Shop Schedule with Per-Stage Worker Groups (Gantt Chart)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # --- Legend for worker groups ---
    if group_assign is not None and worker_capacities is not None and group_colors is not None:
        unique_groups = sorted(set(int(g) for g in np.unique(group_assign)))
        legend_patches = [
            mpatches.Patch(color=group_colors[g % len(group_colors)],
                           label=f"Group {g} (Cap={worker_capacities[g]})")
            for g in unique_groups
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # --- Optional z-activity plot ---
    if z is not None and len(axes) > 1:
        ax2 = axes[1]
        # Aggregate activity across machines
        activity = np.sum(z, axis=1) if isinstance(z, np.ndarray) else None
        if activity is not None:
            im = ax2.imshow(activity, aspect='auto', cmap='Greens')
            ax2.set_title("Job Activity Heatmap")
            ax2.set_xlabel("Time (t)")
            ax2.set_ylabel("Job index")
            plt.colorbar(im, ax=ax2, label="Active on any machine")

    plt.tight_layout()
    plt.show()




def plot_discrete_activity_gantt(z_values, processing_times, group_assign=None, worker_capacities=None):
    """
    Visualizes discrete activity variables z[j, m, t] as a Gantt-like chart,
    showing which worker group (Gx) each job seizes on each machine.

    Args:
        z_values: np.ndarray [num_jobs, num_machines, T], binary activity indicators.
        processing_times: np.ndarray [num_jobs, num_machines].
        group_assign: np.ndarray [num_jobs, num_machines] (optional) – worker group assignment.
        worker_capacities: list[int] (optional) – worker group capacities.
    """
    num_jobs, num_machines, T = z_values.shape
    fig, ax = plt.subplots(figsize=(14, 6))

    # --- Define color scheme ---
    if group_assign is not None:
        group_assign = np.array(group_assign, dtype=int)
        unique_groups = sorted(set(int(g) for g in np.unique(group_assign)))
        group_colors = plt.cm.Set2(np.linspace(0, 1, max(unique_groups) + 1))
        color_by = "group"
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
        color_by = "job"

    # --- Plot each job-machine-time block ---
    for m in range(num_machines):
        for j in range(num_jobs):
            active_times = np.where(z_values[j, m, :] > 0.5)[0]
            if len(active_times) == 0:
                continue

            # Find continuous time segments
            segments = np.split(active_times, np.where(np.diff(active_times) != 1)[0] + 1)

            for seg in segments:
                start_t = seg[0]
                duration = len(seg)

                if group_assign is not None:
                    g = int(group_assign[j, m])
                    color = group_colors[g % len(group_colors)]
                    label = f"J{j}"
                else:
                    color = colors[j % len(colors)]
                    label = f"J{j}"

                # Draw Gantt bar
                ax.barh(
                    y=m,
                    width=duration,
                    left=start_t,
                    height=0.6,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    align="center",
                )

                # Add text label centered on the bar
                ax.text(
                    start_t + duration / 2,
                    m,
                    label,
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="black" if np.mean(color[:3]) > 0.6 else "white",  # contrast-aware text
                    fontweight="bold",
                )

    # --- Axes formatting ---
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)])
    ax.set_xlabel("Discrete Time")
    ax.set_ylabel("Machine")
    ax.set_title("Discrete-Time Flow Shop Activity (Job + Worker Group per Machine)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # --- Legend ---
    if group_assign is not None:
        unique_groups = sorted(set(int(g) for g in np.unique(group_assign)))
        legend_patches = [
            mpatches.Patch(
                color=group_colors[g % len(group_colors)],
                label=f"Group {g}" + (f" (Cap={worker_capacities[g]})" if worker_capacities is not None else "")
            )
            for g in unique_groups
        ]
    else:
        legend_patches = [
            mpatches.Patch(color=colors[j % len(colors)], label=f"Job {j}") for j in range(num_jobs)
        ]

    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()
