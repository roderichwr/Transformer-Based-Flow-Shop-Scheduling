import numpy as np
import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Pool, cpu_count
from functools import partial
import simpy
from collections import defaultdict
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np

sequences = np.load("permutations_with_workers_3.npy", allow_pickle=True)
print(len(sequences))
# ---------- Utility ----------
def deterministic_seed(base_seed: int) -> int:
    rng = np.random.default_rng(base_seed)
    return int(rng.integers(1e9, 1e12))


def generate_job_pool(pool_size: int, num_machines: int, low: int = 1, high: int = 100, seed: int = None):
    rng = np.random.default_rng(seed)
    return rng.integers(low, high + 1, size=(pool_size, num_machines))


def generate_worker_pool(num_groups: int, min_workers: int = 1, max_workers: int = 3, seed: int = None):
    rng = np.random.default_rng(seed)
    capacities = rng.integers(min_workers, max_workers + 1, size=num_groups)
    return capacities


def generate_job_eligibility(pool_size: int, num_groups: int, min_groups_per_job=1, max_groups_per_job=3, seed=None):
    rng = np.random.default_rng(seed)
    eligibility = np.zeros((pool_size, num_groups), dtype=int)
    for j in range(pool_size):
        n = rng.integers(min_groups_per_job, max_groups_per_job + 1)
        eligible_groups = rng.choice(num_groups, size=n, replace=False)
        eligibility[j, eligible_groups] = 1
    return eligibility



def build_event_stream(start_values, C_values, assign_values, round_decimals=6):
    import numpy as _np

    events = []
    is_dict_input = isinstance(start_values, dict)

    if is_dict_input:
        keys = list(start_values.keys())
        jobs = sorted({k[0] for k in keys})
        machines = sorted({k[1] for k in keys})
        for (j, m) in sorted(keys, key=lambda x: (x[0], x[1])):
            start_t = float(start_values[(j, m)])
            end_t = float(C_values[(j, m)])
            start_t = round(start_t, round_decimals)
            end_t = round(end_t, round_decimals)

            g = None
            if isinstance(assign_values, dict):
                if (j, m) in assign_values:
                    g = int(assign_values[(j, m)])
                elif j in assign_values:
                    g = int(assign_values[j])
            else:
                try:
                    g = int(assign_values[j, m])
                except Exception:
                    try:
                        g = int(assign_values[j])
                    except Exception:
                        g = 0
            if g is None:
                g = 0

            events.append(("SEIZE", j, m, g, start_t))
            events.append(("START", j, m, g, start_t))
            events.append(("END", j, m, g, end_t))
            events.append(("RELEASE", j, m, g, end_t))

    else:
        start_arr = _np.asarray(start_values)
        C_arr = _np.asarray(C_values)

        if start_arr.ndim != 2 or C_arr.ndim != 2:
            raise ValueError("start_values and C_values must be 2D arrays or dicts keyed by (job,machine)")

        num_jobs, num_machines = start_arr.shape
        for j in range(num_jobs):
            for m in range(num_machines):
                start_t = float(start_arr[j, m])
                end_t = float(C_arr[j, m])
                start_t = round(start_t, round_decimals)
                end_t = round(end_t, round_decimals)

                if _np.ndim(assign_values) == 2:
                    g = int(assign_values[j, m])
                elif _np.ndim(assign_values) == 1:
                    g = int(assign_values[j])
                else:
                    if isinstance(assign_values, dict):
                        if (j, m) in assign_values:
                            g = int(assign_values[(j, m)])
                        elif j in assign_values:
                            g = int(assign_values[j])
                        else:
                            g = 0
                    else:
                        g = 0

                events.append(("SEIZE", j, m, g, start_t))
                events.append(("START", j, m, g, start_t))
                events.append(("END", j, m, g, end_t))
                events.append(("RELEASE", j, m, g, end_t))

    event_order = {"RELEASE": 0, "END": 1, "SEIZE": 2, "START": 3}

    def sort_key(e):
        etype, job, machine, group, t = e
        return (float(t), event_order.get(etype, 99), int(machine), int(job))

    events.sort(key=sort_key)

    return events



def format_event_token(event):
    etype, j, m, g, t = event
    return f"{etype}[J{j},M{m},G{g}]"


def solve_flowshop_with_workers(processing_times_global, worker_capacities,
                                job_eligibility, selected_indices,
                                time_scale=1, verbose=False):

    num_jobs = len(selected_indices)
    num_machines = processing_times_global.shape[1]
    num_groups = len(worker_capacities)

    jobs = range(num_jobs)
    machines = range(num_machines)
    groups = range(num_groups)

    PT = np.zeros((num_jobs, num_machines))
    for j_local, j_global in enumerate(selected_indices):
        PT[j_local, :] = processing_times_global[j_global, :]

    T = int(np.ceil(np.sum(PT) / time_scale)) + 1

    model = gp.Model("flowshop_with_stage_workers")
    if not verbose:
        model.Params.OutputFlag = 0
        model.Params.MIPFocus = 1
        model.Params.TimeLimit = 1800
        model.Params.Threads = 2
        model.Params.NoRelHeurTime = 300

    assign = model.addVars(num_jobs, num_machines, num_groups, vtype=GRB.BINARY)
    C = model.addVars(num_jobs, num_machines, vtype=GRB.CONTINUOUS)
    makespan = model.addVar(vtype=GRB.CONTINUOUS)

    z = {}
    u = {}
    v = {}
    for j in jobs:
        for m in machines:
            for t in range(T):
                z[j, m, t] = model.addVar(vtype=GRB.BINARY)
                u[j, m, t] = model.addVar(vtype=GRB.BINARY)
                v[j, m, t] = model.addVar(vtype=GRB.BINARY)

    model.addConstrs(assign.sum(j, m, "*") == 1 for j in jobs for m in machines)

    for j in jobs:
        gidx = selected_indices[j]
        for m in machines:
            for g in groups:
                if job_eligibility[gidx, g] == 0:
                    assign[j, m, g].ub = 0

    for j in jobs:
        model.addConstr(C[j, 0] >= PT[j, 0])
        for m in range(1, num_machines):
            model.addConstr(C[j, m] >= C[j, m-1] + PT[j, m])

    bigM = T
    y = model.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY)

    for m in machines:
        for i in jobs:
            for j in jobs:
                if i == j:
                    continue
                model.addConstr(C[i, m] >= C[j, m] + PT[i, m] - bigM*(1 - y[i, j, m]))
                model.addConstr(C[j, m] >= C[i, m] + PT[j, m] - bigM*y[i, j, m])

    for j in jobs:
        model.addConstr(makespan >= C[j, num_machines - 1])

    max_proc = float(np.max(PT))
    bigM_local = T

    for j in jobs:
        for m in machines:
            proc = float(PT[j, m])
            for t in range(T):
                start_jm = C[j, m] - proc
                model.addConstr(start_jm <= t + bigM_local*(1 - u[j, m, t]))
                model.addConstr(start_jm >= (t+1) - bigM_local*u[j, m, t])
                model.addConstr(C[j, m] >= (t+1) - bigM_local*(1 - v[j, m, t]))
                model.addConstr(C[j, m] <= t + bigM_local*v[j, m, t])
                model.addConstr(z[j, m, t] >= u[j, m, t] + v[j, m, t] - 1)
                model.addConstr(z[j, m, t] <= u[j, m, t])
                model.addConstr(z[j, m, t] <= v[j, m, t])

    for g in groups:
        cap = int(worker_capacities[g])
        for t in range(T):
            lhs = gp.quicksum(assign[j, m, g] * z[j, m, t]
                               for j in jobs for m in machines)
            model.addConstr(lhs <= cap)

    model.setObjective(makespan, GRB.MINIMIZE)
    model.optimize()

    # ---------------------- NEW: feasibility check ----------------------
    if model.SolCount==0:
        return None
    # -------------------------------------------------------------------

    C_values = {}
    start_values = {}
    assign_values = {}
    z_values = {}

    for j_local in jobs:
        j_global = int(selected_indices[j_local])
        for m in machines:
            C_values[(j_global, m)] = C[j_local, m].X
            start_values[(j_global, m)] = C_values[(j_global, m)] - PT[j_local, m]
            for g in groups:
                if assign[j_local, m, g].X > 0.5:
                    assign_values[(j_global, m)] = g
                    break

            for t in range(T):
                z_values[(j_global, m, t)] = z[j_local, m, t].X

    makespan_value = makespan.X

    job0_starts = [(j_global, start_values[(j_global, 0)]) 
                   for j_global in selected_indices]
    order = [j for (j, _) in sorted(job0_starts, key=lambda x: x[1])]

    return order, C_values, start_values, assign_values, z_values, makespan_value, model.ObjBound



def _generate_single_instance(job_pool, worker_capacities, eligibility, num_jobs, num_machines, seed):
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(job_pool), size=num_jobs, replace=False)

    # Run MILP ------------------------------
    sol = solve_flowshop_with_workers(job_pool, worker_capacities, eligibility, selected_indices)
    if sol is None:
        return None       # <-- NEW: skip instance if infeasible
    ordered_jobs, C_values, start_values, group_assign, z_values, makespan_milp, bound_milp = sol
    # ---------------------------------------

    events = build_event_stream(start_values, C_values, group_assign)

    tokens = [format_event_token(e) for e in events]

    makespan_des, completion_times, log, start_events = simulate_flowshop_events_simpy(
        events, job_pool, worker_capacities, verbose=True
    )

    start_times, worker_assignments = extract_info_from_events(
        events, completion_times, job_pool
    )

    token_sequence = [[j, m, g] for (_, j, m, g, t) in start_events]
    return token_sequence, makespan_milp, makespan_des, bound_milp



def generate_training_data_with_workers(pool_size: int,
                                        num_instances: int,
                                        num_jobs: int,
                                        num_machines: int,
                                        num_groups: int,
                                        base_seed: int,
                                        save_every: int = 5000,
                                        iteration: int = 0,
                                        n_jobs=None,
                                        start_idx=0):

    rng = np.random.default_rng(deterministic_seed(base_seed))

    job_pool = generate_job_pool(pool_size, num_machines, seed=rng.integers(1e9))
    worker_capacities = generate_worker_pool(num_groups, seed=rng.integers(1e9))
    eligibility = generate_job_eligibility(pool_size, num_groups, seed=rng.integers(1e9))

    seeds = rng.integers(1e9, size=num_instances)

    if n_jobs is None:
        n_jobs = min(cpu_count(), 20)

    sequences = []
    makespans_milp = []
    makespans_des = []
    bounds_milp = []

    with Pool(processes=n_jobs) as pool:

        func = partial(_generate_single_instance,
                       job_pool,
                       worker_capacities,
                       eligibility,
                       num_jobs,
                       num_machines)

        for start in range(start_idx, num_instances, save_every):

            end = min(start + save_every, num_instances)
            chunk_seeds = seeds[start:end]

            print(f"\nProcessing instances {start} → {end}")

            raw_results = list(pool.starmap(func, [(s,) for s in chunk_seeds]))

            results = [r for r in raw_results if r is not None]

            for r in results:
                sequences.append(r[0])
                makespans_milp.append(r[1])
                makespans_des.append(r[2])
                bounds_milp.append(r[3])

            # ---------- SAVE CHECKPOINT ----------
            np.save(f"job_pool_{iteration}.npy", job_pool)
            np.save(f"worker_capacities_{iteration}.npy", worker_capacities)
            np.save(f"eligibility_{iteration}.npy", eligibility)

            np.save(f"permutations_with_workers_{iteration}.npy",
                    np.array(sequences, dtype=object))

            np.save(f"makespans_milp_{iteration}.npy",
                    np.array(makespans_milp))

            np.save(f"makespans_des_{iteration}.npy",
                    np.array(makespans_des))

            np.save(f"bounds_milp_{iteration}.npy",
                    np.array(bounds_milp))

            print(f"Checkpoint saved after {len(sequences)} instances")

    return sequences, job_pool, worker_capacities, eligibility, makespans_milp, makespans_des, bounds_milp



def simulate_flowshop_events_simpy(events,
                                   processing_times_global,
                                   worker_capacities,
                                   verbose=False):

    proc = {}

    if isinstance(processing_times_global, dict):
        for (j, m), pt in processing_times_global.items():
            proc[(j, m)] = float(pt)

        all_ms = sorted({m for (_, m) in processing_times_global.keys()})
        num_machines = max(all_ms) + 1

    else:
        arr = np.asarray(processing_times_global, dtype=float)
        num_jobs_global, num_machines = arr.shape

        for j in range(num_jobs_global):
            for m in range(num_machines):
                proc[(j, m)] = float(arr[j, m])

    start_events = [(etype, j, m, g, t)
                    for (etype, j, m, g, t) in events
                    if etype == "START"]

    machine_sequence = defaultdict(list)
    for step_index, (_, j, m, g, t) in enumerate(start_events):
        machine_sequence[m].append((step_index, j))

    machine_prev = {}
    for m, seq in machine_sequence.items():
        prev = None
        for _, j in seq:
            machine_prev[(j, m)] = prev
            prev = j

    env = simpy.Environment()
    num_groups = len(worker_capacities)

    worker_resources = [
        simpy.Resource(env, capacity=int(worker_capacities[g]))
        for g in range(num_groups)
    ]

    machine_resources = [
        simpy.Resource(env, capacity=1)
        for _ in range(num_machines)
    ]

    op_done = {key: env.event() for key in proc.keys()}

    completion_times = {}
    log = []

    def process_op(j_global, m, g, step_index):
        if (j_global, m - 1) in op_done and m > 0:
            if verbose:
                log.append(f"[{env.now:.4f}] J{j_global}-M{m} waiting job predecessor")
            yield op_done[(j_global, m - 1)]

        prev_j = machine_prev.get((j_global, m))
        if prev_j is not None:
            if verbose:
                log.append(f"[{env.now:.4f}] J{j_global}-M{m} waiting machine predecessor J{prev_j}")
            yield op_done[(prev_j, m)]

        if verbose:
            log.append(f"[{env.now:.4f}] J{j_global}-M{m} requesting worker G{g}")
        req_w = worker_resources[g].request()
        yield req_w

        if verbose:
            log.append(f"[{env.now:.4f}] J{j_global}-M{m} requesting machine {m}")
        req_m = machine_resources[m].request()
        yield req_m

        duration = proc[(j_global, m)]
        if duration < 0:
            raise RuntimeError(f"Negative PT for ({j_global},{m})")

        if verbose:
            log.append(f"[{env.now:.4f}] J{j_global}-M{m} START processing; PT={duration}")

        yield env.timeout(duration)
        end_time = env.now

        machine_resources[m].release(req_m)
        worker_resources[g].release(req_w)

        completion_times[(j_global, m)] = end_time

        if verbose:
            log.append(f"[{env.now:.4f}] J{j_global}-M{m} END at t={end_time}")

        op_done[(j_global, m)].succeed()

    started = []
    for step_index, (_, j, m, g, t) in enumerate(start_events):
        p = env.process(process_op(j, m, g, step_index))
        started.append(p)

    if started:
        env.run(until=simpy.events.AllOf(env, started))

    makespan = max(completion_times.values()) if completion_times else 0.0

    if verbose:
        log.append(f"SIM DONE: Makespan {makespan}")

    return makespan, completion_times, log, start_events




def plot_des_gantt(completion_times, start_times, worker_assignments,
                   figsize=(14,6), title="DES Schedule Gantt Chart"):

    machines = sorted({m for (_, m) in completion_times.keys()})
    jobs = sorted({j for (j, _) in completion_times.keys()})

    prefix_tokens = set(list(completion_times.keys())[:8])

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("tab20")
    job_colors = {j: cmap(j % 20) for j in jobs}

    # ---- Layout params ----
    row_height = 1.6
    box_height = 0.8
    small_gap = 0.05
    short_threshold = 30  # boxes shorter than this considered "small"

    yticks = []
    yticklabels = []

    for i, m in enumerate(machines):

        y = (len(machines) - i - 1) * row_height
        yticks.append(y + box_height / 2)
        yticklabels.append(f"Machine {m+1}")

        ops = [(j, start_times[(j,m)], completion_times[(j,m)]) 
               for (j, mm) in completion_times.keys() if mm == m]

        ops.sort(key=lambda x: x[1])

        consecutive_small_count = 0  # count consecutive small boxes

        for idx, (j, start, end) in enumerate(ops):

            dur = end - start
            color = job_colors[j]
            g = worker_assignments[(j, m)]

            # Prefix highlighting
            if (j, m) in prefix_tokens:
                lw = 2.5
                ls = "--"
            else:
                lw = 1.0
                ls = "-"

            rect = patches.Rectangle(
                (start, y), dur, box_height,
                facecolor=color,
                edgecolor='black',
                linewidth=lw,
                linestyle=ls,
                zorder=1
            )
            ax.add_patch(rect)

            job_label = j + 1
            worker_label = g + 1
            label_text = f"J{job_label}\n(W{worker_label})"

            if dur >= short_threshold:
                # Large box → label inside
                ax.text(
                    start + dur/2,
                    y + box_height/2,
                    label_text,
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='black',
                    zorder=3
                )
                consecutive_small_count = 0  # reset
            else:
                # Small box
                if consecutive_small_count == 1:
                    # Second consecutive small box → place on top
                    text_y = y + box_height + small_gap
                    va = 'bottom'
                else:
                    # Default → below
                    text_y = y - small_gap
                    va = 'top'

                ax.text(
                    start + dur/2,
                    text_y,
                    label_text,
                    ha='center',
                    va=va,
                    fontsize=10,
                    fontweight='bold',
                    color='black',
                    zorder=3
                )

                # Update consecutive counter
                if consecutive_small_count == 1:
                    consecutive_small_count = 0
                else:
                    consecutive_small_count += 1

    prefix_patch = patches.Patch(
    facecolor='white',
    edgecolor='black',
    linewidth=2.5,
    linestyle='--',
    label="Prefix operations"
    )

    ax.legend(
        handles=[prefix_patch],
        handlelength=5,  # makes the legend sample longer horizontally
        handleheight=2,  # increases vertical space for the legend sample
        fontsize=12,   # increases "Prefix jobs" text size
        markerscale=2  # scales legend elements together
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")

    total_height = len(machines) * row_height
    ax.set_ylim(-row_height * 0.5, total_height)
    ax.set_xlim(0, max(completion_times.values()) * 1.05)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("gantt_plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()



def extract_info_from_events(events, completion_times, processing_times_global):
    proc = {}

    if isinstance(processing_times_global, dict):
        for (j, m), pt in processing_times_global.items():
            proc[(j, m)] = float(pt)

    else:
        arr = np.asarray(processing_times_global, dtype=float)
        num_jobs_global, num_machines = arr.shape
        for j in range(num_jobs_global):
            for m in range(num_machines):
                proc[(j, m)] = float(arr[j, m])

    start_times = {}
    worker_assignments = {}

    for etype, j, m, g, t in events:
        if etype == "START":
            end = completion_times[(j, m)]
            pt = proc[(j, m)]
            start = end - pt

            start_times[(j, m)] = start
            worker_assignments[(j, m)] = g

    return start_times, worker_assignments



# ---------- Main ----------
if __name__ == "__main__":
    iteration = 3
    pool_size = 20
    num_instances = 40000
    num_jobs = 8
    num_machines = 4
    num_groups = 3
    import multiprocessing
    multiprocessing.freeze_support()
    sequences, job_pool, worker_cap, eligibility, makespans_milp, makespans_des, bounds_milp = generate_training_data_with_workers(
        pool_size,
        num_instances,
        num_jobs,
        num_machines,
        num_groups,
        base_seed=iteration,
        save_every=5000,
        iteration=iteration,
        start_idx=30000
    )

    np.save(f"job_pool_{iteration}.npy", job_pool)
    np.save(f"worker_capacities_{iteration}.npy", worker_cap)
    np.save(f"eligibility_{iteration}.npy", eligibility)
    np.save(f"permutations_with_workers_{iteration}.npy", np.array(sequences, dtype=object))
    np.save(f"makespans_milp_{iteration}.npy", np.array(makespans_milp))
    np.save(f"makespans_des_{iteration}.npy", np.array(makespans_des))
    np.save(f"bounds_milp_{iteration}.npy", np.array(bounds_milp))

    print(f"Saved {len(sequences)} training sequences with worker assignments.")
