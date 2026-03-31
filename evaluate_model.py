import numpy as np
import torch
import random
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from ga import *
import time

from train_data import TransformerModel, EventSequenceDataset
from generate_data import (
    simulate_flowshop_events_simpy,
    extract_info_from_events,
    plot_des_gantt
)

# ------------------------------------------------------------
# Build vocabulary structure helpers
# ------------------------------------------------------------

def build_job_group_machine_map(idx_to_token):
    """
    Map (job, group) -> sorted list of valid machines
    using training vocabulary only.
    """
    valid_machines = defaultdict(set)
    for job, machine, group in idx_to_token:
        valid_machines[(job, group)].add(machine)

    return {k: sorted(v) for k, v in valid_machines.items()}


# ------------------------------------------------------------
# Structured random prefix generator (VOCAB SAFE)
# ------------------------------------------------------------

def generate_structured_random_prefix(
    idx_to_token,
    token_to_idx,
    job_eligibility,
    max_jobs=8,
    prefix_len=16,
    seed=None
):
    """
    Generate a structurally valid, vocab-safe random prefix (deterministic if seed is set):
    - ≤ max_jobs distinct jobs
    - machines strictly increasing per job
    - worker group fixed per job
    - ONLY tokens seen during training
    - job_eligibility is explicitly enforced
    """

    rng = random.Random(seed)

    # (job, group) -> sorted machines seen in training
    valid_machines = build_job_group_machine_map(idx_to_token)
    job_group_pairs = list(valid_machines.keys())

    # all jobs that appear in the vocabulary
    all_jobs = sorted({job for job, _ in job_group_pairs})

    # choose jobs
    chosen_jobs = rng.sample(all_jobs, k=min(max_jobs, len(all_jobs)))

    # choose a single eligible worker group per job
    job_group = {}
    for job in chosen_jobs:
        eligible_groups = [
            g for (j, g) in job_group_pairs
            if j == job and job_eligibility[j, g] == 1
        ]
        if not eligible_groups:
            raise RuntimeError(f"No eligible worker group for job {job}")
        job_group[job] = rng.choice(eligible_groups)

    # track machine progress per job
    next_machine_idx = {job: 0 for job in chosen_jobs}
    prefix = []

    while len(prefix) < prefix_len:
        active_jobs = [
            job for job in chosen_jobs
            if next_machine_idx[job] < len(valid_machines[(job, job_group[job])])
        ]

        if not active_jobs:
            break

        job = rng.choice(active_jobs)
        group = job_group[job]
        machines = valid_machines[(job, group)]

        m = machines[next_machine_idx[job]]
        tok = (job, m, group)

        # vocab safety
        assert tok in token_to_idx, f"OOV token generated: {tok}"

        prefix.append(tok)
        next_machine_idx[job] += 1

    return prefix




def encode_prefix(prefix_tokens, token_to_idx):
    return [token_to_idx[tok] for tok in prefix_tokens]


def decode_sequence(token_ids, idx_to_token):
    return [idx_to_token[i] for i in token_ids]


# ------------------------------------------------------------
# Schedule state for feasibility checking
# ------------------------------------------------------------
def is_eligible_group(job, group, job_eligibility):
    """
    Check whether job is eligible for the worker group.
    """
    return job_eligibility[job, group] == 1

class ScheduleState:
    """
    Tracks per-job machine progress only.
    Worker groups are decided per operation.
    """

    def __init__(self, job_eligibility):
        self.next_machine = defaultdict(int)
        self.job_eligibility = job_eligibility

    def is_feasible(self, tok):
        job, machine, group = tok

        # 1) Flow-shop order
        if machine != self.next_machine[job]:
            return False

        # 2) Job–worker-group eligibility
        if self.job_eligibility[job, group] != 1:
            return False

        return True

    def apply(self, tok):
        job, machine, group = tok
        self.next_machine[job] += 1


# ------------------------------------------------------------
# Minimal-distortion token selection
# ------------------------------------------------------------

def select_feasible_token(logits, state, idx_to_token):
    """
    Select the highest-probability token that is structurally feasible
    AND respects job–worker-group eligibility.
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_ids = torch.argsort(probs, descending=True)

    for idx in sorted_ids:
        tok = idx_to_token[idx.item()]
        if state.is_feasible(tok):
            return idx.item()

    raise RuntimeError("No feasible token found under eligibility constraints")



# ------------------------------------------------------------
# Autoregressive generation with reconciliation
# ------------------------------------------------------------

def greedy_generate_with_repair(
    model,
    start_prefix_ids,
    idx_to_token,
    max_len,
    device="cpu",
    eligibility=None
):
    model.eval()
    seq = list(start_prefix_ids)

    state = ScheduleState(eligibility)
    for tok_id in seq:
        state.apply(idx_to_token[tok_id])

    with torch.no_grad():
        while len(seq) < max_len:
            x = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            mask = torch.ones_like(x, dtype=torch.float32)

            logits = model(x, mask)[0]
            next_id = select_feasible_token(logits, state, idx_to_token)

            seq.append(next_id)
            state.apply(idx_to_token[next_id])

    return seq

def generate_random_completion_from_prefix(
    prefix_ids,
    active_operations,      # {(job, machine), ...}
    idx_to_token,
    token_to_idx,
    job_eligibility,
    seed=None
):
    """
    Random completion that:

    - Keeps prefix exactly
    - Uses EXACT same active operations as transformer
    - Preserves partial jobs
    - Respects flow-shop precedence
    - Respects worker eligibility
    - Fences prefix block on each machine
    """

    rng = random.Random(seed)

    prefix_tokens = [idx_to_token[i] for i in prefix_ids]

    # -----------------------------------
    # Build job -> machines to schedule
    # -----------------------------------
    job_to_machines = {}
    for (j, m) in active_operations:
        job_to_machines.setdefault(j, []).append(m)

    for j in job_to_machines:
        job_to_machines[j] = sorted(job_to_machines[j])

    # -----------------------------------
    # Initialize state from prefix
    # -----------------------------------
    state = ScheduleState(job_eligibility)

    seq_tokens = list(prefix_tokens)

    for tok in prefix_tokens:
        state.apply(tok)

    # Remove already scheduled prefix operations
    remaining = []
    for j, machines in job_to_machines.items():
        for m in machines:
            if m >= state.next_machine[j]:
                remaining.append((j, m))

    # -----------------------------------
    # Randomized topological scheduling
    # -----------------------------------
    while remaining:

        feasible = []

        for (j, m) in remaining:
            if m == state.next_machine[j]:
                feasible.append((j, m))

        if not feasible:
            break

        j, m = rng.choice(feasible)

        # Random eligible worker group
        eligible_groups = np.where(job_eligibility[j] == 1)[0]
        g = int(rng.choice(eligible_groups))

        tok = (j, m, g)

        # Ensure token exists in vocabulary
        if tok not in token_to_idx:
            # if vocab restricted by training, pick any valid token
            valid_tokens = [
                t for t in idx_to_token
                if t[0] == j and t[1] == m and job_eligibility[j, t[2]] == 1
            ]
            if not valid_tokens:
                remaining.remove((j, m))
                continue
            tok = rng.choice(valid_tokens)

        seq_tokens.append(tok)
        state.apply(tok)
        remaining.remove((j, m))

    # Encode
    return [token_to_idx[tok] for tok in seq_tokens]




# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_unseen_structured_prefixes(
    sequences_file="permutations_with_workers_1.npy",
    model_file="transformer_event_model1.pt",
    job_pool_file="job_pool_1.npy",
    worker_caps_file="worker_capacities_1.npy",
    elibility_file="eligibility_1.npy",
    prefix_len=16,
    target_len=32,
    max_jobs=8,
    device="cpu",
    population_size=50,
    generations=100,
    seed=None
):
    """
    Transformer-only evaluation on unseen, structurally valid prefixes
    with minimal-distortion reconciliation.
    """

    # ---------------- Load dataset + vocab ----------------
    dataset = EventSequenceDataset(sequences_file)

    # ---------------- Load job pool + worker capacities ----------------
    job_pool = np.load(job_pool_file)
    worker_capacities = np.load(worker_caps_file)
    eligibility = np.load(elibility_file)
    # ---------------- Load model ----------------
    model = TransformerModel(dataset.vocab_size)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    print(f"Loaded model (vocab size = {dataset.vocab_size})")
    print(f"Job pool shape = {job_pool.shape}")
    print(f"Worker capacities = {worker_capacities.tolist()}")
    print(f"Max jobs allowed = {max_jobs}")


    prefix_tokens = generate_structured_random_prefix(
        dataset.idx_to_token,
        dataset.token_to_idx,
        job_eligibility=eligibility,
        max_jobs=max_jobs,
        prefix_len=prefix_len,
        seed=seed
    )

    prefix_ids = encode_prefix(prefix_tokens, dataset.token_to_idx)

    tick = time.time()
    generated_ids = greedy_generate_with_repair(
        model,
        start_prefix_ids=prefix_ids,
        idx_to_token=dataset.idx_to_token,
        max_len=target_len,
        device=device,
        eligibility=eligibility
    )
    tock = time.time()
    print(f"Transformer-completion took {tock - tick:.2f} seconds")

    generated_tokens = decode_sequence(generated_ids, dataset.idx_to_token)

    active_operations = {
        (j, m)
        for (j, m, g) in generated_tokens
    }

    selected_indices = sorted({
        j for (j, _, _) in generated_tokens
    })
     
    # ------------------------------------------------------------
    # Plot first transformer-completed schedule using DES pipeline
    # ------------------------------------------------------------
    
    print("\nSimulating and plotting transformer-completed schedule...")

    # Build START events exactly like in training
    start_events = [
        ("START", j, m, g, 0.0)
        for (j, m, g) in generated_tokens
    ]
        
    # Run DES simulation directly on START events
    makespan, completion_times, log, _ = simulate_flowshop_events_simpy(
        start_events,
        job_pool,
        worker_capacities,
        verbose=False
    )

    # Safety check (prevents your crash)
    if not completion_times:
        print("WARNING: DES produced empty schedule — skipping plot.")
    else:
        start_times, worker_assignments = extract_info_from_events(
            start_events,
            completion_times,
            job_pool
        )

        # plot_des_gantt(
        #     completion_times,
        #     start_times,
        #     worker_assignments,
        #     title="Transformer-Completed Flow Shop Schedule (DES)"
        # )

    print("Structured unseen prefix:")
    for e in prefix_tokens:
        print(" ", e)

    print("\nTransformer-generated continuation (reconciled):")
    for e in generated_tokens[prefix_len:]:
        print(" ", e)


    # ------------------------------------------------------------
    # IG heuristic optimization on transformer-selected job set
    # ------------------------------------------------------------
    print("\nOptimizing transformer job set with Iterated Greedy (IG) heuristic...")

    prefix_ops = generated_tokens[:prefix_len]
    tick = time.time()
    start_values_ig, assign_values_ig, makespan_ig = iterated_greedy_with_fixed_prefix_des(
        prefix_ops=prefix_ops,
        all_ops=generated_tokens,
        job_pool=job_pool,
        worker_capacities=worker_capacities,
        eligibility=eligibility,
        max_iter=1000,
        destruction_size=2,
        acceptance_prob=0.1
    )
    tock = time.time()
    print(f"IG heuristic optimization took {tock - tick:.2f} seconds")
    print(f"IG heuristic makespan = {makespan_ig:.2f}")

    # ------------------------------------------------------------
    # Convert heuristic solution to START events
    # ------------------------------------------------------------
    ig_start_events = []
    for (j, m), t_start in start_values_ig.items():
        assigned_group = None
        for (jj, mm, g), v in assign_values_ig.items():
            if jj == j and mm == m and v > 0.5:
                assigned_group = g
                break
        if assigned_group is None:
            raise RuntimeError(f"No worker group assigned for job {j}, machine {m}")
        ig_start_events.append(("START", j, m, assigned_group, float(t_start)))

    ig_start_events.sort(key=lambda x: x[4])

    # ------------------------------------------------------------
    # DES simulation of heuristic schedule
    # ------------------------------------------------------------
    makespan_ig_des, completion_times_ig, log_ig, _ = simulate_flowshop_events_simpy(
        ig_start_events,
        job_pool,
        worker_capacities,
        verbose=False
    )
    print(f"DES makespan (IG schedule) = {makespan_ig_des:.2f}")

    # # ------------------------------------------------------------
    # # Extract info + plot Gantt
    # # ------------------------------------------------------------
    # if completion_times_ig:
    #     start_times_ig, worker_assignments_ig = extract_info_from_events(
    #         ig_start_events,
    #         completion_times_ig,
    #         job_pool
    #     )

    #     plot_des_gantt(
    #         completion_times_ig,
    #         start_times_ig,
    #         worker_assignments_ig,
    #         title="IG-Heuristic Schedule (Transformer Job Set)"
    #     )
    # else:
    #     print("WARNING: IG heuristic DES produced empty schedule — skipping plot.")

    # ------------------------------------------------------------
    # RANDOM BASELINE (same prefix + same job universe)
    # ------------------------------------------------------------

    #Jobs used by transformer (prefix + continuation)
    transformer_jobs = sorted({j for (j, _, _) in generated_tokens})

    
    print("prefix_tokens", prefix_tokens )
    tick = time.time()
    random_ids = generate_random_completion_from_prefix(
        prefix_ids=prefix_ids,
        idx_to_token=dataset.idx_to_token,
        token_to_idx=dataset.token_to_idx,
        job_eligibility=eligibility,
        active_operations=active_operations,
        seed=seed
    )
    tock = time.time()
    print(f"Random completion generation took {tock - tick:.2f} seconds")
    random_tokens = decode_sequence(random_ids, dataset.idx_to_token)

    print("\nRandom completion (same prefix + same jobs):")
    for e in random_tokens[prefix_len:]:
        print(" ", e)

    # ------------------------------------------------------------
    # Plot RANDOM schedule
    # ------------------------------------------------------------
    
    print("\nSimulating and plotting random baseline schedule...")

    start_events_rand = [
        ("START", j, m, g, 0.0)
        for (j, m, g) in random_tokens
    ]

    makespan_rand, completion_times_rand, _, _ = simulate_flowshop_events_simpy(
        start_events_rand,
        job_pool,
        worker_capacities,
        verbose=False
    )

    if not completion_times_rand:
        print("WARNING: Random DES produced empty schedule — skipping plot.")
    else:
        start_times_rand, worker_assignments_rand = extract_info_from_events(
            start_events_rand,
            completion_times_rand,
            job_pool
        )

        # plot_des_gantt(
        #     completion_times_rand,
        #     start_times_rand,
        #     worker_assignments_rand,
        #     title="Random Baseline (Same Prefix & Jobs)"
        # )

    # ------------------------------------------------------------
    # GENETIC ALGORITHM (GA)
    # ------------------------------------------------------------

    print("Running GENETIC ALGORITHM...")
    tick = time.time()
    # In your evaluate_unseen_structured_prefixes function:
    best_schedule_ga = ga_flowshop_schedule(
        processing_times_global=job_pool,  # Processing times for each job
        worker_capacities=worker_capacities,  # Worker capacities
        prefix_tokens=prefix_tokens,  # Fixed job tokens (prefix)
        generated_tokens=generated_tokens,
        population_size=population_size,
        generations=generations
    )
    tock = time.time()
    print(f"GA optimization took {tock - tick:.2f} seconds")
    # Simulate and plot the GA schedule
    print("\nSimulating and plotting GA schedule...")

    start_events_ga = [
        ("START", job_id, machine, worker_group, 0.0)
        for job_id, machine, worker_group in best_schedule_ga
    ]

    makespan_ga, completion_times_ga, _, _ = simulate_flowshop_events_simpy(
        start_events_ga,
        job_pool,
        worker_capacities,
        verbose=False
    )

    if not completion_times_ga:
        print("WARNING: GA DES produced empty schedule — skipping plot.")
    else:
        start_times_ga, worker_assignments_ga = extract_info_from_events(
            start_events_ga,
            completion_times_ga,
            job_pool
        )

        # plot_des_gantt(
        #     completion_times_ga,
        #     start_times_ga,
        #     worker_assignments_ga,
        #     title="GA Optimized Schedule"
        # )
    # ------------------------------------------------------------
    # MILP re-optimization on transformer-selected job set
    # ------------------------------------------------------------
    print("\nRe-optimizing transformer job set with MILP...")

    # 1) Extract UNIQUE jobs from full transformer-completed sequence
    selected_jobs = sorted({j for (j, _, _) in generated_tokens})
    selected_indices = selected_jobs  # MILP expects global job indices

    print(f"Selected {len(selected_indices)} jobs for MILP re-optimization")

    # ------------------------------------------------------------
    # Build prefix tokens with fixed start times for MILP (as it should not optimize the overall schedule)
    # ------------------------------------------------------------
    prefix_tokens_milp = []

    for (j, m, g) in prefix_tokens:   # ONLY the prefix
        t_start = start_times[(j, m)]
        prefix_tokens_milp.append((j, m, g, t_start))

    # 2) Solve MILP from scratch on these jobs
    (
        C_values,
        start_values,
        assign_values,
        makespan_value,
        obj_bound
    ) = solve_flowshop_with_workers_prefixed(
        processing_times_global=job_pool,
        worker_capacities=worker_capacities,
        job_eligibility=eligibility,          # use same eligibility as training
        selected_indices=selected_indices,
        prefix_tokens=prefix_tokens_milp,
        active_operations=active_operations,  # only schedule what transformer scheduled
        time_scale=1,
        verbose=False
    )
    print(f"MILP makespan = {makespan_value:.2f}")
    print(f"MILP objective bound = {obj_bound:.2f}")

    # ------------------------------------------------------------
    # Convert MILP solution to START events
    # ------------------------------------------------------------
    milp_start_events = []

    for (j, m), t_start in start_values.items():
        # Find assigned worker group for (j, m)
        assigned_group = None
        assigned_group = assign_values.get((j, m))
        if assigned_group is None:
            raise RuntimeError(f"No worker group assigned for job {j}, machine {m}")

        milp_start_events.append(
            ("START", j, m, assigned_group, float(t_start))
        )

    # Sort by time to respect DES event order
    milp_start_events.sort(key=lambda x: x[4])

    # ------------------------------------------------------------
    # DES simulation of MILP-optimized schedule
    # ------------------------------------------------------------
    makespan_milp_des, completion_times_milp, log_milp, _ = (
        simulate_flowshop_events_simpy(
            milp_start_events,
            job_pool,
            worker_capacities,
            verbose=False
        )
    )

    print(f"DES makespan (MILP schedule) = {makespan_milp_des:.2f}")

    # ------------------------------------------------------------
    # Extract info + plot Gantt
    #------------------------------------------------------------
    if completion_times_milp:
        start_times_milp, worker_assignments_milp = extract_info_from_events(
            milp_start_events,
            completion_times_milp,
            job_pool
        )

        # plot_des_gantt(
        #     completion_times_milp,
        #     start_times_milp,
        #     worker_assignments_milp,
        #     title="MILP-Optimized Schedule (Transformer Job Set)"
        # )
    else:
        print("WARNING: MILP DES produced empty schedule — skipping plot.")


    # ------------------------------------------------------------
    # Heuristic (NEH) optimization on transformer-selected job set
    # ------------------------------------------------------------
    print("\nOptimizing transformer job set with NEH heuristic...")

    # 1) Extract UNIQUE jobs from full transformer-completed sequence
    selected_jobs = sorted({j for (j, _, _) in generated_tokens})

    print(f"Selected {len(selected_jobs)} jobs for NEH heuristic")
    tick = time.time()
    # 2) Run NEH heuristic from scratch on these jobs
    start_values_heur, assign_values_heur, makespan_heur = (
        neh_heuristic_with_fixed_prefix(
            prefix_tokens=generated_tokens[:prefix_len],
            transformer_tokens=generated_tokens,   # ← pass full set
            job_pool=job_pool,
            worker_capacities=worker_capacities,
            job_eligibility=eligibility,
            active_operations=active_operations
        )
    )
    tock = time.time()
    print(f"NEH optimization took {tock - tick:.2f} seconds")
    print(f"NEH heuristic makespan = {makespan_heur:.2f}")

    # ------------------------------------------------------------
    # Convert heuristic solution to START events
    # ------------------------------------------------------------
    heur_start_events = []

    for (j, m), t_start in start_values_heur.items():
        assigned_group = None

        # assign_values[(j,m,g)] == 1 for exactly one g
        for (jj, mm, g), v in assign_values_heur.items():
            if jj == j and mm == m and v > 0.5:
                assigned_group = g
                break

        if assigned_group is None:
            raise RuntimeError(
                f"No worker group assigned for job {j}, machine {m} (heuristic)"
            )

        heur_start_events.append(
            ("START", j, m, assigned_group, float(t_start))
        )

    # Sort by time to respect DES event order
    heur_start_events.sort(key=lambda x: x[4])

    # ------------------------------------------------------------
    # DES simulation of heuristic schedule
    # ------------------------------------------------------------
    makespan_heur_des, completion_times_heur, log_heur, _ = (
        simulate_flowshop_events_simpy(
            heur_start_events,
            job_pool,
            worker_capacities,
            verbose=False
        )
    )

    print(f"DES makespan (heuristic schedule) = {makespan_heur_des:.2f}")

    # ------------------------------------------------------------
    # Extract info + plot Gantt
    # ------------------------------------------------------------
    # if completion_times_heur:
    #     start_times_heur, worker_assignments_heur = extract_info_from_events(
    #         heur_start_events,
    #         completion_times_heur,
    #         job_pool
    #     )

    #     plot_des_gantt(
    #         completion_times_heur,
    #         start_times_heur,
    #         worker_assignments_heur,
    #         title="NEH-Heuristic Schedule (Transformer Job Set)"
    #     )
    # else:
    #     print("WARNING: Heuristic DES produced empty schedule — skipping plot.")

    return makespan, makespan_milp_des ,makespan_heur_des, makespan_value, obj_bound, makespan_rand, makespan_ga, makespan_ig_des


def simulate_remaining_jobs(
    job_sequence,
    job_pool,
    worker_capacities,
    job_eligibility,
    machine_available,
    job_available,
    next_machine,
    start_values,
    assign_values
):
    num_machines = job_pool.shape[1]
    num_groups = len(worker_capacities)

    for j in job_sequence:
        for m in range(next_machine[j], num_machines):
            best_group = None
            best_finish = float("inf")
            best_start = None

            for g in range(num_groups):
                if job_eligibility[j, g] != 1:
                    continue

                start_t = max(machine_available[m], job_available[j])
                finish_t = start_t + job_pool[j, m] / worker_capacities[g]

                if finish_t < best_finish:
                    best_finish = finish_t
                    best_start = start_t
                    best_group = g

            start_values[(j, m)] = best_start
            assign_values[(j, m, best_group)] = 1

            machine_available[m] = best_finish
            job_available[j] = best_finish

    makespan = max(machine_available)
    return makespan


def neh_heuristic_with_fixed_prefix(
    prefix_tokens,
    transformer_tokens,
    active_operations,        # ← NEW (same as MILP + RANDOM)
    job_pool,
    worker_capacities,
    job_eligibility
):
    """
    NEH heuristic that:

    - Keeps prefix exactly
    - Uses EXACT same active_operations as transformer
    - Allows partially completed jobs
    - Fences prefix block
    - Respects eligibility
    """

    num_machines = job_pool.shape[1]

    # -------------------------------------------------
    # Extract prefix state (DES-consistent state)
    # -------------------------------------------------
    (
        machine_available,
        job_available,
        next_machine,
        start_values,
        assign_values
    ) = extract_prefix_state(
        prefix_tokens,
        job_pool,
        worker_capacities
    )

    # -------------------------------------------------
    # Build job -> machines map from active_operations
    # -------------------------------------------------
    job_to_machines = {}
    for (j, m) in active_operations:
        job_to_machines.setdefault(j, []).append(m)

    for j in job_to_machines:
        job_to_machines[j] = sorted(job_to_machines[j])

    # -------------------------------------------------
    # Determine remaining operations after prefix
    # -------------------------------------------------
    remaining_ops = []

    for j, machines in job_to_machines.items():
        for m in machines:
            if m >= next_machine[j]:
                remaining_ops.append((j, m))

    # Jobs that still have remaining operations
    remaining_jobs = sorted(set(j for (j, _) in remaining_ops))

    # -------------------------------------------------
    # NEH sorting key (remaining workload only!)
    # -------------------------------------------------
    job_scores = {
        j: sum(job_pool[j, m] for m in job_to_machines[j] if m >= next_machine[j])
        for j in remaining_jobs
    }

    sorted_jobs = sorted(remaining_jobs,
                         key=lambda j: job_scores[j],
                         reverse=True)

    sequence = []

    # -------------------------------------------------
    # NEH insertion loop
    # -------------------------------------------------
    for j in sorted_jobs:

        best_seq = None
        best_mk = float("inf")

        for pos in range(len(sequence) + 1):

            trial_seq = sequence[:pos] + [j] + sequence[pos:]

            mk = simulate_remaining_jobs_partial(
                trial_seq,
                job_to_machines,
                job_pool,
                worker_capacities,
                job_eligibility,
                machine_available.copy(),
                job_available.copy(),
                next_machine.copy(),
                start_values.copy(),
                assign_values.copy()
            )

            if mk < best_mk:
                best_mk = mk
                best_seq = trial_seq

        sequence = best_seq

    # -------------------------------------------------
    # Final schedule build
    # -------------------------------------------------
    makespan = simulate_remaining_jobs_partial(
        sequence,
        job_to_machines,
        job_pool,
        worker_capacities,
        job_eligibility,
        machine_available,
        job_available,
        next_machine,
        start_values,
        assign_values
    )

    return start_values, assign_values, makespan

def simulate_remaining_jobs_partial(
    sequence,
    job_to_machines,
    job_pool,
    worker_capacities,
    job_eligibility,
    machine_available,
    job_available,
    next_machine,
    start_values,
    assign_values
):
    """
    Simulates only the active_operations (partial jobs allowed).

    Worker group is chosen by evaluating ALL eligible groups and
    selecting the one that gives the earliest completion time.
    """

    num_groups = len(worker_capacities)

    for j in sequence:

        for m in job_to_machines[j]:

            if m < next_machine[j]:
                continue  # already done in prefix

            ready_time = max(job_available[j], machine_available[m])

            best_g = None
            best_start = None
            best_finish = float("inf")

            # -------------------------------------------------
            # Evaluate all feasible worker groups
            # -------------------------------------------------
            for g in range(num_groups):

                if job_eligibility[j, g] != 1:
                    continue

                start = ready_time
                finish = start + job_pool[j, m] / worker_capacities[g]

                if finish < best_finish:
                    best_finish = finish
                    best_start = start
                    best_g = g

            if best_g is None:
                continue

            # Commit best group
            start_values[(j, m)] = best_start
            assign_values[(j, m, best_g)] = 1

            machine_available[m] = best_finish
            job_available[j] = best_finish
            next_machine[j] = m + 1

    return max(job_available.values()) if job_available else 0

def extract_prefix_state(prefix_tokens, job_pool, worker_capacities):
    """
    Extract machine/job availability and completed machines from prefix.
    """
    num_machines = job_pool.shape[1]

    machine_available = [0.0] * num_machines
    job_available = defaultdict(float)
    next_machine = defaultdict(int)

    start_values = {}
    assign_values = {}

    for (j, m, g) in prefix_tokens:
        start_t = max(machine_available[m], job_available[j])
        finish_t = start_t + job_pool[j, m] / worker_capacities[g]

        start_values[(j, m)] = start_t
        assign_values[(j, m, g)] = 1

        machine_available[m] = finish_t
        job_available[j] = finish_t
        next_machine[j] += 1

    return machine_available, job_available, next_machine, start_values, assign_values


def solve_flowshop_with_workers_prefixed(
    processing_times_global,
    worker_capacities,
    job_eligibility,
    selected_indices,
    prefix_tokens=None,
    active_operations=None,
    time_scale=1,
    verbose=False
):
    """
    MILP flow shop solver with worker groups.

    - Schedules only operations in active_operations
    - Fixes prefix order + worker assignments
    - FENCES prefix on each machine:
        Non-prefix jobs cannot start before prefix block completes
    """

    num_jobs = len(selected_indices)
    num_machines = processing_times_global.shape[1]
    num_groups = len(worker_capacities)

    jobs = range(num_jobs)
    machines = range(num_machines)
    groups = range(num_groups)

    global_to_local = {j: idx for idx, j in enumerate(selected_indices)}

    PT = np.zeros((num_jobs, num_machines))
    for j_local, j_global in enumerate(selected_indices):
        PT[j_local, :] = processing_times_global[j_global, :]

    T = int(np.ceil(np.sum(PT))) + 1

    model = gp.Model("flowshop_with_stage_workers")

    if not verbose:
        model.Params.OutputFlag = 0
        model.Params.MIPFocus = 1
        model.Params.TimeLimit = 600
        model.Params.NoRelHeurTime = 60

    # ---------------- VARIABLES ----------------
    assign = model.addVars(num_jobs, num_machines, num_groups, vtype=GRB.BINARY)
    C = model.addVars(num_jobs, num_machines, vtype=GRB.CONTINUOUS)
    makespan = model.addVar(vtype=GRB.CONTINUOUS)

    z, u, v = {}, {}, {}
    for j in jobs:
        for m in machines:
            for t in range(T):
                z[j,m,t] = model.addVar(vtype=GRB.BINARY)
                u[j,m,t] = model.addVar(vtype=GRB.BINARY)
                v[j,m,t] = model.addVar(vtype=GRB.BINARY)

    # ---------------- FLOW CONSTRAINTS ----------------
    for j_local, j_global in enumerate(selected_indices):
        for m in machines:

            if active_operations and (j_global, m) not in active_operations:
                continue

            if m == 0:
                model.addConstr(C[j_local,0] >= PT[j_local,0])
            else:
                if active_operations is None or (j_global, m-1) in active_operations:
                    model.addConstr(
                        C[j_local,m] >= C[j_local,m-1] + PT[j_local,m]
                    )

    # ---------------- ASSIGNMENT ----------------
    for j_local, j_global in enumerate(selected_indices):
        for m in machines:
            if active_operations and (j_global, m) not in active_operations:
                continue
            model.addConstr(assign.sum(j_local,m,"*") == 1)

    for j_local, j_global in enumerate(selected_indices):
        for m in machines:
            if active_operations and (j_global, m) not in active_operations:
                continue
            for g in groups:
                if job_eligibility[j_global,g] == 0:
                    assign[j_local,m,g].ub = 0

    # ---------------- MACHINE DISJUNCTION ----------------
    bigM = T 
    y = model.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY)

    for m in machines:

        active_jobs_m = [
            j_local
            for j_local, j_global in enumerate(selected_indices)
            if active_operations is None or (j_global, m) in active_operations
        ]

        for i in active_jobs_m:
            for j in active_jobs_m:
                if i < j:
                    model.addConstr(y[i,j,m] + y[j,i,m] == 1)

        for i in active_jobs_m:
            for j in active_jobs_m:
                if i == j:
                    continue
                model.addConstr(
                    C[i,m] >= C[j,m] + PT[i,m] - bigM*(1 - y[i,j,m])
                )
                model.addConstr(
                    C[j,m] >= C[i,m] + PT[j,m] - bigM*y[i,j,m]
                )

    # ---------------- PREFIX FIXING + FENCING ----------------
    if prefix_tokens:

        prefix_by_machine = {}
        prefix_assignments = {}

        for (j_global, m, g, start) in prefix_tokens:
            if j_global not in global_to_local:
                continue

            if active_operations and (j_global, m) not in active_operations:
                continue

            j_local = global_to_local[j_global]
            prefix_by_machine.setdefault(m, []).append(j_local)
            prefix_assignments[(j_local, m)] = g

        # Fix order inside prefix
        for m, jobs_on_m in prefix_by_machine.items():
            for k, j_later in enumerate(jobs_on_m):
                for j_earlier in jobs_on_m[:k]:
                    y[j_later, j_earlier, m].lb = 1
                    y[j_later, j_earlier, m].ub = 1

        # Fix worker assignments
        for (j,m), g in prefix_assignments.items():
            assign[j,m,g].lb = 1
            assign[j,m,g].ub = 1

        # -------- PREFIX FENCING --------
        for m, jobs_on_m in prefix_by_machine.items():

            if not jobs_on_m:
                continue

            # last job in prefix on machine m
            last_prefix_job = jobs_on_m[-1]

            for j_local in jobs:
                if j_local not in jobs_on_m:
                    j_global = selected_indices[j_local]

                    if active_operations and (j_global, m) not in active_operations:
                        continue

                    # FORCE non-prefix job AFTER prefix block
                    model.addConstr(
                        C[j_local, m] >= C[last_prefix_job, m]
                    )

    # ---------------- WORKER CAPACITY ----------------
    max_proc = float(np.max(PT))
    bigM_local = T

    for j in jobs:
        for m in machines:

            j_global = selected_indices[j]
            if active_operations and (j_global, m) not in active_operations:
                continue

            proc = float(PT[j,m])
            for t in range(T):
                start_jm = C[j,m] - proc

                model.addConstr(start_jm <= t + bigM_local*(1 - u[j,m,t]))
                model.addConstr(start_jm >= (t+1) - bigM_local*u[j,m,t])
                model.addConstr(C[j,m] >= (t+1) - bigM_local*(1 - v[j,m,t]))
                model.addConstr(C[j,m] <= t + bigM_local*v[j,m,t])

                model.addConstr(z[j,m,t] >= u[j,m,t] + v[j,m,t] - 1)
                model.addConstr(z[j,m,t] <= u[j,m,t])
                model.addConstr(z[j,m,t] <= v[j,m,t])

    for g in groups:
        cap = int(worker_capacities[g])
        for t in range(T):
            lhs = gp.quicksum(
                assign[j,m,g]*z[j,m,t]
                for j in jobs
                for m in machines
                if active_operations is None or
                   (selected_indices[j], m) in active_operations
            )
            model.addConstr(lhs <= cap)

    # ---------------- OBJECTIVE ----------------
    for j_local, j_global in enumerate(selected_indices):
        for m in machines:
            if active_operations is None or (j_global, m) in active_operations:
                model.addConstr(makespan >= C[j_local,m])

    model.setObjective(makespan, GRB.MINIMIZE)

    # ---------------- SOLVE ----------------
    model.optimize()

    if model.SolCount == 0:
        return None

    C_values = {}
    start_values = {}
    assign_values = {}

    for j_local, j_global in enumerate(selected_indices):
        for m in machines:
            if active_operations and (j_global, m) not in active_operations:
                continue

            C_values[(j_global,m)] = C[j_local,m].X
            start_values[(j_global,m)] = C_values[(j_global,m)] - PT[j_local,m]

            for g in groups:
                if assign[j_local,m,g].X > 0.5:
                    assign_values[(j_global,m)] = g
                    break

    makespan_value = makespan.X

    return C_values, start_values, assign_values, makespan_value, model.ObjBound


def iterated_greedy_with_fixed_prefix_des(
    prefix_ops,          # fixed prefix [(job,machine,worker)]
    all_ops,             # full operation list [(job,machine,worker)]
    job_pool,
    worker_capacities,
    eligibility,
    max_iter=1000,
    destruction_size=2,
    acceptance_prob=0.1
):
    """
    Iterated Greedy for non-permutation flow shop with fixed prefix.
    Fully robust: no operations lost or duplicated.
    Dynamically chooses worker group during simulation using eligibility.
    """

    # --- Step 0: Build complete job -> all machines mapping ---
    job_to_all_machines = {}
    for (j, m, _) in all_ops:
        job_to_all_machines.setdefault(j, []).append(m)
    for j in job_to_all_machines:
        job_to_all_machines[j] = sorted(job_to_all_machines[j])

    # --- Prefix set for skipping fixed operations ---
    prefix_set = set((j, m) for (j, m, _) in prefix_ops)

    # --- Initial sequence: prefix + remaining jobs in arbitrary order ---
    remaining_jobs = list(job_to_all_machines.keys())
    current_sequence = prefix_ops.copy()
    for j in remaining_jobs:
        for m in job_to_all_machines[j]:
            if (j, m) not in prefix_set:
                current_sequence.append((j, m, None))

    best_sequence = current_sequence.copy()
    best_makespan, best_start_values, best_assign_values = simulate_sequence_dynamic_workers(
        best_sequence, job_pool, worker_capacities, eligibility
    )

    # --- IG iterations ---
    for iteration in range(max_iter):
        non_prefix_jobs = [j for j in job_to_all_machines if any((j,m) not in prefix_set for m in job_to_all_machines[j])]
        if len(non_prefix_jobs) == 0:
            continue

        # --- Destruction: remove d jobs from non-prefix jobs ---
        d = min(destruction_size, len(non_prefix_jobs))
        removed_jobs = random.sample(non_prefix_jobs, d)
        remaining_jobs_for_reconstruction = [j for j in non_prefix_jobs if j not in removed_jobs]

        # --- Reconstruction ---
        reconstructed_sequence = prefix_ops.copy()
        # Add remaining jobs first (non-removed)
        for j in remaining_jobs_for_reconstruction:
            for m in job_to_all_machines[j]:
                if (j, m) not in prefix_set:
                    reconstructed_sequence.append((j, m, None))

        # Insert removed jobs one by one
        for j in removed_jobs:
            best_local_seq = None
            best_local_mk = float("inf")

            # Candidate positions after prefix
            for pos in range(len(prefix_ops), len(reconstructed_sequence)+1):
                trial_seq = reconstructed_sequence[:pos]
                # Insert all non-prefix ops of job j
                for m in job_to_all_machines[j]:
                    if (j, m) not in prefix_set:
                        trial_seq.append((j, m, None))
                # Append everything after pos
                trial_seq.extend(reconstructed_sequence[pos:])

                mk, _, _ = simulate_sequence_dynamic_workers(trial_seq, job_pool, worker_capacities, eligibility)
                if mk < best_local_mk:
                    best_local_mk = mk
                    best_local_seq = trial_seq

            reconstructed_sequence = best_local_seq

        current_sequence = reconstructed_sequence

        # --- Acceptance ---
        mk, start_vals, assign_vals = simulate_sequence_dynamic_workers(
            current_sequence, job_pool, worker_capacities, eligibility
        )
        if mk < best_makespan:
            best_sequence = current_sequence.copy()
            best_makespan = mk
            best_start_values = start_vals
            best_assign_values = assign_vals
        else:
            if random.random() < acceptance_prob:
                pass  # keep current_sequence for next iteration

    # --- Final sanity check: all operations included ---
    assert len(best_sequence) == len(all_ops), f"Length mismatch! {len(best_sequence)} vs {len(all_ops)}"

    return best_start_values, best_assign_values, best_makespan

# ----------------------------
# Helper function to convert sequence to DES events
# ----------------------------
def simulate_sequence_dynamic_workers(op_sequence, job_pool, worker_capacities, eligibility):
    """
    Converts op_sequence [(job,machine,worker=None or fixed)] to START events and runs DES.
    Dynamically chooses worker groups for ops where worker=None.
    Returns: start_values, assign_values, makespan
    """

    start_values = {}
    assign_values = {}

    machine_available = {}
    job_available = {}
    next_machine = {}

    # Extract prefix ops (with fixed worker) and remaining
    for op in op_sequence:
        j, m, g = op
        if m not in machine_available:
            machine_available[m] = 0.0
        if j not in job_available:
            job_available[j] = 0.0
        if j not in next_machine:
            next_machine[j] = 0

    start_events = []

    for op in op_sequence:
        j, m, g = op
        if m < next_machine[j]:
            continue  # already done (prefix)

        ready_time = max(job_available[j], machine_available[m])

        # If worker already fixed (prefix), use it
        if g is not None:
            start = ready_time
            finish = start + job_pool[j, m] / worker_capacities[g]
            best_g = g
        else:
            # Choose earliest finishing eligible worker
            best_finish = float("inf")
            best_g = None
            for wg in range(len(worker_capacities)):
                if eligibility[j, wg] != 1:
                    continue
                start = ready_time
                finish = start + job_pool[j, m] / worker_capacities[wg]
                if finish < best_finish:
                    best_finish = finish
                    best_g = wg
            start = ready_time
            finish = start + job_pool[j, m] / worker_capacities[best_g]

        # Commit
        start_values[(j, m)] = start
        assign_values[(j, m, best_g)] = 1
        machine_available[m] = finish
        job_available[j] = finish
        next_machine[j] = m + 1

        start_events.append(("START", j, m, best_g, start))

    makespan = max(job_available.values()) if job_available else 0.0
    return makespan, start_values, assign_values

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    N = 50
    prefix_lens = [4,5,6,7,8,9,10,11,12,13,14,15,16]
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for p in prefix_lens:
        tf_ms, milp_ms, heur_ms, milp_ub, milp_lb, rand_ms, ga_ms, ig_ms = [], [], [], [], [], [], [], []

        for seed in range(N):
            m_tf, m_milp, m_heur, m_milp_ub, m_milp_lb, m_rand, m_ga, m_ig = evaluate_unseen_structured_prefixes(
                device="cpu",
                seed=seed,
                population_size=200,
                generations=1000,
                prefix_len=p
            )
            tf_ms.append(m_tf)
            milp_ms.append(m_milp)
            heur_ms.append(m_heur)
            milp_ub.append(m_milp_ub)
            milp_lb.append(m_milp_lb)
            rand_ms.append(m_rand)
            ga_ms.append(m_ga)
            ig_ms.append(m_ig)

        # store raw data
        np.savez(
            f"{out_dir}/makespan_prefix{p}.npz",
            transformer=tf_ms,
            milp=milp_ms,
            heuristic=heur_ms,
            milp_ub=milp_ub,
            milp_lb=milp_lb,
            rand_ms=rand_ms,
            ga_ms=ga_ms,
            ig_ms=ig_ms
        )
