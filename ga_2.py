import pygad
import random
import numpy as np



# ----------------------------
def simulate_sequence_dynamic_workers(op_sequence, job_pool, worker_capacities, eligibility):
    """
    Fast list-scheduling simulation of an operation sequence
    [(job, machine, worker or None)], CONSISTENT WITH THE DES AND MILP
    SEMANTICS: durations are the raw processing times and each worker group
    is a renewable resource with worker_capacities[g] parallel slots. An
    operation starts at the earliest time >= max(job ready, machine ready)
    at which its group has a free slot for the whole duration, so group
    contention is modeled — the choice of worker group is a real trade-off
    (a busy group delays the start), not just a label.

    worker = None  -> the eligible group with the earliest finish is chosen
                      (greedy fallback).
    worker = g     -> the given group is used (fixed assignments, e.g. the
                      prefix, or assignments searched by NEH/IG/GA).

    Returns: makespan, start_values, assign_values
    """

    start_values = {}
    assign_values = {}

    machine_available = {}
    job_available = {}
    next_machine = {}
    # Each group is modeled as cap parallel worker slots (earliest-free-slot
    # assignment, no gap-filling) — matching the DES, where workers are a
    # capacity-cap resource granting requests in order without backfilling.
    group_slots = {g: [0.0] * int(worker_capacities[g])
                   for g in range(len(worker_capacities))}

    for op in op_sequence:
        j, m, g = op
        machine_available.setdefault(m, 0.0)
        job_available.setdefault(j, 0.0)
        next_machine.setdefault(j, 0)

    for op in op_sequence:
        j, m, g = op
        if m < next_machine[j]:
            continue  # already done (prefix)

        ready_time = max(job_available[j], machine_available[m])
        duration = float(job_pool[j, m])

        if g is not None:
            candidates = [g]
        else:
            candidates = [wg for wg in range(len(worker_capacities))
                          if eligibility[j, wg] == 1]

        best_g, best_slot, best_start, best_finish = None, None, None, float("inf")
        for wg in candidates:
            slots = group_slots[wg]
            si = min(range(len(slots)), key=lambda i: slots[i])
            t = max(ready_time, slots[si])
            if t + duration < best_finish:
                best_g, best_slot, best_start, best_finish = wg, si, t, t + duration

        # Commit
        start_values[(j, m)] = best_start
        assign_values[(j, m, best_g)] = 1
        group_slots[best_g][best_slot] = best_finish
        machine_available[m] = best_finish
        job_available[j] = best_finish
        next_machine[j] = m + 1

    makespan = max((v for v in job_available.values()), default=0.0)
    return makespan, start_values, assign_values


# GA Fitness Evaluation Function
def evaluate_schedule(ga_instance, solution, solution_idx):
    """
    Evaluate the schedule by calculating its makespan.
    - ga_instance: The instance of the PyGAD GA class.
    - solution: The solution (individual) to evaluate.
    - solution_idx: The index of the solution within the population.

    The decoded schedule is simulated with the same eligibility-aware
    list scheduler used by the IG and NEH baselines
    (simulate_sequence_dynamic_workers), so the GA operates in exactly the
    same decision space as all other methods
    (job selection + sequencing + worker assignment).

    Returns -makespan, since PyGAD maximizes fitness.
    """
    # Decode the individual into the actual schedule (prefix + completion)
    schedule = decode_individual(solution, ga_instance.ga_context)

    # Simulate the schedule and calculate the makespan.
    # Worker groups of non-prefix operations (g=None) are chosen dynamically
    # among the eligible groups, identical to the IG/NEH baselines.
    makespan, _, _ = simulate_sequence_dynamic_workers(
        schedule,
        ga_instance.processing_times_global,
        ga_instance.worker_capacities,
        ga_instance.eligibility
    )

    return -makespan  # PyGAD maximizes fitness -> minimize makespan


def build_ga_context(prefix_tokens, candidate_jobs, num_jobs_total, num_machines, eligibility):
    """
    Precomputed decoding context shared by all individuals.

    The genome consists of two parts (random keys):
    - selection genes: one key per NON-PREFIX candidate job of the FULL pool;
      the (num_jobs_total - #prefix_jobs) jobs with the smallest keys are
      selected. Prefix jobs are always selected (mandatory).
    - order genes: num_jobs_total * num_machines keys addressing the
      operations of the selected jobs via (job slot, machine); they define
      the priority used in topological decoding.
    - group genes: num_jobs_total * num_machines keys addressing the same
      operations; each key selects one of the job's ELIGIBLE worker groups
      (the key's fractional part is mapped onto the eligible-group list), so
      the worker-group assignment is a searched degree of freedom exactly
      like selection and sequencing.
    """
    prefix_jobs = sorted({j for (j, m, g) in prefix_tokens})

    next_machine = {}
    for (j, m, g) in prefix_tokens:
        next_machine[j] = max(next_machine.get(j, 0), m + 1)

    others = [c for c in candidate_jobs if c not in prefix_jobs]
    n_add = num_jobs_total - len(prefix_jobs)

    eligible_groups = {j: [g for g in range(eligibility.shape[1]) if eligibility[j, g] == 1]
                       for j in candidate_jobs}

    return {
        "eligible_groups": eligible_groups,
        "prefix_tokens": list(prefix_tokens),
        "prefix_jobs": prefix_jobs,
        "next_machine": next_machine,
        "others": others,          # selectable (non-prefix) candidates
        "n_add": n_add,            # number of jobs the GA must select
        "num_jobs_total": num_jobs_total,
        "num_machines": num_machines,
        "num_sel_genes": len(others),
        "num_order_genes": num_jobs_total * num_machines,
        "num_group_genes": num_jobs_total * num_machines,
    }


# Decode the GA individual into a full schedule (prefix + completion)
def decode_individual(genes, ctx):
    """
    Decodes a GA individual into a schedule.

    1) Job selection: prefix jobs are mandatory; among the non-prefix pool
       candidates, the n_add jobs with the smallest selection keys are chosen.
    2) Sequencing: the remaining operations of all selected jobs (every
       selected job must be completed on all machines) are decoded
       topologically: at each step, among the operations whose flow-shop
       predecessor is already scheduled, the one with the smallest order key
       is appended. This guarantees feasibility and full completion.

    Returns the decoded schedule (list of tuples: (job_id, machine, worker_group)),
    where worker_group is None for non-prefix operations (decided during simulation).
    """
    others = ctx["others"]
    n_sel = ctx["num_sel_genes"]
    M = ctx["num_machines"]

    sel_keys = genes[:n_sel]
    order_keys = genes[n_sel:n_sel + ctx["num_order_genes"]]
    group_keys = genes[n_sel + ctx["num_order_genes"]:]

    # 1) Job selection
    chosen_idx = sorted(range(len(others)), key=lambda i: (sel_keys[i], i))[:ctx["n_add"]]
    selected = sorted(ctx["prefix_jobs"] + [others[i] for i in chosen_idx])
    slot = {j: i for i, j in enumerate(selected)}

    # 2) Remaining operations (all selected jobs completed on all machines)
    next_machine = dict(ctx["next_machine"])
    remaining_ops = []
    for j in selected:
        next_machine.setdefault(j, 0)
        for m in range(next_machine[j], M):
            remaining_ops.append((j, m))

    schedule = [(j, m, g) for (j, m, g) in ctx["prefix_tokens"]]

    def priority(op):
        j, m = op
        return order_keys[slot[j] * M + m]

    def group_of(op):
        j, m = op
        elig = ctx["eligible_groups"][j]
        key = float(group_keys[slot[j] * M + m])
        frac = key - np.floor(key)  # map any real key into [0, 1)
        return elig[min(int(frac * len(elig)), len(elig) - 1)]

    # Random-key topological decoding
    pending = set(range(len(remaining_ops)))
    while pending:
        feasible = [i for i in pending
                    if remaining_ops[i][1] == next_machine[remaining_ops[i][0]]]
        if not feasible:
            raise RuntimeError("GA decoding stalled: no feasible operation")

        i_best = min(feasible, key=lambda i: (priority(remaining_ops[i]), i))
        j, m = remaining_ops[i_best]

        schedule.append((j, m, group_of((j, m))))  # searched worker group
        next_machine[j] = m + 1
        pending.remove(i_best)

    return schedule


# Create a new individual for the GA (random keys)
def create_individual(num_genes):
    """
    Creates a new individual: uniform random keys for both the selection part
    and the order part of the genome.
    """
    return [random.random() for _ in range(num_genes)]


# Main GA Function for Flowshop Scheduling
def ga_flowshop_schedule(processing_times_global, worker_capacities, eligibility,
                         prefix_tokens, candidate_jobs, num_jobs_total,
                         population_size=50, generations=100):
    """
    Run the GA to complete a fixed prefix in the FULL decision space:
    - processing_times_global: job processing times
    - worker_capacities: worker capacities
    - eligibility: job–worker-group eligibility matrix
    - prefix_tokens: fixed job tokens (prefix)
    - candidate_jobs: the FULL job pool from which jobs may be selected
    - num_jobs_total: total number of jobs to use (prefix jobs included);
      every selected job must be completed on all machines

    The GA evolves the job selection, the operation order, AND the worker
    group assignment of every operation (restricted to eligible groups).

    Returns (start_values, assign_values, makespan, selected_jobs) of the best
    schedule found, in the same format as the IG and NEH baselines.
    """
    num_machines = processing_times_global.shape[1]
    ctx = build_ga_context(prefix_tokens, candidate_jobs, num_jobs_total, num_machines, eligibility)

    num_genes = ctx["num_sel_genes"] + ctx["num_order_genes"] + ctx["num_group_genes"]

    # Create the population of individuals (random keys)
    population = [create_individual(num_genes) for _ in range(population_size)]

    # Initialize the GA instance with the correct parameters
    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=min(15, population_size),  # Increased from 10 to improve mating selection
        fitness_func=evaluate_schedule,
        sol_per_pop=population_size,
        num_genes=num_genes,
        initial_population=population,  # Initial population
        crossover_type="uniform",  # Uniform crossover
        crossover_probability=0.7,  # Keeps the probability of crossover
        mutation_type="random",  # Random mutation (consider other types if needed)
        mutation_probability=0.3,  # Increased mutation probability to encourage diversity
        parent_selection_type="tournament",  # Tournament selection (standard)
        keep_parents=2,  # Keep the best 2 parents to preserve good solutions
        mutation_by_replacement=True  # Mutated individuals replace the original ones
    )
    ga_instance.ga_context = ctx
    ga_instance.processing_times_global = processing_times_global
    ga_instance.worker_capacities = worker_capacities
    ga_instance.eligibility = eligibility
    # Run the genetic algorithm
    ga_instance.run()

    # Get the best solution found
    best_solution = ga_instance.best_solution()

    # Decode the best solution to get the schedule and resolve worker groups
    best_schedule = decode_individual(best_solution[0], ctx)

    makespan, start_values, assign_values = simulate_sequence_dynamic_workers(
        best_schedule,
        processing_times_global,
        worker_capacities,
        eligibility
    )

    selected_jobs = sorted({j for (j, m, g) in best_schedule})

    return start_values, assign_values, makespan, selected_jobs