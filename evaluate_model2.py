import numpy as np
import torch
import random
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from ga_2 import *
import time

from train_data2 import TransformerModel, EventSequenceDataset
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

    If max_distinct_jobs is given, opening a NEW job is only feasible while
    fewer than max_distinct_jobs jobs have been started. Jobs may be selected
    freely from the whole pool (job selection is part of the decision space),
    but with a token budget of max_distinct_jobs * num_machines this forces
    every started job to be completed on all machines — the same requirement
    imposed on all baseline methods.
    """

    def __init__(self, job_eligibility, max_distinct_jobs=None, num_machines=None):
        self.next_machine = defaultdict(int)
        self.job_eligibility = job_eligibility
        self.max_distinct_jobs = max_distinct_jobs
        self.num_machines = num_machines

    def is_feasible(self, tok):
        job, machine, group = tok

        # 0) Cap on the number of distinct jobs (job selection budget)
        if self.max_distinct_jobs is not None:
            started = sum(1 for v in self.next_machine.values() if v > 0)
            if self.next_machine[job] == 0 and started >= self.max_distinct_jobs:
                return False

        # 0b) Job must not already be fully scheduled
        if self.num_machines is not None and self.next_machine[job] >= self.num_machines:
            return False

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

def select_feasible_token(logits, state, idx_to_token, rng=None, temperature=1.0, top_p=0.9):
    """
    Select a token that is structurally feasible AND respects
    job–worker-group eligibility.

    rng is None  -> greedy: the feasible token with the highest probability.
    rng given    -> nucleus (top-p) sampling: among the feasible tokens, keep
                    the smallest set of highest-probability tokens whose
                    cumulative (renormalized) mass reaches top_p, and sample
                    from it. Randomness is therefore injected only among
                    continuations the model itself considers plausible, which
                    yields diverse but high-quality completions for best-of-K
                    search (plain full-distribution sampling wastes budget on
                    tail tokens the model effectively rules out).
                    top_p = 1.0 recovers plain sampling; temperature controls
                    the spread within the nucleus.
    """
    if rng is None:
        probs = torch.softmax(logits, dim=-1)
        sorted_ids = torch.argsort(probs, descending=True)

        for idx in sorted_ids:
            tok = idx_to_token[idx.item()]
            if state.is_feasible(tok):
                return idx.item()

        raise RuntimeError("No feasible token found under eligibility constraints")

    # ---- nucleus sampling over the feasible token set ----
    feasible_ids = [i for i, tok in enumerate(idx_to_token) if state.is_feasible(tok)]
    if not feasible_ids:
        raise RuntimeError("No feasible token found under eligibility constraints")

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    p = probs[feasible_ids].detach().cpu().numpy().astype(float)
    total = p.sum()
    if total <= 0:
        p = np.ones(len(feasible_ids)) / len(feasible_ids)
    else:
        p = p / total

    # top-p filtering on the feasible-renormalized distribution
    order = np.argsort(-p)
    cumulative = np.cumsum(p[order])
    cutoff = int(np.searchsorted(cumulative, min(max(top_p, 1e-6), 1.0)) + 1)
    keep = order[:cutoff]
    p_keep = p[keep] / p[keep].sum()

    return int(feasible_ids[keep[rng.choice(len(keep), p=p_keep)]])



# ------------------------------------------------------------
# Autoregressive generation with reconciliation
# ------------------------------------------------------------

def greedy_generate_with_repair(
    model,
    start_prefix_ids,
    idx_to_token,
    max_len,
    device="cpu",
    eligibility=None,
    max_distinct_jobs=None,
    num_machines=None,
    rng=None,
    temperature=1.0,
    top_p=0.9
):
    """
    Autoregressive generation with feasibility reconciliation.
    Greedy decoding when rng is None; stochastic (feasible-set sampling)
    when a numpy Generator is passed — used to draw diverse completions.

    Job selection remains free (any job of the pool/vocabulary may be opened),
    but if max_distinct_jobs and num_machines are given, at most
    max_distinct_jobs distinct jobs can be started. With
    max_len = max_distinct_jobs * num_machines this forces exactly
    max_distinct_jobs jobs, each completed on all machines — the same
    decision space (job selection + sequencing + worker assignment) in which
    all baseline methods operate.
    """
    model.eval()
    seq = list(start_prefix_ids)

    state = ScheduleState(eligibility, max_distinct_jobs=max_distinct_jobs,
                          num_machines=num_machines)
    for tok_id in seq:
        state.apply(idx_to_token[tok_id])

    with torch.no_grad():
        while len(seq) < max_len:
            x = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            mask = torch.ones_like(x, dtype=torch.float32)

            logits = model(x, mask)[0]
            next_id = select_feasible_token(logits, state, idx_to_token,
                                            rng=rng, temperature=temperature,
                                            top_p=top_p)

            seq.append(next_id)
            state.apply(idx_to_token[next_id])

    return seq


@torch.inference_mode()
def batched_sample_completions(
    model,
    start_prefix_ids,
    idx_to_token,
    max_len,
    num_samples,
    eligibility,
    max_distinct_jobs,
    num_machines,
    temperature=1.0,
    top_p=0.9,
    seed=None,
    device="cpu",
    greedy_first=True,
    seed_offset=0
):
    """
    Generate num_samples completions of the same prefix IN PARALLEL (one
    batched forward pass per decoding step instead of one per sample).

    Sample 0 uses deterministic greedy decoding; samples k >= 1 use nucleus
    sampling with an independent RNG stream seeded by (seed, k). Each sample's
    logits depend only on its own sequence, so the trajectories are identical
    to generating the samples one by one -- batching only changes the speed
    (roughly an order of magnitude on CPU), which is what makes large search
    budgets (hundreds of completions) affordable per instance.

    Returns a list of num_samples token-id sequences of length max_len.
    """
    model.eval()

    seqs = [list(start_prefix_ids) for _ in range(num_samples)]
    states = []
    rngs = []
    for k in range(num_samples):
        st = ScheduleState(eligibility, max_distinct_jobs=max_distinct_jobs,
                           num_machines=num_machines)
        for tok_id in start_prefix_ids:
            st.apply(idx_to_token[tok_id])
        states.append(st)
        rngs.append(None if (k == 0 and greedy_first) else
                    np.random.default_rng([0 if seed is None else seed, seed_offset + k]))

    while len(seqs[0]) < max_len:
        x = torch.tensor(seqs, dtype=torch.long).to(device)
        mask = torch.ones_like(x, dtype=torch.float32)
        logits = model(x, mask)  # (num_samples, vocab)

        for k in range(num_samples):
            next_id = select_feasible_token(logits[k], states[k], idx_to_token,
                                            rng=rngs[k], temperature=temperature,
                                            top_p=top_p)
            seqs[k].append(next_id)
            states[k].apply(idx_to_token[next_id])

    return seqs

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
    - Completes the EXACT same fixed instance operations as all other methods
      (all instance jobs finished on all machines)
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
                # Dropping the operation would violate the requirement that all
                # methods complete every instance operation.
                raise RuntimeError(f"No vocabulary token for operation ({j}, {m})")
            tok = rng.choice(valid_tokens)

        seq_tokens.append(tok)
        state.apply(tok)
        remaining.remove((j, m))

    # Encode
    return [token_to_idx[tok] for tok in seq_tokens]


# ------------------------------------------------------------
# Shared helpers for job selection + sequencing methods
# ------------------------------------------------------------

def prefix_next_machine(prefix_tokens):
    """Per-job machine progress implied by the prefix."""
    nm = defaultdict(int)
    for (j, m, g) in prefix_tokens:
        nm[j] = max(nm[j], m + 1)
    return nm


def jobs_in_prefix(prefix_tokens):
    return sorted({j for (j, m, g) in prefix_tokens})


def random_completion_with_selection(
    prefix_ids,
    prefix_tokens,
    candidate_jobs,
    num_jobs_total,
    num_machines,
    idx_to_token,
    token_to_idx,
    job_eligibility,
    seed=None
):
    """
    Random baseline with FREE job selection from the whole pool:
    randomly selects additional jobs (uniformly among all candidates) until
    num_jobs_total jobs are used, then completes all selected jobs on all
    machines in random topological order.
    """
    rng = random.Random(seed)

    pjobs = jobs_in_prefix(prefix_tokens)
    others = [c for c in candidate_jobs if c not in pjobs]
    n_add = num_jobs_total - len(pjobs)
    extra = rng.sample(others, n_add) if n_add > 0 else []

    selected = sorted(pjobs + extra)
    active_operations = {(j, m) for j in selected for m in range(num_machines)}

    return generate_random_completion_from_prefix(
        prefix_ids=prefix_ids,
        active_operations=active_operations,
        idx_to_token=idx_to_token,
        token_to_idx=token_to_idx,
        job_eligibility=job_eligibility,
        seed=seed
    )




# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_unseen_structured_prefixes(
    sequences_file="permutations_with_workers_3.npy",
    model_file="transformer_event_model3.pt",
    job_pool_file="job_pool_3.npy",
    worker_caps_file="worker_capacities_3.npy",
    elibility_file="eligibility_3.npy",
    prefix_len=16,
    max_jobs=8,
    device="cpu",
    population_size=50,
    generations=100,
    transformer_samples=1600,
    transformer_temperature=1.0,
    transformer_top_p=1.0,
    random_samples=None,   # None -> same number of completions as the transformer search
    seed=None
):

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
    # Every method receives the same prefix and the FULL job pool, and
    # independently performs BOTH job selection and scheduling: it selects
    # jobs from the pool until num_jobs_total = max_jobs jobs are used
    # (prefix jobs included) and must complete all selected jobs on all
    # machines, respecting worker eligibility. Different methods may
    # therefore complete the prefix with different jobs.
    num_machines = job_pool.shape[1]
    num_jobs_total = max_jobs
    candidate_jobs = sorted({j for (j, m, g) in dataset.idx_to_token})  # full pool (vocab-safe)
    target_len = num_jobs_total * num_machines  # full completion length

    prefix_jobs = jobs_in_prefix(prefix_tokens)
    next_machine_prefix = prefix_next_machine(prefix_tokens)

    print(f"Prefix jobs = {prefix_jobs}")
    print(f"Candidate job pool = {candidate_jobs}")
    print(f"Each method selects {num_jobs_total} jobs in total and completes them "
          f"on all {num_machines} machines")

    prefix_ids = encode_prefix(prefix_tokens, dataset.token_to_idx)

    print(f"Transformer sampling: temperature = {transformer_temperature}, "
          f"top_p = {transformer_top_p}")

    # ------------------------------------------------------------
    # TRANSFORMER SEARCH: best of transformer_samples completions
    # ------------------------------------------------------------
    tick = time.time()

    makespan = float("inf")
    generated_tokens = None
    start_events = None
    completion_times = None
    sample_makespans = []

    all_ids = batched_sample_completions(
        model,
        start_prefix_ids=prefix_ids,
        idx_to_token=dataset.idx_to_token,
        max_len=target_len,
        num_samples=max(1, int(transformer_samples)),
        eligibility=eligibility,
        max_distinct_jobs=num_jobs_total,
        num_machines=num_machines,
        temperature=transformer_temperature,
        top_p=transformer_top_p,
        seed=seed,
        device=device
    )

    for generated_ids_k in all_ids:
        tokens_k = decode_sequence(generated_ids_k, dataset.idx_to_token)

        # Sanity check: exactly num_jobs_total jobs, all completed on all
        # machines (no partial jobs)
        ops_k = [(j, m) for (j, m, g) in tokens_k]
        jobs_k = sorted({j for (j, m) in ops_k})
        assert len(ops_k) == len(set(ops_k)), \
            "Transformer scheduled an operation twice"
        assert len(jobs_k) == num_jobs_total, \
            f"Transformer used {len(jobs_k)} jobs instead of {num_jobs_total}"
        assert set(ops_k) == {(j, m) for j in jobs_k for m in range(num_machines)}, \
            "Transformer left a selected job partially unfinished"

        # Build START events exactly like in training and evaluate with DES
        events_k = [("START", j, m, g, 0.0) for (j, m, g) in tokens_k]
        makespan_k, completion_k, log_k, _ = simulate_flowshop_events_simpy(
            events_k,
            job_pool,
            worker_capacities,
            verbose=False
        )
        sample_makespans.append(makespan_k)

        if makespan_k < makespan:
            makespan = makespan_k
            generated_tokens = tokens_k
            start_events = events_k
            completion_times = completion_k

    tock = time.time()
    runtime_transformer = tock - tick
    print(f"Transformer search: {len(sample_makespans)} completions in {tock - tick:.2f} s "
          f"| best = {min(sample_makespans):.2f}, mean = {np.mean(sample_makespans):.2f}, "
          f"worst = {max(sample_makespans):.2f} (greedy = {sample_makespans[0]:.2f})")

    transformer_jobs = sorted({j for (j, m, g) in generated_tokens})
    print(f"Transformer selected jobs (best completion) = {transformer_jobs}")

    # ------------------------------------------------------------
    # Plot best transformer-completed schedule using DES pipeline
    # ------------------------------------------------------------

    print("\nSimulating and plotting transformer-completed schedule...")

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

    print("\nTransformer-generated continuation (best of samples, reconciled):")
    for e in generated_tokens[prefix_len:]:
        print(" ", e)


    # ------------------------------------------------------------
    # IG heuristic: job selection from the full pool + sequencing
    # ------------------------------------------------------------
    print("\nCompleting prefix with Iterated Greedy (IG) heuristic (free job selection)...")

    tick = time.time()
    start_values_ig, assign_values_ig, makespan_ig, jobs_ig = iterated_greedy_with_selection(
        prefix_tokens=prefix_tokens,
        candidate_jobs=candidate_jobs,
        num_jobs_total=num_jobs_total,
        job_pool=job_pool,
        worker_capacities=worker_capacities,
        eligibility=eligibility,
        max_iter=int(transformer_samples),
        destruction_size=2,
        acceptance_prob=0.1
    )
    tock = time.time()
    runtime_ig = tock - tick
    print(f"IG heuristic optimization took {tock - tick:.2f} seconds")
    print(f"IG heuristic makespan = {makespan_ig:.2f}")
    print(f"IG selected jobs = {jobs_ig}")

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
    # RANDOM BASELINE (same prefix, free job selection from the full pool)
    # ------------------------------------------------------------

    print("prefix_tokens", prefix_tokens )

    # Best-of-N random search: like the other methods, the random baseline
    # receives a search budget. random_samples completions (random job
    # selection, random topological order, random eligible worker groups)
    # are generated with independent seeds and DES-evaluated; the one with
    # the minimal makespan is kept.
    tick = time.time()

    makespan_rand = float("inf")
    random_tokens = None
    start_events_rand = None
    completion_times_rand = None
    rand_makespans = []

    if random_samples is None:
        random_samples = transformer_samples  # equal search budget (same number of completions)

    base = 0 if seed is None else seed
    for k in range(max(1, int(random_samples))):
        random_ids_k = random_completion_with_selection(
            prefix_ids=prefix_ids,
            prefix_tokens=prefix_tokens,
            candidate_jobs=candidate_jobs,
            num_jobs_total=num_jobs_total,
            num_machines=num_machines,
            idx_to_token=dataset.idx_to_token,
            token_to_idx=dataset.token_to_idx,
            job_eligibility=eligibility,
            seed=base * 1_000_003 + k
        )
        tokens_k = decode_sequence(random_ids_k, dataset.idx_to_token)

        events_k = [("START", j, m, g, 0.0) for (j, m, g) in tokens_k]
        mk_k, completion_k, _, _ = simulate_flowshop_events_simpy(
            events_k,
            job_pool,
            worker_capacities,
            verbose=False
        )
        rand_makespans.append(mk_k)

        if mk_k < makespan_rand:
            makespan_rand = mk_k
            random_tokens = tokens_k
            start_events_rand = events_k
            completion_times_rand = completion_k

    tock = time.time()
    runtime_random = tock - tick
    print(f"Random search: {len(rand_makespans)} completions in {tock - tick:.2f} s "
          f"| best = {min(rand_makespans):.2f}, mean = {np.mean(rand_makespans):.2f}, "
          f"worst = {max(rand_makespans):.2f}")

    print("\nRandom completion (best of samples):")
    for e in random_tokens[prefix_len:]:
        print(" ", e)

    # ------------------------------------------------------------
    # Plot RANDOM schedule
    # ------------------------------------------------------------

    print("\nSimulating and plotting random baseline schedule...")

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
    # GA operates in the same decision space as all other methods:
    # it selects jobs from the FULL candidate pool (job selection genes) and
    # sequences their operations (order genes), with worker groups chosen
    # dynamically among eligible groups only.
    start_values_ga, assign_values_ga, makespan_ga_internal, jobs_ga = ga_flowshop_schedule(
        processing_times_global=job_pool,  # Processing times for each job
        worker_capacities=worker_capacities,  # Worker capacities
        eligibility=eligibility,  # Job–worker-group eligibility
        prefix_tokens=prefix_tokens,  # Fixed job tokens (prefix)
        candidate_jobs=candidate_jobs,  # Full job pool (free job selection)
        num_jobs_total=num_jobs_total,
        population_size=population_size,
        generations=generations
    )
    tock = time.time()
    runtime_ga = tock - tick
    print(f"GA optimization took {tock - tick:.2f} seconds")
    print(f"GA internal makespan = {makespan_ga_internal:.2f}")
    print(f"GA selected jobs = {jobs_ga}")
    # Simulate and plot the GA schedule
    print("\nSimulating and plotting GA schedule...")

    start_events_ga = []
    for (j, m), t_start in start_values_ga.items():
        assigned_group = None
        for (jj, mm, g), v in assign_values_ga.items():
            if jj == j and mm == m and v > 0.5:
                assigned_group = g
                break
        if assigned_group is None:
            raise RuntimeError(f"No worker group assigned for job {j}, machine {m} (GA)")
        start_events_ga.append(("START", j, m, assigned_group, float(t_start)))

    start_events_ga.sort(key=lambda x: x[4])

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
    # Heuristic (NEH) with job selection from the full pool
    # ------------------------------------------------------------
    print("\nCompleting prefix with NEH heuristic (free job selection)...")

    tick = time.time()
    start_values_heur, assign_values_heur, makespan_heur, jobs_neh, order_neh = (
        neh_heuristic_with_selection(
            prefix_tokens=prefix_tokens,
            candidate_jobs=candidate_jobs,
            num_jobs_total=num_jobs_total,
            job_pool=job_pool,
            worker_capacities=worker_capacities,
            job_eligibility=eligibility
        )
    )
    tock = time.time()
    runtime_neh = tock - tick
    print(f"NEH optimization took {tock - tick:.2f} seconds")
    print(f"NEH heuristic makespan = {makespan_heur:.2f}")
    print(f"NEH selected jobs = {jobs_neh}")

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
    # MILP: job selection from the full pool + sequencing
    # ------------------------------------------------------------
    print("\nCompleting prefix with MILP (free job selection)...")

    print(f"MILP selects {num_jobs_total} jobs from a pool of {len(candidate_jobs)} candidates")

    tick = time.time()

    # ------------------------------------------------------------
    # Build prefix tokens with fixed start times for MILP (as it should not optimize the overall schedule).
    # Start times come from a prefix-only DES simulation, so they are
    # method-independent (identical regardless of any completion).
    # ------------------------------------------------------------
    prefix_start_events = [("START", j, m, g, 0.0) for (j, m, g) in prefix_tokens]
    _, prefix_completion_times, _, _ = simulate_flowshop_events_simpy(
        prefix_start_events,
        job_pool,
        worker_capacities,
        verbose=False
    )
    prefix_start_times, _ = extract_info_from_events(
        prefix_start_events,
        prefix_completion_times,
        job_pool
    )

    prefix_tokens_milp = []

    for (j, m, g) in prefix_tokens:   # ONLY the prefix
        t_start = prefix_start_times[(j, m)]
        prefix_tokens_milp.append((j, m, g, t_start))

    # The MILP starts from scratch: it receives ONLY the problem data
    # (pool, prefix, eligibility, capacities) and derives its own bounds
    # internally — no information from NEH or any other method.

    # Solve MILP with free job selection over the full candidate pool
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
        candidate_indices=candidate_jobs,
        num_select=num_jobs_total,
        prefix_tokens=prefix_tokens_milp,
        time_scale=1,
        verbose=False
    )
    tock = time.time()
    runtime_milp = tock - tick
    print(f"MILP optimization took {tock - tick:.2f} seconds")

    jobs_milp = sorted({j for (j, m) in start_values.keys()})
    print(f"MILP makespan = {makespan_value:.2f}")
    print(f"MILP objective bound = {obj_bound:.2f}")
    print(f"MILP selected jobs = {jobs_milp}")

    if np.isnan(makespan_value):
        # No MILP solution within the time limit: mark result files with NaN
        # (obj_bound remains a valid lower bound) and skip the DES simulation.
        makespan_milp_des = float("nan")
        completion_times_milp = {}
    else:
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
        #     title="MILP-Optimized Schedule"
        # )
    else:
        print("WARNING: MILP DES produced empty schedule — skipping plot.")

    runtimes = {
        "transformer": runtime_transformer,
        "milp": runtime_milp,
        "neh": runtime_neh,
        "random": runtime_random,
        "ga": runtime_ga,
        "ig": runtime_ig,
    }

    return makespan, makespan_milp_des ,makespan_heur_des, makespan_value, obj_bound, makespan_rand, makespan_ga, makespan_ig_des, runtimes


class _BudgetExhausted(Exception):
    """Raised when the evaluation-call budget of a search is used up."""
    pass


class _EvalBudget:
    """Counts makespan-evaluation calls (simulator calls) against a limit.
    limit=None means unlimited (used by the exempt NEH construction)."""
    def __init__(self, limit=None):
        self.limit = limit
        self.used = 0

    def spend(self):
        if self.limit is not None and self.used >= self.limit:
            raise _BudgetExhausted()
        self.used += 1


def _remaining_machines(j, next_machine, num_machines):
    return list(range(next_machine.get(j, 0), num_machines))


def _insert_job_oplevel(comp_seq, j, machines_list, prefix_tokens,
                        job_pool, worker_capacities, job_eligibility,
                        budget=None):
    """
    Insert the remaining operations of job j into the completion sequence at
    the OPERATION level: each operation (in flow order) is tried at every
    valid position (after the job's previous operation) AND with every
    ELIGIBLE worker group, and committed at the (position, group) pair with
    the minimum simulated makespan. Both the interleaving of operations and
    the worker-group assignment are therefore searched degrees of freedom.

    Returns (new_comp_seq, makespan).
    """
    seq = list(comp_seq)
    prev_pos = -1  # position of the job's previously inserted operation

    eligible = [g for g in range(len(worker_capacities)) if job_eligibility[j, g] == 1]

    if budget is None:
        budget = _EvalBudget(None)

    if not machines_list:
        budget.spend()
        mk, _, _ = simulate_sequence_dynamic_workers(
            list(prefix_tokens) + seq, job_pool, worker_capacities, job_eligibility)
        return seq, mk

    for m in machines_list:
        best_seq, best_mk, best_pos = None, float("inf"), None
        for pos in range(prev_pos + 1, len(seq) + 1):
            for g in eligible:
                trial = seq[:pos] + [(j, m, g)] + seq[pos:]
                budget.spend()
                mk, _, _ = simulate_sequence_dynamic_workers(
                    list(prefix_tokens) + trial,
                    job_pool, worker_capacities, job_eligibility)
                if mk < best_mk:
                    best_mk, best_seq, best_pos = mk, trial, pos
        seq = best_seq
        prev_pos = best_pos

    return seq, best_mk


def _greedy_select_and_insert_oplevel(comp_seq, unselected, n_add, next_machine,
                                      num_machines, prefix_tokens,
                                      job_pool, worker_capacities, job_eligibility,
                                      budget=None):
    """
    Greedy job selection with operation-level insertion: repeatedly evaluate,
    for every unselected candidate job, the best op-level insertion of its
    operations, and commit the job with the minimum resulting makespan,
    until n_add jobs have been added. Returns (comp_seq, unselected).
    """
    unselected = list(unselected)
    for _ in range(n_add):
        best_seq, best_mk, best_job = None, float("inf"), None
        for c in unselected:
            trial_seq, mk = _insert_job_oplevel(
                comp_seq, c, _remaining_machines(c, next_machine, num_machines),
                prefix_tokens, job_pool, worker_capacities, job_eligibility,
                budget=budget)
            if mk < best_mk:
                best_mk, best_seq, best_job = mk, trial_seq, c
        comp_seq = best_seq
        unselected.remove(best_job)
    return comp_seq, unselected


def neh_heuristic_with_selection(
    prefix_tokens,
    candidate_jobs,          # FULL job pool (same for all methods)
    num_jobs_total,          # jobs to use in total (prefix jobs included)
    job_pool,
    worker_capacities,
    job_eligibility
):
    """
    NEH-style heuristic operating in the full decision space
    (job selection + sequencing + worker assignment), with operation-level
    insertion:

    - Keeps prefix exactly (prefix jobs are mandatory and must be completed)
    - First inserts the completions of the prefix jobs (descending remaining
      workload); each operation is placed individually at its best position,
      so operations of different jobs may interleave
    - Then greedily SELECTS additional jobs from the full candidate pool:
      at each step, every unselected candidate is trially inserted (op-level)
      and the job with the minimum makespan is committed, until
      num_jobs_total jobs are used
    - All selected jobs are completed on all machines
    - Respects worker eligibility (dynamic eligible-group choice)

    Returns (start_values, assign_values, makespan, selected_jobs, comp_seq),
    where comp_seq is the completion operation sequence (without prefix).
    """
    num_machines = job_pool.shape[1]
    next_machine = prefix_next_machine(prefix_tokens)
    pjobs = jobs_in_prefix(prefix_tokens)

    def remaining_pt(j):
        return sum(job_pool[j, m] for m in _remaining_machines(j, next_machine, num_machines))

    # 1) Op-level insertion of prefix-job completions (descending workload)
    comp_seq = []
    for j in sorted(pjobs, key=remaining_pt, reverse=True):
        comp_seq, _ = _insert_job_oplevel(
            comp_seq, j, _remaining_machines(j, next_machine, num_machines),
            prefix_tokens, job_pool, worker_capacities, job_eligibility)

    # 2) Greedy job selection from the full pool (op-level insertion)
    n_add = num_jobs_total - len(pjobs)
    others = [c for c in candidate_jobs if c not in pjobs]

    comp_seq, _ = _greedy_select_and_insert_oplevel(
        comp_seq, others, n_add, next_machine, num_machines,
        prefix_tokens, job_pool, worker_capacities, job_eligibility)

    # Final schedule
    makespan, start_values, assign_values = simulate_sequence_dynamic_workers(
        list(prefix_tokens) + comp_seq, job_pool, worker_capacities, job_eligibility)

    selected = sorted({j for (j, m, g) in comp_seq} | set(pjobs))
    return start_values, assign_values, makespan, selected, comp_seq


def _random_initial_completion(prefix_tokens, candidate_jobs, num_jobs_total,
                               next_machine, num_machines, job_eligibility):
    """
    Random from-scratch initial solution for IG: uniformly random job
    selection (prefix jobs mandatory), a uniformly random topological
    operation order, and a uniformly random ELIGIBLE worker group per
    operation (the group assignment is part of the searched solution).
    """
    pjobs = jobs_in_prefix(prefix_tokens)
    others = [c for c in candidate_jobs if c not in pjobs]
    n_add = num_jobs_total - len(pjobs)
    selected = sorted(pjobs + (random.sample(others, n_add) if n_add > 0 else []))

    num_groups = job_eligibility.shape[1]
    nm = {j: next_machine.get(j, 0) for j in selected}
    comp_seq = []
    pending = {j for j in selected if nm[j] < num_machines}
    while pending:
        j = random.choice(sorted(pending))
        g = random.choice([g for g in range(num_groups) if job_eligibility[j, g] == 1])
        comp_seq.append((j, nm[j], g))
        nm[j] += 1
        if nm[j] >= num_machines:
            pending.remove(j)
    return comp_seq


def iterated_greedy_with_selection(
    prefix_tokens,
    candidate_jobs,          # FULL job pool (same for all methods)
    num_jobs_total,          # jobs to use in total (prefix jobs included)
    job_pool,
    worker_capacities,
    eligibility,
    max_iter=1000,
    destruction_size=2,
    acceptance_prob=0.1,
    max_evaluations=None
):
    """
    Iterated Greedy operating in the full decision space
    (job selection + sequencing + worker assignment), FROM SCRATCH and with
    OPERATION-LEVEL reconstruction:

    - Starts from a RANDOM solution (random job selection, random topological
      operation order) — no information from NEH or any other method
    - Keeps prefix exactly (prefix jobs mandatory, completions repositionable)
    - Destruction: removes all operations of d randomly chosen jobs
    - Reconstruction: removed PREFIX jobs are reinserted op-level at their
      best positions (they are mandatory); the freed slots are refilled by
      greedy op-level insertion competing over ALL currently unselected
      candidates, so the job SET itself is explored across iterations and
      operations of different jobs may interleave; each reinserted operation
      is tried with every eligible worker group, so the group assignment is
      searched as well
    - Acceptance: better solutions always, worse ones with acceptance_prob
    - All selected jobs are completed on all machines; eligibility respected

    max_evaluations caps the TOTAL number of makespan-evaluation calls
    (simulator calls), including every insertion trial during reconstruction.
    When the budget is exhausted, the search stops immediately (a partially
    reconstructed candidate is discarded) and the best complete solution
    found so far is returned.

    Returns (start_values, assign_values, makespan, selected_jobs).
    """
    num_machines = job_pool.shape[1]
    next_machine = prefix_next_machine(prefix_tokens)
    pjobs = set(jobs_in_prefix(prefix_tokens))

    budget = _EvalBudget(max_evaluations)

    def evaluate(comp_seq):
        budget.spend()
        return simulate_sequence_dynamic_workers(
            list(prefix_tokens) + comp_seq, job_pool, worker_capacities, eligibility)

    # --- Random from-scratch initial solution ---
    current_seq = _random_initial_completion(
        prefix_tokens, candidate_jobs, num_jobs_total, next_machine, num_machines,
        eligibility)
    current_mk, best_start_values, best_assign_values = evaluate(current_seq)

    best_seq = list(current_seq)
    best_mk = current_mk

    # --- IG iterations (stopped by max_iter or the evaluation budget) ---
    try:
        for iteration in range(max_iter):

            selected_now = sorted({j for (j, m, g) in current_seq} | pjobs)
            d = min(destruction_size, len(selected_now))
            removed = random.sample(selected_now, d)
            seq = [op for op in current_seq if op[0] not in removed]

            # Mandatory prefix jobs must be reinserted (op-level, best positions)
            for j in removed:
                if j in pjobs:
                    seq, _ = _insert_job_oplevel(
                        seq, j, _remaining_machines(j, next_machine, num_machines),
                        prefix_tokens, job_pool, worker_capacities, eligibility,
                        budget=budget)

            # Refill remaining slots by greedy op-level selection over ALL
            # unselected candidates (removed non-prefix jobs compete with the pool)
            jobs_in_seq = {j for (j, m, g) in seq} | pjobs
            n_add = num_jobs_total - len(jobs_in_seq)
            unselected = [c for c in candidate_jobs if c not in jobs_in_seq]
            seq, _ = _greedy_select_and_insert_oplevel(
                seq, unselected, n_add, next_machine, num_machines,
                prefix_tokens, job_pool, worker_capacities, eligibility,
                budget=budget)

            mk, start_vals, assign_vals = evaluate(seq)

            # --- Acceptance ---
            if mk < current_mk or random.random() < acceptance_prob:
                current_seq = seq
                current_mk = mk

            if mk < best_mk:
                best_seq = list(seq)
                best_mk = mk
                best_start_values = start_vals
                best_assign_values = assign_vals

    except _BudgetExhausted:
        print(f"IG evaluation budget exhausted after {budget.used} calls "
              f"({iteration} completed iterations)")

    # --- Final sanity check: correct number of jobs, all completed ---
    best_jobs = sorted({j for (j, m, g) in best_seq} | pjobs)
    assert len(best_jobs) == num_jobs_total, \
        f"IG used {len(best_jobs)} jobs instead of {num_jobs_total}"

    return best_start_values, best_assign_values, best_mk, best_jobs


def solve_flowshop_with_workers_prefixed(
    processing_times_global,
    worker_capacities,
    job_eligibility,
    candidate_indices,
    num_select,
    prefix_tokens=None,
    time_scale=1,
    verbose=False
):
    """
    MILP flow shop solver with worker groups operating in the FULL decision
    space (job selection + sequencing + worker assignment):

    - Binary selection variables s_j over the whole candidate pool;
      exactly num_select jobs are selected, prefix jobs are forced selected
    - Every selected job is completed on all machines (no partial jobs)
    - Fixes prefix order + worker assignments
    - FENCES prefix on each machine:
        non-prefix selected jobs cannot start before the prefix block completes

    EVENT-BASED CONTINUOUS-TIME FORMULATION.
    The worker-capacity constraints are expressed exactly via the interval
    property: more than cap operations of a group overlap at some
    point in time IFF some cap+1 of them overlap pairwise. Hence capacity
    cap is enforced by requiring, for every (cap+1)-subset of operations
    assignable to the group, that at least one pair is sequenced:

        sum_{pairs {p,q} in Q} (delta_pq + delta_qp) >= sum_{op in Q} x_op,g - cap

    Additional valid inequalities (load cuts) tighten the LP relaxation:
        makespan >= sum_j PT[j,m] * s_j                    for every machine m
        makespan >= (1/cap_g) * sum_{j,m} PT[j,m]*x_j,m,g  for every group g

    - The MILP is fully self-contained (starts from scratch): it takes NO
      information from any other method. Its time horizon / big-M values come
      from an internally computed, always-valid upper bound: the makespan of
      a purely sequential schedule of the prefix jobs plus the additional
      jobs with the smallest total workload (a trivially feasible solution of
      this problem, derived from the instance data only).
    - If no incumbent is found within the time limit, the function returns
      empty schedules and makespan = NaN (a mark for the result files),
      together with the valid dual bound.
    """

    num_jobs = len(candidate_indices)
    num_machines = processing_times_global.shape[1]
    num_groups = len(worker_capacities)

    jobs = range(num_jobs)
    machines = range(num_machines)
    groups = range(num_groups)

    global_to_local = {j: idx for idx, j in enumerate(candidate_indices)}

    PT = np.zeros((num_jobs, num_machines))
    for j_local, j_global in enumerate(candidate_indices):
        PT[j_local, :] = processing_times_global[j_global, :]

    max_pt = float(PT.max())

    # ------------------------------------------------------------------
    prefix_local_jobs = set()
    if prefix_tokens:
        for (j_global, m, g, start) in prefix_tokens:
            if j_global in global_to_local:
                prefix_local_jobs.add(global_to_local[j_global])

    job_totals = PT.sum(axis=1)
    n_add = num_select - len(prefix_local_jobs)
    other_totals = sorted(float(job_totals[j]) for j in jobs if j not in prefix_local_jobs)
    UB = (sum(float(job_totals[j]) for j in prefix_local_jobs)
          + sum(other_totals[:max(0, n_add)])) / time_scale

    bigM = UB + max_pt

    model = gp.Model("flowshop_with_stage_workers_selection")

    if not verbose:
        model.Params.OutputFlag = 0
        model.Params.MIPFocus = 1
        model.Params.TimeLimit = 600
        model.Params.Threads=8


    # ---------------- VARIABLES ----------------
    s = model.addVars(num_jobs, vtype=GRB.BINARY)              # job selection
    assign = model.addVars(num_jobs, num_machines, num_groups, vtype=GRB.BINARY)
    C = model.addVars(num_jobs, num_machines, vtype=GRB.CONTINUOUS)
    makespan = model.addVar(vtype=GRB.CONTINUOUS, ub=UB)

    # ---------------- JOB SELECTION ----------------
    model.addConstr(s.sum() == num_select)

    # ---------------- FLOW CONSTRAINTS ----------------
    # Selected jobs are completed on ALL machines; unselected jobs collapse to C = 0
    for j in jobs:
        model.addConstr(C[j, 0] >= PT[j, 0] * s[j])
        for m in range(1, num_machines):
            model.addConstr(C[j, m] >= C[j, m-1] + PT[j, m] - bigM*(1 - s[j]))
        for m in machines:
            model.addConstr(C[j, m] <= UB * s[j])   # horizon + C=0 if unselected

    # ---------------- ASSIGNMENT ----------------
    for j in jobs:
        for m in machines:
            model.addConstr(assign.sum(j, m, "*") == s[j])

    eligible = {}  # (j_local, g) -> bool, for constraint pruning
    for j_local, j_global in enumerate(candidate_indices):
        for g in groups:
            eligible[(j_local, g)] = (job_eligibility[j_global, g] == 1)
            if not eligible[(j_local, g)]:
                for m in machines:
                    assign[j_local, m, g].ub = 0

    # ---------------- MACHINE DISJUNCTION ----------------
    y = model.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY)

    for m in machines:
        for i in jobs:
            for j in jobs:
                if i < j:
                    model.addConstr(y[i,j,m] + y[j,i,m] == 1)

        for i in jobs:
            for j in jobs:
                if i == j:
                    continue
                model.addConstr(
                    C[i,m] >= C[j,m] + PT[i,m] - bigM*(1 - y[i,j,m])
                              - bigM*(1 - s[i]) - bigM*(1 - s[j])
                )

    # ---------------- PREFIX FIXING + FENCING ----------------
    if prefix_tokens:

        prefix_by_machine = {}
        prefix_assignments = {}
        prefix_locals = set()

        for (j_global, m, g, start) in prefix_tokens:
            if j_global not in global_to_local:
                continue

            j_local = global_to_local[j_global]
            prefix_by_machine.setdefault(m, []).append(j_local)
            prefix_assignments[(j_local, m)] = g
            prefix_locals.add(j_local)

        # Prefix jobs are mandatory
        for j_local in prefix_locals:
            s[j_local].lb = 1

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
                    # FORCE non-prefix (selected) job AFTER prefix block
                    model.addConstr(
                        C[j_local, m] >= C[last_prefix_job, m] - bigM*(1 - s[j_local])
                    )

    # ---------------- WORKER CAPACITY ----------------
    ops = [(j, m) for j in jobs for m in machines]

    pair_list = [
        (p, q)
        for pi, p in enumerate(ops)
        for q in ops[pi+1:]
        if p[0] != q[0] and p[1] != q[1]
    ]

    delta = {}
    for (p, q) in pair_list:
        delta[(p, q)] = model.addVar(vtype=GRB.BINARY)
        delta[(q, p)] = model.addVar(vtype=GRB.BINARY)

        # delta[p,q] = 1  =>  C_p <= S_q = C_q - PT_q
        model.addConstr(
            C[p[0], p[1]] <= C[q[0], q[1]] - PT[q[0], q[1]] + bigM*(1 - delta[(p, q)])
        )
        model.addConstr(
            C[q[0], q[1]] <= C[p[0], p[1]] - PT[p[0], p[1]] + bigM*(1 - delta[(q, p)])
        )

    def pairs_within(subset):
        return [(subset[a], subset[b])
                for a in range(len(subset)) for b in range(a+1, len(subset))]

    from itertools import combinations

    for g in groups:
        cap = int(worker_capacities[g])

        # At most num_machines operations can ever run simultaneously
        if cap >= num_machines:
            continue

        # Operations of jobs eligible for group g
        ops_g = [(j, m) for (j, m) in ops if eligible[(j, g)]]

        # Forbid (cap+1)-cliques of pairwise-overlapping operations in group g:
        # in every subset Q of cap+1 ops (pairwise different jobs AND machines),
        # at least one pair must be sequenced whenever all of Q is assigned to g.
        for Q in combinations(ops_g, cap + 1):
            js = [p[0] for p in Q]
            ms = [p[1] for p in Q]
            if len(set(js)) < len(Q) or len(set(ms)) < len(Q):
                continue  # contains a same-job or same-machine pair: never a clique

            lhs = gp.quicksum(
                delta[(p, q)] + delta[(q, p)] for (p, q) in pairs_within(list(Q))
            )
            rhs = gp.quicksum(assign[p[0], p[1], g] for p in Q) - cap
            model.addConstr(lhs >= rhs)

    # ---------------- LOAD CUTS (valid inequalities) ----------------
    # Machine load: all selected operations of machine m run sequentially
    for m in machines:
        model.addConstr(makespan >= gp.quicksum(PT[j, m] * s[j] for j in jobs))

    # Group load: total work assigned to group g, divided by its capacity
    for g in groups:
        cap = float(worker_capacities[g])
        model.addConstr(
            makespan >= gp.quicksum(
                PT[j, m] * assign[j, m, g] for j in jobs for m in machines
            ) / cap
        )

    # ---------------- OBJECTIVE ----------------
    for j in jobs:
        for m in machines:
            model.addConstr(makespan >= C[j,m])

    model.setObjective(makespan, GRB.MINIMIZE)

    # ---------------- SOLVE (from scratch, no warm start) ----------------
    model.optimize()

    if model.SolCount == 0:
        # No incumbent found within the limits: mark the result as
        # "no solution found" (NaN) and report the valid dual bound.
        try:
            fallback_bound = float(model.ObjBound)
        except (AttributeError, gp.GurobiError):
            fallback_bound = 0.0

        print("WARNING: MILP found no incumbent within the limits "
              f"(status {model.Status}); marking MILP result as NaN, "
              f"bound = {fallback_bound:.2f}")
        return {}, {}, {}, float("nan"), fallback_bound

    C_values = {}
    start_values = {}
    assign_values = {}

    for j_local, j_global in enumerate(candidate_indices):
        if s[j_local].X < 0.5:
            continue

        for m in machines:
            C_values[(j_global,m)] = C[j_local,m].X
            start_values[(j_global,m)] = C_values[(j_global,m)] - PT[j_local,m]

            for g in groups:
                if assign[j_local,m,g].X > 0.5:
                    assign_values[(j_global,m)] = g
                    break

    makespan_value = makespan.X

    return C_values, start_values, assign_values, makespan_value, model.ObjBound


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    N = 50
    prefix_lens = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for p in prefix_lens:
        tf_ms, milp_ms, heur_ms, milp_ub, milp_lb, rand_ms, ga_ms, ig_ms = [], [], [], [], [], [], [], []
        t_tf, t_milp, t_neh, t_rand, t_ga, t_ig = [], [], [], [], [], []

        for seed in range(N):
            m_tf, m_milp, m_heur, m_milp_ub, m_milp_lb, m_rand, m_ga, m_ig, rt = evaluate_unseen_structured_prefixes(
                device="cpu",
                seed=seed,
                population_size=40,
                generations=39,   # 40 x (39 generations + initial population) = 1600 fitness calls

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
            t_tf.append(rt["transformer"])
            t_milp.append(rt["milp"])
            t_neh.append(rt["neh"])
            t_rand.append(rt["random"])
            t_ga.append(rt["ga"])
            t_ig.append(rt["ig"])

        # store raw data (makespans + runtimes per instance)
        np.savez(
            f"{out_dir}/makespan_prefix{p}.npz",
            transformer=tf_ms,
            milp=milp_ms,
            heuristic=heur_ms,
            milp_ub=milp_ub,
            milp_lb=milp_lb,
            rand_ms=rand_ms,
            ga_ms=ga_ms,
            ig_ms=ig_ms,
            time_transformer=t_tf,
            time_milp=t_milp,
            time_heuristic=t_neh,
            time_rand=t_rand,
            time_ga=t_ga,
            time_ig=t_ig
        )