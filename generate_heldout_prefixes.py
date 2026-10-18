"""
Held-out prefix generation.

Builds a test set of prefixes that provably never occur as model input during
training, to address the memorization concern.

Training samples are (prefix -> next token) pairs, where every prefix is the
LEADING part seq[:t] of a training schedule (train_data2.EventSequenceDataset).
A prefix of length p is therefore "seen" iff it is identical to the first p
tokens of at least one training schedule. A held-out prefix of length p is one
that matches no training schedule's first p tokens.

Prefixes are drawn with exactly the same generator as the main benchmark
(evaluate_model2.generate_structured_random_prefix), using seeds disjoint from
the benchmark seeds (0..49), and kept only if they are unseen and distinct.
If fewer than N unseen prefixes exist at a length (short prefixes, whose space
is fully covered by training data), that length is skipped. The held-out test
starts at the first prefix length for which N unseen prefixes can be found.

Output: heldout_prefixes_{ITERATION}.json (prefixes + generator seeds +
diagnostics), consumed by evaluate_heldout.py.
"""

import json
import numpy as np

from evaluate_model2 import generate_structured_random_prefix

# ---------------------------------------
# SETTINGS
# ---------------------------------------
ITERATION = 3
SEQUENCES_FILE = f"permutations_with_workers_{ITERATION}.npy"
ELIGIBILITY_FILE = f"eligibility_{ITERATION}.npy"
OUT_FILE = f"heldout_prefixes_{ITERATION}.json"

N_PER_LENGTH = 50
PREFIX_LENS = list(range(1, 17))
MAX_JOBS = 8

BENCHMARK_SEEDS = range(50)    # seeds used by the main benchmark (for diagnostics)
SEED_OFFSET = 1_000_000        # held-out seeds start here -> disjoint from benchmark
MAX_ATTEMPTS = 200_000         # generator calls per prefix length at most
PATIENCE = 20_000              # stop a length early if no new unseen prefix for this many calls


# ---------------------------------------
# LOAD TRAINING DATA + VOCABULARY
# ---------------------------------------
sequences = np.load(SEQUENCES_FILE, allow_pickle=True)
eligibility = np.load(ELIGIBILITY_FILE)

# Same tokenization and vocabulary as train_data2.EventSequenceDataset
train_seqs = [tuple(tuple(int(x) for x in tok) for tok in seq) for seq in sequences]
idx_to_token = sorted({tok for seq in train_seqs for tok in seq})
token_to_idx = {tok: i for i, tok in enumerate(idx_to_token)}

max_p = max(PREFIX_LENS)
train_prefixes = {p: set() for p in range(1, max_p + 1)}
for seq in train_seqs:
    for p in range(1, min(max_p, len(seq)) + 1):
        train_prefixes[p].add(seq[:p])

print(f"Loaded {len(train_seqs)} training schedules, vocab size {len(idx_to_token)}")


def generate(p, seed):
    return tuple(generate_structured_random_prefix(
        idx_to_token, token_to_idx,
        job_eligibility=eligibility,
        max_jobs=MAX_JOBS,
        prefix_len=p,
        seed=seed,
    ))


def first_unseen_position(prefix):
    """1-based position of the first token at which the prefix leaves all
    training prefixes (i.e. the smallest q with prefix[:q] unseen)."""
    for q in range(1, len(prefix) + 1):
        if prefix[:q] not in train_prefixes[q]:
            return q
    return None  # fully seen


# ---------------------------------------
# GENERATE HELD-OUT PREFIXES
# ---------------------------------------
heldout = {}
diagnostics = {}

print(f"\n{'p':>3} | {'train prefixes':>14} | {'benchmark seen':>14} | "
      f"{'held-out found':>14} | {'attempts':>8} | {'mean first unseen pos':>21}")
print("-" * 92)

for p in PREFIX_LENS:
    # How much of the existing benchmark at this length was seen in training
    bench_seen = sum(generate(p, s) in train_prefixes[p] for s in BENCHMARK_SEEDS)

    found, seen_here = [], set()
    attempts, since_last_new = 0, 0
    while len(found) < N_PER_LENGTH and attempts < MAX_ATTEMPTS and since_last_new < PATIENCE:
        seed = SEED_OFFSET + p * MAX_ATTEMPTS + attempts
        attempts += 1
        since_last_new += 1

        prefix = generate(p, seed)
        if len(prefix) < p:            # generator ran out of operations
            continue
        if prefix in train_prefixes[p] or prefix in seen_here:
            continue

        seen_here.add(prefix)
        found.append({"seed": seed,
                      "prefix": [list(tok) for tok in prefix],
                      "first_unseen_pos": first_unseen_position(prefix)})
        since_last_new = 0

    mean_pos = (np.mean([f["first_unseen_pos"] for f in found]) if found else float("nan"))
    diagnostics[p] = {
        "n_distinct_training_prefixes": len(train_prefixes[p]),
        "benchmark_prefixes_seen_in_training": int(bench_seen),
        "n_benchmark_prefixes": len(BENCHMARK_SEEDS),
        "heldout_found": len(found),
        "attempts": attempts,
        "mean_first_unseen_pos": float(mean_pos),
    }
    if len(found) == N_PER_LENGTH:
        heldout[p] = found

    print(f"{p:>3} | {len(train_prefixes[p]):>14} | "
          f"{bench_seen:>7}/{len(BENCHMARK_SEEDS):<6} | "
          f"{len(found):>14} | {attempts:>8} | {mean_pos:>21.2f}")

# Held-out test starts at the first length with N unseen prefixes
complete = sorted(heldout)
if not complete:
    raise SystemExit(f"No prefix length yields {N_PER_LENGTH} unseen prefixes.")
start_len = complete[0]
heldout_lens = [p for p in PREFIX_LENS if p >= start_len]
missing = [p for p in heldout_lens if p not in heldout]
if missing:
    print(f"WARNING: lengths {missing} >= {start_len} did not reach {N_PER_LENGTH} "
          f"held-out prefixes and are skipped.")
heldout_lens = [p for p in heldout_lens if p in heldout]

print(f"\nHeld-out test starts at prefix length {start_len}; "
      f"lengths used: {heldout_lens}")

with open(OUT_FILE, "w") as f:
    json.dump({
        "iteration": ITERATION,
        "n_per_length": N_PER_LENGTH,
        "max_jobs": MAX_JOBS,
        "seen_definition": "prefix equals the first p tokens of a training schedule",
        "start_len": start_len,
        "lengths": heldout_lens,
        "prefixes": {str(p): heldout[p] for p in heldout_lens},
        "diagnostics": {str(p): d for p, d in diagnostics.items()},
    }, f, indent=1)

print(f"Saved {OUT_FILE}")
