"""
Held-out benchmark: runs the full method comparison (transformer, IG, NEH, GA,
MILP, random) on the prefixes produced by generate_heldout_prefixes.py, i.e.
prefixes that never occur as model input in the training data.

Everything (methods, budgets, time limits, DES re-evaluation) is identical to
the main benchmark in evaluate_model2.py; only the prefixes differ. Each
instance is run with its stored generator seed, which reproduces the held-out
prefix through the unchanged generator; a guard asserts this for every
instance, so no other code path is altered.

Results are written to results_heldout/makespan_prefix{p}.npz with the same
keys as the main benchmark (plus "seeds"), so plot_benchmark.py can be reused
by setting data_dir = "results_heldout".
"""

import json
import os
import numpy as np

import evaluate_model2 as ev

ITERATION = 3
HELDOUT_FILE = f"heldout_prefixes_{ITERATION}.json"
OUT_DIR = "results_heldout"

with open(HELDOUT_FILE) as f:
    heldout = json.load(f)

os.makedirs(OUT_DIR, exist_ok=True)

# Guard: the benchmark generates its prefix internally from (prefix_len, seed).
# Wrap the generator to assert it reproduces the stored held-out prefix.
_original_generator = ev.generate_structured_random_prefix
_expected = {"prefix": None}


def _checked_generator(*args, **kwargs):
    prefix = _original_generator(*args, **kwargs)
    expected = [tuple(tok) for tok in _expected["prefix"]]
    assert [tuple(tok) for tok in prefix] == expected, \
        "Generated prefix differs from the stored held-out prefix"
    return prefix


ev.generate_structured_random_prefix = _checked_generator

print(f"Held-out lengths: {heldout['lengths']} "
      f"({heldout['n_per_length']} instances each)")

for p in heldout["lengths"]:
    out_file = os.path.join(OUT_DIR, f"makespan_prefix{p}.npz")
    if os.path.exists(out_file):
        print(f"Prefix length {p}: {out_file} exists, skipping")
        continue

    keys = ["transformer", "milp", "heuristic", "milp_ub", "milp_lb",
            "rand_ms", "ga_ms", "ig_ms"]
    tkeys = ["transformer", "milp", "neh", "random", "ga", "ig"]
    res = {k: [] for k in keys}
    times = {k: [] for k in tkeys}
    seeds = []

    for inst in heldout["prefixes"][str(p)]:
        _expected["prefix"] = inst["prefix"]
        *makespans, rt = ev.evaluate_unseen_structured_prefixes(
            device="cpu",
            seed=inst["seed"],
            population_size=40,
            generations=39,      # 40 x (39 + 1) = 1600 fitness calls
            prefix_len=p,
            max_jobs=heldout["max_jobs"],
        )
        for k, v in zip(keys, makespans):
            res[k].append(v)
        for k in tkeys:
            times[k].append(rt[k])
        seeds.append(inst["seed"])

    np.savez(
        out_file,
        **res,
        time_transformer=times["transformer"],
        time_milp=times["milp"],
        time_heuristic=times["neh"],
        time_rand=times["random"],
        time_ga=times["ga"],
        time_ig=times["ig"],
        seeds=seeds,
    )
    print(f"Saved {out_file}")
