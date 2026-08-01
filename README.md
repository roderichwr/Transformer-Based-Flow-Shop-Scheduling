# Transformer-Based Flow Shop Scheduling Using MILP-Generated Training Data

Code and data for the paper "Transformer-Based Flow Shop Scheduling Using MILP-Generated Training Data".

The repository contains three experimental iterations using independently generated flow shop instances of the same size (20 jobs 4 machines 3 worker groups). Iteration 1 restricted baseline methods to the transformer's selected job set; iteration 2 and 3 have all methods independently select and schedule jobs from the full pool. Iteration 3 (published) uses 40,000 training schedules at 1,800 s solve time; iteration 2 used 100,000 at 600 s.

## Files

| File | Description |
|------|-------------|
| `transformer_event_model3.pt` | Trained transformer model |
| `permutations_with_workers_3.npy` | MILP-generated training schedules (40,000 instances, 1800 s solve time) |
| `makespans_milp_3.npy` | MILP primal objective values (makespans) from training data generation |
| `bounds_milp_3.npy` | MILP dual bounds from training data generation |
| `makespans_des_3.npy` | DES-simulated makespans of the training schedules |
| `job_pool_3.npy` | Processing times for all 20 pool jobs on 4 machines |
| `eligibility_3.npy` | Job–worker-group eligibility matrix |
| `worker_capacities_3.npy` | Capacities of the 3 worker groups |
| `results/` | Evaluation results (one `.npz` file per prefix length) |
| `generate_data.py` | Generates MILP training schedules |
| `train_data2.py` | Trains the transformer |
| `evaluate_model2.py` | Evaluates all methods; writes results to `results/` |
| `ga_2.py` | GA baseline (imported by `evaluate_model2.py`) |
| `plot_benchmark.py` | Plots makespan and runtime distributions, prints summary statistics |
| `plot_MILP_gaps.py` | Plots MILP optimality gap distributions for training and evaluation data |