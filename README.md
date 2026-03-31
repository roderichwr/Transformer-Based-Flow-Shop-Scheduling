This repository contains the code for the publication "Transformer-Based Flow Shop Scheduling Using MILP-Generated Training Data"

It contains:

- The trained transformer model transformer_event_model1.pt
- The training data (schedules) permutations_with_workers1.pt as well as the performance data from MILP training data generation makespans_milp-1.npy, makespans_des_1.npy, bounds_milp_1.npy
- The underlying flow shop data to generate the training data and perform the evaluations job_pool_1.npy, eligibility_1.npy, worker_capacities_1.npy
- The evaluation script which compares the different methods to the transformer evaluate_model.py
- Helper scripts for plotting
- The result files shown in the article 
