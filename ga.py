import pygad
import random
import numpy as np
from deap import base, creator, tools
from generate_data import *

# GA Fitness Evaluation Function (Updated for PyGAD 2.20.0)
def evaluate_schedule(ga_instance, solution, solution_idx):
    """
    Evaluate the schedule by calculating its makespan.
    - ga_instance: The instance of the PyGAD GA class.
    - solution: The solution (individual) to evaluate.
    - solution_idx: The index of the solution within the population.
    
    Returns the makespan as the fitness.
    """
    # Decode the individual (ranking) into the actual schedule (including prefix + generated)
    schedule = decode_individual(solution, ga_instance.generated_tokens, ga_instance.prefix_tokens)

    # Simulate the schedule and calculate the makespan
    start_events = [("START", job_id, machine, worker_group, 0.0) for job_id, machine, worker_group in schedule]
    
    makespan, _, _, _ = simulate_flowshop_events_simpy(
        start_events,
        ga_instance.processing_times_global,
        ga_instance.worker_capacities,
        verbose=False
    )
    
    return makespan,  # Return the makespan for fitness

# Decode the GA individual (ranking) into a full schedule (prefix + new sequence)
def decode_individual(ranking, generated_tokens, prefix_tokens):
    """
    Decodes a GA individual (ranking of jobs) into a schedule.
    - ranking: list of ranks representing the order of jobs
    - generated_tokens: list of generated tokens for remaining jobs
    - prefix_tokens: list of fixed tokens (prefix)
    
    Returns the decoded schedule (list of tuples: (job_id, machine, worker_group)).
    """
    remaining_jobs = generated_tokens[len(prefix_tokens):]  # Take all jobs after the prefix

    # Create the full schedule: start with prefix tokens
    schedule = [(j_global, m, g) for j_global, m, g in prefix_tokens]

    # Create a dictionary to group jobs by machine
    jobs_by_machine = {}
    for i in range(len(ranking)):
        job = remaining_jobs[i]
        job_id, machine_id, worker_group = job
        if machine_id not in jobs_by_machine:
            jobs_by_machine[machine_id] = []
        jobs_by_machine[machine_id].append((job_id, machine_id, worker_group))
    
    # Sort each machine's jobs based on their job_id (preserving flowshop constraint)
    for machine_id in jobs_by_machine:
        jobs_by_machine[machine_id].sort(key=lambda x: x[0])  # Sort jobs by job_id for each machine

    # Add the jobs in sorted order per machine to the schedule
    for machine_id in sorted(jobs_by_machine.keys()):
        machine_jobs = jobs_by_machine[machine_id]
        for job in machine_jobs:
            schedule.append(job)

    return schedule

# Create a new individual for the GA (operating on ranks)
def create_individual(num_jobs, generated_tokens, prefix_tokens):
    """
    Creates a new individual with a fitness attribute.
    - num_jobs: number of remaining jobs
    - generated_tokens: list of generated tokens for the remaining jobs
    - prefix_tokens: list of fixed tokens (prefix)
    
    Returns an Individual object representing the order of jobs to schedule.
    """
    remaining_jobs = generated_tokens[len(prefix_tokens):]  # Take all jobs after the prefix
    
    # Create an individual represented by a random ranking of jobs (from 0 to N-1)
    individual = list(range(num_jobs))  # Start with a list of ranks: [0, 1, 2, ..., N-1]
    random.shuffle(individual)  # Shuffle to randomize the initial order
    
    assert len(individual) == len(set(individual))  # Ensure uniqueness of individual
    return individual

# Mutation Step: Modify Mutation applied to ranks
def mutate(ranking, mutation_rate):
    if random.random() < mutation_rate:
        # Randomly select one index to modify its rank value
        idx = random.randint(0, len(ranking) - 1)
        new_rank = random.randint(0, len(ranking) - 1)  # Generate a new rank
        while new_rank == ranking[idx]:  # Ensure the new rank is different from the current one
            new_rank = random.randint(0, len(ranking) - 1)
        ranking[idx] = new_rank  # Assign the new rank to the chosen index
    return ranking

# Main GA Function for Flowshop Scheduling
def ga_flowshop_schedule(processing_times_global, worker_capacities, prefix_tokens, generated_tokens, population_size=50, generations=100):
    """
    Run the GA to optimize the job sequence for flowshop scheduling.
    - processing_times_global: job processing times
    - worker_capacities: worker capacities
    - prefix_tokens: fixed job tokens
    - generated_tokens: generated job tokens for the remaining jobs
    
    Returns the best schedule found by the GA.
    """
    num_genes = len(generated_tokens) - len(prefix_tokens)  # Number of genes per individual (remaining jobs)
    
    # Create the necessary DEAP components (Fitness, Individual)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize makespan
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create the population of individuals (represented by ranks)
    population = [create_individual(num_genes, generated_tokens, prefix_tokens) for _ in range(population_size)]

    # Initialize the GA instance with the correct parameters
    ga_instance = pygad.GA(
        num_generations=100,  # You can adjust this based on your needs
        num_parents_mating=15,  # Increased from 10 to improve mating selection
        fitness_func=evaluate_schedule,  # Updated fitness function
        sol_per_pop=100,  # Increased population size for better exploration
        num_genes=num_genes,  # Number of genes in an individual (remaining jobs)
        initial_population=population,  # Initial population
        crossover_type="uniform",  # Uniform crossover
        crossover_probability=0.7,  # Keeps the probability of crossover
        mutation_type="random",  # Random mutation (consider other types if needed)
        mutation_probability=0.3,  # Increased mutation probability to encourage diversity
        parent_selection_type="tournament",  # Tournament selection (standard)
        keep_parents=2,  # Keep the best 2 parents to preserve good solutions
        mutation_by_replacement=True  # Mutated individuals replace the original ones
    )
    ga_instance.generated_tokens = generated_tokens
    ga_instance.prefix_tokens = prefix_tokens
    ga_instance.processing_times_global = processing_times_global
    ga_instance.worker_capacities = worker_capacities
    # Run the genetic algorithm
    ga_instance.run()

    # Get the best solution found
    best_solution = ga_instance.best_solution()

    # Decode the best solution to get the schedule
    best_schedule = decode_individual(best_solution[0], generated_tokens, prefix_tokens)

    return best_schedule