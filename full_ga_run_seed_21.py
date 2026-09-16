
import time
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from deap import base, creator, tools
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight


# Set the main folder and create a folder for the GA results
main_folder = Path.cwd()
results_folder = main_folder / "full_ga_results_seed_21"
results_folder.mkdir(exist_ok=True)

checkpoint_file = results_folder / "full_ga_checkpoint.pkl"


# Load the training and validation data
X_train = pd.read_pickle(
    main_folder / "mordred_training_features_filtered.pkl"
).astype(np.float32)

X_valid = pd.read_pickle(
    main_folder / "mordred_validation_features_filtered.pkl"
).astype(np.float32)

y_train = pd.read_pickle(
    main_folder / "mordred_training_labels.pkl"
).squeeze().astype(np.int8)

y_valid = pd.read_pickle(
    main_folder / "mordred_validation_labels.pkl"
).squeeze().astype(np.int8)

descriptor_names_df = pd.read_csv(
    main_folder / "mordred_filtered_descriptor_names.csv"
)

descriptor_names_df = descriptor_names_df.loc[
    :,
    ~descriptor_names_df.columns.str.startswith("Unnamed")
]

descriptor_names = (
    descriptor_names_df.iloc[:, 0]
    .astype(str)
    .tolist()
)


# Convert the datasets to NumPy arrays for repeated model training
X_train_array = X_train.to_numpy(copy=False)
X_valid_array = X_valid.to_numpy(copy=False)

y_train_array = y_train.to_numpy()
y_valid_array = y_valid.to_numpy()

train_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_array
)


# Use the same XGBoost settings as the mRMR assessment
xgb_params = {
    "objective": "binary:logistic",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 1.00,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1
}


# Full GA settings
population_size = 40
number_of_generations = 30
crossover_probability = 0.80
mutation_probability = 0.20
chromosome_length = 626


# Store evaluated feature subsets so they are not trained twice
fitness_cache = {}


def evaluate_feature_subset(individual):
    feature_mask = np.asarray(individual, dtype=bool)

    # A chromosome must contain at least one descriptor
    if feature_mask.sum() == 0:
        return -1.0

    cache_key = feature_mask.tobytes()

    if cache_key in fitness_cache:
        return fitness_cache[cache_key]

    X_train_selected = X_train_array[:, feature_mask]
    X_valid_selected = X_valid_array[:, feature_mask]

    model = XGBClassifier(**xgb_params)

    model.fit(
        X_train_selected,
        y_train_array,
        sample_weight=train_weights,
        eval_set=[(X_valid_selected, y_valid_array)],
        verbose=False
    )

    validation_probabilities = model.predict_proba(
        X_valid_selected
    )[:, 1]

    validation_predictions = (
        validation_probabilities >= 0.50
    ).astype(np.int8)

    mcc = matthews_corrcoef(
        y_valid_array,
        validation_predictions
    )

    fitness_cache[cache_key] = float(mcc)

    return float(mcc)


# Create the DEAP classes
if not hasattr(creator, "FullGAFitness"):
    creator.create(
        "FullGAFitness",
        base.Fitness,
        weights=(1.0,)
    )

if not hasattr(creator, "FullGAIndividual"):
    creator.create(
        "FullGAIndividual",
        list,
        fitness=creator.FullGAFitness
    )


toolbox = base.Toolbox()

toolbox.register(
    "attribute",
    random.randint,
    0,
    1
)

toolbox.register(
    "individual",
    tools.initRepeat,
    creator.FullGAIndividual,
    toolbox.attribute,
    n=chromosome_length
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)


def ga_fitness(individual):
    return (evaluate_feature_subset(individual),)


toolbox.register("evaluate", ga_fitness)
toolbox.register("mate", tools.cxTwoPoint)

toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=1 / chromosome_length
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)


def evaluate_invalid_individuals(population):
    invalid_individuals = [
        individual
        for individual in population
        if not individual.fitness.valid
    ]

    fitness_values = map(
        toolbox.evaluate,
        invalid_individuals
    )

    for individual, fitness_value in zip(
        invalid_individuals,
        fitness_values
    ):
        individual.fitness.values = fitness_value

    return len(invalid_individuals)


def save_progress(
    generation,
    population,
    hall_of_fame,
    history,
    evaluations,
    elapsed_seconds
):
    hall_of_fame.update(population)

    fitness_values = np.array([
        individual.fitness.values[0]
        for individual in population
    ])

    best_individual = hall_of_fame[0]

    history.append({
        "generation": generation,
        "evaluations": evaluations,
        "best_mcc": best_individual.fitness.values[0],
        "mean_mcc": fitness_values.mean(),
        "minimum_mcc": fitness_values.min(),
        "maximum_mcc": fitness_values.max(),
        "selected_descriptors": sum(best_individual),
        "unique_subsets_evaluated": len(fitness_cache),
        "elapsed_minutes": elapsed_seconds / 60
    })

    pd.DataFrame(history).to_csv(
        results_folder / "full_ga_history.csv",
        index=False
    )

    selected_descriptor_names = [
        descriptor_names[index]
        for index, selected in enumerate(best_individual)
        if selected == 1
    ]

    pd.DataFrame({
        "descriptor": selected_descriptor_names
    }).to_csv(
        results_folder / "full_ga_best_features.csv",
        index=False
    )

    np.save(
        results_folder / "full_ga_best_chromosome.npy",
        np.asarray(best_individual, dtype=np.int8)
    )

    checkpoint = {
        "generation": generation,
        "population": population,
        "hall_of_fame": hall_of_fame,
        "history": history,
        "fitness_cache": fitness_cache,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "elapsed_seconds": elapsed_seconds
    }

    with open(checkpoint_file, "wb") as checkpoint_output:
        pickle.dump(checkpoint, checkpoint_output)

    print(
        f"Generation {generation} | "
        f"Evaluations: {evaluations} | "
        f"Best MCC: {best_individual.fitness.values[0]:.4f} | "
        f"Selected: {sum(best_individual)} | "
        f"Unique subsets: {len(fitness_cache)} | "
        f"Elapsed: {elapsed_seconds / 60:.1f} minutes",
        flush=True
    )


# Continue from the checkpoint if the run was interrupted
if checkpoint_file.exists():
    print("Loading the saved GA checkpoint.", flush=True)

    with open(checkpoint_file, "rb") as checkpoint_input:
        checkpoint = pickle.load(checkpoint_input)

    population = checkpoint["population"]
    hall_of_fame = checkpoint["hall_of_fame"]
    history = checkpoint["history"]
    fitness_cache = checkpoint["fitness_cache"]

    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])

    completed_generation = checkpoint["generation"]
    previous_elapsed_seconds = checkpoint["elapsed_seconds"]

    first_generation = completed_generation + 1

else:
    print("Starting a new full GA run.", flush=True)

    random.seed(21)
    np.random.seed(21)

    population = toolbox.population(n=population_size)
    hall_of_fame = tools.HallOfFame(1)
    history = []

    previous_elapsed_seconds = 0
    first_generation = 1

    initial_start = time.time()

    initial_evaluations = evaluate_invalid_individuals(
        population
    )

    initial_elapsed = (
        previous_elapsed_seconds
        + time.time()
        - initial_start
    )

    save_progress(
        generation=0,
        population=population,
        hall_of_fame=hall_of_fame,
        history=history,
        evaluations=initial_evaluations,
        elapsed_seconds=initial_elapsed
    )

    previous_elapsed_seconds = initial_elapsed


# Run the remaining generations
run_start_time = time.time()

for generation in range(
    first_generation,
    number_of_generations + 1
):
    elite = toolbox.clone(hall_of_fame[0])

    offspring = toolbox.select(
        population,
        len(population)
    )

    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover
    for first, second in zip(
        offspring[::2],
        offspring[1::2]
    ):
        if random.random() < crossover_probability:
            toolbox.mate(first, second)

            if first.fitness.valid:
                del first.fitness.values

            if second.fitness.valid:
                del second.fitness.values

    # Apply mutation
    for individual in offspring:
        if random.random() < mutation_probability:
            toolbox.mutate(individual)

            if individual.fitness.valid:
                del individual.fitness.values

    generation_evaluations = evaluate_invalid_individuals(
        offspring
    )

    # Keep the best result found so far
    worst_index = min(
        range(len(offspring)),
        key=lambda index: offspring[index].fitness.values[0]
    )

    if (
        elite.fitness.values[0]
        > offspring[worst_index].fitness.values[0]
    ):
        offspring[worst_index] = elite

    population[:] = offspring

    total_elapsed_seconds = (
        previous_elapsed_seconds
        + time.time()
        - run_start_time
    )

    save_progress(
        generation=generation,
        population=population,
        hall_of_fame=hall_of_fame,
        history=history,
        evaluations=generation_evaluations,
        elapsed_seconds=total_elapsed_seconds
    )


best_individual = hall_of_fame[0]

print("\nFull GA completed.", flush=True)
print(
    "Best validation MCC:",
    round(best_individual.fitness.values[0], 4),
    flush=True
)
print(
    "Number of selected descriptors:",
    sum(best_individual),
    flush=True
)
print(
    "Unique subsets evaluated:",
    len(fitness_cache),
    flush=True
)
