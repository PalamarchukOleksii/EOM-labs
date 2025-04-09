import random
import math
import os
import sys

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_TO_FILE_FLAG = False
LOG_FILE_NAME = "output_log.txt"

OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "outputs")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIRECTORY, LOG_FILE_NAME)
CSV_SAVE_PATH = os.path.join(OUTPUT_DIRECTORY, "comprehensive_results.csv")

POPULATION_SIZES = [30, 60]

FI_SELECTION = 0.1
FI_CROSSOVER = 0.8
FI_MUTATION = 0.1

D = 0.5

GEN_LIMIT = 50

LB = [-10, -10]
UB = [10, 10]


def f3(x, y):
    return math.sqrt(math.pow(x - 1, 2) + math.pow(y - 7, 4))


def f6(x, y):
    return math.pow(x + y - 2, 2) + x


def generate_population(lower_bound, upper_bound, population_size, functions):
    population = []

    for i in range(population_size):
        point = []
        for d in range(len(lower_bound)):
            r_id = random.uniform(0, 1)
            coord_id = lower_bound[d] + r_id * (upper_bound[d] - lower_bound[d])
            point.append(coord_id)

        funcs_values = []
        for func in functions:
            funcs_values.append(func(point[0], point[1]))

        population.append((point, funcs_values, 0))

    return population


def check_non_dominance(population):
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if (
                population[i][1][0] >= population[j][1][0]
                and population[i][1][1] >= population[j][1][1]
            ):
                population[i][2] += 1
            if (
                population[j][1][0] >= population[i][1][0]
                and population[j][1][1] >= population[i][1][1]
            ):
                population[j][2] += 1

    return population


def sort_population(population, sort_index):
    return sorted(population, key=lambda x: x[sort_index])


def rank_select(population):
    p = len(population)
    ranks = [i + 1 for i in range(p)]

    probabilities_sum = sum(p - rank + 1 for rank in ranks)
    probabilities = [(p - rank + 1) / probabilities_sum for rank in ranks]

    selected_index = random.choices(range(p), probabilities)[0]

    return population[selected_index][0]


def uniform_crossover_in_natural_coding(
    parent_1, parent_2, d, lower_bound, upper_bound
):
    new_point = []
    for i in range(len(parent_1)):
        beta = random.uniform(-d, 1 + d)

        new_point_coord = parent_1[i] + beta * (parent_2[i] - parent_1[i])
        new_point_coord = max(min(new_point_coord, upper_bound[i]), lower_bound[i])

        new_point.append(new_point_coord)

    return new_point


def crossover_population(population, d, lower_bound, upper_bound, function):
    new_population = []

    while len(new_population) < len(population):
        parent1 = rank_select(population)
        parent2 = rank_select(population)
        while parent1 == parent2:
            parent2 = rank_select(population)

        child_coords = uniform_crossover_in_natural_coding(
            parent1, parent2, d, lower_bound, upper_bound
        )

        child_value = function(child_coords[0], child_coords[1])
        new_population.append((child_coords, child_value))

    return new_population


def mutation_in_natural_coding(point, lower_bound, upper_bound, sigma):
    mutated_point = []

    for i in range(len(point)):
        mutation_value = random.gauss(0, sigma[i])

        new_value = point[i] + mutation_value
        new_value = max(min(new_value, upper_bound[i]), lower_bound[i])

        mutated_point.append(new_value)

    return mutated_point


def mutate_population(population, lower_bound, upper_bound, sigma, function):
    mutated_population = []

    for point, _ in population:
        mutated_coords = mutation_in_natural_coding(
            point, lower_bound, upper_bound, sigma
        )
        mutated_value = function(mutated_coords[0], mutated_coords[1])
        mutated_population.append((mutated_coords, mutated_value))

    return mutated_population


def run_experiment(
    population_size,
    lower_bound,
    upper_bound,
    generation_limit,
    function,
    crossover_area_expansion,
    elite_fraction,
    crossover_fraction,
    mutation_fraction,
):
    population = generate_population(
        lower_bound, upper_bound, population_size, function
    )
    sorted_population = sort_population(population)

    sigma = [(upper_bound[i] - lower_bound[i]) / 10 for i in range(len(lower_bound))]

    for i in range(generation_limit):
        elite_count = int(len(sorted_population) * elite_fraction)
        crossover_count = int(len(sorted_population) * crossover_fraction)

        elite_points = sorted_population[:elite_count]
        crossover_points = sorted_population[
            elite_count : elite_count + crossover_count
        ]
        mutation_points = sorted_population[elite_count + crossover_count :]

        crossovered_points = crossover_population(
            crossover_points,
            crossover_area_expansion,
            lower_bound,
            upper_bound,
            function,
        )
        mutated_points = mutate_population(
            mutation_points, lower_bound, upper_bound, sigma, function
        )

        population = elite_points + crossovered_points + mutated_points
        sorted_population = sort_population(population)

    return sorted_population


if __name__ == "__main__":
    original_stdout = sys.stdout
    log_file = open(LOG_PATH, "w")

    if LOG_TO_FILE_FLAG:
        print(f"Logging output to {LOG_PATH}...")
        sys.stdout = log_file
    else:
        log_file.close()

    results_data = []

    for P in POPULATION_SIZES:
        print(
            f"\nRunning Experiments with Parameters:\n"
            f"- Population Size: {P}\n"
            f"- Fi Selection: {FI_SELECTION}\n"
            f"- Fi Crossover: {FI_CROSSOVER}\n"
            f"- Fi Mutation: {FI_MUTATION}\n"
        )

        experiment_name = (
            f"p{P}_fiSel{FI_SELECTION}_fiCross{FI_CROSSOVER}_fiMut{FI_MUTATION}"
        )

        # TODO: add call of run experimnt func

    if LOG_TO_FILE_FLAG:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {LOG_PATH}")
