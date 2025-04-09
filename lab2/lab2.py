import random
import math
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

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


FUNCTION_LIST = [f3, f6]


def evaluate_functions_at_point(point, function_list=FUNCTION_LIST):
    values = []
    for func in function_list:
        values.append(func(point[0], point[1]))
    return values


def generate_population(
    lower_bound, upper_bound, population_size, function_list=FUNCTION_LIST
):
    population = []

    for i in range(population_size):
        point = []
        for d in range(len(lower_bound)):
            r_id = random.uniform(0, 1)
            coord_id = lower_bound[d] + r_id * (upper_bound[d] - lower_bound[d])
            point.append(coord_id)

        population.append(
            [
                point,
                evaluate_functions_at_point(point, function_list),
                0,
            ]
        )

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


def sort_population(population, sort_index=2):
    return sorted(population, key=lambda x: x[sort_index])


def rank_select(population):
    p = len(population)
    ranks = [i + 1 for i in range(p)]

    probabilities_sum = sum(p - rank + 1 for rank in ranks)
    probabilities = [(p - rank + 1) / probabilities_sum for rank in ranks]

    selected_index = random.choices(range(p), probabilities)[0]

    return population[selected_index][0]


def uniform_crossover_in_natural_coding(
    parent_1, parent_2, d=D, lower_bound=LB, upper_bound=UB
):
    new_point = []
    for i in range(len(parent_1)):
        beta = random.uniform(-d, 1 + d)

        new_point_coord = parent_1[i] + beta * (parent_2[i] - parent_1[i])
        new_point_coord = max(min(new_point_coord, upper_bound[i]), lower_bound[i])

        new_point.append(new_point_coord)

    return new_point


def crossover_population(
    population, d=D, lower_bound=LB, upper_bound=UB, function_list=FUNCTION_LIST
):
    new_population = []

    while len(new_population) < len(population):
        parent1 = rank_select(population)
        parent2 = rank_select(population)
        while parent1 == parent2:
            parent2 = rank_select(population)

        child_coords = uniform_crossover_in_natural_coding(
            parent1, parent2, d, lower_bound, upper_bound
        )

        new_population.append(
            [
                child_coords,
                evaluate_functions_at_point(child_coords, function_list),
                0,
            ]
        )

    return new_population


def mutation_in_natural_coding(point, sigma, lower_bound=LB, upper_bound=UB):
    mutated_point = []

    for i in range(len(point)):
        mutation_value = random.gauss(0, sigma[i])

        new_value = point[i] + mutation_value
        new_value = max(min(new_value, upper_bound[i]), lower_bound[i])

        mutated_point.append(new_value)

    return mutated_point


def mutate_population(
    population, sigma, lower_bound=LB, upper_bound=UB, function_list=FUNCTION_LIST
):
    mutated_population = []

    for point, _, _ in population:
        mutated_coords = mutation_in_natural_coding(
            point, sigma, lower_bound, upper_bound
        )

        mutated_population.append(
            [
                mutated_coords,
                evaluate_functions_at_point(mutated_coords, function_list),
                0,
            ]
        )

    return mutated_population


def plot_pareto_set(
    population,
    experiment_name,
    upper_bound=UB,
    lower_bound=LB,
    save_dir=OUTPUT_DIRECTORY,
):
    pareto_front = [ind for ind in population if ind[2] == 0]
    dominated_set = [ind for ind in population if ind[2] != 0]

    pareto_set_x, pareto_set_y = zip(*[(ind[0][0], ind[0][1]) for ind in pareto_front])
    dominated_x, dominated_y = zip(*[(ind[0][0], ind[0][1]) for ind in dominated_set])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pareto_set_x, pareto_set_y, color="blue", label="Non-dominated (Pareto Set)"
    )
    plt.scatter(dominated_x, dominated_y, color="red", label="Dominated")

    plt.title(f"Pareto Set - {experiment_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.xlim(lower_bound[0], upper_bound[0])
    plt.ylim(lower_bound[1], upper_bound[1])

    if save_dir:
        save_path = os.path.join(save_dir, f"{experiment_name}.png")
        plt.savefig(save_path)

    plt.show()


def plot_pareto_front(population, experiment_name, save_dir=OUTPUT_DIRECTORY):
    pareto_front = [ind for ind in population if ind[2] == 0]
    dominated_set = [ind for ind in population if ind[2] != 0]

    pareto_front_f3, pareto_front_f6 = zip(
        *[(ind[1][0], ind[1][1]) for ind in pareto_front]
    )
    dominated_f3, dominated_f6 = zip(*[(ind[1][0], ind[1][1]) for ind in dominated_set])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pareto_front_f3,
        pareto_front_f6,
        color="blue",
        label="Non-dominated (Pareto Front)",
    )
    plt.scatter(dominated_f3, dominated_f6, color="red", label="Dominated")

    plt.title(f"Pareto Front - {experiment_name}")
    plt.xlabel("F3")
    plt.ylabel("F6")
    plt.legend()
    plt.grid(True)

    if save_dir:
        save_path = os.path.join(save_dir, f"{experiment_name}.png")
        plt.savefig(save_path)

    plt.show()


def results_to_table(population, experiment_name, save_dir=OUTPUT_DIRECTORY):
    data = []

    for item in population:
        x, y = item[0]
        f3_value, f6_value = item[1]
        dominance = item[2]
        data.append([x, y, f3_value, f6_value, dominance])

    df = pd.DataFrame(data, columns=["x", "y", "f3_value", "f6_value", "dominance"])

    if save_dir:
        save_path = os.path.join(save_dir, f"{experiment_name}_results.csv")
        df.to_csv(save_path, index=False)

    print(f"{experiment_name} Results:")
    print(df)


def run_experiment(
    population_size,
    lower_bound=LB,
    upper_bound=UB,
    generation_limit=GEN_LIMIT,
    function_list=FUNCTION_LIST,
    crossover_area_expansion=D,
    elite_fraction=FI_SELECTION,
    crossover_fraction=FI_CROSSOVER,
    mutation_fraction=FI_MUTATION,
):
    population = generate_population(
        lower_bound, upper_bound, population_size, function_list
    )
    checked_population = check_non_dominance(population)
    sorted_population = sort_population(checked_population, 2)

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
            function_list,
        )
        mutated_points = mutate_population(
            mutation_points, sigma, lower_bound, upper_bound, function_list
        )

        population = elite_points + crossovered_points + mutated_points
        checked_population = check_non_dominance(population)
        sorted_population = sort_population(checked_population, 2)

    return sorted_population


def test_params_on_experiments(
    population_sizes=POPULATION_SIZES,
    lower_bound=LB,
    upper_bound=UB,
    generation_limit=GEN_LIMIT,
    function_list=FUNCTION_LIST,
    crossover_area_expansion=D,
    elite_fraction=FI_SELECTION,
    crossover_fraction=FI_CROSSOVER,
    mutation_fraction=FI_MUTATION,
    save_directory=OUTPUT_DIRECTORY,
):
    for p in population_sizes:
        print(f"\nRunning Experiments with population size: {p}\n")

        experiment_name = f"population_size_{p}"

        result_population = run_experiment(
            p,
            lower_bound,
            upper_bound,
            generation_limit,
            function_list,
            crossover_area_expansion,
            elite_fraction,
            crossover_fraction,
            mutation_fraction,
        )

        plot_pareto_set(
            result_population, experiment_name, upper_bound, lower_bound, save_directory
        )
        plot_pareto_front(result_population, experiment_name, save_directory)
        results_to_table(result_population, experiment_name, save_directory)


class OutputLogger:
    def __init__(
        self,
        log_to_file=LOG_TO_FILE_FLAG,
        output_directory=OUTPUT_DIRECTORY,
        log_filename=LOG_FILE_NAME,
    ):
        self.output_directory = output_directory
        self.log_to_file = log_to_file
        self.log_filename = log_filename
        self.log_path = os.path.join(output_directory, log_filename)
        self.original_stdout = sys.stdout
        self.log_file = None

    def start_logging(self):
        print(f"Logging output to {self.log_path}...")
        os.makedirs(self.output_directory, exist_ok=True)
        self.log_file = open(self.log_path, "w")
        sys.stdout = self.log_file

    def stop_logging(self):
        if self.log_to_file and self.log_file:
            self.log_file.close()
            sys.stdout = self.original_stdout
            print(f"All output is logged to {self.log_path}")


if __name__ == "__main__":
    logger = OutputLogger()

    logger.start_logging()

    test_params_on_experiments()

    logger.stop_logging()
