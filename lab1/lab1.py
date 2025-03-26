import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_TO_FILE_FLAG = True
LOG_FILE_NAME = "output_log.txt"

OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "outputs")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIRECTORY, LOG_FILE_NAME)
CSV_SAVE_PATH = os.path.join(OUTPUT_DIRECTORY, "comprehensive_results.csv")

POPULATION_SIZES = [30, 60, 120, 240]
PARAMETERS_SETS = [
    {"name": "Low Elite Selection", "fi_sel": 0.1, "fi_cross": 0.8, "fi_mut": 0.1},
    {
        "name": "Medium Elite Selection",
        "fi_sel": 0.2,
        "fi_cross": 0.7,
        "fi_mut": 0.1,
    },
    {"name": "High Elite Selection", "fi_sel": 0.3, "fi_cross": 0.6, "fi_mut": 0.1},
]

D = 0.5

GEN_LIMIT = 50
NUM_EXPERIMENTS = 100

F4_CALL_COUNTER = 0
F12_CALL_COUNTER = 0

LB_F4 = [-10, -10]
UB_F4 = [10, 10]

TRUE_MINIMUM_F4 = ([1, 1], 0)
EPSILON_RADIUS_F4 = 0.05


def f4(x, y):
    global F4_CALL_COUNTER
    F4_CALL_COUNTER += 1
    return (
        (np.sin(3 * np.pi * x)) ** 2
        + ((x - 1) ** 2 * (1 + (np.sin(3 * np.pi * y)) ** 2))
        + ((y - 1) ** 2 * (1 + (np.sin(2 * np.pi * y)) ** 2))
    )


LB_F12 = [0, 0]
UB_F12 = [np.pi, np.pi]

TRUE_MINIMUM_F12 = ([2.20, 1.57], -1.8013)
EPSILON_RADIUS_F12 = 0.01


def f12(x, y):
    global F12_CALL_COUNTER
    F12_CALL_COUNTER += 1
    return -np.sin(x) * np.pow(np.sin((x**2) / np.pi), 20) - np.sin(y) * np.pow(
        np.sin((2 * y**2) / np.pi), 20
    )


def generate_population(lower_bound, upper_bound, population_size, function):
    population = []

    for i in range(population_size):
        point = []
        for d in range(len(lower_bound)):
            r_id = random.uniform(0, 1)
            coord_id = lower_bound[d] + r_id * (upper_bound[d] - lower_bound[d])
            point.append(coord_id)

        func_value = function(point[0], point[1])
        population.append((point, func_value))

    return population


def sort_population(population):
    return sorted(population, key=lambda x: x[1])


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


def plot_function_landscape(
    function, lower_bound, upper_bound, title, filename, points=None
):
    resolution = 1000
    x = np.linspace(lower_bound[0], upper_bound[0], resolution)
    y = np.linspace(lower_bound[1], upper_bound[1], resolution)

    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        Z,
        extent=[lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]],
        origin="lower",
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, label="Function Value", shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    if points:
        x_points = np.array([point[0][0] for point in points])
        y_points = np.array([point[0][1] for point in points])

        plt.scatter(
            x_points,
            y_points,
            color="black",
            alpha=0.7,
            s=30,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.title(title, fontsize=12)
    plt.xlabel("X", fontsize=10)
    plt.ylabel("Y", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {filename}")


def check_epsilon_sphere(point, true_minimum, epsilon_radius):
    distance = np.sqrt(
        (point[0][0] - true_minimum[0][0]) ** 2
        + (point[0][1] - true_minimum[0][1]) ** 2
        + (point[1] - true_minimum[1]) ** 2
    )

    return distance <= epsilon_radius


def run_multiple_experiments(
    short_experiment_name,
    NUM_EXPERIMENTS,
    epsilon_radius,
    true_minimum,
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
    function_name = function.__name__
    plots_output_dir = os.path.join(
        SCRIPT_DIRECTORY, f"plots/{function_name}/{short_experiment_name}"
    )
    os.makedirs(plots_output_dir, exist_ok=True)

    global F4_CALL_COUNTER, F12_CALL_COUNTER
    if function_name == "f4":
        F4_CALL_COUNTER = 0
    elif function_name == "f12":
        F12_CALL_COUNTER = 0

    results = []
    for i in range(NUM_EXPERIMENTS):
        result = run_experiment(
            population_size,
            lower_bound,
            upper_bound,
            generation_limit,
            function,
            crossover_area_expansion,
            elite_fraction,
            crossover_fraction,
            mutation_fraction,
        )
        results.append(result)

        print(f"\nFunction {function_name} experiment {i+1} results:")
        for res in result:
            print(res)

        plot_function_landscape(
            function,
            lower_bound,
            upper_bound,
            f"Function {function_name} Landscape experiment {i+1}",
            os.path.join(plots_output_dir, f"experiment_{i+1}.png"),
            result,
        )

    best_result = float("inf")
    best_result_coords = None
    successful_experiments = 0
    total_points = 0
    successful_points = 0

    for res in results:
        current_best = res[0]
        if current_best[1] < best_result:
            best_result = current_best[1]
            best_result_coords = current_best[0]

        num_mutation_points = int(len(res) * mutation_fraction)
        non_mutated_points = (
            res[:-num_mutation_points] if num_mutation_points > 0 else res
        )

        if all(
            check_epsilon_sphere(point, true_minimum, epsilon_radius)
            for point in non_mutated_points
        ):
            successful_experiments += 1

        for point in res:
            if check_epsilon_sphere(point, true_minimum, epsilon_radius):
                successful_points += 1
            total_points += 1

    success_rate = successful_experiments / NUM_EXPERIMENTS * 100
    overall_accuracy = successful_points / total_points * 100

    function_call_count = F4_CALL_COUNTER if function_name == "f4" else F12_CALL_COUNTER

    return {
        "num_experiments": NUM_EXPERIMENTS,
        "results": results,
        "best_result": best_result,
        "best_result_coords": best_result_coords,
        "success_rate": success_rate,
        "overall_accuracy": overall_accuracy,
        "function_calls_per_experiment": function_call_count / NUM_EXPERIMENTS,
    }


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
        for param_set in PARAMETERS_SETS:
            print(
                f"\nRunning experiments for Population Size {P} with {param_set['name']}:"
            )

            experiment_name = f"p{P}_fiElite{param_set['fi_sel']}_fiCross{param_set['fi_cross']}_fiMut{param_set['fi_mut']}"

            metrics_f4 = run_multiple_experiments(
                experiment_name,
                NUM_EXPERIMENTS,
                EPSILON_RADIUS_F4,
                TRUE_MINIMUM_F4,
                P,
                LB_F4,
                UB_F4,
                GEN_LIMIT,
                f4,
                D,
                elite_fraction=param_set["fi_sel"],
                crossover_fraction=param_set["fi_cross"],
                mutation_fraction=param_set["fi_mut"],
            )

            metrics_f12 = run_multiple_experiments(
                experiment_name,
                NUM_EXPERIMENTS,
                EPSILON_RADIUS_F12,
                TRUE_MINIMUM_F12,
                P,
                LB_F12,
                UB_F12,
                GEN_LIMIT,
                f12,
                D,
                elite_fraction=param_set["fi_sel"],
                crossover_fraction=param_set["fi_cross"],
                mutation_fraction=param_set["fi_mut"],
            )

            result_entry = {
                "Number of Experiments": NUM_EXPERIMENTS,
                "Population Size": P,
                "Selection Strategy": param_set["name"],
                "Selection Fraction": param_set["fi_sel"],
                "Crossover Fraction": param_set["fi_cross"],
                "Mutation Fraction": param_set["fi_mut"],
                "F4 Best Result": metrics_f4["best_result"],
                "F4 Best Result Coords": metrics_f4["best_result_coords"],
                "F4 Success Rate (%)": metrics_f4["success_rate"],
                "F4 Overall Accuracy (%)": metrics_f4["overall_accuracy"],
                "F4 Calls per Experiment": metrics_f4["function_calls_per_experiment"],
                "F12 Best Result": metrics_f12["best_result"],
                "F12 Best Result Coords": metrics_f12["best_result_coords"],
                "F12 Success Rate (%)": metrics_f12["success_rate"],
                "F12 Overall Accuracy (%)": metrics_f12["overall_accuracy"],
                "F12 Calls per Experiment": metrics_f12[
                    "function_calls_per_experiment"
                ],
            }

            print(f"\nResults:")
            for key, value in result_entry.items():
                print(f"{key}: {value}")

            results_data.append(result_entry)

    results_df = pd.DataFrame(results_data)
    float_columns = [
        "F4 Best Result",
        "F4 Best Result Coords",
        "F4 Success Rate (%)",
        "F4 Overall Accuracy (%)",
        "F12 Best Result",
        "F12 Best Result Coords",
        "F12 Success Rate (%)",
        "F12 Overall Accuracy (%)",
    ]
    for col in float_columns:
        results_df[col] = results_df[col].round(4)

    print("\nComprehensive Comparative Results Table:")
    print(results_df.to_string(index=False))

    results_df.to_csv(CSV_SAVE_PATH, index=False)
    print(f"\nResults saved to {CSV_SAVE_PATH}")

    if LOG_TO_FILE_FLAG:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {LOG_PATH}")
