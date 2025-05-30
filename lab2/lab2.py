import random
import math
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


class Function:
    @staticmethod
    def f3(x, y):
        return math.sqrt(math.pow(x - 1, 2) + math.pow(y - 7, 4))

    @staticmethod
    def f6(x, y):
        return math.pow(x + y - 2, 2) + x


class MultiObjectiveGeneticAlgorithm:
    def __init__(
        self,
        population_size,
        lower_bound,
        upper_bound,
        generation_limit,
        function_list,
        crossover_area_expansion,
        elite_fraction,
        crossover_fraction,
        mutation_fraction,
    ):
        self.__population_size = population_size
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__generation_limit = generation_limit
        self.__function_list = function_list
        self.__crossover_area_expansion = crossover_area_expansion
        self.__elite_fraction = elite_fraction
        self.__crossover_fraction = crossover_fraction
        self.__mutation_fraction = mutation_fraction
        self.__sigma = [
            (upper_bound[i] - lower_bound[i]) / 10 for i in range(len(lower_bound))
        ]
        self.__population = []

    def __evaluate_functions_at_point(self, point):
        return [func(*point) for func in self.__function_list]

    def __generate_population(self):
        for _ in range(self.__population_size):
            point = []
            for d in range(len(self.__lower_bound)):
                r_id = random.uniform(0, 1)
                coord_id = self.__lower_bound[d] + r_id * (
                    self.__upper_bound[d] - self.__lower_bound[d]
                )
                point.append(coord_id)

            self.__population.append(
                [
                    point,
                    self.__evaluate_functions_at_point(point),
                    0,
                ]
            )

    def __check_non_dominance(self):
        for i in range(len(self.__population)):
            self.__population[i][2] = 0

        for i in range(len(self.__population)):
            for j in range(len(self.__population)):
                if i == j:
                    continue

                p_i = self.__population[i][1]
                p_j = self.__population[j][1]
                if all(p_i[k] >= p_j[k] for k in range(len(p_i))) and any(
                    p_i[k] > p_j[k] for k in range(len(p_i))
                ):
                    self.__population[i][2] += 1

    def __sort_population(self):
        self.__population = sorted(self.__population, key=lambda x: x[2])

    def __rank_select(self, population):
        p = len(population)
        ranks = [i + 1 for i in range(p)]

        probabilities_sum = sum(p - rank + 1 for rank in ranks)
        probabilities = [(p - rank + 1) / probabilities_sum for rank in ranks]

        selected_index = random.choices(range(p), probabilities)[0]

        return population[selected_index][0]

    def __uniform_crossover_in_natural_coding(self, parent_1, parent_2):
        new_point = []
        for i in range(len(parent_1)):
            beta = random.uniform(
                -self.__crossover_area_expansion, 1 + self.__crossover_area_expansion
            )

            new_point_coord = parent_1[i] + beta * (parent_2[i] - parent_1[i])
            new_point_coord = max(
                min(new_point_coord, self.__upper_bound[i]), self.__lower_bound[i]
            )

            new_point.append(new_point_coord)

        return new_point

    def __crossover_population(self, crossover_population):
        new_population = []

        while len(new_population) < len(crossover_population):
            parent1 = self.__rank_select(crossover_population)
            parent2 = self.__rank_select(crossover_population)
            while parent1 == parent2:
                parent2 = self.__rank_select(crossover_population)

            child_coords = self.__uniform_crossover_in_natural_coding(parent1, parent2)

            new_population.append(
                [
                    child_coords,
                    self.__evaluate_functions_at_point(child_coords),
                    0,
                ]
            )

        return new_population

    def __mutation_in_natural_coding(self, point):
        mutated_point = []

        for i in range(len(point)):
            mutation_value = random.gauss(0, self.__sigma[i])

            new_value = point[i] + mutation_value
            new_value = max(
                min(new_value, self.__upper_bound[i]), self.__lower_bound[i]
            )

            mutated_point.append(new_value)

        return mutated_point

    def __mutate_population(self, mutation_population):
        mutated_population = []

        for point, _, _ in mutation_population:
            mutated_coords = self.__mutation_in_natural_coding(point)

            mutated_population.append(
                [
                    mutated_coords,
                    self.__evaluate_functions_at_point(mutated_coords),
                    0,
                ]
            )

        return mutated_population

    def run(self):
        self.__generate_population()
        self.__check_non_dominance()
        self.__sort_population()

        for _ in range(self.__generation_limit):
            elite_count = int(len(self.__population) * self.__elite_fraction)
            crossover_count = int(len(self.__population) * self.__crossover_fraction)

            elite_points = self.__population[:elite_count]
            crossover_points = self.__population[
                elite_count : elite_count + crossover_count
            ]
            mutation_points = self.__population[elite_count + crossover_count :]

            crossovered_points = self.__crossover_population(
                crossover_points,
            )
            mutated_points = self.__mutate_population(mutation_points)

            self.__population = elite_points + crossovered_points + mutated_points
            self.__check_non_dominance()
            self.__sort_population()

        return self.__population


class FunctionsMinimumsInfo:
    def __init__(self, lower_bound, upper_bound, functions_list):
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__functions_list = functions_list

    def __find_minimum(self, func, initial_guess=None):
        def wrapped_func(point):
            return func(*point)

        if initial_guess is None:
            initial_guess = [
                random.uniform(self.__lower_bound[i], self.__upper_bound[i])
                for i in range(len(self.__lower_bound))
            ]

        bounds = [
            (self.__lower_bound[i], self.__upper_bound[i])
            for i in range(len(self.__lower_bound))
        ]

        result = minimize(wrapped_func, initial_guess, bounds=bounds)

        min_point = result.x
        min_value = result.fun

        return min_point, min_value

    def get_minimums_info(self):
        min_info = {}
        min_info["functions_names"] = [func.__name__ for func in self.__functions_list]

        min_points = []
        min_values = []

        for i, func in enumerate(self.__functions_list):
            min_point, min_value = self.__find_minimum(func)
            min_points.append(min_point)
            min_values.append(min_value)

        grouped_min_info = []

        for i in range(len(self.__functions_list)):
            grouped_min_info.append((min_points[i]))

        min_info["grouped_minimums"] = grouped_min_info

        grouped_value_in_min_info = []

        for i in range(len(self.__functions_list)):
            values_at_min = [func(*min_points[i]) for func in self.__functions_list]
            grouped_value_in_min_info.append(tuple(values_at_min))

        min_info["grouped_values_in_minimums"] = grouped_value_in_min_info

        return min_info


class ResultVisualizer:
    def __init__(self, output_directory):
        self.__output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        self.__min_colors = [
            "green",
            "purple",
            "orange",
            "cyan",
            "magenta",
            "lime",
            "brown",
            "olive",
        ]

    def __extract_data(self, population):
        pareto_front = [ind for ind in population if ind[2] == 0]
        dominated_set = [ind for ind in population if ind[2] != 0]

        pareto_set_x, pareto_set_y = (
            zip(*[(ind[0][0], ind[0][1]) for ind in pareto_front])
            if pareto_front
            else ([], [])
        )
        dominated_x, dominated_y = (
            zip(*[(ind[0][0], ind[0][1]) for ind in dominated_set])
            if dominated_set
            else ([], [])
        )

        pareto_f3, pareto_f6 = (
            zip(*[(ind[1][0], ind[1][1]) for ind in pareto_front])
            if pareto_front
            else ([], [])
        )
        dominated_f3, dominated_f6 = (
            zip(*[(ind[1][0], ind[1][1]) for ind in dominated_set])
            if dominated_set
            else ([], [])
        )

        return (
            pareto_set_x,
            pareto_set_y,
            dominated_x,
            dominated_y,
            pareto_f3,
            pareto_f6,
            dominated_f3,
            dominated_f6,
        )

    def __save_plot(self, experiment_name, plot_type):
        save_path = os.path.join(
            self.__output_directory, f"{plot_type}_{experiment_name}.png"
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def __plot_pareto_set(
        self,
        x,
        y,
        title,
        xlabel,
        ylabel,
        legend_labels,
        lower_bound,
        upper_bound,
        ax,
        min_info=None,
    ):
        ax.scatter(x[0], y[0], color="blue", label=legend_labels[0])
        ax.scatter(x[1], y[1], color="red", label=legend_labels[1])

        if min_info:
            for i in range(len(min_info["functions_names"])):
                color = self.__min_colors[i % len(self.__min_colors)]
                ax.scatter(
                    min_info["grouped_minimums"][i][0],
                    min_info["grouped_minimums"][i][1],
                    color=color,
                    marker="x",
                    s=100,
                    label=f"Min of {min_info["functions_names"][i]}",
                )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

        if lower_bound is not None and upper_bound is not None:
            ax.set_xlim(lower_bound[0], upper_bound[0])
            ax.set_ylim(lower_bound[1], upper_bound[1])

    def __plot_pareto_front(
        self,
        x,
        y,
        title,
        xlabel,
        ylabel,
        legend_labels,
        ax,
        min_info=None,
    ):
        ax.scatter(x[0], y[0], color="blue", label=legend_labels[0])
        ax.scatter(x[1], y[1], color="red", label=legend_labels[1])

        if min_info:
            for i in range(len(min_info["functions_names"])):
                color = self.__min_colors[i % len(self.__min_colors)]

                ax.scatter(
                    min_info["grouped_values_in_minimums"][i][0],
                    min_info["grouped_values_in_minimums"][i][1],
                    color=color,
                    marker="x",
                    s=100,
                    label=f"Point based on {min_info["functions_names"][0]} and {min_info["functions_names"][1]} values in min of {min_info["functions_names"][i]}",
                )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    def plot_pareto_all_points(
        self,
        population,
        experiment_name,
        lower_bound,
        upper_bound,
        min_info=None,
    ):
        (
            pareto_set_x,
            pareto_set_y,
            dominated_x,
            dominated_y,
            pareto_f3,
            pareto_f6,
            dominated_f3,
            dominated_f6,
        ) = self.__extract_data(population)

        _, axes = plt.subplots(1, 2, figsize=(12, 6))

        self.__plot_pareto_set(
            [pareto_set_x, dominated_x],
            [pareto_set_y, dominated_y],
            f"Pareto Set - All Points - {experiment_name}",
            "X",
            "Y",
            ["Pareto-optimal", "Dominated"],
            lower_bound,
            upper_bound,
            axes[0],
            min_info,
        )

        self.__plot_pareto_front(
            [pareto_f3, dominated_f3],
            [pareto_f6, dominated_f6],
            f"Pareto Front - All Points - {experiment_name}",
            min_info["functions_names"][0],
            min_info["functions_names"][1],
            ["Pareto-optimal", "Dominated"],
            axes[1],
            min_info,
        )

        self.__save_plot(experiment_name, "pareto_all_points")

    def plot_pareto_non_dominated_only(
        self,
        population,
        experiment_name,
        lower_bound,
        upper_bound,
        min_info=None,
    ):
        (
            pareto_set_x,
            pareto_set_y,
            _,
            _,
            pareto_f3,
            pareto_f6,
            _,
            _,
        ) = self.__extract_data(population)

        _, axes = plt.subplots(1, 2, figsize=(12, 6))

        self.__plot_pareto_set(
            [pareto_set_x, []],
            [pareto_set_y, []],
            f"Pareto Set - Non-Dominated - {experiment_name}",
            "X",
            "Y",
            ["Pareto-optimal", ""],
            lower_bound,
            upper_bound,
            axes[0],
            min_info,
        )

        self.__plot_pareto_front(
            [pareto_f3, []],
            [pareto_f6, []],
            f"Pareto Front - Non-Dominated - {experiment_name}",
            min_info["functions_names"][0],
            min_info["functions_names"][1],
            ["Pareto-optimal", ""],
            axes[1],
            min_info,
        )

        self.__save_plot(experiment_name, "pareto_non_dominated")

    def results_to_table(self, population, experiment_name):
        data = [
            [item[0][0], item[0][1], item[1][0], item[1][1], item[2]]
            for item in population
        ]

        df = pd.DataFrame(data, columns=["x", "y", "f3_value", "f6_value", "dominance"])

        save_path = os.path.join(
            self.__output_directory, f"results_{experiment_name}.csv"
        )
        df.to_csv(save_path, index=False)

        print("\n" + "-" * 50)
        print(f"RESULTS: {experiment_name}")
        print("-" * 50)

        pd.set_option("display.precision", 4)
        pd.set_option("display.max_rows", 10)
        pd.set_option("display.width", 120)
        print(df)
        print("-" * 50)


class OutputLogger:
    def __init__(
        self,
        log_to_file_flag,
        output_directory,
        log_filename,
    ):
        self.__output_directory = output_directory
        self.__log_to_file_flag = log_to_file_flag
        self.__log_path = os.path.join(output_directory, log_filename)
        self.__original_stdout = sys.stdout
        self.__log_file = None

    def start_logging(self):
        if not self.__log_to_file_flag:
            print("Logging to file is disabled.")
            return

        if self.__log_file:
            print("Logging has already started.")
            return

        print(f"Logging output to: {self.__log_path}")
        os.makedirs(self.__output_directory, exist_ok=True)
        self.__log_file = open(self.__log_path, "w", encoding="utf-8")
        sys.stdout = self.__log_file

    def stop_logging(self):
        if not self.__log_to_file_flag:
            print("Logging to file is disabled.")
            return

        if not self.__log_file:
            print("Logging is not started yet.")
            return

        self.__log_file.close()
        self.__log_file = None
        sys.stdout = self.__original_stdout
        print(f"All output is logged to: {self.__log_path}")


class Experiment:
    def __init__(self, log_to_file_flag, output_directory, log_filename):
        self.__output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)

        self.__logger = OutputLogger(
            log_to_file_flag, self.__output_directory, log_filename
        )

        self.__visualizer = ResultVisualizer(output_directory)

    def __run_experiment(
        self,
        population_size,
        lower_bound,
        upper_bound,
        generation_limit,
        function_list,
        crossover_area_expansion,
        elite_fraction,
        crossover_fraction,
        mutation_fraction,
    ):
        experiment_name = f"population_size_{population_size}"

        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {experiment_name}")
        print("=" * 70)
        print(f"Population Size: {population_size}")
        print(f"Generation Limit: {generation_limit}")
        print(f"Domain Bounds: {lower_bound} to {upper_bound}")
        print(f"Parameters:")
        print(f"  Elite Fraction: {elite_fraction:.2f}")
        print(f"  Crossover Fraction: {crossover_fraction:.2f}")
        print(f"  Mutation Fraction: {mutation_fraction:.2f}")
        print(f"  Crossover Area Expansion: {crossover_area_expansion:.2f}")
        print("=" * 70)

        optimizer = MultiObjectiveGeneticAlgorithm(
            population_size,
            lower_bound,
            upper_bound,
            generation_limit,
            function_list,
            crossover_area_expansion,
            elite_fraction,
            crossover_fraction,
            mutation_fraction,
        )

        result_population = optimizer.run()

        minimizer = FunctionsMinimumsInfo(lower_bound, upper_bound, function_list)
        min_info = minimizer.get_minimums_info()

        print("\n" + "-" * 60)
        print(f"MINIMUMS INFORMATION FOR {experiment_name}")
        print("-" * 60)
        print(f"Functions: {', '.join(min_info['functions_names'])}")

        print("\nFUNCTION MINIMUMS:")
        for i, func_name in enumerate(min_info["functions_names"]):
            coords = min_info["grouped_minimums"][i]
            formatted_coords = [f"{x:.4f}" for x in coords]
            print(f"  {func_name} minimum at point: ({', '.join(formatted_coords)})")

        print("\nFUNCTION VALUES AT MINIMUMS:")
        for i, func_name in enumerate(min_info["functions_names"]):
            values = min_info["grouped_values_in_minimums"][i]
            formatted_values = [f"{x:.4f}" for x in values]
            print(f"  Values at {func_name} minimum: ({', '.join(formatted_values)})")
        print("-" * 60)

        self.__visualizer.plot_pareto_all_points(
            result_population,
            experiment_name,
            lower_bound,
            upper_bound,
            min_info,
        )
        self.__visualizer.plot_pareto_non_dominated_only(
            result_population,
            experiment_name,
            lower_bound,
            upper_bound,
            min_info,
        )
        self.__visualizer.results_to_table(result_population, experiment_name)

    def run_multiple_experiments(
        self,
        population_sizes,
        lower_bound,
        upper_bound,
        generation_limit,
        function_list,
        crossover_area_expansion,
        elite_fraction,
        crossover_fraction,
        mutation_fraction,
    ):
        self.__logger.start_logging()

        print("\n" + "*" * 70)
        print("MULTI-OBJECTIVE GENETIC ALGORITHM EXPERIMENTS")
        print("*" * 70)
        print(
            f"Running {len(population_sizes)} experiments with different population sizes:"
        )
        print(f"   {', '.join(map(str, population_sizes))}")
        print("*" * 70)

        for p in population_sizes:
            self.__run_experiment(
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

        print("\n" + "*" * 70)
        print("ALL EXPERIMENTS COMPLETED")
        print("*" * 70)
        self.__logger.stop_logging()


class Utils:
    @staticmethod
    def set_random_seed(seed=42):
        random.seed(seed)
        print("Random seed set to", seed)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE GENETIC ALGORITHM")
    print("=" * 70)

    Utils.set_random_seed()

    SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "outputs")
    LOG_TO_FILE_FLAG = True
    LOG_FILENAME = "log.txt"

    POPULATION_SIZES = [30, 60, 120]
    LOWER_BOUND = [-10, -10]
    UPPER_BOUND = [10, 10]
    GEN_LIMIT = 50
    FUNCTION_LIST = [Function.f3, Function.f6]
    CROSSOVER_AREA_EXPANSION = 0.5
    FI_SELECTION = 0.1
    FI_CROSSOVER = 0.8
    FI_MUTATION = 0.1

    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print(f"Functions: {[func.__name__ for func in FUNCTION_LIST]}")
    print("-" * 70)

    experiment = Experiment(LOG_TO_FILE_FLAG, OUTPUT_DIRECTORY, LOG_FILENAME)
    experiment.run_multiple_experiments(
        POPULATION_SIZES,
        LOWER_BOUND,
        UPPER_BOUND,
        GEN_LIMIT,
        FUNCTION_LIST,
        CROSSOVER_AREA_EXPANSION,
        FI_SELECTION,
        FI_CROSSOVER,
        FI_MUTATION,
    )
