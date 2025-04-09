import random
import math
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


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
        for i in range(self.__population_size):
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

        for i in range(self.__generation_limit):
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


class ResultVisualizer:
    def __init__(self, output_directory):
        self.__output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)

    def plot_pareto_set(self, population, experiment_name, lower_bound, upper_bound):
        pareto_front = [ind for ind in population if ind[2] == 0]
        dominated_set = [ind for ind in population if ind[2] != 0]

        pareto_set_x, pareto_set_y = zip(
            *[(ind[0][0], ind[0][1]) for ind in pareto_front]
        )
        dominated_x, dominated_y = zip(
            *[(ind[0][0], ind[0][1]) for ind in dominated_set]
        )

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

        save_path = os.path.join(
            self.__output_directory, f"pareto_set_{experiment_name}.png"
        )
        plt.savefig(save_path)

        plt.show()

    def plot_pareto_front(self, population, experiment_name):
        pareto_front = [ind for ind in population if ind[2] == 0]
        dominated_set = [ind for ind in population if ind[2] != 0]

        pareto_front_f3, pareto_front_f6 = zip(
            *[(ind[1][0], ind[1][1]) for ind in pareto_front]
        )
        dominated_f3, dominated_f6 = zip(
            *[(ind[1][0], ind[1][1]) for ind in dominated_set]
        )

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

        save_path = os.path.join(
            self.__output_directory, f"pareto_front_{experiment_name}.png"
        )
        plt.savefig(save_path)

        plt.show()

    def results_to_table(self, population, experiment_name):
        data = []

        for item in population:
            x, y = item[0]
            f3_value, f6_value = item[1]
            dominance = item[2]
            data.append([x, y, f3_value, f6_value, dominance])

        df = pd.DataFrame(data, columns=["x", "y", "f3_value", "f6_value", "dominance"])

        save_path = os.path.join(
            self.__output_directory, f"results_{experiment_name}.csv"
        )
        df.to_csv(save_path, index=False)

        print(f"{experiment_name} Results:")
        print(df)


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

        print(f"Logging output to {self.__log_path}...")
        os.makedirs(self.__output_directory, exist_ok=True)
        self.__log_file = open(self.__log_path, "w")
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
        print(f"All output is logged to {self.__log_path}")


class Experiment:
    def __init__(self, log_to_file_flag, output_directory, log_filename):
        self.__output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)

        self.__logger = OutputLogger(
            log_to_file_flag, self.__output_directory, log_filename
        )

        self.__visualizer = ResultVisualizer(output_directory)

    def _run_experiment(
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

        self.__visualizer.plot_pareto_set(
            result_population, experiment_name, lower_bound, upper_bound
        )
        self.__visualizer.plot_pareto_front(result_population, experiment_name)
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

        for p in population_sizes:
            self._run_experiment(
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

        self.__logger.stop_logging()


if __name__ == "__main__":
    SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "outputs")
    LOG_TO_FILE_FLAG = False
    LOG_FILENAME = "log.txt"

    POPULATION_SIZES = [30, 60]
    LOWER_BOUND = [-10, -10]
    UPPER_BOUND = [10, 10]
    GEN_LIMIT = 50
    FUNCTION_LIST = [Function.f3, Function.f6]
    CROSSOVER_AREA_EXPANSION = 0.5
    FI_SELECTION = 0.1
    FI_CROSSOVER = 0.8
    FI_MUTATION = 0.1

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
