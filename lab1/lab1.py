import random
import math

P = 50

fi_sel = 0.2
fi_cross = 0.7
fi_mut = 0.1

d = 0.5

gen_limit = 50
num_experiments = 1000

lb_f4 = [-10, -10]
ub_f4 = [10, 10]

true_minimum_f4 = 0
epsilon_radius_f4 = 0.05


def f4(x, y):
    return (
        (math.sin(3 * math.pi * x)) ** 2
        + ((x - 1) ** 2 * (1 + (math.sin(3 * math.pi * y)) ** 2))
        + ((y - 1) ** 2 * (1 + (math.sin(2 * math.pi * y)) ** 2))
    )


lb_f12 = [0, 0]
ub_f12 = [math.pi, math.pi]

true_minimum_f12 = -1.8013
epsilon_radius_f12 = 0.01


def f12(x, y):
    return -math.sin(x) * math.pow(math.sin((x**2) / math.pi), 20) - math.sin(
        y
    ) * math.pow(math.sin((2 * y**2) / math.pi), 20)


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
    elite_fraction,
    crossover_fraction,
    mutation_fraction,
    lower_bound,
    upper_bound,
    generation_limit,
    function,
    crossover_area_expansion,
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


def run_multiple_experiments(num_experiments, epsilon_radius, true_minimum, *args):
    results = [run_experiment(*args) for _ in range(num_experiments)]

    best_result = float("inf")
    successful_experiments = 0
    total_points = 0
    successful_points = 0

    for res in results:
        best_result = min(best_result, res[0][1])

        num_mutation_points = int(len(res) * fi_mut)
        non_mutated_points = (
            res[:-num_mutation_points] if num_mutation_points > 0 else res
        )

        if all(
            abs(point[1] - true_minimum) <= epsilon_radius
            for point in non_mutated_points
        ):
            successful_experiments += 1

        for point in res:
            if abs(point[1] - true_minimum) <= epsilon_radius:
                successful_points += 1
            total_points += 1

    success_rate = successful_experiments / num_experiments * 100
    overall_accuracy = successful_points / total_points * 100

    return {
        "results": results,
        "best_result": best_result,
        "success_rate": success_rate,
        "overall_accuracy": overall_accuracy,
    }


if __name__ == "__main__":
    print("Running multiple experiments for function f4...")
    metrics_f4 = run_multiple_experiments(
        num_experiments,
        epsilon_radius_f4,
        true_minimum_f4,
        P,
        fi_sel,
        fi_cross,
        fi_mut,
        lb_f4,
        ub_f4,
        gen_limit,
        f4,
        d,
    )
    print(f"Best result for f4: {metrics_f4['best_result']}")
    print(f"Success rate for f4: {metrics_f4['success_rate']}%")
    print(f"Overall accuracy for f4: {metrics_f4['overall_accuracy']}%")

    print("\nRunning multiple experiments for function f12...")
    metrics_f12 = run_multiple_experiments(
        num_experiments,
        epsilon_radius_f12,
        true_minimum_f12,
        P,
        fi_sel,
        fi_cross,
        fi_mut,
        lb_f12,
        ub_f12,
        gen_limit,
        f12,
        d,
    )
    print(f"Best result for f12: {metrics_f12['best_result']}")
    print(f"Success rate for f12: {metrics_f12['success_rate']}%")
    print(f"Overall accuracy for f12: {metrics_f12['overall_accuracy']}%")
