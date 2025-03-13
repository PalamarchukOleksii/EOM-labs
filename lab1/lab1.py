import random
import math

P = 25

fi_sel = 0.2
fi_cross = 0.7
fi_mut = 0.1

lb_f4 = [-10, -10]
ub_f4 = [10, 10]

d = 0.5

gen_limit = 50


def f4(x, y):
    return (
        (math.sin(3 * math.pi * x)) ** 2
        + ((x - 1) ** 2 * (1 + (math.sin(3 * math.pi * y)) ** 2))
        + ((y - 1) ** 2 * (1 + (math.sin(2 * math.pi * y)) ** 2))
    )


lb_f12 = [0, 0]
ub_f12 = [math.pi, math.pi]


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
        if i % 10 == 0:
            print(f"Generation {i}...")

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
    print("Running experiment for function f4...")
    result = run_experiment(P, fi_sel, fi_cross, fi_mut, lb_f4, ub_f4, gen_limit, f4, d)

    for point, value in result:
        print(f"Point: {point}, Value: {value}")

    print("\nRunning experiment for function f12...")
    result = run_experiment(
        P, fi_sel, fi_cross, fi_mut, lb_f12, ub_f12, gen_limit, f12, d
    )

    for point, value in result:
        print(f"Point: {point}, Value: {value}")
