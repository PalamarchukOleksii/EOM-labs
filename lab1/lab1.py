import random
import math

P = 10

fi_sel = 0.2
fi_cross = 0.4
fi_mut = 0.4

lb_f4 = {"x": -10, "y": -10}
ub_f4 = {"x": 10, "y": 10}

param_list = ["x", "y"]

d = 1

gen_limit = 100


def f4(x, y):
    return (
        (math.sin(3 * math.pi * x)) ** 2
        + ((x - 1) ** 2 * (1 + (math.sin(3 * math.pi * y)) ** 2))
        + ((y - 1) ** 2 * (1 + (math.sin(2 * math.pi * y)) ** 2))
    )


def generate_population(lower_bound, upper_bound, population_size, parameters_list):
    population = []

    for i in range(population_size):
        point = []
        for d in parameters_list:
            r_id = random.uniform(0, 1)

            coord_id = lower_bound[d] + r_id * (upper_bound[d] - lower_bound[d])

            point.append(coord_id)

        population.append(point)

    return population


def sort_population(function, population):
    population_values = []
    for point in population:
        population_values.append((point, function(point[0], point[1])))

    population_values.sort(key=lambda x: x[1])

    sorted_population = []
    for value in population_values:
        sorted_population.append(value[0])

    return sorted_population


def uniform_crossover_in_natural_coding(parent_1, parent_2, d):
    beta_1 = random.uniform(-d, 1 + d)
    beta_2 = random.uniform(-d, 1 + d)

    new_point_x = parent_1[0] + beta_1 * (parent_2[0] - parent_1[0])
    new_point_y = parent_1[1] + beta_2 * (parent_2[1] - parent_1[1])

    return [new_point_x, new_point_y]


def crossover_population(population, d):
    new_population = []

    while len(new_population) < len(population):
        parent1 = random.choice(population)
        parent2 = random.choice(population)

        child = uniform_crossover_in_natural_coding(parent1, parent2, d)

        new_population.append(child)

    return new_population


def mutation_in_natural_coding(point, lower_bound, upper_bound, sigma):
    mutated_point = []

    for i in len(point):
        mutation_value = random.gauss(0, sigma[i])

        new_value = point[i] + mutation_value
        new_value = max(min(new_value, upper_bound[i]), lower_bound[i])

        mutated_point.append(new_value)

    return mutated_point


def mutate_population(population, lower_bound, upper_bound, sigma):
    mutate_population = []

    for point in population:
        mutated_point = mutation_in_natural_coding(
            point, upper_bound, lower_bound, sigma
        )

        mutate_population.append(mutated_point)

    return mutate_population


def run_experiment(
    population_size,
    elite_fraction,
    crossover_fraction,
    mutation_fraction,
    lower_bound,
    upper_bound,
    parameters_list,
    generation_limit,
    function,
    crossover_area_expansion,
):
    population = generate_population(
        lower_bound, upper_bound, population_size, parameters_list
    )
    sigma = [(upper_bound[key] - lower_bound[key]) / 6 for key in lower_bound]

    for i in range(generation_limit):
        sorted_population = sort_population(function, population)

        elite_count = int(len(sorted_population) * elite_fraction)
        crossover_count = int(len(sorted_population) * crossover_fraction)

        elite_points = sorted_population[0:elite_count]
        crossover_points = sorted_population[
            elite_count : elite_count + crossover_count
        ]
        mutation_points = sorted_population[elite_count + crossover_count :]

        crossovered_points = crossover_population(
            crossover_points, crossover_area_expansion
        )
        mutated_points = mutate_population(
            mutation_points, lower_bound, upper_bound, sigma
        )

        population = elite_points + crossovered_points + mutated_points

    return population


result = run_experiment(
    P, fi_sel, fi_cross, fi_mut, lb_f4, ub_f4, param_list, gen_limit, f4, d
)

for point in result:
    print(point)
