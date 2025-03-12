import random
import math

P = 100

fi_sel = 0.3
fi_cross = 0.4
fi_mut = 0.3

lb_f4 = [-10, -10]
ub_f4 = [10, 10]

d = 0.5

gen_limit = 500


def f4(x, y):
    return (
        (math.sin(3 * math.pi * x)) ** 2
        + ((x - 1) ** 2 * (1 + (math.sin(3 * math.pi * y)) ** 2))
        + ((y - 1) ** 2 * (1 + (math.sin(2 * math.pi * y)) ** 2))
    )


def generate_population(lower_bound, upper_bound, population_size):
    population = []

    for i in range(population_size):
        point = []
        for d in range(len(lower_bound)):
            r_id = random.uniform(0, 1)

            coord_id = lower_bound[d] + r_id * (upper_bound[d] - lower_bound[d])

            point.append(coord_id)

        population.append(point)

    return population


def sort_population(function, population):
    population_values = [(point, function(point[0], point[1])) for point in population]

    population_values.sort(key=lambda x: x[1])

    return [value[0] for value in population_values]


def uniform_crossover_in_natural_coding(parent_1, parent_2, d):
    beta_1 = random.uniform(-d, 1 + d)
    beta_2 = random.uniform(-d, 1 + d)

    new_point_x = parent_1[0] + beta_1 * (parent_2[0] - parent_1[0])
    new_point_y = parent_1[1] + beta_2 * (parent_2[1] - parent_1[1])

    return [new_point_x, new_point_y]


def crossover_population(population, d):
    new_population = []

    while len(new_population) < len(population):
        parent1, parent2 = random.sample(population, 2)

        child = uniform_crossover_in_natural_coding(parent1, parent2, d)

        new_population.append(child)

    return new_population


def mutation_in_natural_coding(point, lower_bound, upper_bound, sigma):
    mutated_point = []

    for i in range(len(point)):
        mutation_value = random.gauss(0, sigma[i])

        new_value = point[i] + mutation_value
        new_value = max(min(new_value, upper_bound[i]), lower_bound[i])

        mutated_point.append(new_value)

    return mutated_point


def mutate_population(population, lower_bound, upper_bound, sigma):
    return [
        mutation_in_natural_coding(point, lower_bound, upper_bound, sigma)
        for point in population
    ]


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
    population = generate_population(lower_bound, upper_bound, population_size)
    sigma = [(upper_bound[i] - lower_bound[i]) / 10 for i in range(len(lower_bound))]

    for i in range(generation_limit):
        if i % 10 == 0:
            print(f"Generation {i}...")

        sorted_population = sort_population(function, population)

        elite_count = int(len(sorted_population) * elite_fraction)
        crossover_count = int(len(sorted_population) * crossover_fraction)

        elite_points = sorted_population[:elite_count]
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

    population = sort_population(function, population)
    return population


result = run_experiment(P, fi_sel, fi_cross, fi_mut, lb_f4, ub_f4, gen_limit, f4, d)

for i in range(0, 10):
    print(result[i])
