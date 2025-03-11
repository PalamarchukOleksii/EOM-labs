import random
import math

P = 10

fi_sel = 0.2
fi_cross = 0.4
fi_mut = 0.4

lb_f4 = {"x": -10, "y": -10}
ub_f4 = {"x": 10, "y": 10}

param_list = ["x", "y"]


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


population = generate_population(lb_f4, ub_f4, P, param_list)

sorted_population = sort_population(f4, population)

for el in sorted_population:
    print(el)
