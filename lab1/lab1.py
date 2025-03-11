import random

P = 10

fi_sel = 0.2
fi_cross = 0.4
fi_mut = 0.4

lb_f4 = {"x": -10, "y": -10}
ub_f4 = {"x": 10, "y": 10}

param_list = ["x", "y"]


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


population = generate_population(lb_f4, ub_f4, P, param_list)

for point in population:
    print(point)
