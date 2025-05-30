import random
import math
import matplotlib.pyplot as plt


def generate_random_coordinates(n=100, width=10000, height=10000):
    return [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]


def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def create_distance_matrix(coordinates):
    n = len(coordinates)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    return matrix


def calculate_distance(route, distance_matrix):
    total_distance = 0
    num_cities = len(route)
    for i in range(num_cities):
        total_distance += distance_matrix[route[i]][route[(i + 1) % num_cities]]
    return total_distance


def get_neighbors_2opt(route):
    n = len(route)
    neighbors = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            new_route = route[:]
            new_route[i : j + 1] = new_route[i : j + 1][::-1]

            neighbors.append((new_route, (i, j)))
    return neighbors


def tabu_search_flowchart(coordinates, iterations=50000, tabu_size=30):
    distance_matrix = create_distance_matrix(coordinates)
    n = len(coordinates)

    current_solution = list(range(n))
    random.shuffle(current_solution)
    start_solution = current_solution[:]

    tabu_list = []

    best_solution = current_solution[:]
    best_distance = calculate_distance(best_solution, distance_matrix)

    distances_history = [best_distance]

    for iter_num in range(iterations):
        if iter_num % 100 == 0:
            print(f"{iter_num=}")

        best_candidate_neighbor_route = None
        best_candidate_distance = float("inf")
        best_candidate_move = None

        neighbors = get_neighbors_2opt(current_solution)

        for neighbor_route, move in neighbors:
            if move not in tabu_list:
                neighbor_distance = calculate_distance(neighbor_route, distance_matrix)
                if neighbor_distance < best_candidate_distance:
                    best_candidate_distance = neighbor_distance
                    best_candidate_neighbor_route = neighbor_route[:]
                    best_candidate_move = move

        if best_candidate_neighbor_route is not None:
            current_solution = best_candidate_neighbor_route[:]
            current_distance = best_candidate_distance

            if current_distance < best_distance:
                best_solution = current_solution[:]
                best_distance = current_distance

            tabu_list.append(best_candidate_move)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
        else:
            pass

        distances_history.append(best_distance)

    return start_solution, best_solution, best_distance, distances_history


def plot_route(coordinates, route, title):
    x = [coordinates[i][0] for i in route] + [coordinates[route[0]][0]]
    y = [coordinates[i][1] for i in route] + [coordinates[route[0]][1]]

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, marker="o", linestyle="-", color="blue")

    for i, (xi, yi) in enumerate(coordinates):
        plt.text(xi + 0.5, yi + 0.5, str(i), fontsize=9, ha="center", va="bottom")

    plt.title(title)
    plt.xlabel("X-координата")
    plt.ylabel("Y-координата")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_cities(coordinates):
    x = [city[0] for city in coordinates]
    y = [city[1] for city in coordinates]

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, marker="o", color="red", s=50, zorder=5)

    for i, (xi, yi) in enumerate(coordinates):
        plt.text(xi + 0.5, yi + 0.5, str(i), fontsize=9, ha="center", va="bottom")

    plt.title("Координати міст (без маршруту)")
    plt.xlabel("X-координата")
    plt.ylabel("Y-координата")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_progress(distances_history, title="Прогрес алгоритму табу-пошуку"):
    plt.figure(figsize=(10, 6))
    plt.plot(distances_history, color="green", linewidth=2)
    plt.title(title)
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраща довжина маршруту")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    num_cities = 100

    coordinates = generate_random_coordinates(n=num_cities)

    iterations_count = 50000
    tabu_list_size = num_cities / 2

    print(f"Запускаємо табу-пошук для {num_cities} міст...")
    print(
        f"Кількість ітерацій: {iterations_count}, Розмір табу-списку: {tabu_list_size}"
    )

    start_solution, best_route, best_cost, history = tabu_search_flowchart(
        coordinates, iterations=iterations_count, tabu_size=tabu_list_size
    )

    initial_distance = calculate_distance(
        start_solution, create_distance_matrix(coordinates)
    )

    print("\n--- Результати табу-пошуку ---")
    print(f"Кількість міст: {num_cities}")
    print(f"Початкова довжина маршруту: {round(initial_distance, 2)}")
    print(f"Найкраща довжина маршруту, знайдена алгоритмом: {round(best_cost, 2)}")
    print(
        f"Покращення: {round((initial_distance - best_cost) / initial_distance * 100, 2)}%"
    )

    print("\nПочатковий маршрут (перші 5 та останні 5 міст):")
    print(start_solution[:5], "...", start_solution[-5:])
    print("\nНайкращий знайдений маршрут (перші 5 та останні 5 міст):")
    print(best_route[:5], "...", best_route[-5:])

    print("\n--- Візуалізація результатів ---")
    plot_cities(coordinates)
    plot_route(
        coordinates,
        start_solution,
        f"Початковий маршрут (Довжина: {round(initial_distance, 2)})",
    )
    plot_route(
        coordinates, best_route, f"Найкращий маршрут (Довжина: {round(best_cost, 2)})"
    )
    plot_progress(
        history, f"Прогрес табу-пошуку ({num_cities} міст, {iterations_count} ітерацій)"
    )
