import os, sys, random, time, math
import matplotlib.pyplot as plt


class OutputLogger:
    def __init__(self, enabled, directory, filename):
        self.enabled = enabled
        self.path = os.path.join(directory, filename)
        self.original_stdout = sys.stdout
        self.log_file = None

    def start(self):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.log_file = open(self.path, "w", encoding="utf-8")
        sys.stdout = self.log_file

    def stop(self):
        if not self.enabled or not self.log_file:
            return
        sys.stdout = self.original_stdout
        self.log_file.close()
        print(f"Logs saved to: {self.path}")


class TSPHelper:
    @staticmethod
    def generate_cities(n, width, height):
        return [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]

    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def distance_matrix(coords):
        n = len(coords)
        return [
            [
                TSPHelper.euclidean(coords[i], coords[j]) if i != j else 0
                for j in range(n)
            ]
            for i in range(n)
        ]

    @staticmethod
    def route_distance(route, matrix):
        return sum(
            matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route))
        )

    @staticmethod
    def generate_neighbors_2opt(route):
        n, neighbors = len(route), []
        for i in range(n - 1):
            for j in range(i + 1, n):
                new = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                neighbors.append((new, (i, j)))
        return neighbors


class TabuSearch:
    def __init__(self, coords, iterations, tabu_size):
        self.coords = coords
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.matrix = TSPHelper.distance_matrix(coords)
        self.tabu = []

    def solve(self):
        route = list(range(len(self.coords)))
        random.shuffle(route)
        best = route[:]
        best_dist = TSPHelper.route_distance(best, self.matrix)
        history = [best_dist]

        for i in range(self.iterations):
            if i % 100 == 0:
                print(f"{i=}")

            neighbors = TSPHelper.generate_neighbors_2opt(route)
            candidates = [
                (r, TSPHelper.route_distance(r, self.matrix), m)
                for r, m in neighbors
                if m not in self.tabu
            ]
            if not candidates:
                continue
            best_candidate = min(candidates, key=lambda x: x[1])
            route = best_candidate[0]
            self.tabu.append(best_candidate[2])
            if len(self.tabu) > self.tabu_size:
                self.tabu.pop(0)
            if best_candidate[1] < best_dist:
                best, best_dist = route[:], best_candidate[1]
            history.append(best_dist)

        return best, best_dist, history


class Visualizer:
    @staticmethod
    def plot_route(coords, route, title="TSP Route"):
        x = [coords[i][0] for i in route + [route[0]]]
        y = [coords[i][1] for i in route + [route[0]]]
        plt.figure(figsize=(10, 8))
        plt.plot(x, y, "o-", markersize=4)
        plt.title(title)
        plt.gca().set_aspect("equal")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    @staticmethod
    def plot_history(histories, labels):
        plt.figure(figsize=(12, 6))
        for h, label in zip(histories, labels):
            plt.plot(h, label=label)
        plt.title("Tabu Search Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.grid(True)
        plt.legend()
        plt.show()


class Configuration:
    def __init__(self, name, iterations, tabu_size):
        self.name, self.iterations, self.tabu_size = name, iterations, tabu_size


def run_experiment(configs, num_cities, width, height):
    coords = TSPHelper.generate_cities(num_cities, width, height)
    histories, labels = [], []

    for cfg in configs:
        print(f"\nRunning: {cfg.name}")
        solver = TabuSearch(coords, cfg.iterations, cfg.tabu_size)
        best_route, best_dist, history = solver.solve()
        init_dist = TSPHelper.route_distance(list(range(num_cities)), solver.matrix)
        print(
            f"Initial: {init_dist:.2f}, Final: {best_dist:.2f}, Improvement: {(init_dist - best_dist) / init_dist * 100:.2f}%"
        )
        histories.append(history)
        labels.append(f"{cfg.name} ({best_dist:.0f})")
        Visualizer.plot_route(coords, best_route, f"{cfg.name} - Best Route")

    Visualizer.plot_history(histories, labels)


if __name__ == "__main__":
    random.seed(42)
    NUM_CITIES = 100
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logger = OutputLogger(
        enabled=False,
        directory=os.path.join(BASE_DIR, "outputs"),
        filename="tsp_log.txt",
    )
    logger.start()

    configurations = [
        Configuration("Conservative", iterations=10000, tabu_size=30),
        Configuration("Aggressive", iterations=25000, tabu_size=70),
    ]
    run_experiment(configurations, NUM_CITIES, 10000, 10000)

    logger.stop()
