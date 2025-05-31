import os, sys, random, math, itertools, csv
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
    def save_cities(coords, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(coords)

    @staticmethod
    def load_cities(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            return [(float(x), float(y)) for x, y in reader]

    @staticmethod
    def save_matrix(matrix, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(matrix)

    @staticmethod
    def load_matrix(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            return [[float(cell) for cell in row] for row in reader]

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
    def __init__(self, coords, matrix, iterations, tabu_size):
        self.coords = coords
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.matrix = matrix
        self.tabu = []

    def solve(self):
        route = list(range(len(self.coords)))
        random.shuffle(route)
        best = route[:]
        best_dist = TSPHelper.route_distance(best, self.matrix)
        history = [best_dist]

        for i in range(self.iterations):
            if i % 10 == 0:
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
    def plot_routes_grid(coords, routes, labels, filename="routes_grid.png", output_dir="outputs"):
        cols = 3
        rows = (len(routes) + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, (route, label) in enumerate(zip(routes, labels)):
            x = [coords[i][0] for i in route + [route[0]]]
            y = [coords[i][1] for i in route + [route[0]]]
            ax = axes[idx]
            ax.plot(x, y, "o-", markersize=4)
            ax.set_title(label)
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.6)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def plot_histories_grid(histories, labels, filename="histories_grid.png", output_dir="outputs"):
        cols = 3
        rows = (len(histories) + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, (history, label) in enumerate(zip(histories, labels)):
            axes[idx].plot(history)
            axes[idx].set_title(label)
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Best Distance")
            axes[idx].grid(True)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Configuration:
    def __init__(self, name, iterations, tabu_size):
        self.name, self.iterations, self.tabu_size = name, iterations, tabu_size


class Experiment:
    @staticmethod
    def run_experiment(configs, coords, matrix, output_dir="outputs"):
        histories, labels, best_routes = [], [], []
        summary_table = []
        num_cities = len(coords)

        for cfg in configs:
            print(f"\nRunning: {cfg.name}")
            solver = TabuSearch(coords, matrix, cfg.iterations, cfg.tabu_size)
            best_route, best_dist, history = solver.solve()
            init_dist = TSPHelper.route_distance(list(range(num_cities)), matrix)
            improvement = (init_dist - best_dist) / init_dist * 100

            print(f"Initial: {init_dist:.2f}, Final: {best_dist:.2f}, Improvement: {improvement:.2f}%")

            histories.append(history)
            labels.append(f"{cfg.name} ({best_dist:.0f})")
            best_routes.append(best_route)

            summary_table.append({
                "Name": cfg.name,
                "Iterations": cfg.iterations,
                "TabuSize": cfg.tabu_size,
                "InitialDistance": round(init_dist, 2),
                "FinalDistance": round(best_dist, 2),
                "Improvement(%)": round(improvement, 2)
            })

        Visualizer.plot_routes_grid(coords, best_routes, labels, filename="routes_grid.png", output_dir=output_dir)
        Visualizer.plot_histories_grid(histories, labels, filename="histories_grid.png", output_dir=output_dir)
        Experiment.save_summary(summary_table, output_dir)

    @staticmethod
    def save_summary(table, output_dir):
        print("\nComparison Table:")
        headers = table[0].keys()
        row_format = "{:<20}" * len(headers)
        print(row_format.format(*headers))
        for row in table:
            print(row_format.format(*[str(row[h]) for h in headers]))

        csv_path = os.path.join(output_dir, "summary.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(table)


class Utils:
    @staticmethod
    def generate_configurations(iter_opt, tabu_ratio_opt, num_cities):
        configurations_list = []
        for i, (iters, ratio) in enumerate(itertools.product(iter_opt, tabu_ratio_opt), 1):
            tabu_size = int(num_cities * ratio)
            name = f"Cfg{i}_it{iters}_tb{tabu_size}"
            configurations_list.append(Configuration(name=name, iterations=iters, tabu_size=tabu_size))
        return configurations_list
    
    @staticmethod
    def prepare_inputs(num_cities, width, height, inputs_dir="inputs"):
        os.makedirs(inputs_dir, exist_ok=True)
        cities_path = os.path.join(inputs_dir, "cities.csv")
        matrix_path = os.path.join(inputs_dir, "matrix.csv")

        if os.path.exists(cities_path):
            cities = TSPHelper.load_cities(cities_path)
            print("Loaded cities from file.")
        else:
            cities = TSPHelper.generate_cities(num_cities, width, height)
            TSPHelper.save_cities(cities, cities_path)
            print("Generated and saved cities.")

        matrix = TSPHelper.distance_matrix(cities)
        TSPHelper.save_matrix(matrix, matrix_path)
        print("Generated and saved distance matrix.")

        return cities, matrix


if __name__ == "__main__":
    random.seed(42)

    ENABLE_LOGGING = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "inputs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    OUTPUT_FILENAME = "tsp_log.txt"

    NUM_CITIES = 100
    WIDTH, HEIGHT = 10000, 10000

    logger = OutputLogger(
        enabled=ENABLE_LOGGING,
        directory=OUTPUT_DIR,
        filename=OUTPUT_FILENAME,
    )
    logger.start()

    iteration_options = [125, 250]
    tabu_ratio_options = [0.25, 0.5, 0.75]
    configurations = Utils.generate_configurations(iteration_options, tabu_ratio_options, NUM_CITIES)
    cities, matrix = Utils.prepare_inputs(NUM_CITIES, WIDTH, HEIGHT, inputs_dir=INPUT_DIR)

    Experiment.run_experiment(configurations, cities, matrix, output_dir=OUTPUT_DIR)

    logger.stop()
