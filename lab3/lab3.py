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
        print(f"Logging started. Output will be saved to: {self.path}")
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
        print(f"Generating {n} random cities within {width}x{height} area.")
        return [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]

    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def distance_matrix(coords):
        print("Creating distance matrix.")
        n = len(coords)
        return [
            [TSPHelper.euclidean(coords[i], coords[j]) if i != j else 0 for j in range(n)]
            for i in range(n)
        ]

    @staticmethod
    def save_cities(coords, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(coords)
        print(f"Cities saved to {path}")

    @staticmethod
    def load_cities(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            print(f"Loaded cities from {path}")
            return [(float(x), float(y)) for x, y in reader]

    @staticmethod
    def save_matrix(matrix, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(matrix)
        print(f"Distance matrix saved to {path}")

    @staticmethod
    def load_matrix(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            print(f"Loaded distance matrix from {path}")
            return [[float(cell) for cell in row] for row in reader]

    @staticmethod
    def route_distance(route, matrix):
        return sum(matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

    @staticmethod
    def generate_neighbors_2opt(route):
        n, neighbors = len(route), []
        for i in range(n - 1):
            for j in range(i + 1, n):
                new = route[:i] + route[i:j+1][::-1] + route[j+1:]
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
        print(f"Initial random route generated.")
        best = route[:]
        best_dist = TSPHelper.route_distance(best, self.matrix)
        print(f"Initial distance: {best_dist:.2f}")
        history = [best_dist]

        for i in range(self.iterations):
            neighbors = TSPHelper.generate_neighbors_2opt(route)

            candidates = [
                (r, TSPHelper.route_distance(r, self.matrix), m)
                for r, m in neighbors
                if m not in self.tabu
            ]
            if not candidates:
                print(f"Iteration {i}/{self.iterations}: No valid candidates (all in Tabu). Skipping.")
                continue

            best_candidate = min(candidates, key=lambda x: x[1])
            route = best_candidate[0]
            self.tabu.append(best_candidate[2])
            if len(self.tabu) > self.tabu_size:
                removed = self.tabu.pop(0)

            if best_candidate[1] < best_dist:
                best, best_dist = route[:], best_candidate[1]

            history.append(best_dist)

            if i % 10 == 0:
                print(f"Iteration {i}/{self.iterations}, Current Best: {best_dist:.2f}")
                print(f"Generated {len(neighbors)} neighbors.")
                print(f"Tabu added: {best_candidate[2]}")
                if len(self.tabu) > self.tabu_size:
                    print(f"Tabu list full. Removed: {removed}")
                print(f"New best distance: {best_dist:.2f}")

        print(f"Finished Tabu Search. Best Distance: {best_dist:.2f}")
        return best, best_dist, history


class Visualizer:
    @staticmethod
    def plot_cities(coords, filename="cities.png", output_dir="outputs"):
        plt.figure(figsize=(8, 6))
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        plt.plot(x, y, "o", markersize=6, color="blue", label="Cities")
        plt.title("Cities Distribution")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"Cities plot saved to {path}")

    @staticmethod
    def plot_initial_route(coords, initial_route, distance, filename="initial_route.png", output_dir="outputs"):
        plt.figure(figsize=(8, 6))
        x = [coords[i][0] for i in initial_route + [initial_route[0]]]
        y = [coords[i][1] for i in initial_route + [initial_route[0]]]
        plt.plot(x, y, "o-", markersize=4, label=f"Distance: {distance:.2f}")
        plt.title("Initial Random Route")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"Initial route plot saved to {path}")

    @staticmethod
    def plot_routes_grid(coords, routes, labels, filename="routes_grid.png", output_dir="outputs"):
        cols = 3
        rows = (len(routes) + cols - 1) // cols
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

        for i in range(len(routes), len(axes)):
            fig.delaxes(axes[i])

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Route plots saved to {path}")

    @staticmethod
    def plot_histories_grid(histories, labels, filename="histories_grid.png", output_dir="outputs"):
        cols = 3
        rows = (len(histories) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, (history, label) in enumerate(zip(histories, labels)):
            axes[idx].plot(history)
            axes[idx].set_title(label)
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Best Distance")
            axes[idx].grid(True)
        
        for i in range(len(histories), len(axes)):
            fig.delaxes(axes[i])

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"History plots saved to {path}")

class Configuration:
    def __init__(self, name, iterations, tabu_size):
        self.name = name
        self.iterations = iterations
        self.tabu_size = tabu_size

class Experiment:
    @staticmethod
    def run_experiment(configs, coords, matrix, output_dir="outputs"):
        print(f"Running {len(configs)} configurations...")
        histories, labels, best_routes = [], [], []
        summary_table = []
        num_cities = len(coords)

        Visualizer.plot_cities(coords, output_dir=output_dir)

        initial_random_route = list(range(num_cities))
        random.shuffle(initial_random_route)
        initial_random_distance = TSPHelper.route_distance(initial_random_route, matrix)
        Visualizer.plot_initial_route(coords, initial_random_route, initial_random_distance, output_dir=output_dir)


        for cfg in configs:
            print(f"\n=== Running configuration: {cfg.name} ===")
            solver = TabuSearch(coords, matrix, cfg.iterations, cfg.tabu_size)
            best_route, best_dist, history = solver.solve()
            init_dist_for_cfg = history[0]
            improvement = (init_dist_for_cfg - best_dist) / init_dist_for_cfg * 100

            print(f"Initial: {init_dist_for_cfg:.2f}, Final: {best_dist:.2f}, Improvement: {improvement:.2f}%")

            histories.append(history)
            labels.append(f"{cfg.name} ({best_dist:.0f})")
            best_routes.append(best_route)

            summary_table.append({
                "Name": cfg.name,
                "Iterations": cfg.iterations,
                "TabuSize": cfg.tabu_size,
                "InitialDistance": round(init_dist_for_cfg, 2),
                "FinalDistance": round(best_dist, 2),
                "Improvement(%)": round(improvement, 2)
            })

        print("Creating visualization plots...")
        Visualizer.plot_routes_grid(coords, best_routes, labels, filename="routes_grid.png", output_dir=output_dir)
        Visualizer.plot_histories_grid(histories, labels, filename="histories_grid.png", output_dir=output_dir)
        print("Plots saved.")

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
        print(f"Summary saved to {csv_path}")

class Utils:
    @staticmethod
    def generate_configurations(iter_opt, tabu_ratio_opt, num_cities):
        print("Generating configurations...")
        configurations_list = []
        for i, (iters, ratio) in enumerate(itertools.product(iter_opt, tabu_ratio_opt), 1):
            tabu_size = int(num_cities * ratio)
            if tabu_size == 0 and num_cities > 0:
                tabu_size = 1

            name = f"Cfg{i}_it{iters}_tb{tabu_size}"
            configurations_list.append(Configuration(name=name, iterations=iters, tabu_size=tabu_size))
        print(f"{len(configurations_list)} configurations generated.")
        return configurations_list

    @staticmethod
    def prepare_inputs(num_cities, width, height, inputs_dir="inputs"):
        os.makedirs(inputs_dir, exist_ok=True)
        cities_path = os.path.join(inputs_dir, "cities.csv")
        matrix_path = os.path.join(inputs_dir, "matrix.csv")

        if os.path.exists(cities_path):
            cities = TSPHelper.load_cities(cities_path)
        else:
            cities = TSPHelper.generate_cities(num_cities, width, height)
            TSPHelper.save_cities(cities, cities_path)

        matrix = TSPHelper.distance_matrix(cities)
        TSPHelper.save_matrix(matrix, matrix_path)

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

    iteration_options = [125, 250, 500]
    tabu_ratio_options = [0.25, 0.5, 0.75]
    configurations = Utils.generate_configurations(iteration_options, tabu_ratio_options, NUM_CITIES)
    cities, matrix = Utils.prepare_inputs(NUM_CITIES, WIDTH, HEIGHT, inputs_dir=INPUT_DIR)

    Experiment.run_experiment(configurations, cities, matrix, output_dir=OUTPUT_DIR)

    logger.stop()
