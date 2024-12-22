import pygad
import numpy as np
import networkx as nx
import heapq

# Directions for moving: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def dijkstra(grid, start, target):
    rows, cols = len(grid), len(grid[0])

    # Initialize distance matrix, set all values to infinity
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[start[0]][start[1]] = 0  # Start point cost

    # Priority queue (min-heap) for Dijkstra's algorithm
    pq = [(dist[start[0]][start[1]], start)]  # (cost, (row, col))

    # While there are nodes to process in the queue
    while pq:
        current_dist, (r, c) = heapq.heappop(pq)

        # If we reach the target, return the distance
        if (r, c) == target:
            return current_dist

        # Explore all neighbors (up, down, left, right)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Boundary check
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                new_dist = current_dist + 1

                # If a shorter path is found
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(pq, (new_dist, (nr, nc)))

    return float('inf')  # Return infinity if no path exists

class MazeGenGA(pygad.GA):
    def __init__(self, map_shape, config) -> None:
        self.map_objects = ["air", "obstacles", "start point", "flag"]
        self.map_shape = map_shape
        
        self.gene_space = [i for i in range(len(self.map_objects))]

        if config["mutation"] == "one_point":
            mut_type = self._one_point_mutation
        else:
            mut_type = config["mutation"]

        super().__init__(num_generations = config["num_gen"],
            num_parents_mating = config["num_par_mat"],
            fitness_func = self._fitness_func,
            sol_per_pop = config["population"],
            num_genes = self.map_shape[0] * self.map_shape[1],
            gene_space = self.gene_space,
            gene_type = int,
            on_generation = self.callback_generation,
            K_tournament = config["K_tour"] if config["parent_selection_type"] == "tournament" else None,
            parent_selection_type = config["parent_selection_type"],
            crossover_type = config["Xover"],
            mutation_type = mut_type,
            mutation_probability = config["mut_prob"],
            keep_elitism = config["keep_elitism"],
        )

        self.records = {"best_fit_val": [],
                        }
    
    def _fitness_func(self, ga_inst, sol, sol_idx) -> float:
        fit_val = 0
        maze = self.geno2pheno(sol)

        no_start = False
        no_goal = False

        infeasible_pen = len(sol) * 2
        # 1 start
        indices = np.where(maze == 2)
        start_coords = list(zip(indices[0], indices[1]))
        if len(start_coords) >= 1:
            start_coord = start_coords[0]
            fit_val -= (len(start_coords) - 1) * infeasible_pen
        else:
            no_start = True

        # 1 goal
        indices = np.where(maze == 3)
        goal_coords = list(zip(indices[0], indices[1]))
        if len(goal_coords) >= 1:
            goal_coord = goal_coords[0]
            fit_val -= (len(goal_coords) - 1) * infeasible_pen
        else:
            no_goal = True

        if no_start or no_goal:
            fit_val -= 2 * infeasible_pen
            return fit_val

        # maze complexity
        # count the min len between start and goal
        dist = dijkstra(maze, start_coord, goal_coord)
        if dist == float('inf'):
            fit_val -= infeasible_pen
        else:
            fit_val += 2 * dist

        # count number of air
        air_num = np.count_nonzero(maze == 0)
        fit_val += air_num

        return fit_val


    def geno2pheno(self, geno:np.ndarray) -> np.ndarray:
        pheno = geno.reshape(self.map_shape)
        return pheno

    def _one_point_mutation(self, offspring, ga_instance):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.mutation_probability:
                mut_idx = np.random.randint(0, offspring.shape[1])
                tmp = np.random.choice(self.gene_space, 1)[0]
                while tmp == offspring[idx][mut_idx]:
                    tmp = np.random.choice(self.gene_space, 1)[0]
                offspring[idx][mut_idx] = tmp
                
        return offspring

    def callback_generation(self, ga_instance):
        sol, best_fit, _ = ga_instance.best_solution()
        self.records["best_fit_val"].append(best_fit)