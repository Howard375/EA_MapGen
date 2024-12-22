import numpy as np
from MapGen import MazeGenGA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

def main():
    map_shape = (5, 5)
    figure_name = "5_5"
    config = {
        "population": 50,
        "num_gen": 2000,
        "num_par_mat": 50,
        "parent_selection_type": "tournament", # "tournament"
        "K_tour": 2,
        "Xover": "single_point", # "single_point"
        "mutation": "one_point", # "one_point", "random"
        "mut_prob": 0.5, # 0.3, 0.1, 0.01
        "keep_elitism": 10,
    }
    ga_inst = MazeGenGA(map_shape, config)
    ga_inst.run()
    sol, sol_fit, sol_idx = ga_inst.best_solution()
    records = ga_inst.records
    maze = ga_inst.geno2pheno(sol)
    print(maze)
    print(sol_fit)
    plt.figure()
    plt.plot(list(range(config["num_gen"])), records["best_fit_val"])
    plt.title("AVG Best Fitness Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    
    plt.figure()
    colors = ['white', 'black', 'red', 'green']
    custom_cmap = ListedColormap(colors)
    norm = Normalize(vmin=np.min(maze), vmax=np.max(maze))
    plt.imshow(maze, cmap = custom_cmap, interpolation = 'nearest', norm = norm)
    plt.grid(True)
    plt.savefig(f'./output/{figure_name}')
    plt.show()
    

if __name__ == "__main__":
    main()
