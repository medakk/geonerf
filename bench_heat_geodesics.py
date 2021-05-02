import sys
import pickle
import random
import time

import numpy as np
import igl
import potpourri3d as pp3d
from tqdm import trange

def main():
    in_fname = sys.argv[1]
    V, F = igl.read_triangle_mesh(in_fname)
    
    # parameter for heat equation
    solver = pp3d.MeshHeatMethodDistanceSolver(V,F)

    N = V.shape[0]
    D = np.ndarray((N, N))
    start_time = time.time()
    n_trials = 100
    for i in range(n_trials):
        solver.compute_distance(random.randrange(0, N))
    end_time = time.time()
    avg_time = (end_time - start_time) / n_trials
    print(f'Average time for geodesics from a given source: {avg_time}')
    
if __name__ == '__main__':
    main()