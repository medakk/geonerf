import sys
import pickle

import numpy as np
import igl
import potpourri3d as pp3d
from tqdm import trange

def main():
    in_fname, subdivs, out_fname = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    V, F = igl.read_triangle_mesh(in_fname)

    if subdivs != 0:
        print('before upsampling:', V.shape)
        V, F = igl.upsample(V, F, subdivs)
        print('after upsampling: ', V.shape)
    
    # parameter for heat equation
    solver = pp3d.MeshHeatMethodDistanceSolver(V,F)

    N = V.shape[0]
    D = np.ndarray((N, N))
    for i in trange(N):
        D[i] = solver.compute_distance(i)
    
    pickle.dump({
        'V': V,
        'D': D
    }, open(out_fname, 'wb'))

if __name__ == '__main__':
    main()