from cleanNewProject.config import Modelvariables
from Kernels.knn import knn
from Kernels.knn1 import knn1
import numpy as np

def predictor(x_star):


    x_u = Modelvariables['x_u']
    x_b = Modelvariables['x_b']

    u = Modelvariables['u']
    u_b = Modelvariables['u_b']
    y = np.concatenate((u_b, u), axis=0)

    n_u = len(u)
    n_b = len(u_b)

    S0 = Modelvariables['S0']

    S = np.block([
        [np.zeros((n_b, n_b)), np.zeros((n_b, n_u))],
        [np.zeros((n_u, n_b)), S0]
    ])

    hyp = Modelvariables['hyp']

    D = x_u.shape[1]

    K1 = knn(x_star, x_b, hyp, 0)
    K2 = knn1(x_star, x_u, hyp, u, 0)



    # Assuming K1 and K2 are already defined as numpy arrays
    psi = np.hstack((K1, K2))

    L = Modelvariables['L']

    # Solve for f
    # L \ y is equivalent to np.linalg.solve(L, y)
    # L' \ (L \ y) is equivalent to np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ly = np.linalg.solve(L, y)
    f = np.dot(psi, np.linalg.solve(L.T, Ly))

    # Solve for alpha
    # L' \ (L \ psi') is equivalent to np.linalg.solve(L.T, np.linalg.solve(L, psi.T))
    Lpsi = np.linalg.solve(L, psi.T)
    alpha = np.linalg.solve(L.T, Lpsi)

    # Compute v
    # knn is a placeholder for the appropriate kernel function, replace it accordingly
    v = knn(x_star, x_star, Modelvariables['hyp'], 0) - np.dot(psi, alpha) + np.dot(alpha.T, np.dot(S, alpha))


    return f, v