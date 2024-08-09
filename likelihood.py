import numpy as np
from scipy.linalg import cholesky, solve_triangular

from cleanNewProject.config import Modelvariables
from cleanNewProject.Kernels.knn import knn
from cleanNewProject.Kernels.knn1 import knn1
from cleanNewProject.Kernels.kn1n1 import kn1n1



def likelihood(hyp):

    x_u = Modelvariables['x_u']
    x_b = Modelvariables['x_b']
    u = Modelvariables['u']
    u_b = Modelvariables['u_b']
    jitter = Modelvariables['jitter']
    D = Modelvariables['D']

    y = np.concatenate([u_b, u])

    sigma_n = np.exp(Modelvariables['hyp'][2])

    n_u = len(x_u)
    n_b = len(x_b)
    n = n_b + n_u


    Knn = knn(x_b, x_b, hyp, 0)
    Knn1 = knn1(x_b, x_u, hyp, u, 0)
    Kn1n1 = kn1n1(x_u, x_u, hyp, u, u, 0)

    K = np.block([
        [Knn, Knn1],
        [Knn1.T, Kn1n1]
    ])



    K = K + np.eye(n) * jitter


    # Cholesky factorisation
    try:

        #print(f" Eigenwerte: {np.linalg.eigvalsh(K)}")
        L = cholesky(K, lower=True)
        Modelvariables['L'] = L
    except np.linalg.LinAlgError:

        print('Covariance is ill-conditioned')
        return None, None

    # Solve for alpha
    #alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
    #alpha = np.linalg.solve(L, np.dot(L.T, y))
    # Solve L * temp = y for temp
    temp = solve_triangular(L, y, lower=True)

    # Solve L.T * alpha = temp for alpha
    alpha = solve_triangular(L.T, temp, lower=False)


    # Compute NLML
    NLML = 0.5 * np.dot(y.T, alpha) + np.sum(np.log(np.diag(L))) + np.log(2 * np.pi) * y.size / 2

    # Initialize derivatives
    D_NLML = np.zeros_like(hyp)

    # Compute Q
    eye_n = np.eye(len(y))
    Q = solve_triangular(L.T, solve_triangular(L, eye_n, lower=True)) - np.outer(alpha, alpha)

    for i in range(0,len(hyp)-1):

        DKnn = knn(x_b, x_b, hyp[:D + 1], i+1)
        DKnn1 = knn1(x_b, x_u, hyp[:D + 1], u, i+1)
        DKn1n1 = kn1n1(x_u, x_u, hyp[:D + 1], u, u, i+1)

        #print(f"Dknn: {DKnn}")
        #print(f"DKnn1: {DKnn1}")
        #print(f"DKn1n1: {DKn1n1}")

        DK = np.block([
            [DKnn, DKnn1],
            [DKnn1.T, DKn1n1]
        ])

        D_NLML[i] = 0.5 * np.sum(Q * DK)

    # Add noise term
    n_b = x_b.shape[0]
    D_NLML[-1] = 0.5 * sigma_n * np.trace(Q[n_b:, n_b:])

    return NLML, D_NLML





