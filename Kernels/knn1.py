import numpy as np
from cleanNewProject.Kernels.DkDy import DkDy
from cleanNewProject.Kernels.D2kDy2 import D2kDy2
from cleanNewProject.config import Modelvariables
from cleanNewProject.Kernels.k import k



def knn1(x, y, hyp, uny, i):
    """
    Calculate the kernel knn1 function.

    Parameters:
    x : array_like
        Input data points.
    y : array_like
        Input data points.
    hyp : array_like
        Hyperparameters [logsigma, logtheta].
    uny : array_like
        Auxiliary parameter.
    i : int
        Index to select the formula to be used.

    Returns:
    K : array_like
        The resulting kernel matrix.
    """

    nu = Modelvariables['nu']
    dt = Modelvariables['dt']

    n_x = x.shape[0]
    uny = np.outer(np.ones(n_x), uny)

    K = k(x, y, hyp, i) + dt * uny * DkDy(x, y, hyp, i) - nu * dt * D2kDy2(x, y, hyp, i)

    return K
