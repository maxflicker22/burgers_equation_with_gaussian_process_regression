import numpy as np
from cleanNewProject.config import Modelvariables
from cleanNewProject.Kernels.k import k
from cleanNewProject.Kernels.DkDy import DkDy
from cleanNewProject.Kernels.DkDx import DkDx
from cleanNewProject.Kernels.DkDxDy import DkDxDy
from cleanNewProject.Kernels.D2kDy2 import D2kDy2
from cleanNewProject.Kernels.D2kDx2 import D2kDx2
from cleanNewProject.Kernels.D3kDx2Dy import D3kDx2Dy
from cleanNewProject.Kernels.D3kDxDy2 import D3kDxDy2
from cleanNewProject.Kernels.D4kDx2Dy2 import D4kDx2Dy2



def kn1n1(x, y, hyp, unx, uny, i):
    """
    Calculate the kernel kn1n1 function.

    Parameters:
    x : array_like
        Input data points.
    y : array_like
        Input data points.
    hyp : array_like
        Hyperparameters [logsigma, logtheta].
    unx : array_like
        Input data points.
    uny : array_like
        Input data points.
    i : int
        Index to select the formula to be used.

    Returns:
    K : array_like
        The resulting kernel matrix.
    """

    nu = Modelvariables['nu']
    dt = Modelvariables['dt']

    n_x = x.shape[0]
    n_y = y.shape[0]

    uny = np.outer(np.ones(n_x), uny)
    unx = np.outer(unx, np.ones(n_y))


    K = k(x, y, hyp, i) + dt * uny * DkDy(x, y, hyp, i) - nu * dt * D2kDy2(x, y, hyp, i) \
        + dt * unx * DkDx(x, y, hyp, i) + dt**2 * unx * uny * DkDxDy(x, y, hyp, i) \
        - nu * dt**2 * unx * D3kDxDy2(x, y, hyp, i) - nu * dt * D2kDx2(x, y, hyp, i) \
        - nu * dt**2 * uny * D3kDx2Dy(x, y, hyp, i) + nu**2 * dt**2 * D4kDx2Dy2(x, y, hyp, i)

    return K
