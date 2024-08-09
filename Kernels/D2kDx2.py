import numpy as np
from cleanNewProject.Kernels.D2kDy2 import D2kDy2


def D2kDx2(x, y, hyp, i):
    """
    Calculate the second derivative of the kernel function with respect to x.

    Parameters:
    x : array_like
        Input data points.
    y : array_like
        Input data points.
    hyp : array_like
        Hyperparameters [logsigma, logtheta].
    i : int
        Index to select the formula to be used.

    Returns:
    K : array_like
        The resulting kernel matrix.
    """

    logsigma = hyp[0]
    logtheta = hyp[1]

    n_x = x.shape[0]
    n_y = y.shape[0]

    x = np.outer(x, np.ones(n_y))
    y = np.outer(np.ones(n_x), y)

    ehoch1 = np.e
    pi = np.pi

    if i == 0:

        e_logtheta = np.exp(logtheta)
        e_logsigma = np.exp(logsigma)

        a = 2 * e_logsigma * e_logtheta

        b1 = np.sqrt(e_logsigma + (y ** 2 * e_logtheta) + 1)
        b2 = np.power((e_logtheta * x ** 2) + e_logsigma + 1, 3 / 2)
        b31n = (y * e_logtheta * x + e_logsigma) ** 2
        b31z = (e_logsigma + (y ** 2 * e_logtheta) + 1)
        b32z = (e_logtheta * x ** 2) + e_logsigma + 1
        b3 = np.pi * (np.sqrt(1 - (b31n / (b31z * b32z))))
        b = b1 * b2 * b3

        c = 6 * e_logtheta * e_logtheta * x * (e_logsigma * x - (y * e_logsigma) - y)

        d1 = b1
        d2 = np.power((e_logtheta * x ** 2) + e_logsigma + 1, 5 / 2)
        d3 = b3
        d = d1 * d2 * d3

        e1 = e_logtheta * (e_logsigma * x - (y * e_logsigma) - y)
        e21n = ((y * e_logtheta * x + e_logsigma) ** 2) * 2 * e_logtheta * x
        e21z = (e_logsigma + (y ** 2 * e_logtheta) + 1) * ((e_logtheta * x ** 2) + e_logsigma + 1) ** 2
        e22n = (y * e_logtheta * x + e_logsigma) * 2 * e_logtheta * y
        e22z = (e_logsigma + (y ** 2 * e_logtheta) + 1) * (((e_logtheta * x ** 2) + e_logsigma + 1))
        e2 = (e21n / e21z) - (e22n / e22z)
        e = e1 * (e2)

        f1 = b1
        f2 = b2
        f3 = np.pi * np.power(1 - ((y * e_logtheta * x + e_logsigma) ** 2 /
                                   ((e_logsigma + (y ** 2 * e_logtheta) + 1) *
                                    ((e_logtheta * x ** 2) + e_logsigma + 1))), 3 / 2)
        f = f1 * f2 * f3

        K = -(a / b) + (c / d) + (e / f)


    elif i == 1:

        # Compute the derivative with respect to logsigma

        K = 2 * ehoch1 ** logtheta * (
                    -6 * ehoch1 ** (3 * logsigma) - 6 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + x ** 4 * (
                        2 * ehoch1 ** (logsigma + 2 * logtheta) + 4 * ehoch1 ** (
                            2 * logsigma + 2 * logtheta)) + x ** 3 * (
                                -6 * ehoch1 ** (logsigma + 2 * logtheta) * y - 12 * ehoch1 ** (
                                    2 * logsigma + 2 * logtheta) * y) + x ** 2 * (2 * ehoch1 ** (2 * logsigma) * (
                        6 * ehoch1 ** (2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * (
                                                                                              6 * ehoch1 ** (
                                                                                                  2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                                2 * ehoch1 ** (2 * logsigma) * (-2 * ehoch1 ** (
                                    2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * (
                                            -4 * ehoch1 ** (2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y))) / (
                    pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / (
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                            ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * (
                                ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (7 / 2) * (
                                ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (
                                3 / 2)) - 3 * ehoch1 ** logtheta * (
                    ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / (
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                            ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2) + ehoch1 ** logsigma * (
                                ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / (
                                (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2 * (
                                    ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) - 2 * ehoch1 ** logsigma * (
                                ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / (
                                (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                                    ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) * (
                    -2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + x ** 4 * (
                        2 * ehoch1 ** (logsigma + 2 * logtheta) + 2 * ehoch1 ** (
                            2 * logsigma + 2 * logtheta)) + x ** 3 * (
                                -3 * ehoch1 ** (2 * logtheta) * y - 6 * ehoch1 ** (
                                    logsigma + 2 * logtheta) * y - 6 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta) * y) + x ** 2 * (ehoch1 ** (2 * logsigma) * (
                        6 * ehoch1 ** (2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * (
                                                                                                      6 * ehoch1 ** (
                                                                                                          2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                                ehoch1 ** (2 * logsigma) * (-2 * ehoch1 ** (
                                    2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * (
                                            -4 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - 2 * ehoch1 ** (
                                            2 * logtheta) * y ** 3 - 3 * ehoch1 ** logtheta * y)) / (pi * (
                    -(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / (
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                            ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) * (
                                                                                                                 ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (
                                                                                                                 7 / 2) * (
                                                                                                                 ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (
                                                                                                                 3 / 2)) - 3 * ehoch1 ** (
                    logsigma + logtheta) * (
                    -2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + x ** 4 * (
                        2 * ehoch1 ** (logsigma + 2 * logtheta) + 2 * ehoch1 ** (
                            2 * logsigma + 2 * logtheta)) + x ** 3 * (
                                -3 * ehoch1 ** (2 * logtheta) * y - 6 * ehoch1 ** (
                                    logsigma + 2 * logtheta) * y - 6 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta) * y) + x ** 2 * (ehoch1 ** (2 * logsigma) * (
                        6 * ehoch1 ** (2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * (
                                                                                                      6 * ehoch1 ** (
                                                                                                          2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                                ehoch1 ** (2 * logsigma) * (-2 * ehoch1 ** (
                                    2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * (
                                            -4 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - 2 * ehoch1 ** (
                                            2 * logtheta) * y ** 3 - 3 * ehoch1 ** logtheta * y)) / (pi * (
                    -(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / (
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                            ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * (
                                                                                                                 ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (
                                                                                                                 7 / 2) * (
                                                                                                                 ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (
                                                                                                                 5 / 2)) - 7 * ehoch1 ** (
                    logsigma + logtheta) * (
                    -2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + x ** 4 * (
                        2 * ehoch1 ** (logsigma + 2 * logtheta) + 2 * ehoch1 ** (
                            2 * logsigma + 2 * logtheta)) + x ** 3 * (
                                -3 * ehoch1 ** (2 * logtheta) * y - 6 * ehoch1 ** (
                                    logsigma + 2 * logtheta) * y - 6 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta) * y) + x ** 2 * (ehoch1 ** (2 * logsigma) * (
                        6 * ehoch1 ** (2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * (
                                                                                                      6 * ehoch1 ** (
                                                                                                          2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                                ehoch1 ** (2 * logsigma) * (-2 * ehoch1 ** (
                                    2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * (
                                            -4 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - 2 * ehoch1 ** (
                                            2 * logtheta) * y ** 3 - 3 * ehoch1 ** logtheta * y)) / (pi * (
                    -(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / (
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                            ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * (
                                                                                                                 ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (
                                                                                                                 9 / 2) * (
                                                                                                                 ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (
                                                                                                                 3 / 2))




    elif i == 2:

        # Compute the derivative with respect to logtheta

        K = -7 * ehoch1 ** (2 * logtheta) * x ** 2 * (-2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) \
                                                      - ehoch1 ** logsigma + x ** 4 * (
                                                                  2 * ehoch1 ** (logsigma + 2 * logtheta) + \
                                                                  2 * ehoch1 ** (
                                                                              2 * logsigma + 2 * logtheta)) + x ** 3 * (
                                                                  -3 * ehoch1 ** (2 * logtheta) * y - \
                                                                  6 * ehoch1 ** (
                                                                              logsigma + 2 * logtheta) * y - 6 * ehoch1 ** (
                                                                              2 * logsigma + 2 * logtheta) * y) \
                                                      + x ** 2 * (ehoch1 ** (2 * logsigma) * (
                            6 * ehoch1 ** (2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) \
                                                                  + ehoch1 ** logsigma * (6 * ehoch1 ** (
                                    2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + \
                                                      x * (ehoch1 ** (2 * logsigma) * (
                            -2 * ehoch1 ** (2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + \
                                                           ehoch1 ** logsigma * (-4 * ehoch1 ** (
                                    2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - \
                                                           2 * ehoch1 ** (
                                                                       2 * logtheta) * y ** 3 - 3 * ehoch1 ** logtheta * y)) / \
            (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / ((ehoch1 ** logsigma + \
                                                                               ehoch1 ** logtheta * x ** 2 + 1) * (
                                                                                          ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) \
                   + 1) ** (3 / 2) * (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) - 3 * ehoch1 ** (2 * logtheta) \
            * y ** 2 * (-2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + \
                        x ** 4 * (2 * ehoch1 ** (logsigma + 2 * logtheta) + 2 * ehoch1 ** (
                            2 * logsigma + 2 * logtheta)) + \
                        x ** 3 * (-3 * ehoch1 ** (2 * logtheta) * y - 6 * ehoch1 ** (logsigma + 2 * logtheta) * y - \
                                  6 * ehoch1 ** (2 * logsigma + 2 * logtheta) * y) + x ** 2 * (
                                    ehoch1 ** (2 * logsigma) * \
                                    (6 * ehoch1 ** (
                                                2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * \
                                    (6 * ehoch1 ** (2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                                    ehoch1 ** (2 * logsigma) * \
                                    (-2 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * \
                                    (-4 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - 2 * ehoch1 ** (
                                                2 * logtheta) * \
                                    y ** 3 - 3 * ehoch1 ** logtheta * y)) / (
                        pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                              ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (ehoch1 ** logsigma + \
                                                                                         ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (
                                    3 / 2) * (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 \
                                              + 1) ** (7 / 2) * (
                                    ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (5 / 2)) + \
            2 * ehoch1 ** logtheta * (x ** 4 * (4 * ehoch1 ** (logsigma + 2 * logtheta) + \
                                                4 * ehoch1 ** (2 * logsigma + 2 * logtheta)) + x ** 3 * (
                                                  -6 * ehoch1 ** (2 * logtheta) * y - \
                                                  12 * ehoch1 ** (logsigma + 2 * logtheta) * y - 12 * ehoch1 ** (
                                                              2 * logsigma + 2 * logtheta) * y) + \
                                      x ** 2 * (ehoch1 ** (2 * logsigma) * (
                            12 * ehoch1 ** (2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + \
                                                ehoch1 ** logsigma * (12 * ehoch1 ** (
                                    2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + \
                                      x * (ehoch1 ** (2 * logsigma) * (
                            -4 * ehoch1 ** (2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + \
                                           ehoch1 ** logsigma * (-8 * ehoch1 ** (
                                    2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - \
                                           4 * ehoch1 ** (2 * logtheta) * y ** 3 - 3 * ehoch1 ** logtheta * y)) / (
                        pi * (-(ehoch1 ** logsigma + \
                                ehoch1 ** logtheta * x * y) ** 2 / (
                                          (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                                          (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * (
                                    ehoch1 ** logsigma + \
                                    ehoch1 ** logtheta * x ** 2 + 1) ** (7 / 2) * (
                                    ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) + \
            2 * ehoch1 ** logtheta * (
                        -2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + \
                        x ** 4 * (2 * ehoch1 ** (logsigma + 2 * logtheta) + 2 * ehoch1 ** (
                            2 * logsigma + 2 * logtheta)) + \
                        x ** 3 * (-3 * ehoch1 ** (2 * logtheta) * y - 6 * ehoch1 ** (logsigma + 2 * logtheta) * y - \
                                  6 * ehoch1 ** (2 * logsigma + 2 * logtheta) * y) + x ** 2 * (
                                    ehoch1 ** (2 * logsigma) * \
                                    (6 * ehoch1 ** (
                                                2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * \
                                    (6 * ehoch1 ** (2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                                    ehoch1 ** (2 * logsigma) * \
                                    (-2 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * \
                                    (-4 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - 2 * ehoch1 ** (
                                                2 * logtheta) * y ** 3 - \
                                    3 * ehoch1 ** logtheta * y)) / (
                        pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                              ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (ehoch1 ** logsigma + \
                                                                                         ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (
                                    3 / 2) * (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 \
                                              + 1) ** (7 / 2) * (
                                    ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) - \
            3 * ehoch1 ** logtheta * (
                        ehoch1 ** logtheta * x ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                        ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2 * (ehoch1 ** logsigma + \
                                                                                        ehoch1 ** logtheta * y ** 2 + 1)) - 2 * ehoch1 ** logtheta * x * y * (
                                    ehoch1 ** logsigma + \
                                    ehoch1 ** logtheta * x * y) / (
                                    (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                                    (
                                                ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + ehoch1 ** logtheta * y ** 2 * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / ((ehoch1 ** logsigma + \
                                                                                   ehoch1 ** logtheta * x ** 2 + 1) * (
                                                                                              ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2)) * \
            (-2 * ehoch1 ** (3 * logsigma) - 3 * ehoch1 ** (2 * logsigma) - ehoch1 ** logsigma + x ** 4 * \
             (2 * ehoch1 ** (logsigma + 2 * logtheta) + 2 * ehoch1 ** (2 * logsigma + 2 * logtheta)) + \
             x ** 3 * (-3 * ehoch1 ** (2 * logtheta) * y - 6 * ehoch1 ** (logsigma + 2 * logtheta) * y - \
                       6 * ehoch1 ** (2 * logsigma + 2 * logtheta) * y) + x ** 2 * (ehoch1 ** (2 * logsigma) * \
                                                                                    (6 * ehoch1 ** (
                                                                                                2 * logtheta) * y ** 2 + 2 * ehoch1 ** logtheta) + ehoch1 ** logsigma * \
                                                                                    (6 * ehoch1 ** (
                                                                                                2 * logtheta) * y ** 2 + ehoch1 ** logtheta)) + x * (
                         ehoch1 ** (2 * logsigma) * \
                         (-2 * ehoch1 ** (2 * logtheta) * y ** 3 - 6 * ehoch1 ** logtheta * y) + ehoch1 ** logsigma * \
                         (-4 * ehoch1 ** (2 * logtheta) * y ** 3 - 9 * ehoch1 ** logtheta * y) - 2 * ehoch1 ** (
                                     2 * logtheta) * \
                         y ** 3 - 3 * ehoch1 ** logtheta * y)) / (
                        pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                              ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * (
                                          ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) \
                              + 1) ** (5 / 2) * (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (7 / 2) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2))

    return K
