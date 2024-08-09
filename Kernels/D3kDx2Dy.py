import numpy as np

def D3kDx2Dy(x, y, hyp, i):
    """
    Calculate the third derivative of the kernel function with respect to x twice and y.

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

        
        term1 = (
                (2 * ehoch1 ** (2 * logtheta + logsigma) * y) /
                (np.pi * np.sqrt(1 - (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2 /
                                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) *
                 (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                 (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2))
        )

        term2 = (
                ehoch1 ** logtheta * (-ehoch1 ** logsigma - 1) *
                ((2 * ehoch1 ** logtheta * x * (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2) /
                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** 2) -
                 2 * ehoch1 ** logtheta * y * (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) /
                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) /
                (np.pi * (1 - (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2 /
                          ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                           (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) ** (3 / 2) *
                 np.sqrt(ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                 (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2))
        )

        term3 = (
                (6 * ehoch1 ** (2 * logtheta) * x * (-ehoch1 ** logsigma - 1)) /
                (np.pi * np.sqrt(1 - (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2 /
                                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) *
                 np.sqrt(ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                 (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2))
        )

        term4 = (
                -3 * ehoch1 ** (2 * logtheta) * x *
                ((2 * ehoch1 ** logtheta * y * (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2) /
                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** 2 *
                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1)) -
                 2 * ehoch1 ** logtheta * x * (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) /
                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** 2)) *
                (-ehoch1 ** logsigma * y + ehoch1 ** logsigma * x - y) /
                (2 * np.pi * (1 - (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2 /
                              ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                               (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) ** (5 / 2) *
                 (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                 (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2))
        )

        term5 = (
                -ehoch1 ** (2 * logtheta) * y *
                ((2 * ehoch1 ** logtheta * x * (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2) /
                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** 2) -
                 2 * ehoch1 ** logtheta * y * (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) /
                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) *
                (-ehoch1 ** logsigma * y + ehoch1 ** logsigma * x - y) /
                (np.pi * (1 - (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2 /
                          ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                           (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) ** (3 / 2) *
                 (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                 (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2))
        )

        term6 = (
                -6 * ehoch1 ** (3 * logtheta) * y * x *
                (-ehoch1 ** logsigma * y + ehoch1 ** logsigma * x - y) /
                (np.pi * np.sqrt(1 - (ehoch1 ** logtheta * y * x + ehoch1 ** logsigma) ** 2 /
                                 ((ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) *
                                  (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1))) *
                 (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                 (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2))
        )

        K = term1 + term2 + term3 + term4 + term5 + term6




    elif i == 1:

        # Compute the derivative with respect to logsigma
        e = np.e


        term1 = (6 * e ** (2 * logtheta) *
                 (6 * e ** (3 * logsigma) +
                  2 * e ** (2 * logsigma) *
                  (4 * e ** logtheta * x ** 2 + 5) +
                  e ** logsigma *
                  (2 * e ** (2 * logtheta) * x ** 4 +
                   6 * e ** logtheta * x ** 2 + 4)) *
                 (-e ** (2 * logsigma) * x -
                  2 * e ** logsigma * x +
                  e ** (logsigma + logtheta) * y ** 3 - x +
                  y ** 2 * (-e ** logtheta * x -
                            e ** (logsigma + logtheta) * x) +
                  y * (e ** (2 * logsigma) + e ** logsigma)) /
                 (pi * (-(e ** logsigma + e ** logtheta * x * y) ** 2 /
                        ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                         (e ** logsigma + e ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                  (e ** logsigma + e ** logtheta * x ** 2 + 1) ** (9 / 2) *
                  (e ** logsigma + e ** logtheta * y ** 2 + 1) ** (7 / 2)))

        term2 = (6 * e ** (2 * logtheta) *
                 (-2 * e ** (2 * logsigma) * x -
                  2 * e ** logsigma * x -
                  e ** (logsigma + logtheta) * x * y ** 2 +
                  e ** (logsigma + logtheta) * y ** 3 +
                  y * (2 * e ** (2 * logsigma) + e ** logsigma)) *
                 (2 * e ** (3 * logsigma) +
                  e ** (2 * logsigma) *
                  (4 * e ** logtheta * x ** 2 + 5) +
                  e ** logsigma *
                  (2 * e ** (2 * logtheta) * x ** 4 +
                   6 * e ** logtheta * x ** 2 + 4) +
                  e ** (2 * logtheta) * x ** 4 +
                  2 * e ** logtheta * x ** 2 + 1) /
                 (pi * (-(e ** logsigma + e ** logtheta * x * y) ** 2 /
                        ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                         (e ** logsigma + e ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                  (e ** logsigma + e ** logtheta * x ** 2 + 1) ** (9 / 2) *
                  (e ** logsigma + e ** logtheta * y ** 2 + 1) ** (7 / 2)))

        term3 = (-15 * e ** (2 * logtheta) *
                 (e ** logsigma *
                  (e ** logsigma + e ** logtheta * x * y) ** 2 /
                  ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                   (e ** logsigma + e ** logtheta * y ** 2 + 1) ** 2) +
                  e ** logsigma *
                  (e ** logsigma + e ** logtheta * x * y) ** 2 /
                  ((e ** logsigma + e ** logtheta * x ** 2 + 1) ** 2 *
                   (e ** logsigma + e ** logtheta * y ** 2 + 1)) -
                  2 * e ** logsigma *
                  (e ** logsigma + e ** logtheta * x * y) /
                  ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                   (e ** logsigma + e ** logtheta * y ** 2 + 1))) *
                 (2 * e ** (3 * logsigma) +
                  e ** (2 * logsigma) *
                  (4 * e ** logtheta * x ** 2 + 5) +
                  e ** logsigma *
                  (2 * e ** (2 * logtheta) * x ** 4 +
                   6 * e ** logtheta * x ** 2 + 4) +
                  e ** (2 * logtheta) * x ** 4 +
                  2 * e ** logtheta * x ** 2 + 1) *
                 (-e ** (2 * logsigma) * x -
                  2 * e ** logsigma * x +
                  e ** (logsigma + logtheta) * y ** 3 - x +
                  y ** 2 * (-e ** logtheta * x -
                            e ** (logsigma + logtheta) * x) +
                  y * (e ** (2 * logsigma) + e ** logsigma)) /
                 (pi * (-(e ** logsigma + e ** logtheta * x * y) ** 2 /
                        ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                         (e ** logsigma + e ** logtheta * y ** 2 + 1)) + 1) ** (7 / 2) *
                  (e ** logsigma + e ** logtheta * x ** 2 + 1) ** (9 / 2) *
                  (e ** logsigma + e ** logtheta * y ** 2 + 1) ** (7 / 2)))

        term4 = (-21 * e ** (logsigma + 2 * logtheta) *
                 (2 * e ** (3 * logsigma) +
                  e ** (2 * logsigma) *
                  (4 * e ** logtheta * x ** 2 + 5) +
                  e ** logsigma *
                  (2 * e ** (2 * logtheta) * x ** 4 +
                   6 * e ** logtheta * x ** 2 + 4) +
                  e ** (2 * logtheta) * x ** 4 +
                  2 * e ** logtheta * x ** 2 + 1) *
                 (-e ** (2 * logsigma) * x -
                  2 * e ** logsigma * x +
                  e ** (logsigma + logtheta) * y ** 3 - x +
                  y ** 2 * (-e ** logtheta * x -
                            e ** (logsigma + logtheta) * x) +
                  y * (e ** (2 * logsigma) + e ** logsigma)) /
                 (pi * (-(e ** logsigma + e ** logtheta * x * y) ** 2 /
                        ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                         (e ** logsigma + e ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                  (e ** logsigma + e ** logtheta * x ** 2 + 1) ** (9 / 2) *
                  (e ** logsigma + e ** logtheta * y ** 2 + 1) ** (9 / 2)))

        term5 = (-27 * e ** (logsigma + 2 * logtheta) *
                 (2 * e ** (3 * logsigma) +
                  e ** (2 * logsigma) *
                  (4 * e ** logtheta * x ** 2 + 5) +
                  e ** logsigma *
                  (2 * e ** (2 * logtheta) * x ** 4 +
                   6 * e ** logtheta * x ** 2 + 4) +
                  e ** (2 * logtheta) * x ** 4 +
                  2 * e ** logtheta * x ** 2 + 1) *
                 (-e ** (2 * logsigma) * x -
                  2 * e ** logsigma * x +
                  e ** (logsigma + logtheta) * y ** 3 - x +
                  y ** 2 * (-e ** logtheta * x -
                            e ** (logsigma + logtheta) * x) +
                  y * (e ** (2 * logsigma) + e ** logsigma)) /
                 (pi * (-(e ** logsigma + e ** logtheta * x * y) ** 2 /
                        ((e ** logsigma + e ** logtheta * x ** 2 + 1) *
                         (e ** logsigma + e ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                  (e ** logsigma + e ** logtheta * x ** 2 + 1) ** (11 / 2) *
                  (e ** logsigma + e ** logtheta * y ** 2 + 1) ** (7 / 2)))

        K = term1 + term2 + term3 + term4 + term5



    elif i == 2:

        # Compute the derivative with respect to logtheta

        term1 = (
                -27 * ehoch1 ** (3 * logtheta) * x ** 2 *
                (2 * ehoch1 ** (3 * logsigma) +
                 ehoch1 ** (2 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 + 5) +
                 ehoch1 ** logsigma * (2 * ehoch1 ** (2 * logtheta) * x ** 4 + 6 * ehoch1 ** logtheta * x ** 2 + 4) +
                 ehoch1 ** (2 * logtheta) * x ** 4 + 2 * ehoch1 ** logtheta * x ** 2 + 1) *
                (-ehoch1 ** (2 * logsigma) * x - 2 * ehoch1 ** logsigma * x +
                 ehoch1 ** (logsigma + logtheta) * y ** 3 - x +
                 y ** 2 * (-ehoch1 ** logtheta * x - ehoch1 ** (logsigma + logtheta) * x) +
                 y * (ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma)) /
                (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (11 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (7 / 2))
        )

        term2 = (
                -21 * ehoch1 ** (3 * logtheta) * y ** 2 *
                (2 * ehoch1 ** (3 * logsigma) +
                 ehoch1 ** (2 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 + 5) +
                 ehoch1 ** logsigma * (2 * ehoch1 ** (2 * logtheta) * x ** 4 + 6 * ehoch1 ** logtheta * x ** 2 + 4) +
                 ehoch1 ** (2 * logtheta) * x ** 4 + 2 * ehoch1 ** logtheta * x ** 2 + 1) *
                (-ehoch1 ** (2 * logsigma) * x - 2 * ehoch1 ** logsigma * x +
                 ehoch1 ** (logsigma + logtheta) * y ** 3 - x +
                 y ** 2 * (-ehoch1 ** logtheta * x - ehoch1 ** (logsigma + logtheta) * x) +
                 y * (ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma)) /
                (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (9 / 2))
        )

        term3 = (
                6 * ehoch1 ** (2 * logtheta) *
                (ehoch1 ** (logsigma + logtheta) * y ** 3 +
                 y ** 2 * (-ehoch1 ** logtheta * x - ehoch1 ** (logsigma + logtheta) * x)) *
                (2 * ehoch1 ** (3 * logsigma) +
                 ehoch1 ** (2 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 + 5) +
                 ehoch1 ** logsigma * (2 * ehoch1 ** (2 * logtheta) * x ** 4 + 6 * ehoch1 ** logtheta * x ** 2 + 4) +
                 ehoch1 ** (2 * logtheta) * x ** 4 + 2 * ehoch1 ** logtheta * x ** 2 + 1) /
                (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (7 / 2))
        )

        term4 = (
                6 * ehoch1 ** (2 * logtheta) *
                (ehoch1 ** logsigma * (4 * ehoch1 ** (2 * logtheta) * x ** 4 + 6 * ehoch1 ** logtheta * x ** 2) +
                 2 * ehoch1 ** (2 * logtheta) * x ** 4 + 2 * ehoch1 ** logtheta * x ** 2 +
                 4 * ehoch1 ** (2 * logsigma + logtheta) * x ** 2) *
                (-ehoch1 ** (2 * logsigma) * x - 2 * ehoch1 ** logsigma * x +
                 ehoch1 ** (logsigma + logtheta) * y ** 3 - x +
                 y ** 2 * (-ehoch1 ** logtheta * x - ehoch1 ** (logsigma + logtheta) * x) +
                 y * (ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma)) /
                (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (7 / 2))
        )

        term5 = (
                12 * ehoch1 ** (2 * logtheta) *
                (2 * ehoch1 ** (3 * logsigma) +
                 ehoch1 ** (2 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 + 5) +
                 ehoch1 ** logsigma * (2 * ehoch1 ** (2 * logtheta) * x ** 4 + 6 * ehoch1 ** logtheta * x ** 2 + 4) +
                 ehoch1 ** (2 * logtheta) * x ** 4 + 2 * ehoch1 ** logtheta * x ** 2 + 1) *
                (-ehoch1 ** (2 * logsigma) * x - 2 * ehoch1 ** logsigma * x +
                 ehoch1 ** (logsigma + logtheta) * y ** 3 - x +
                 y ** 2 * (-ehoch1 ** logtheta * x - ehoch1 ** (logsigma + logtheta) * x) +
                 y * (ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma)) /
                (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (7 / 2))
        )

        term6 = (
                -15 * ehoch1 ** (2 * logtheta) *
                (ehoch1 ** logtheta * x ** 2 *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                 ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2 *
                  (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) -
                 2 * ehoch1 ** logtheta * x * y *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) /
                 ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                  (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) +
                 ehoch1 ** logtheta * y ** 2 *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                 ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                  (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2)) *
                (2 * ehoch1 ** (3 * logsigma) +
                 ehoch1 ** (2 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 + 5) +
                 ehoch1 ** logsigma * (2 * ehoch1 ** (2 * logtheta) * x ** 4 + 6 * ehoch1 ** logtheta * x ** 2 + 4) +
                 ehoch1 ** (2 * logtheta) * x ** 4 + 2 * ehoch1 ** logtheta * x ** 2 + 1) *
                (-ehoch1 ** (2 * logsigma) * x - 2 * ehoch1 ** logsigma * x +
                 ehoch1 ** (logsigma + logtheta) * y ** 3 - x +
                 y ** 2 * (-ehoch1 ** logtheta * x - ehoch1 ** (logsigma + logtheta) * x) +
                 y * (ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma)) /
                (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (7 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                 (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (7 / 2))
        )

        K = term1 + term2 + term3 + term4 + term5 + term6


    return K