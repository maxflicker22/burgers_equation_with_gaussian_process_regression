import numpy as np

def D4kDx2Dy2(x, y, hyp, i):
    """
    Calculate the fourth mixed derivative of the kernel function with respect to x and y.

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
                6 * ehoch1 ** (2 * logtheta + logsigma) * (2 * ehoch1 ** logsigma + 1) /
                (
                        np.pi * (
                        1 - (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2 /
                        ((ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                         (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1))
                ) ** (5 / 2) *
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2) *
                        (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2)
                )
        )

        term2 = (
                -15 * ehoch1 ** (2 * logtheta) * (2 * ehoch1 ** logsigma + 1) *
                (
                        (2 * ehoch1 ** logtheta * x * (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2) /
                        (
                                (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** 2 *
                                (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                        ) -
                        2 * ehoch1 ** logtheta * y * (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) /
                        (
                                (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                                (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                        )
                ) * (ehoch1 ** logsigma * x - ehoch1 ** logsigma * y - y) /
                (
                        np.pi * (
                        1 - (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2 /
                        ((ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                         (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1))
                ) ** (7 / 2) *
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2) *
                        (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2)
                )
        )

        term3 = (
                -30 * ehoch1 ** (3 * logtheta) * x * (2 * ehoch1 ** logsigma + 1) *
                (ehoch1 ** logsigma * x - ehoch1 ** logsigma * y - y) /
                (
                        np.pi * (
                        1 - (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2 /
                        ((ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                         (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1))
                ) ** (5 / 2) *
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (7 / 2) *
                        (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (5 / 2)
                )
        )

        K = term1 + term2 + term3


    elif i == 1:

        term1 = (
                        -6 * ehoch1 ** (2 * logtheta) *
                        (4 * ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma * (2 * ehoch1 ** logtheta * x ** 2 + 3)) *
                        (ehoch1 ** (3 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 - 2) +
                         ehoch1 ** (2 * logsigma) * (8 * ehoch1 ** logtheta * x ** 2 - 3) +
                         ehoch1 ** logsigma * (4 * ehoch1 ** logtheta * x ** 2 - 1) +
                         y ** 4 * (4 * ehoch1 ** (logsigma + 2 * logtheta) + 4 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta)) +
                         y ** 3 * (-5 * ehoch1 ** (2 * logtheta) * x - 10 * ehoch1 ** (logsigma + 2 * logtheta) * x -
                                   8 * ehoch1 ** (2 * logsigma + 2 * logtheta) * x) +
                         y ** 2 * (ehoch1 ** (2 * logsigma) * (
                                            4 * ehoch1 ** (2 * logtheta) * x ** 2 + 6 * ehoch1 ** logtheta) +
                                   ehoch1 ** logsigma * (
                                               4 * ehoch1 ** (2 * logtheta) * x ** 2 + 3 * ehoch1 ** logtheta) +
                                   4 * ehoch1 ** (3 * logsigma + logtheta)) +
                         y * (-5 * ehoch1 ** logtheta * x - 15 * ehoch1 ** (logsigma + logtheta) * x -
                              18 * ehoch1 ** (2 * logsigma + logtheta) * x - 8 * ehoch1 ** (
                                          3 * logsigma + logtheta) * x))
                ) / (
                        pi *
                        (-((ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                           ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                            (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) + 1) ** (7 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (9 / 2)
                )

        term2 = (
                        -6 * ehoch1 ** (2 * logtheta) *
                        (2 * ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma * (2 * ehoch1 ** logtheta * x ** 2 + 3) +
                         ehoch1 ** logtheta * x ** 2 + 1) *
                        (3 * ehoch1 ** (3 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 - 2) +
                         2 * ehoch1 ** (2 * logsigma) * (8 * ehoch1 ** logtheta * x ** 2 - 3) +
                         ehoch1 ** logsigma * (4 * ehoch1 ** logtheta * x ** 2 - 1) +
                         y ** 4 * (4 * ehoch1 ** (logsigma + 2 * logtheta) + 8 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta)) +
                         y ** 3 * (-10 * ehoch1 ** (logsigma + 2 * logtheta) * x - 16 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta) * x) +
                         y ** 2 * (2 * ehoch1 ** (2 * logsigma) * (
                                            4 * ehoch1 ** (2 * logtheta) * x ** 2 + 6 * ehoch1 ** logtheta) +
                                   ehoch1 ** logsigma * (
                                               4 * ehoch1 ** (2 * logtheta) * x ** 2 + 3 * ehoch1 ** logtheta) +
                                   12 * ehoch1 ** (3 * logsigma + logtheta)) +
                         y * (-15 * ehoch1 ** (logsigma + logtheta) * x - 36 * ehoch1 ** (2 * logsigma + logtheta) * x -
                              24 * ehoch1 ** (3 * logsigma + logtheta) * x))
                ) / (
                        pi *
                        (-((ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                           ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                            (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) + 1) ** (7 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (9 / 2)
                )

        term3 = (
                        21 * ehoch1 ** (2 * logtheta) *
                        (ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                         ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                          (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2) +
                         ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                         ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2 *
                          (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) -
                         2 * ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) /
                         ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                          (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) *
                        (2 * ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma * (2 * ehoch1 ** logtheta * x ** 2 + 3) +
                         ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** (3 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 - 2) +
                         ehoch1 ** (2 * logsigma) * (8 * ehoch1 ** logtheta * x ** 2 - 3) +
                         ehoch1 ** logsigma * (4 * ehoch1 ** logtheta * x ** 2 - 1) +
                         y ** 4 * (4 * ehoch1 ** (logsigma + 2 * logtheta) + 4 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta)) +
                         y ** 3 * (-5 * ehoch1 ** (2 * logtheta) * x - 10 * ehoch1 ** (logsigma + 2 * logtheta) * x -
                                   8 * ehoch1 ** (2 * logsigma + 2 * logtheta) * x) +
                         y ** 2 * (ehoch1 ** (2 * logsigma) * (
                                            4 * ehoch1 ** (2 * logtheta) * x ** 2 + 6 * ehoch1 ** logtheta) +
                                   ehoch1 ** logsigma * (
                                               4 * ehoch1 ** (2 * logtheta) * x ** 2 + 3 * ehoch1 ** logtheta) +
                                   4 * ehoch1 ** (3 * logsigma + logtheta)) +
                         y * (-5 * ehoch1 ** logtheta * x - 15 * ehoch1 ** (logsigma + logtheta) * x -
                              18 * ehoch1 ** (2 * logsigma + logtheta) * x - 8 * ehoch1 ** (
                                          3 * logsigma + logtheta) * x))
                ) / (
                        pi *
                        (-((ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                           ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                            (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) + 1) ** (9 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (9 / 2)
                )

        term4 = (
                        27 * ehoch1 ** (logsigma + 2 * logtheta) *
                        (2 * ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma * (2 * ehoch1 ** logtheta * x ** 2 + 3) +
                         ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** (3 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 - 2) +
                         ehoch1 ** (2 * logsigma) * (8 * ehoch1 ** logtheta * x ** 2 - 3) +
                         ehoch1 ** logsigma * (4 * ehoch1 ** logtheta * x ** 2 - 1) +
                         y ** 4 * (4 * ehoch1 ** (logsigma + 2 * logtheta) + 4 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta)) +
                         y ** 3 * (-5 * ehoch1 ** (2 * logtheta) * x - 10 * ehoch1 ** (logsigma + 2 * logtheta) * x -
                                   8 * ehoch1 ** (2 * logsigma + 2 * logtheta) * x) +
                         y ** 2 * (ehoch1 ** (2 * logsigma) * (
                                            4 * ehoch1 ** (2 * logtheta) * x ** 2 + 6 * ehoch1 ** logtheta) +
                                   ehoch1 ** logsigma * (
                                               4 * ehoch1 ** (2 * logtheta) * x ** 2 + 3 * ehoch1 ** logtheta) +
                                   4 * ehoch1 ** (3 * logsigma + logtheta)) +
                         y * (-5 * ehoch1 ** logtheta * x - 15 * ehoch1 ** (logsigma + logtheta) * x -
                              18 * ehoch1 ** (2 * logsigma + logtheta) * x - 8 * ehoch1 ** (
                                          3 * logsigma + logtheta) * x))
                ) / (
                        pi *
                        (-((ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                           ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                            (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) + 1) ** (7 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (9 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (11 / 2)
                )

        term5 = (
                        27 * ehoch1 ** (logsigma + 2 * logtheta) *
                        (2 * ehoch1 ** (2 * logsigma) + ehoch1 ** logsigma * (2 * ehoch1 ** logtheta * x ** 2 + 3) +
                         ehoch1 ** logtheta * x ** 2 + 1) *
                        (ehoch1 ** (3 * logsigma) * (4 * ehoch1 ** logtheta * x ** 2 - 2) +
                         ehoch1 ** (2 * logsigma) * (8 * ehoch1 ** logtheta * x ** 2 - 3) +
                         ehoch1 ** logsigma * (4 * ehoch1 ** logtheta * x ** 2 - 1) +
                         y ** 4 * (4 * ehoch1 ** (logsigma + 2 * logtheta) + 4 * ehoch1 ** (
                                            2 * logsigma + 2 * logtheta)) +
                         y ** 3 * (-5 * ehoch1 ** (2 * logtheta) * x - 10 * ehoch1 ** (logsigma + 2 * logtheta) * x -
                                   8 * ehoch1 ** (2 * logsigma + 2 * logtheta) * x) +
                         y ** 2 * (ehoch1 ** (2 * logsigma) * (
                                            4 * ehoch1 ** (2 * logtheta) * x ** 2 + 6 * ehoch1 ** logtheta) +
                                   ehoch1 ** logsigma * (
                                               4 * ehoch1 ** (2 * logtheta) * x ** 2 + 3 * ehoch1 ** logtheta) +
                                   4 * ehoch1 ** (3 * logsigma + logtheta)) +
                         y * (-5 * ehoch1 ** logtheta * x - 15 * ehoch1 ** (logsigma + logtheta) * x -
                              18 * ehoch1 ** (2 * logsigma + logtheta) * x - 8 * ehoch1 ** (
                                          3 * logsigma + logtheta) * x))
                ) / (
                        pi *
                        (-((ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                           ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                            (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) + 1) ** (7 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (11 / 2) *
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (9 / 2)
                )

        K = term1 + term2 + term3 + term4 + term5


    elif i == 2:

        numerator = (
                3 * ehoch1 ** (2 * logtheta) * (2 * ehoch1 ** logsigma + 1) * (
                16 * ehoch1 ** (5 * logsigma) + 48 * ehoch1 ** (4 * logsigma) +
                52 * ehoch1 ** (3 * logsigma) + 24 * ehoch1 ** (2 * logsigma) +
                4 * ehoch1 ** logsigma +
                ehoch1 ** (4 * logtheta) * (
                        ehoch1 ** (3 * logsigma) * (
                        4 * x ** 6 * y ** 2 - 16 * x ** 5 * y ** 3 + 24 * x ** 4 * y ** 4 -
                        16 * x ** 3 * y ** 5 + 4 * x ** 2 * y ** 6
                ) +
                        ehoch1 ** (2 * logsigma) * (
                                8 * x ** 6 * y ** 2 - 26 * x ** 5 * y ** 3 + 36 * x ** 4 * y ** 4 -
                                26 * x ** 3 * y ** 5 + 8 * x ** 2 * y ** 6
                        ) +
                        ehoch1 ** logsigma * (
                                4 * x ** 6 * y ** 2 - 15 * x ** 5 * y ** 3 + 18 * x ** 4 * y ** 4 -
                                15 * x ** 3 * y ** 5 + 4 * x ** 2 * y ** 6
                        ) -
                        5 * x ** 5 * y ** 3 - 5 * x ** 3 * y ** 5
                ) +
                ehoch1 ** (3 * logtheta) * (
                        ehoch1 ** (4 * logsigma) * (
                        4 * x ** 6 - 16 * x ** 5 * y + 28 * x ** 4 * y ** 2 - 32 * x ** 3 * y ** 3 +
                        28 * x ** 2 * y ** 4 - 16 * x * y ** 5 + 4 * y ** 6
                ) +
                        ehoch1 ** (3 * logsigma) * (
                                12 * x ** 6 - 42 * x ** 5 * y + 18 * x ** 4 * y ** 2 + 24 * x ** 3 * y ** 3 +
                                18 * x ** 2 * y ** 4 - 42 * x * y ** 5 + 12 * y ** 6
                        ) +
                        ehoch1 ** (2 * logsigma) * (
                                12 * x ** 6 - 41 * x ** 5 * y - 15 * x ** 4 * y ** 2 + 92 * x ** 3 * y ** 3 -
                                15 * x ** 2 * y ** 4 - 41 * x * y ** 5 + 12 * y ** 6
                        ) +
                        ehoch1 ** logsigma * (
                                4 * x ** 6 - 20 * x ** 5 * y - 5 * x ** 4 * y ** 2 + 80 * x ** 3 * y ** 3 -
                                5 * x ** 2 * y ** 4 - 20 * x * y ** 5 + 4 * y ** 6
                        ) -
                        5 * x ** 5 * y + 20 * x ** 3 * y ** 3 - 5 * x * y ** 5
                ) +
                ehoch1 ** (2 * logtheta) * (
                        ehoch1 ** (5 * logsigma) * (
                        4 * x ** 4 - 16 * x ** 3 * y + 24 * x ** 2 * y ** 2 - 16 * x * y ** 3 + 4 * y ** 4
                ) +
                        ehoch1 ** (4 * logsigma) * (
                                -38 * x ** 4 + 50 * x ** 3 * y - 24 * x ** 2 * y ** 2 + 50 * x * y ** 3 - 38 * y ** 4
                        ) +
                        ehoch1 ** (3 * logsigma) * (
                                -111 * x ** 4 + 199 * x ** 3 * y - 140 * x ** 2 * y ** 2 + 199 * x * y ** 3 - 111 * y ** 4
                        ) +
                        ehoch1 ** (2 * logsigma) * (
                                -92 * x ** 4 + 233 * x ** 3 * y - 128 * x ** 2 * y ** 2 + 233 * x * y ** 3 - 92 * y ** 4
                        ) +
                        ehoch1 ** logsigma * (
                                -23 * x ** 4 + 125 * x ** 3 * y - 32 * x ** 2 * y ** 2 + 125 * x * y ** 3 - 23 * y ** 4
                        ) +
                        25 * x ** 3 * y + 25 * x * y ** 3
                ) +
                ehoch1 ** logtheta * (
                        ehoch1 ** (5 * logsigma) * (
                        -54 * x ** 2 + 108 * x * y - 54 * y ** 2
                ) +
                        ehoch1 ** (4 * logsigma) * (
                                -173 * x ** 2 + 390 * x * y - 173 * y ** 2
                        ) +
                        ehoch1 ** (3 * logsigma) * (
                                -211 * x ** 2 + 576 * x * y - 211 * y ** 2
                        ) +
                        ehoch1 ** (2 * logsigma) * (
                                -115 * x ** 2 + 444 * x * y - 115 * y ** 2
                        ) +
                        ehoch1 ** logsigma * (
                                -23 * x ** 2 + 180 * x * y - 23 * y ** 2
                        ) +
                        30 * x * y
                )
        )
        )

        denominator = (
                pi *
                (-(
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 /
                        ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) *
                         (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))
                ) + 1) ** (9 / 2) *
                (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (11 / 2) *
                (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (11 / 2)
        )

        K = numerator / denominator
        

    return K
