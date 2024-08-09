import numpy as np

def DkDxDy(x, y, hyp, i):
    """
    Calculate the mixed derivative of the kernel function with respect to x and y.

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
                -2 * ehoch1 ** logtheta * (-ehoch1 ** logsigma - 1) /
                (
                        np.pi * np.sqrt(
                    1 - (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2 /
                    (
                            (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                            (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                    )
                ) *
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                        np.sqrt(ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                )
        )

        term2 = (
                ehoch1 ** logtheta * (
                (2 * ehoch1 ** logtheta * y * (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2) /
                (
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                        (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** 2
                ) -
                2 * ehoch1 ** logtheta * x * (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) /
                (
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                        (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                )
        ) * (ehoch1 ** logsigma * x - ehoch1 ** logsigma * y - y) /
                (
                        np.pi * (
                        1 - (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2 /
                        (
                                (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                                (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                        )
                ) ** (3 / 2) *
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                        np.sqrt(ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                )
        )

        term3 = (
                2 * ehoch1 ** (2 * logtheta) * y * (ehoch1 ** logsigma * x - ehoch1 ** logsigma * y - y) /
                (
                        np.pi * np.sqrt(
                    1 - (ehoch1 ** logtheta * x * y + ehoch1 ** logsigma) ** 2 /
                    (
                            (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) *
                            (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1)
                    )
                ) *
                        (ehoch1 ** logtheta * x ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2) *
                        (ehoch1 ** logtheta * y ** 2 + ehoch1 ** logsigma + 1) ** (3 / 2)
                )
        )

        K =  term1 + term2 + term3


    elif i == 1:

        K = -3 * ehoch1 ** logsigma * (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) / \
            (2 * pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (5 / 2)) - \
            3 * ehoch1 ** logsigma * (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) / \
            (2 * pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (5 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) + \
            4 * ehoch1 ** (logsigma + logtheta) / (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                                                         ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                                                          (
                                                                      ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (
                                                               3 / 2) * \
                                                   (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
                                                   (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) - \
            3 * (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) * \
            (ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2) + \
             ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2 * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) - \
             2 * ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) / \
            (2 * pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2))



    elif i == 2:

        K = -3 * ehoch1 ** logtheta * x ** 2 * (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) / \
            (2 * pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (5 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) - \
            3 * ehoch1 ** logtheta * y ** 2 * (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) / \
            (2 * pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (5 / 2)) + \
            (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) / \
            (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                   ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                    (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) - \
            3 * (2 * ehoch1 ** logtheta + 4 * ehoch1 ** (logsigma + logtheta)) * \
            (ehoch1 ** logtheta * x ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2 * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) - \
             2 * ehoch1 ** logtheta * x * y * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + \
             ehoch1 ** logtheta * y ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2)) / \
            (2 * pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                       ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                        (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1) ** (5 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2))

    return K
