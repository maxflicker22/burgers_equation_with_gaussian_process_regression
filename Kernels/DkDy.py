import numpy as np

def DkDy(x, y, hyp, i):
    """
    Calculate the derivative of the kernel function with respect to y.

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

        a = x * e_logtheta

        b1 = (e_logsigma + (x ** 2 * e_logtheta) + 1)
        b2 = (e_logsigma + (e_logtheta * y ** 2) + 1)
        b = (b1 * b2) ** (0.5)

        c = e_logtheta * y * (x * y * e_logtheta + e_logsigma)

        d1 = (e_logsigma + (x ** 2 * e_logtheta) + 1) ** (0.5)
        d2 = ((e_logtheta * y ** 2) + e_logsigma + 1) ** (3 / 2)
        d = d1 * d2

        e = (x * e_logtheta * y + e_logsigma) ** 2

        f1 = e_logsigma + (x ** 2 * e_logtheta) + 1
        f2 = (e_logtheta * y ** 2) + e_logsigma + 1
        f = f1 * f2

        nenner = 2 * ((a / b) - (c / d))
        zähler = np.pi * (1 - (e / f)) ** (0.5)


        # Suppress warnings and handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.divide(nenner, zähler)
            # Replace inf and nan with appropriate values, for example np.nan
            K[~np.isfinite(K)] = np.inf  # or use K[~np.isfinite(K)] = 0 to replace with 0


    elif i == 1:

        K = -2 * ehoch1 ** logtheta * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2) * \
             np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + \
            ehoch1 ** logtheta * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) * \
            (ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2) + \
             ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2 * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) - \
             2 * ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1))) / \
            (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                   ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                    (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2) * \
             np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + \
            ehoch1 ** (logsigma + logtheta) * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2)) + \
            3 * ehoch1 ** (logsigma + logtheta) * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (5 / 2) * \
             np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1))


    elif i == 2:

        K = 3 * ehoch1 ** (2 * logtheta) * y ** 2 * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (5 / 2) * \
             np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + \
            ehoch1 ** (2 * logtheta) * x ** 2 * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2)) - \
            2 * ehoch1 ** logtheta * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2) * \
             np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + \
            ehoch1 ** logtheta * (ehoch1 ** logsigma * y - ehoch1 ** logsigma * x - x) * \
            (ehoch1 ** logtheta * y ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** 2 * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) - \
             2 * ehoch1 ** logtheta * y * x * (ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + \
             ehoch1 ** logtheta * x ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** 2)) / \
            (pi * (-(ehoch1 ** logsigma + ehoch1 ** logtheta * y * x) ** 2 / \
                   ((ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) * \
                    (ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1)) + 1) ** (3 / 2) * \
             (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2) * \
             np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1))

    return K
