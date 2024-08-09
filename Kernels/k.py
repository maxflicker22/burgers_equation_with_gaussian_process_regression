import numpy as np




def k(x, y, hyp, i):
    """
    Calculate the kernel function.

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
        # Evaluate the kernel function numerically
        arcsin_argument = np.clip((np.exp(logsigma) + np.exp(logtheta) * x * y) * \
                          ((1 + np.exp(logsigma) + np.exp(logtheta) * x ** 2) * \
                           (1 + np.exp(logsigma) + np.exp(logtheta) * y ** 2)) ** (-1 / 2),-1,1)

        K =  2 * np.pi ** (-1) * np.arcsin(arcsin_argument)

    elif i == 1:
        # Compute the derivative with respect to logsigma
        K = (-ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / \
             (np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2)) - \
             ehoch1 ** logsigma * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
              np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + \
             2 * ehoch1 ** logsigma / (np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                                       np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1))) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1))

    elif i == 2:
        # Compute the derivative with respect to logtheta
        K = (-ehoch1 ** logtheta * x ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / \
             ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) ** (3 / 2) * \
              np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + \
             2 * ehoch1 ** logtheta * x * y / (np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                                               np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) - \
             ehoch1 ** logtheta * y ** 2 * (ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) / \
             (np.sqrt(ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
              (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1) ** (3 / 2))) / \
            (pi * np.sqrt(-(ehoch1 ** logsigma + ehoch1 ** logtheta * x * y) ** 2 / \
                          ((ehoch1 ** logsigma + ehoch1 ** logtheta * x ** 2 + 1) * \
                           (ehoch1 ** logsigma + ehoch1 ** logtheta * y ** 2 + 1)) + 1))


    return K
